import threading
from typing import Any, Callable, Dict, Optional, Tuple


class EngineCache:
    """Thread-safe LRU / singleton cache for initialized model instances."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._cache: Dict[Tuple, Any] = {}
        return cls._instance

    def _make_key(self, family: str, name: str, device: str, kwargs: dict) -> Tuple:
        kw_items = tuple(sorted((k, str(v)) for k, v in kwargs.items() if k != "model_path"))
        model_path = kwargs.get("model_path")
        return (family.lower(), name.lower(), str(device), model_path, kw_items)

    def get_or_create(
        self,
        family: str,
        name: str,
        factory: Callable[[], Any],
        device: str = "auto",
        **kwargs
    ) -> Any:
        """Retrieve existing cached instance or instantiate via factory."""
        key = self._make_key(family, name, device, kwargs)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            instance = factory()
            self._cache[key] = instance
            return instance

    def clear(self) -> None:
        """Clear all cached model instances and release resources."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)
