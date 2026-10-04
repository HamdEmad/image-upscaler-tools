from typing import Any, Callable, Dict, List, Optional, Type
from image_upscaler_tools.core.base import BaseProcessor, BaseUpscaler, BaseDenoiser

_UPSCALER_REGISTRY: Dict[str, Type[BaseUpscaler]] = {}
_UPSCALER_ALIASES: Dict[str, str] = {}

_DENOISER_REGISTRY: Dict[str, Type[BaseDenoiser]] = {}
_DENOISER_ALIASES: Dict[str, str] = {}


def register_upscaler(name: str, aliases: Optional[List[str]] = None) -> Callable:
    """Decorator to register an upscaler class."""
    def decorator(cls: Type[BaseUpscaler]) -> Type[BaseUpscaler]:
        canonical = name.lower().strip()
        _UPSCALER_REGISTRY[canonical] = cls
        if aliases:
            for alias in aliases:
                _UPSCALER_ALIASES[alias.lower().strip()] = canonical
        return cls
    return decorator


def register_denoiser(name: str, aliases: Optional[List[str]] = None) -> Callable:
    """Decorator to register a denoiser class."""
    def decorator(cls: Type[BaseDenoiser]) -> Type[BaseDenoiser]:
        canonical = name.lower().strip()
        _DENOISER_REGISTRY[canonical] = cls
        if aliases:
            for alias in aliases:
                _DENOISER_ALIASES[alias.lower().strip()] = canonical
        return cls
    return decorator


def _ensure_upscalers_loaded() -> None:
    if not _UPSCALER_REGISTRY:
        import image_upscaler_tools.upscalers  # noqa: F401


def _ensure_denoisers_loaded() -> None:
    if not _DENOISER_REGISTRY:
        import image_upscaler_tools.denoisers  # noqa: F401


def is_upscaler(name: str) -> bool:
    """Check if name corresponds to a registered upscaler."""
    _ensure_upscalers_loaded()
    key = name.lower().strip()
    return key in _UPSCALER_REGISTRY or key in _UPSCALER_ALIASES


def is_denoiser(name: str) -> bool:
    """Check if name corresponds to a registered denoiser."""
    _ensure_denoisers_loaded()
    key = name.lower().strip()
    return key in _DENOISER_REGISTRY or key in _DENOISER_ALIASES


def list_upscalers() -> List[str]:
    """Return sorted list of all registered upscaler names."""
    _ensure_upscalers_loaded()
    return sorted(list(_UPSCALER_REGISTRY.keys()))


def list_denoisers() -> List[str]:
    """Return sorted list of all registered denoiser names."""
    _ensure_denoisers_loaded()
    return sorted(list(_DENOISER_REGISTRY.keys()))


def get_upscaler(name: str, device: str = "auto", **kwargs) -> BaseUpscaler:
    """Retrieve and instantiate an upscaler engine by name."""
    _ensure_upscalers_loaded()
    key = name.lower().strip()
    if key in _UPSCALER_ALIASES:
        key = _UPSCALER_ALIASES[key]

    if key not in _UPSCALER_REGISTRY:
        avail = list_upscalers()
        raise ValueError(f"Unknown upscaler '{name}'. Available: {avail}")

    cls = _UPSCALER_REGISTRY[key]
    return cls(device=device, **kwargs)


def get_denoiser(name: str, device: str = "auto", **kwargs) -> BaseDenoiser:
    """Retrieve and instantiate a denoiser engine by name."""
    _ensure_denoisers_loaded()
    key = name.lower().strip()
    if key in _DENOISER_ALIASES:
        key = _DENOISER_ALIASES[key]

    if key not in _DENOISER_REGISTRY:
        avail = list_denoisers()
        raise ValueError(f"Unknown denoiser '{name}'. Available: {avail}")

    cls = _DENOISER_REGISTRY[key]
    return cls(device=device, **kwargs)


def get_processor(name: str, device: str = "auto", **kwargs) -> BaseProcessor:
    """
    Unified lookup: returns either an upscaler or denoiser instance
    by inspecting both registries.
    """
    _ensure_upscalers_loaded()
    _ensure_denoisers_loaded()
    key = name.lower().strip()

    if is_denoiser(key):
        return get_denoiser(key, device=device, **kwargs)
    elif is_upscaler(key):
        return get_upscaler(key, device=device, **kwargs)
    else:
        all_avail = list_denoisers() + list_upscalers()
        raise ValueError(f"Unknown tool '{name}'. Available tools: {all_avail}")


def describe(name: str) -> Dict[str, Any]:
    """Return descriptive dictionary for any denoiser or upscaler."""
    from image_upscaler_tools.core.model_manager import ModelManager
    _ensure_upscalers_loaded()
    _ensure_denoisers_loaded()
    key = name.lower().strip()

    if is_denoiser(key):
        canonical = _DENOISER_ALIASES.get(key, key)
        cls = _DENOISER_REGISTRY[canonical]
        info = ModelManager.get_model_info(canonical) or {}
        return {
            "name": canonical,
            "category": "denoiser",
            "class": cls.__name__,
            "description": info.get("description", cls.__doc__ or "Denoiser"),
            "license": info.get("license", "Built-in algorithm"),
            "weights": info.get("filename", None)
        }
    elif is_upscaler(key):
        canonical = _UPSCALER_ALIASES.get(key, key)
        cls = _UPSCALER_REGISTRY[canonical]
        info = ModelManager.get_model_info(canonical) or {}
        return {
            "name": canonical,
            "category": "upscaler",
            "class": cls.__name__,
            "description": info.get("description", cls.__doc__ or "Upscaler"),
            "license": info.get("license", "Open Source"),
            "weights": info.get("filename", None)
        }
    raise ValueError(f"Unknown tool '{name}'")
