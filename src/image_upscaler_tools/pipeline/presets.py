from typing import Any, Dict, List, Tuple

_PRESET_DEFINITIONS: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {
    "catalog_fast": [
        ("fast_nlm", {"h": 3.0, "h_color": 3.0}),
        ("span", {})
    ],
    "catalog_premium": [
        ("fast_nlm", {"h": 3.0, "h_color": 3.0}),
        ("hat", {})
    ],
    "restore_heavy": [
        ("scunet", {}),
        ("hat", {})
    ],
    "gentle_bilateral_hat": [
        ("skimage_bilateral", {"sigma_color": 0.03, "sigma_spatial": 5.0}),
        ("hat", {})
    ],
    "impulse_clean_fast": [
        ("adaptive_median", {"s_max": 7}),
        ("span", {})
    ],
    # Anime & Face Illustration Presets
    "anime_crisp": [
        ("scunet", {}),
        ("ultrasharp", {}),
        ("bilateral", {"d": 3, "sigma_color": 15, "sigma_space": 15})
    ],
    "anime_smooth": [
        ("scunet", {}),
        ("realesrgan", {}),
        ("bilateral", {"d": 3, "sigma_color": 15, "sigma_space": 15})
    ],
    "anime_psnr": [
        ("scunet", {}),
        ("realesrnet", {})
    ],
    # Hardware, PCB & Macro Photo Presets
    "hardware_text": [
        ("scunet", {}),
        ("hat", {}),
        ("bilateral", {"d": 3, "sigma_color": 15, "sigma_space": 15})
    ],
    "hardware_clean": [
        ("scunet", {}),
        ("span", {}),
        ("bilateral", {"d": 3, "sigma_color": 15, "sigma_space": 15})
    ],
    "hardware_fast": [
        ("wavelet", {"method": "BayesShrink", "mode": "soft"}),
        ("span", {}),
        ("bilateral", {"d": 3, "sigma_color": 15, "sigma_space": 15})
    ]
}


def list_presets() -> List[str]:
    """Return list of all registered preset names."""
    return sorted(list(_PRESET_DEFINITIONS.keys()))


def get_preset_steps(name: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Return step definitions for a given preset."""
    key = name.lower().strip()
    if key not in _PRESET_DEFINITIONS:
        avail = list_presets()
        raise ValueError(f"Unknown preset '{name}'. Available presets: {avail}")
    return _PRESET_DEFINITIONS[key]
