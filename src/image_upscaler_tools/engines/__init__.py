# Backward compatibility re-export shim
from image_upscaler_tools.upscalers import (
    BaseUpscaler,
    SpanUpscaler,
    HatUpscaler,
    RealEsrganUpscaler,
    RealEsrnetUpscaler,
    UltraSharpUpscaler,
    EdsrUpscaler,
    PilUpscaler,
)

__all__ = [
    "BaseUpscaler",
    "SpanUpscaler",
    "HatUpscaler",
    "RealEsrganUpscaler",
    "RealEsrnetUpscaler",
    "UltraSharpUpscaler",
    "EdsrUpscaler",
    "PilUpscaler",
]
