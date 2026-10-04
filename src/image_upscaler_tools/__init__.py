"""
image-upscaler-tools: A modular, composable Python toolkit for image denoising (11 methods)
and super-resolution upscaling (7 engines).
"""

from image_upscaler_tools.core.base import BaseProcessor, BaseUpscaler, BaseDenoiser
from image_upscaler_tools.core.imageio import load_image, save_image, split_alpha, merge_alpha
from image_upscaler_tools.core.model_manager import (
    ModelManager,
    WeightsNotFoundError,
    WeightsVerificationError,
)
from image_upscaler_tools.core.registry import (
    get_upscaler,
    get_denoiser,
    get_processor,
    list_upscalers,
    list_denoisers,
    register_upscaler,
    register_denoiser,
    is_upscaler,
    is_denoiser,
    describe,
)
from image_upscaler_tools.pipeline.pipeline import ImagePipeline
from image_upscaler_tools.pipeline.orchestrator import process_image, upscale_image
from image_upscaler_tools.pipeline.presets import list_presets, get_preset_steps
from image_upscaler_tools.pipeline.parser import parse_step_string

# Ensure denoisers and upscalers are loaded and registered
import image_upscaler_tools.denoisers  # noqa: F401
import image_upscaler_tools.upscalers  # noqa: F401

__version__ = "0.2.0"

__all__ = [
    "__version__",
    # Primary Composable API
    "ImagePipeline",
    "process_image",
    "upscale_image",
    # Factory & Discovery
    "get_denoiser",
    "get_upscaler",
    "get_processor",
    "list_denoisers",
    "list_upscalers",
    "list_presets",
    "get_preset_steps",
    "parse_step_string",
    "describe",
    "is_denoiser",
    "is_upscaler",
    # Core Classes & Utilities
    "BaseProcessor",
    "BaseUpscaler",
    "BaseDenoiser",
    "ModelManager",
    "WeightsNotFoundError",
    "WeightsVerificationError",
    "load_image",
    "save_image",
    "split_alpha",
    "merge_alpha",
    "register_upscaler",
    "register_denoiser",
]
