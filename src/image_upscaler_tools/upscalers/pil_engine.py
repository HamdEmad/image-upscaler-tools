from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image

from image_upscaler_tools.core.base import BaseUpscaler
from image_upscaler_tools.core.registry import register_upscaler


@register_upscaler("pil", aliases=["pil_lanczos", "lanczos", "bicubic"])
class PilUpscaler(BaseUpscaler):
    """
    Standard PIL Classical Interpolation Upscaler (Lanczos / Bicubic).
    Zero dependencies, immediate execution, flexible arbitrary scaling.
    """

    def __init__(self, method: str = "lanczos", **kwargs):
        super().__init__(device="cpu")
        self.method = method.lower().strip()

    @property
    def name(self) -> str:
        return "pil"

    @property
    def scale_factor(self) -> int:
        return 4

    def load_model(self) -> None:
        pass

    def _get_resample_filter(self) -> Image.Resampling:
        if self.method in ("bicubic", "cubic"):
            return Image.Resampling.BICUBIC
        elif self.method in ("bilinear", "linear"):
            return Image.Resampling.BILINEAR
        return Image.Resampling.LANCZOS

    def _process_pil(self, img: Image.Image, **kwargs) -> Image.Image:
        resample = self._get_resample_filter()
        w, h = img.size
        return img.resize((w * self.scale_factor, h * self.scale_factor), resample)
