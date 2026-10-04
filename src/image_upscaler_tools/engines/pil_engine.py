from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image
from image_upscaler_tools.core.base import BaseUpscaler
from image_upscaler_tools.core.registry import register_upscaler


@register_upscaler(name="pil", aliases=["lanczos", "bicubic"])
class PilUpscaler(BaseUpscaler):
    """Classical mathematical interpolation upscaler using PIL Lanczos filter."""

    @property
    def name(self) -> str:
        return "PIL (Lanczos)"

    def load_model(self) -> None:
        # PIL requires no external model weights
        self.model = True

    def _process_tensor(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        # Not used because upscale() is optimized directly on PIL Image
        return tensor

    def upscale(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        target_size: Optional[Tuple[int, int]] = None,
        **kwargs
    ) -> Image.Image:
        """Upscale image using PIL Lanczos resampling."""
        if isinstance(image, (str, Path)):
            pil_img = Image.open(str(image)).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        elif isinstance(image, np.ndarray):
            rgb = self._normalize_input(image)
            pil_img = Image.fromarray(rgb)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        w, h = pil_img.size
        if target_size is not None:
            final_w, final_h = target_size
        else:
            final_w, final_h = w * self.scale_factor, h * self.scale_factor

        return pil_img.resize((final_w, final_h), Image.Resampling.LANCZOS)
