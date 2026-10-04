import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np
import torch
from PIL import Image

from image_upscaler_tools.core.imageio import load_image, split_alpha, merge_alpha, resize_with_fit, apply_scaling
from image_upscaler_tools.core.tiling import tile_process_tensor


class BaseProcessor(ABC):
    """
    Abstract Base Class for both Upscalers and Denoisers.
    Standardizes device selection, tensor/PIL conversions, and output resizing.
    """

    def __init__(self, device: str = "auto"):
        self.device = self._resolve_device(device)
        self.model = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical name of the processor."""
        pass

    @property
    def scale_factor(self) -> int:
        """Spatial resolution scaling factor (1 for denoisers, typically 4 for upscalers)."""
        return 1

    def _resolve_device(self, device_str: str) -> torch.device:
        """Select appropriate PyTorch execution device."""
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)

    def _to_tensor(self, rgb_array: np.ndarray) -> torch.Tensor:
        """Convert [H, W, C] uint8 RGB array to [1, C, H, W] float32 tensor on device."""
        tensor = torch.from_numpy(rgb_array).float() / 255.0
        return tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

    def _to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert [1, C, H, W] float32 tensor back to PIL Image."""
        t = tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        uint8_array = (t * 255.0).round().astype(np.uint8)
        return Image.fromarray(uint8_array)

    def _resize_with_fit(
        self,
        pil_img: Image.Image,
        target_size: Tuple[int, int],
        fit: str = "stretch"
    ) -> Image.Image:
        return resize_with_fit(pil_img, target_size, fit=fit)

    def _apply_scaling(
        self,
        img: Image.Image,
        target_size: Optional[Tuple[int, int]] = None,
        scale: Optional[float] = None,
        fit: str = "stretch"
    ) -> Image.Image:
        return apply_scaling(img, target_size=target_size, scale=scale, fit=fit)


class BaseUpscaler(BaseProcessor):
    """
    Abstract Base Class for all super-resolution engines.
    """

    @property
    def scale_factor(self) -> int:
        return 4

    def load_model(self) -> None:
        """Load weights onto device (if model uses weights)."""
        pass

    def _process_tensor(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """Default tensor forward pass. Subclasses should override."""
        raise NotImplementedError

    def _process_pil(self, img: Image.Image, **kwargs) -> Image.Image:
        """Fallback PIL processing for non-neural / OpenCV engines."""
        raise NotImplementedError

    def upscale(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        target_size: Optional[Tuple[int, int]] = None,
        scale: Optional[float] = None,
        fit: str = "stretch",
        tile: Optional[int] = None,
        tile_overlap: int = 16,
        **kwargs
    ) -> Image.Image:
        """
        Upscale an input image. Preserves alpha channel if present.
        """
        pil_img = load_image(image, preserve_alpha=True)
        orig_w, orig_h = pil_img.size
        if scale is not None:
            if target_size is not None:
                raise ValueError("target_size and scale are mutually exclusive; specify only one.")
            target_size = (max(1, int(round(orig_w * scale))), max(1, int(round(orig_h * scale))))
            scale = None

        rgb_img, alpha_mask = split_alpha(pil_img)

        # Execute upscaling
        if type(self)._process_pil != BaseUpscaler._process_pil:
            out_rgb = self._process_pil(rgb_img, **kwargs)
        else:
            if self.model is None:
                self.load_model()
            rgb_arr = np.array(rgb_img)
            tensor = self._to_tensor(rgb_arr)
            with torch.no_grad():
                if tile and tile > 0:
                    out_tensor = tile_process_tensor(
                        tensor,
                        lambda t: self._process_tensor(t, **kwargs),
                        tile_size=tile,
                        tile_overlap=tile_overlap,
                        scale=self.scale_factor
                    )
                else:
                    out_tensor = self._process_tensor(tensor, **kwargs)
            out_rgb = self._to_pil(out_tensor)

        # Merge alpha if previously present
        out_img = merge_alpha(out_rgb, alpha_mask)

        # Apply target_size or scale if requested
        return self._apply_scaling(out_img, target_size=target_size, scale=scale, fit=fit)


class BaseDenoiser(BaseProcessor):
    """
    Abstract Base Class for all image denoisers.
    Denoising strictly preserves input dimensions.
    """

    @property
    def scale_factor(self) -> int:
        return 1

    def load_model(self) -> None:
        pass

    @abstractmethod
    def denoise(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        **kwargs
    ) -> Image.Image:
        """Clean noise from the input image, preserving resolution and alpha."""
        pass
