from pathlib import Path
from typing import Optional, Union
import numpy as np
import spandrel
import torch
from PIL import Image

from image_upscaler_tools.core.base import BaseDenoiser
from image_upscaler_tools.core.imageio import load_image, split_alpha, merge_alpha
from image_upscaler_tools.core.model_manager import ModelManager
from image_upscaler_tools.core.registry import register_denoiser
from image_upscaler_tools.core.tiling import tile_process_tensor


@register_denoiser("scunet", aliases=["scunet_real", "scunet_color_real_psnr"])
class ScunetDenoiser(BaseDenoiser):
    """
    SCUNet: Practical Blind Denoising via Swin-Conv-UNet.
    State-of-the-art deep neural network for complex real-world sensor noise and artifacts.
    """

    def __init__(
        self,
        device: str = "auto",
        model_path: Optional[str] = None,
        tile: Optional[int] = None,
        tile_overlap: int = 16,
        **kwargs
    ):
        super().__init__(device=device)
        self.model_path = model_path
        self.tile = tile
        self.tile_overlap = tile_overlap

    @property
    def name(self) -> str:
        return "scunet"

    def load_model(self) -> None:
        if self.model is not None:
            return
        checkpoint_path = ModelManager.get_model_path("scunet", custom_path=self.model_path)
        loader = spandrel.ModelLoader(device=self.device)
        descriptor = loader.load_from_file(checkpoint_path)
        self.model = descriptor.model.to(self.device).eval()

    def _process_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        # SCUNet requires input dimensions to be multiple of 32 for Swin window attention
        _, _, h, w = tensor.shape
        pad_h = (32 - (h % 32)) % 32
        pad_w = (32 - (w % 32)) % 32

        if pad_h > 0 or pad_w > 0:
            tensor_padded = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
            out = self.model(tensor_padded)
            return out[:, :, :h, :w]
        return self.model(tensor)

    def denoise(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        tile: Optional[int] = None,
        tile_overlap: Optional[int] = None,
        **kwargs
    ) -> Image.Image:
        pil_img = load_image(image, preserve_alpha=True)
        rgb_img, alpha_mask = split_alpha(pil_img)

        self.load_model()
        rgb_arr = np.array(rgb_img)
        tensor = self._to_tensor(rgb_arr)

        curr_tile = tile if tile is not None else self.tile
        curr_overlap = tile_overlap if tile_overlap is not None else self.tile_overlap

        with torch.no_grad():
            if curr_tile and curr_tile > 0:
                out_tensor = tile_process_tensor(
                    tensor,
                    self._process_tensor,
                    tile_size=curr_tile,
                    tile_overlap=curr_overlap,
                    scale=1
                )
            else:
                out_tensor = self._process_tensor(tensor)

        out_rgb = self._to_pil(out_tensor)
        return merge_alpha(out_rgb, alpha_mask)
