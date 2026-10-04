from typing import Optional
import cv2
import numpy as np
import torch
from spandrel import ModelLoader
from image_upscaler_tools.core.base import BaseUpscaler
from image_upscaler_tools.core.model_manager import ModelManager
from image_upscaler_tools.core.registry import register_upscaler


@register_upscaler(name="span", aliases=["span_x4", "purephoto", "4xpurephoto"])
class SpanUpscaler(BaseUpscaler):
    """
    Swift Parameter-free Attention Network (SPAN) 4x Upscaler.
    Won 1st place in CVPR 2024 NTIRE Efficient Super-Resolution Challenge.
    Lightweight (~8.6 MB), sub-second CPU inference, with zero cartoon artifacts.
    """

    def __init__(self, device: str = "auto", model_path: Optional[str] = None):
        super().__init__(device=device)
        self.model_path = model_path

    @property
    def name(self) -> str:
        return "SPAN (4xPurePhoto)"

    def load_model(self) -> None:
        path = ModelManager.get_model_path("span", self.model_path)
        loader = ModelLoader()
        model_descriptor = loader.load_from_file(path)
        self.model = model_descriptor.to(self.device)
        self.model.eval()

    def _process_tensor(self, tensor: torch.Tensor, pre_denoise: bool = False, **kwargs) -> torch.Tensor:
        if pre_denoise:
            # Gentle background noise suppression before super-resolution
            np_rgb = (tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            bgr = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)
            denoised_bgr = cv2.fastNlMeansDenoisingColored(bgr, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21)
            denoised_rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(denoised_rgb).float() / 255.0
            tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        return self.model(tensor)
