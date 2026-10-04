from typing import Optional
import spandrel
import torch
from image_upscaler_tools.core.base import BaseUpscaler
from image_upscaler_tools.core.model_manager import ModelManager
from image_upscaler_tools.core.registry import register_upscaler


@register_upscaler("realesrgan", aliases=["realesrgan_x4plus", "real-esrgan"])
class RealEsrganUpscaler(BaseUpscaler):
    """
    Real-ESRGAN 4x Plus Super-Resolution Engine.
    Trained on higher-order degradation models for real-world photo restoration.
    """

    def __init__(self, device: str = "auto", model_path: Optional[str] = None, **kwargs):
        super().__init__(device=device)
        self.model_path = model_path

    @property
    def name(self) -> str:
        return "realesrgan"

    @property
    def scale_factor(self) -> int:
        return 4

    def load_model(self) -> None:
        if self.model is not None:
            return
        checkpoint = ModelManager.get_model_path("realesrgan", custom_path=self.model_path)
        loader = spandrel.ModelLoader(device=self.device)
        descriptor = loader.load_from_file(checkpoint)
        self.model = descriptor.model.to(self.device).eval()

    def _process_tensor(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(tensor)


@register_upscaler("realesrnet", aliases=["realesrnet_x4plus", "real-esrnet"])
class RealEsrnetUpscaler(BaseUpscaler):
    """
    Real-ESRNet 4x Plus Engine.
    Non-GAN version of Real-ESRGAN optimized for PSNR fidelity without hallucination.
    """

    def __init__(self, device: str = "auto", model_path: Optional[str] = None, **kwargs):
        super().__init__(device=device)
        self.model_path = model_path

    @property
    def name(self) -> str:
        return "realesrnet"

    @property
    def scale_factor(self) -> int:
        return 4

    def load_model(self) -> None:
        if self.model is not None:
            return
        checkpoint = ModelManager.get_model_path("realesrnet", custom_path=self.model_path)
        loader = spandrel.ModelLoader(device=self.device)
        descriptor = loader.load_from_file(checkpoint)
        self.model = descriptor.model.to(self.device).eval()

    def _process_tensor(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(tensor)
