from typing import Optional
import spandrel
import torch
from image_upscaler_tools.core.base import BaseUpscaler
from image_upscaler_tools.core.model_manager import ModelManager
from image_upscaler_tools.core.registry import register_upscaler


@register_upscaler("span", aliases=["4xpurephoto-span", "purephoto"])
class SpanUpscaler(BaseUpscaler):
    """
    SPAN 4x PurePhoto Super-Resolution Engine.
    Based on Swift Parameter-free Attention Network (CVPR NTIRE winner).
    Extremely fast, robust, and artifact-free.
    """

    def __init__(self, device: str = "auto", model_path: Optional[str] = None, **kwargs):
        super().__init__(device=device)
        self.model_path = model_path

    @property
    def name(self) -> str:
        return "span"

    @property
    def scale_factor(self) -> int:
        return 4

    def load_model(self) -> None:
        if self.model is not None:
            return
        checkpoint = ModelManager.get_model_path("span", custom_path=self.model_path)
        loader = spandrel.ModelLoader(device=self.device)
        descriptor = loader.load_from_file(checkpoint)
        self.model = descriptor.model.to(self.device).eval()

    def _process_tensor(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(tensor)
