from typing import Optional
import torch
from spandrel import ModelLoader
from image_upscaler_tools.core.base import BaseUpscaler
from image_upscaler_tools.core.model_manager import ModelManager
from image_upscaler_tools.core.registry import register_upscaler


@register_upscaler(name="ultrasharp", aliases=["4x_ultrasharp", "ultrasharp4x"])
class UltraSharpUpscaler(BaseUpscaler):
    """
    4x-UltraSharp Upscaler.
    Community fine-tuned ESRGAN model renowned for razor-sharp edge contrast.
    """

    def __init__(self, device: str = "auto", model_path: Optional[str] = None):
        super().__init__(device=device)
        self.model_path = model_path

    @property
    def name(self) -> str:
        return "4x-UltraSharp"

    def load_model(self) -> None:
        path = ModelManager.get_model_path("ultrasharp", self.model_path)
        loader = ModelLoader()
        model_descriptor = loader.load_from_file(path)
        self.model = model_descriptor.to(self.device)
        self.model.eval()

    def _process_tensor(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(tensor)
