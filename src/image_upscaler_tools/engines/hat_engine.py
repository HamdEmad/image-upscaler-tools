from typing import Optional
import torch
from spandrel import ModelLoader
from image_upscaler_tools.core.base import BaseUpscaler
from image_upscaler_tools.core.model_manager import ModelManager
from image_upscaler_tools.core.registry import register_upscaler


@register_upscaler(name="hat", aliases=["hat_srx4", "hat_x4", "hybrid_attention"])
class HatUpscaler(BaseUpscaler):
    """
    Hybrid Attention Transformer (HAT) 4x Upscaler.
    State-of-the-art transformer architecture combining channel attention
    with window-based self-attention for highest fidelity micro-texture reconstruction.
    """

    def __init__(self, device: str = "auto", model_path: Optional[str] = None):
        super().__init__(device=device)
        self.model_path = model_path

    @property
    def name(self) -> str:
        return "HAT (Hybrid Attention Transformer)"

    def load_model(self) -> None:
        path = ModelManager.get_model_path("hat", self.model_path)
        loader = ModelLoader()
        model_descriptor = loader.load_from_file(path)
        self.model = model_descriptor.to(self.device)
        self.model.eval()

    def _process_tensor(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(tensor)
