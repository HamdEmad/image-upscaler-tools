from typing import Optional
import spandrel
import torch
from image_upscaler_tools.core.base import BaseUpscaler
from image_upscaler_tools.core.model_manager import ModelManager
from image_upscaler_tools.core.registry import register_upscaler


@register_upscaler("hat", aliases=["hat_srx4"])
class HatUpscaler(BaseUpscaler):
    """
    HAT 4x: Hybrid Attention Transformer Super-Resolution Engine.
    Combines channel attention and self-attention for maximum structural fidelity.
    """

    def __init__(self, device: str = "auto", model_path: Optional[str] = None, **kwargs):
        super().__init__(device=device)
        self.model_path = model_path

    @property
    def name(self) -> str:
        return "hat"

    @property
    def scale_factor(self) -> int:
        return 4

    def load_model(self) -> None:
        if self.model is not None:
            return
        checkpoint = ModelManager.get_model_path("hat", custom_path=self.model_path)
        loader = spandrel.ModelLoader(device=self.device)
        descriptor = loader.load_from_file(checkpoint)
        self.model = descriptor.model.to(self.device).eval()

    def _process_tensor(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        # Pad if needed to align with window size (typically 16 or 32)
        _, _, h, w = tensor.shape
        pad_h = (16 - (h % 16)) % 16
        pad_w = (16 - (w % 16)) % 16

        if pad_h > 0 or pad_w > 0:
            tensor_padded = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
            out = self.model(tensor_padded)
            return out[:, :, : h * 4, : w * 4]
        return self.model(tensor)
