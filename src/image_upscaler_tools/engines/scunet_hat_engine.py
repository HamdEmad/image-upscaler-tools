from typing import Optional
import torch
from spandrel import ModelLoader
from image_upscaler_tools.core.base import BaseUpscaler
from image_upscaler_tools.core.model_manager import ModelManager
from image_upscaler_tools.core.registry import register_upscaler


@register_upscaler(name="scunet_before_hat", aliases=["scunet_hat", "scunet+hat", "scunet"])
class ScunetHatUpscaler(BaseUpscaler):
    """
    Two-Stage Restoration Pipeline:
    1. SCUNet (Swin-Conv UNet) Real-World Denoiser: cleans JPEG compression and sensor noise.
    2. HAT (Hybrid Attention Transformer): super-resolves the clean latent into high-fidelity 4x.
    """

    def __init__(
        self,
        device: str = "auto",
        scunet_model_path: Optional[str] = None,
        hat_model_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__(device=device)
        self.scunet_model_path = scunet_model_path
        self.hat_model_path = hat_model_path
        self.scunet_model = None
        self.hat_model = None

    @property
    def name(self) -> str:
        return "SCUNet Denoise + HAT 4x"

    def load_model(self) -> None:
        loader = ModelLoader()

        # 1. Load SCUNet Denoiser
        scunet_path = ModelManager.get_model_path("scunet", self.scunet_model_path)
        scunet_descriptor = loader.load_from_file(scunet_path)
        self.scunet_model = scunet_descriptor.to(self.device)
        self.scunet_model.eval()

        # 2. Load HAT Super-Resolution
        hat_path = ModelManager.get_model_path("hat", self.hat_model_path)
        hat_descriptor = loader.load_from_file(hat_path)
        self.hat_model = hat_descriptor.to(self.device)
        self.hat_model.eval()

        self.model = True

    def _process_tensor(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        # Stage 1: Denoise with SCUNet
        denoised = self.scunet_model(tensor)
        # Stage 2: Super-resolve with HAT
        sr_output = self.hat_model(denoised)
        return sr_output
