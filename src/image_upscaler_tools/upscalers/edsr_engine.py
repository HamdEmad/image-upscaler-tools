import os
from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from image_upscaler_tools.core.base import BaseUpscaler
from image_upscaler_tools.core.model_manager import ModelManager
from image_upscaler_tools.core.registry import register_upscaler


@register_upscaler("edsr", aliases=["edsr_x4"])
class EdsrUpscaler(BaseUpscaler):
    """
    EDSR 4x Super-Resolution Engine via OpenCV dnn_superres.
    Runs efficiently on CPU.
    """

    def __init__(self, device: str = "cpu", model_path: Optional[str] = None, **kwargs):
        super().__init__(device="cpu")
        self.model_path = model_path
        self._sr = None

    @property
    def name(self) -> str:
        return "edsr"

    @property
    def scale_factor(self) -> int:
        return 4

    def load_model(self) -> None:
        if self._sr is not None:
            return

        has_sr = False
        try:
            if hasattr(cv2, "dnn_superres") and hasattr(cv2.dnn_superres, "DnnSuperResImpl_create"):
                self._sr = cv2.dnn_superres.DnnSuperResImpl_create()
                has_sr = True
            elif hasattr(cv2, "dnn_superres") and hasattr(cv2.dnn_superres, "DnnSuperResImpl"):
                self._sr = cv2.dnn_superres.DnnSuperResImpl.create()
                has_sr = True
        except Exception:
            pass

        if not has_sr:
            raise RuntimeError(
                "EDSR engine requires OpenCV with dnn_superres support.\n"
                "Please install opencv-contrib-python: pip install opencv-contrib-python\n"
                "Or use other upscalers such as 'span', 'hat', 'realesrgan', or 'pil'."
            )

        checkpoint = ModelManager.get_model_path("edsr", custom_path=self.model_path)
        self._sr.readModel(checkpoint)
        self._sr.setModel("edsr", 4)

    def _process_pil(self, img: Image.Image, **kwargs) -> Image.Image:
        self.load_model()
        rgb_arr = np.array(img)
        bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        upscaled_bgr = self._sr.upsample(bgr)
        upscaled_rgb = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(upscaled_rgb)
