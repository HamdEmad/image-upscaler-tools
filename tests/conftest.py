import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def dummy_rgb_image():
    """Create a 64x64 synthetic RGB test image."""
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[16:48, 16:48] = [200, 100, 50]
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def dummy_rgba_image():
    """Create a 64x64 synthetic RGBA test image with transparency."""
    arr = np.zeros((64, 64, 4), dtype=np.uint8)
    arr[16:48, 16:48] = [200, 100, 50, 200]
    return Image.fromarray(arr, mode="RGBA")


@pytest.fixture
def noisy_impulse_image():
    """Create an image corrupted with salt-and-pepper impulse noise."""
    np.random.seed(42)
    arr = np.ones((64, 64, 3), dtype=np.uint8) * 128
    noise = np.random.rand(64, 64)
    arr[noise < 0.05] = 0
    arr[noise > 0.95] = 255
    return Image.fromarray(arr, mode="RGB")
