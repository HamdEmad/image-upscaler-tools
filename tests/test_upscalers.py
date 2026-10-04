import pytest
from PIL import Image

from image_upscaler_tools import get_upscaler, list_upscalers
from image_upscaler_tools import ModelManager


def test_pil_upscaler(dummy_rgb_image):
    pil_up = get_upscaler("pil")
    assert pil_up.scale_factor == 4

    # Native 4x
    out_native = pil_up.upscale(dummy_rgb_image)
    assert out_native.size == (256, 256)

    # Custom target_size
    out_target = pil_up.upscale(dummy_rgb_image, target_size=(100, 100))
    assert out_target.size == (100, 100)

    # Custom scale
    out_scale = pil_up.upscale(dummy_rgb_image, scale=2.0)
    assert out_scale.size == (128, 128)


def test_pil_fit_contain(dummy_rgb_image):
    pil_up = get_upscaler("pil")
    out = pil_up.upscale(dummy_rgb_image, target_size=(200, 100), fit="contain")
    assert out.size == (200, 100)


def test_pil_fit_cover(dummy_rgb_image):
    pil_up = get_upscaler("pil")
    out = pil_up.upscale(dummy_rgb_image, target_size=(200, 100), fit="cover")
    assert out.size == (200, 100)


@pytest.mark.parametrize("model_name", ["span", "hat", "realesrgan", "realesrnet", "ultrasharp"])
def test_neural_upscalers(model_name, dummy_rgb_image):
    if not ModelManager.has_weights(model_name):
        pytest.skip(f"Weights for {model_name} not available locally")

    engine = get_upscaler(model_name, device="cpu")
    out = engine.upscale(dummy_rgb_image, target_size=(128, 128))
    assert isinstance(out, Image.Image)
    assert out.size == (128, 128)
