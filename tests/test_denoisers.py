import numpy as np
import pytest
from PIL import Image

from image_upscaler_tools import get_denoiser, list_denoisers
from image_upscaler_tools import ModelManager


CLASSICAL_DENOISERS = [
    "fast_nlm",
    "bilateral",
    "skimage_bilateral",
    "tv_chambolle",
    "wavelet",
    "median",
    "adaptive_median",
    "gaussian",
    "mean",
    "skimage_nlm",
]


@pytest.mark.parametrize("name", CLASSICAL_DENOISERS)
def test_classical_denoisers_preserve_shape_and_mode(name, dummy_rgb_image):
    denoiser = get_denoiser(name)
    out = denoiser.denoise(dummy_rgb_image)
    assert isinstance(out, Image.Image)
    assert out.size == dummy_rgb_image.size
    assert out.mode == "RGB"


@pytest.mark.parametrize("name", ["fast_nlm", "bilateral", "median", "adaptive_median"])
def test_denoisers_preserve_alpha(name, dummy_rgba_image):
    denoiser = get_denoiser(name)
    out = denoiser.denoise(dummy_rgba_image)
    assert out.size == dummy_rgba_image.size
    assert out.mode == "RGBA"
    # Alpha mask dimension matches
    assert out.getchannel("A").size == (64, 64)


def test_median_parameter_validation(dummy_rgb_image):
    with pytest.raises(ValueError):
        get_denoiser("median", ksize=4)  # even ksize invalid

    median = get_denoiser("median", ksize=3)
    with pytest.raises(ValueError):
        median.denoise(dummy_rgb_image, ksize=2)


def test_adaptive_median_noise_removal(noisy_impulse_image):
    arr_before = np.array(noisy_impulse_image)
    noise_count_before = np.sum((arr_before == 0) | (arr_before == 255))
    assert noise_count_before > 0

    amf = get_denoiser("adaptive_median", s_max=7)
    out = amf.denoise(noisy_impulse_image)
    arr_after = np.array(out)
    noise_count_after = np.sum((arr_after == 0) | (arr_after == 255))

    # Adaptive median should remove virtually all salt & pepper noise
    assert noise_count_after < noise_count_before * 0.05


def test_scunet_denoiser(dummy_rgb_image):
    if not ModelManager.has_weights("scunet"):
        pytest.skip("SCUNet weights not present locally")

    scunet = get_denoiser("scunet", device="cpu")
    out = scunet.denoise(dummy_rgb_image)
    assert out.size == dummy_rgb_image.size
    assert out.mode == "RGB"
