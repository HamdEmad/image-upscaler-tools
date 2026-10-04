import numpy as np
import pytest
from PIL import Image

from image_upscaler_tools import ImagePipeline, process_image
from image_upscaler_tools import ModelManager


def test_empty_pipeline_raises():
    pipe = ImagePipeline()
    with pytest.raises(ValueError):
        pipe.run(Image.new("RGB", (32, 32)))


def test_unknown_stage_raises():
    with pytest.raises(ValueError):
        ImagePipeline().add("non_existent_engine_name")


def test_mutually_exclusive_size_and_scale(dummy_rgb_image):
    pipe = ImagePipeline().add("median")
    with pytest.raises(ValueError):
        pipe.run(dummy_rgb_image, target_size=(100, 100), scale=2.0)


def test_denoise_only_pipeline(dummy_rgb_image):
    pipe = ImagePipeline().add("median", ksize=3).add("bilateral", d=3)
    out = pipe.run(dummy_rgb_image)
    assert out.size == dummy_rgb_image.size
    assert len(pipe.last_timings) == 2


def test_pipeline_chaining_and_timings(dummy_rgb_image):
    pipe = ImagePipeline()
    pipe.add("fast_nlm", h=2).add("pil").add("bilateral", d=3)
    out = pipe.run(dummy_rgb_image, target_size=(128, 128))

    assert out.size == (128, 128)
    assert len(pipe.last_timings) == 3
    assert pipe.last_timings[0][0] == "fast_nlm"
    assert pipe.last_timings[1][0] == "pil"
    assert pipe.last_timings[2][0] == "bilateral"
    assert pipe.total_time > 0


def test_from_string_parser(dummy_rgb_image):
    pipe = ImagePipeline.from_string("fast_nlm(h=2) > pil > bilateral(d=3)")
    out = pipe.run(dummy_rgb_image, target_size=(80, 80))
    assert out.size == (80, 80)
    assert len(pipe.last_timings) == 3


def test_from_preset(dummy_rgb_image):
    pipe = ImagePipeline.from_preset("catalog_fast")
    if not ModelManager.has_weights("span"):
        pytest.skip("SPAN weights not available for catalog_fast")
    out = pipe.run(dummy_rgb_image, target_size=(128, 128))
    assert out.size == (128, 128)


def test_process_image_functional(dummy_rgb_image):
    # Denoise only
    out1 = process_image(dummy_rgb_image, denoiser="fast_nlm")
    assert out1.size == dummy_rgb_image.size

    # Upscale only
    out2 = process_image(dummy_rgb_image, upscaler="pil", target_size=(100, 100))
    assert out2.size == (100, 100)

    # Order validation
    with pytest.raises(ValueError):
        process_image(dummy_rgb_image, denoiser="fast_nlm", order="upscale_first")
