import numpy as np
import pytest
from PIL import Image

from image_upscaler_tools.core.imageio import (
    load_image,
    save_image,
    split_alpha,
    merge_alpha,
    resize_with_fit
)


def test_load_numpy_arrays():
    # 2D Grayscale
    arr2d = np.zeros((32, 32), dtype=np.uint8)
    img = load_image(arr2d)
    assert img.mode == "RGB"
    assert img.size == (32, 32)

    # 3D RGB Float
    arr_float = np.ones((32, 32, 3), dtype=np.float32) * 0.5
    img_f = load_image(arr_float)
    assert img_f.mode == "RGB"
    assert np.allclose(np.array(img_f)[0, 0], [128, 128, 128], atol=2)

    # 16-bit uint16
    arr16 = np.ones((32, 32, 3), dtype=np.uint16) * 32768
    with pytest.warns(UserWarning):
        img16 = load_image(arr16)
    assert img16.mode == "RGB"


def test_split_and_merge_alpha():
    rgba = Image.new("RGBA", (64, 64), (100, 150, 200, 128))
    rgb, alpha = split_alpha(rgba)
    assert rgb.mode == "RGB"
    assert alpha is not None
    assert alpha.size == (64, 64)

    # Upscale rgb to (128, 128) and recombine
    rgb_large = rgb.resize((128, 128))
    merged = merge_alpha(rgb_large, alpha)
    assert merged.mode == "RGBA"
    assert merged.size == (128, 128)
    assert merged.getchannel("A").size == (128, 128)


def test_save_jpeg_rgba_white_background(tmp_path):
    rgba = Image.new("RGBA", (32, 32), (255, 0, 0, 0))  # Fully transparent red
    out_file = tmp_path / "test.jpg"
    save_image(rgba, out_file)
    assert out_file.exists()

    # Loaded image should have white background, not black
    reloaded = Image.open(out_file)
    corner_pixel = reloaded.getpixel((0, 0))
    assert corner_pixel == (255, 255, 255)
