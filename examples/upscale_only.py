"""
Upscale-Only Example:
Super-resolve directly using neural or classical upscaling engines.
"""

from PIL import Image
import numpy as np
from image_upscaler_tools import ImagePipeline, upscale_image

# Synthetic input image (64x64)
image = Image.fromarray(np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8))

# Method A: Upscale to exact dimensions with fit
up_500 = ImagePipeline().add("span").run(image, target_size=(500, 500), fit="contain")
print(f"Upscaled to 500x500: {up_500.size}")

# Method B: Upscale by relative multiplier
up_2x = ImagePipeline().add("span").run(image, scale=2.0)
print(f"Upscaled 2x: {up_2x.size}")

# Method C: Backward-compatible one-liner
up_pil = upscale_image(image, model="pil", target_size=(256, 256))
print(f"PIL Lanczos output: {up_pil.size}")
