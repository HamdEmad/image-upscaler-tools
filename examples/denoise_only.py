"""
Denoise-Only Example:
Clean noise without altering image dimensions.
"""

from PIL import Image
import numpy as np
from image_upscaler_tools import ImagePipeline, get_denoiser, process_image

# Generate a synthetic noisy test image
image = Image.fromarray(np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8))

# Method A: Composable ImagePipeline
clean_img = ImagePipeline().add("fast_nlm", h=3).run(image)
print(f"Cleaned image size: {clean_img.size} (resolution preserved)")

# Method B: Pre-configured denoiser instance
denoiser = get_denoiser("adaptive_median", s_max=7)
clean_amf = ImagePipeline().add(denoiser).run(image)
print(f"Adaptive median cleaned size: {clean_amf.size}")

# Method C: Functional one-liner
clean_func = process_image(image, denoiser="bilateral", denoiser_params={"d": 5})
print(f"Bilateral cleaned size: {clean_func.size}")
