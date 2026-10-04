"""
Quickstart Guide for image-upscaler-tools
"""
from PIL import Image
from image_upscaler_tools import get_upscaler, list_upscalers, upscale_image

print("Available Models:", list_upscalers())

# Example 1: One-liner upscale with SPAN (fastest production model)
# out = upscale_image("input.jpg", model="span", target_size=(500, 500))
# out.save("output_span_500.jpg")

# Example 2: Reusable engine instance for high-throughput batch processing
upscaler = get_upscaler("span", device="auto")
upscaler.load_model()
print(f"Loaded: {upscaler.name} on {upscaler.device}")

# Process an in-memory PIL image
img = Image.new("RGB", (150, 150), color=(73, 109, 137))
result = upscaler.upscale(img, target_size=(500, 500))
print(f"Upscaled image size: {result.size}")

# Example 3: Flagship quality with HAT
# hat_upscaler = get_upscaler("hat", device="auto")
# high_fidelity_out = hat_upscaler.upscale("product.jpg", target_size=(500, 500))
