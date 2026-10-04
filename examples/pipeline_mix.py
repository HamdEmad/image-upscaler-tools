"""
Pipeline Mix Example:
Demonstrating arbitrary chaining where call order = execution order.
"""

from PIL import Image
import numpy as np
from image_upscaler_tools import ImagePipeline, process_image

image = Image.fromarray(np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8))

# 1. Fluent Chaining: Denoise -> Upscale -> Denoise
pipeline = ImagePipeline()
pipeline.add("fast_nlm", h=3)
pipeline.add("span")
pipeline.add("bilateral", d=5)

output = pipeline.run(image, target_size=(500, 500), fit="stretch")
print(f"Mix pipeline output: {output.size}")
print("Per-stage execution timings:")
for name, elapsed in pipeline.last_timings:
    print(f"  - {name}: {elapsed:.3f}s")
print(f"Total time: {pipeline.total_time:.3f}s")

# 2. Step string syntax
step_out = process_image(image, steps="fast_nlm(h=2) > span", target_size=(400, 400))
print(f"Step string output: {step_out.size}")

# 3. Named Preset (Catalog Fast: fast_nlm > span)
preset_out = process_image(image, preset="catalog_fast", target_size=(500, 500))
print(f"Preset output: {preset_out.size}")
