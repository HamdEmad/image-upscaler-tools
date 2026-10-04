import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from image_upscaler_tools.core.base import BaseProcessor, BaseUpscaler, BaseDenoiser
from image_upscaler_tools.core.cache import EngineCache
from image_upscaler_tools.core.imageio import load_image, split_alpha, merge_alpha, save_image, apply_scaling
from image_upscaler_tools.core.registry import get_processor, is_denoiser, is_upscaler
from image_upscaler_tools.pipeline.parser import parse_step_string
from image_upscaler_tools.pipeline.presets import get_preset_steps


class ImagePipeline:
    """
    Composable multi-stage image processing pipeline.
    Execution order strictly matches the order of steps added.
    
    Can be used for:
    - Denoise-only (resolution preserved)
    - Upscale-only (raw super-resolution)
    - Denoise then upscale (pre-cleanup avoids noise amplification)
    - Upscale then denoise (post-smoothing residual artifacts)
    - Arbitrary multi-step chaining
    """

    def __init__(
        self,
        steps: Optional[List[Union[str, Tuple[str, Dict[str, Any]], BaseProcessor]]] = None,
        device: str = "auto"
    ):
        self.device = device
        self._steps: List[Tuple[Union[str, BaseProcessor], Dict[str, Any]]] = []
        self.last_timings: List[Tuple[str, float]] = []
        self.total_time: float = 0.0

        if steps:
            for s in steps:
                if isinstance(s, tuple) and len(s) == 2:
                    name_or_proc, params = s
                    self.add(name_or_proc, **params)
                else:
                    self.add(s)

    def add(self, step: Union[str, BaseProcessor], **params) -> "ImagePipeline":
        """
        Add a processing stage to the pipeline.
        
        Parameters
        ----------
        step : str or BaseProcessor instance
            Step name (e.g. 'fast_nlm', 'hat', 'median') or processor object.
        **params :
            Keyword arguments passed to the processor.
            
        Returns
        -------
        ImagePipeline
            Self, allowing fluent method chaining (.add().add().run()).
        """
        if isinstance(step, str):
            # Validate tool name exists
            canonical = step.lower().strip()
            # Test that it can be looked up
            get_processor(canonical, device=self.device)
            self._steps.append((canonical, params))
        elif isinstance(step, BaseProcessor):
            self._steps.append((step, params))
        else:
            raise TypeError(f"Expected step to be a string name or BaseProcessor, got {type(step)}")
        return self

    def __len__(self) -> int:
        return len(self._steps)

    @classmethod
    def from_preset(cls, preset_name: str, device: str = "auto") -> "ImagePipeline":
        """Instantiate a pipeline configured from a named preset."""
        preset_steps = get_preset_steps(preset_name)
        pipe = cls(device=device)
        for name, params in preset_steps:
            pipe.add(name, **params)
        return pipe

    @classmethod
    def from_string(cls, step_str: str, device: str = "auto") -> "ImagePipeline":
        """Instantiate a pipeline from a step string (e.g. 'fast_nlm(h=3) > hat')."""
        parsed = parse_step_string(step_str)
        pipe = cls(device=device)
        for name, params in parsed:
            pipe.add(name, **params)
        return pipe

    def _resolve_processor(
        self,
        item: Union[str, BaseProcessor],
        init_kwargs: Dict[str, Any]
    ) -> BaseProcessor:
        """Resolve processor instance using EngineCache."""
        if isinstance(item, BaseProcessor):
            return item

        name = item.lower().strip()
        cache = EngineCache()

        family = "denoiser" if is_denoiser(name) else "upscaler"
        return cache.get_or_create(
            family=family,
            name=name,
            factory=lambda: get_processor(name, device=self.device, **init_kwargs),
            device=self.device,
            **init_kwargs
        )

    def run(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        target_size: Optional[Tuple[int, int]] = None,
        scale: Optional[float] = None,
        fit: str = "stretch",
        save: Optional[Union[str, Path]] = None,
        tile: Optional[int] = None,
        tile_overlap: int = 16,
    ) -> Image.Image:
        """
        Execute the pipeline on an input image.

        Parameters
        ----------
        image : str, Path, PIL.Image, or np.ndarray
            Input image.
        target_size : tuple of (width, height), optional
            Exact final output size (applied once at the very end).
        scale : float, optional
            Multiplier for final resolution (applied once at the end).
        fit : str, default 'stretch'
            Fit strategy ('stretch', 'contain', 'cover').
        save : str or Path, optional
            If provided, saves output image to this filepath.
        tile : int, optional
            Tile size for neural engines (e.g. 256).
        tile_overlap : int, default 16
            Tile overlap for seamless blending.

        Returns
        -------
        PIL.Image.Image
            Final processed image.
        """
        if not self._steps:
            raise ValueError("Cannot run empty ImagePipeline. Add at least one step via pipeline.add(...).")

        if target_size is not None and scale is not None:
            raise ValueError("target_size and scale are mutually exclusive; specify only one.")

        # Check for multiple consecutive upscalers
        upscaler_count = sum(
            1 for s, _ in self._steps
            if (isinstance(s, str) and is_upscaler(s)) or isinstance(s, BaseUpscaler)
        )
        if upscaler_count > 1:
            warnings.warn(
                f"Pipeline contains {upscaler_count} upscaler stages. "
                "Output resolution may be large."
            )

        pil_img = load_image(image, preserve_alpha=True)
        orig_w, orig_h = pil_img.size
        if scale is not None:
            if target_size is not None:
                raise ValueError("target_size and scale are mutually exclusive; specify only one.")
            target_size = (max(1, int(round(orig_w * scale))), max(1, int(round(orig_h * scale))))
            scale = None

        current_img = pil_img

        self.last_timings = []
        overall_start = time.perf_counter()

        for step_idx, (step_item, step_params) in enumerate(self._steps):
            step_name = step_item.name if isinstance(step_item, BaseProcessor) else step_item
            step_start = time.perf_counter()

            # Merge per-step params with runtime tile params
            runtime_params = dict(step_params)
            if tile is not None and "tile" not in runtime_params:
                runtime_params["tile"] = tile
            if "tile_overlap" not in runtime_params:
                runtime_params["tile_overlap"] = tile_overlap

            processor = self._resolve_processor(step_item, step_params)

            if isinstance(processor, BaseDenoiser):
                # Denoiser preserves size
                current_img = processor.denoise(current_img, **runtime_params)
            elif isinstance(processor, BaseUpscaler):
                # Upscaler scales natively; final target_size is held until after the loop
                current_img = processor.upscale(
                    current_img,
                    target_size=None,
                    scale=None,
                    **runtime_params
                )
            else:
                raise RuntimeError(f"Unknown processor type: {type(processor)}")

            elapsed = time.perf_counter() - step_start
            self.last_timings.append((step_name, elapsed))

        # Apply target_size or scale ONCE at the end of the entire pipeline
        if target_size is not None or scale is not None:
            if upscaler_count == 0:
                warnings.warn(
                    "Pipeline contains only denoisers; applying Lanczos resize to reach target dimensions."
                )
            current_img = apply_scaling(
                current_img,
                target_size=target_size,
                scale=scale,
                fit=fit
            )

        self.total_time = time.perf_counter() - overall_start

        if save:
            save_image(current_img, save)

        return current_img

    def run_folder(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        target_size: Optional[Tuple[int, int]] = None,
        scale: Optional[float] = None,
        fit: str = "stretch",
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp"),
        tile: Optional[int] = None,
        tile_overlap: int = 16,
    ) -> List[Path]:
        """Process all images in a folder using the same loaded pipeline."""
        in_path = Path(input_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if not in_path.exists() or not in_path.is_dir():
            raise FileNotFoundError(f"Input directory not found: {in_path}")

        image_files = [p for p in in_path.iterdir() if p.suffix.lower() in extensions and p.is_file()]
        if not image_files:
            print(f"No matching images found in '{in_path}'.")
            return []

        print(f"Processing {len(image_files)} images from '{in_path}' -> '{out_path}'...")
        saved_paths = []
        total_folder_time = 0.0

        for idx, img_path in enumerate(image_files, 1):
            dest_file = out_path / img_path.name
            out_img = self.run(
                img_path,
                target_size=target_size,
                scale=scale,
                fit=fit,
                save=dest_file,
                tile=tile,
                tile_overlap=tile_overlap
            )
            saved_paths.append(dest_file)
            total_folder_time += self.total_time
            print(f"[{idx}/{len(image_files)}] {img_path.name} -> {dest_file.name} ({self.total_time:.2f}s)")

        avg_time = total_folder_time / max(1, len(image_files))
        print(f"Finished {len(saved_paths)} images in {total_folder_time:.2f}s (avg {avg_time:.2f}s/img).")
        return saved_paths
