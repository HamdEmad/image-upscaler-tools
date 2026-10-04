from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image

from image_upscaler_tools.pipeline.pipeline import ImagePipeline


def process_image(
    image: Union[str, Path, Image.Image],
    denoiser: Optional[str] = None,
    upscaler: Optional[str] = None,
    order: str = "denoise_first",
    steps: Optional[str] = None,
    preset: Optional[str] = None,
    target_size: Optional[Tuple[int, int]] = None,
    scale: Optional[float] = None,
    fit: str = "stretch",
    denoiser_params: Optional[Dict[str, Any]] = None,
    upscaler_params: Optional[Dict[str, Any]] = None,
    device: str = "auto",
    tile: Optional[int] = None,
    tile_overlap: int = 16,
    save: Optional[Union[str, Path]] = None,
) -> Image.Image:
    """
    High-level functional API for processing images with denoisers and upscalers.
    
    Parameters
    ----------
    image : str, Path, or PIL.Image
        Input image.
    denoiser : str, optional
        Denoiser name (e.g. 'fast_nlm', 'bilateral', 'median', 'scunet').
    upscaler : str, optional
        Upscaler name (e.g. 'span', 'hat', 'realesrgan', 'ultrasharp', 'pil').
    order : str, default 'denoise_first'
        Execution order when both denoiser and upscaler are provided:
        'denoise_first' or 'upscale_first'.
    steps : str, optional
        Explicit step string, e.g. 'median > bilateral > span'.
        Mutually exclusive with denoiser/upscaler/preset.
    preset : str, optional
        Named preset, e.g. 'catalog_fast', 'catalog_premium', 'restore_heavy'.
        Mutually exclusive with denoiser/upscaler/steps.
    target_size : tuple of (width, height), optional
        Output dimensions.
    scale : float, optional
        Scale factor.
    fit : str, default 'stretch'
        Fit strategy ('stretch', 'contain', 'cover').
    denoiser_params : dict, optional
        Specific parameters for the denoiser stage.
    upscaler_params : dict, optional
        Specific parameters for the upscaler stage.
    device : str, default 'auto'
        Execution device ('auto', 'cpu', 'cuda').
    tile : int, optional
        Tile size for neural inference.
    tile_overlap : int, default 16
        Tile overlap in pixels.
    save : str or Path, optional
        Output path to write image.

    Returns
    -------
    PIL.Image.Image
        Processed output image.
    """
    # Mutual exclusivity checks
    specified_modes = sum(bool(x) for x in [steps is not None, preset is not None, (denoiser or upscaler)])
    if specified_modes > 1:
        raise ValueError("Parameters 'steps', 'preset', and '(denoiser/upscaler)' are mutually exclusive.")

    den_kwargs = denoiser_params or {}
    up_kwargs = upscaler_params or {}

    if preset:
        pipeline = ImagePipeline.from_preset(preset, device=device)

    elif steps:
        pipeline = ImagePipeline.from_string(steps, device=device)

    elif denoiser and upscaler:
        norm_order = order.lower().strip()
        pipeline = ImagePipeline(device=device)
        if norm_order == "denoise_first":
            pipeline.add(denoiser, **den_kwargs)
            pipeline.add(upscaler, **up_kwargs)
        elif norm_order == "upscale_first":
            pipeline.add(upscaler, **up_kwargs)
            pipeline.add(denoiser, **den_kwargs)
        else:
            raise ValueError(f"Unknown order '{order}'. Use 'denoise_first' or 'upscale_first'.")

    elif denoiser:
        if order != "denoise_first":
            raise ValueError(
                f"Specifying 'order={order}' with only a denoiser is invalid."
            )
        pipeline = ImagePipeline(device=device)
        pipeline.add(denoiser, **den_kwargs)

    elif upscaler:
        if order != "denoise_first":
            raise ValueError(
                f"Specifying 'order={order}' with only an upscaler is invalid."
            )
        pipeline = ImagePipeline(device=device)
        pipeline.add(upscaler, **up_kwargs)

    else:
        raise ValueError(
            "Must provide at least one of: 'denoiser', 'upscaler', 'steps', or 'preset'."
        )

    return pipeline.run(
        image=image,
        target_size=target_size,
        scale=scale,
        fit=fit,
        save=save,
        tile=tile,
        tile_overlap=tile_overlap
    )


def upscale_image(
    image: Union[str, Path, Image.Image],
    model: str = "span",
    target_size: Optional[Tuple[int, int]] = None,
    device: str = "auto",
    tile: Optional[int] = None,
    **kwargs
) -> Image.Image:
    """
    Backward-compatible convenience function for upscaling.
    """
    # Special handle legacy 'scunet_before_hat'
    if model.lower().strip() == "scunet_before_hat":
        return process_image(
            image=image,
            denoiser="scunet",
            upscaler="hat",
            order="denoise_first",
            target_size=target_size,
            device=device,
            tile=tile,
            **kwargs
        )

    return process_image(
        image=image,
        upscaler=model,
        target_size=target_size,
        device=device,
        tile=tile,
        upscaler_params=kwargs
    )
