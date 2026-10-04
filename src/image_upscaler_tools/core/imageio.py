import os
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image, ImageOps


def load_image(
    image: Union[str, Path, Image.Image, np.ndarray],
    preserve_alpha: bool = True
) -> Image.Image:
    """
    Load an image from a filepath, PIL Image, or NumPy array.
    
    Features:
    - Auto-applies EXIF orientation transposition.
    - Handles 16-bit uint16 / float images with graceful downsampling and warning.
    - Preserves RGBA alpha channel if present (or converts to RGB if preserve_alpha=False).
    - Converts grayscale 1-channel to RGB/RGBA.
    """
    if isinstance(image, (str, Path)):
        p = Path(image)
        if not p.exists():
            raise FileNotFoundError(f"Input image not found: {p}")
        pil_img = Image.open(p)
        pil_img = ImageOps.exif_transpose(pil_img)
        if pil_img.mode in ("RGBA", "LA") and preserve_alpha:
            return pil_img.convert("RGBA")
        elif pil_img.mode == "L":
            return pil_img.convert("RGB")
        elif pil_img.mode not in ("RGB", "RGBA"):
            return pil_img.convert("RGB")
        return pil_img

    elif isinstance(image, Image.Image):
        pil_img = ImageOps.exif_transpose(image)
        if pil_img.mode in ("RGBA", "LA") and preserve_alpha:
            return pil_img.convert("RGBA")
        elif pil_img.mode == "L":
            return pil_img.convert("RGB")
        elif pil_img.mode not in ("RGB", "RGBA"):
            return pil_img.convert("RGB")
        return pil_img

    elif isinstance(image, np.ndarray):
        arr = image.copy()
        if arr.dtype == np.uint16:
            warnings.warn("Converting 16-bit image array to 8-bit for processing.")
            arr = (arr / 256.0).clip(0, 255).astype(np.uint8)
        elif np.issubdtype(arr.dtype, np.floating):
            if arr.max() <= 1.01:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)

        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L").convert("RGB")
        elif arr.ndim == 3:
            if arr.shape[2] == 1:
                return Image.fromarray(arr[:, :, 0], mode="L").convert("RGB")
            elif arr.shape[2] == 3:
                return Image.fromarray(arr, mode="RGB")
            elif arr.shape[2] == 4:
                if preserve_alpha:
                    return Image.fromarray(arr, mode="RGBA")
                else:
                    return Image.fromarray(arr[:, :, :3], mode="RGB")
        raise ValueError(f"Unsupported NumPy array shape for image: {image.shape}")

    else:
        raise TypeError(f"Unsupported image input type: {type(image)}")


def split_alpha(img: Image.Image) -> Tuple[Image.Image, Optional[Image.Image]]:
    """
    Split a PIL Image into RGB content and an optional Alpha mask.
    Returns:
        (rgb_image, alpha_channel_or_None)
    """
    if img.mode == "RGBA":
        r, g, b, a = img.split()
        rgb = Image.merge("RGB", (r, g, b))
        return rgb, a
    return img.convert("RGB"), None


def merge_alpha(rgb_img: Image.Image, alpha: Optional[Image.Image]) -> Image.Image:
    """
    Merge an RGB PIL Image with an Alpha channel.
    If the dimensions differ, alpha is resized to match rgb_img using LANCZOS.
    """
    if alpha is None:
        return rgb_img

    if alpha.size != rgb_img.size:
        alpha = alpha.resize(rgb_img.size, Image.Resampling.LANCZOS)

    r, g, b = rgb_img.convert("RGB").split()
    return Image.merge("RGBA", (r, g, b, alpha))


def save_image(
    img: Image.Image,
    output_path: Union[str, Path],
    quality: int = 95,
    **kwargs
) -> Path:
    """
    Save a PIL Image to disk with safe defaults.
    - If output format is JPEG and image has alpha, composites over white background.
    - Creates parent directories automatically.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    ext = out.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        if img.mode in ("RGBA", "LA"):
            # Composite over white background to avoid black background artifacts
            bg = Image.new("RGB", img.size, (255, 255, 255))
            alpha = img.getchannel("A") if "A" in img.getbands() else None
            if alpha:
                bg.paste(img.convert("RGB"), mask=alpha)
            else:
                bg.paste(img.convert("RGB"))
            bg.save(out, quality=quality, **kwargs)
            return out
        img.convert("RGB").save(out, quality=quality, **kwargs)
    else:
        img.save(out, **kwargs)
    return out


def resize_with_fit(
    pil_img: Image.Image,
    target_size: Tuple[int, int],
    fit: str = "stretch"
) -> Image.Image:
    """
    Resize PIL Image to target_size (width, height) using the requested fit policy:
    - 'stretch': exact non-uniform scaling (default).
    - 'contain': preserve aspect ratio, fit inside target_size, pad borders.
    - 'cover': preserve aspect ratio, fill target_size, center crop.
    """
    target_w, target_h = target_size
    orig_w, orig_h = pil_img.size

    if fit == "stretch":
        return pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

    elif fit == "contain":
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        mode = pil_img.mode
        color = (0, 0, 0, 0) if mode == "RGBA" else (255, 255, 255)
        canvas = Image.new(mode, (target_w, target_h), color)
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        canvas.paste(resized, (paste_x, paste_y))
        return canvas

    elif fit == "cover":
        scale = max(target_w / orig_w, target_h / orig_h)
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        crop_x = (new_w - target_w) // 2
        crop_y = (new_h - target_h) // 2
        return resized.crop((crop_x, crop_y, crop_x + target_w, crop_y + target_h))

    else:
        raise ValueError(f"Unknown fit mode '{fit}'. Supported: 'stretch', 'contain', 'cover'.")


def apply_scaling(
    img: Image.Image,
    target_size: Optional[Tuple[int, int]] = None,
    scale: Optional[float] = None,
    fit: str = "stretch"
) -> Image.Image:
    """Apply final target_size or scale multiplier if specified."""
    if target_size is not None and scale is not None:
        raise ValueError("target_size and scale are mutually exclusive; specify only one.")

    if target_size is not None:
        return resize_with_fit(img, target_size, fit=fit)

    if scale is not None:
        w, h = img.size
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    return img
