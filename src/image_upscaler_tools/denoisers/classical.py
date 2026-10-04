from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from image_upscaler_tools.core.base import BaseDenoiser
from image_upscaler_tools.core.imageio import load_image, split_alpha, merge_alpha
from image_upscaler_tools.core.registry import register_denoiser

try:
    from skimage.restoration import (
        denoise_nl_means,
        estimate_sigma,
        denoise_bilateral as sk_denoise_bilateral,
        denoise_tv_chambolle as sk_denoise_tv_chambolle,
        denoise_wavelet as sk_denoise_wavelet,
    )
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False


@register_denoiser("fast_nlm", aliases=["nlm", "fastnlm", "opencv_nlm"])
class FastNlmDenoiser(BaseDenoiser):
    """
    Fast Non-Local Means Denoising using OpenCV (cv2.fastNlMeansDenoisingColored).
    Recommended for clean white-background JPEG compression cleanup and color noise.
    """

    def __init__(
        self,
        h: float = 3.0,
        h_color: float = 3.0,
        template_window_size: int = 7,
        search_window_size: int = 21,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.h = float(h)
        self.h_color = float(h_color)
        self.template_window_size = int(template_window_size)
        self.search_window_size = int(search_window_size)

    @property
    def name(self) -> str:
        return "fast_nlm"

    def denoise(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        h: Optional[float] = None,
        h_color: Optional[float] = None,
        **kwargs
    ) -> Image.Image:
        pil_img = load_image(image, preserve_alpha=True)
        rgb_img, alpha_mask = split_alpha(pil_img)

        curr_h = float(h) if h is not None else self.h
        curr_h_color = float(h_color) if h_color is not None else self.h_color

        rgb_arr = np.array(rgb_img)
        bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        denoised_bgr = cv2.fastNlMeansDenoisingColored(
            bgr,
            None,
            curr_h,
            curr_h_color,
            self.template_window_size,
            self.search_window_size
        )
        denoised_rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
        out_img = Image.fromarray(denoised_rgb)
        return merge_alpha(out_img, alpha_mask)


@register_denoiser("bilateral", aliases=["cv2_bilateral", "bilateral_filter"])
class BilateralDenoiser(BaseDenoiser):
    """
    OpenCV Bilateral Filter (cv2.bilateralFilter).
    Smooths flat regions while preserving sharp edges.
    """

    def __init__(
        self,
        d: int = 5,
        sigma_color: float = 30.0,
        sigma_space: float = 30.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d = int(d)
        self.sigma_color = float(sigma_color)
        self.sigma_space = float(sigma_space)

    @property
    def name(self) -> str:
        return "bilateral"

    def denoise(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        d: Optional[int] = None,
        sigma_color: Optional[float] = None,
        sigma_space: Optional[float] = None,
        **kwargs
    ) -> Image.Image:
        pil_img = load_image(image, preserve_alpha=True)
        rgb_img, alpha_mask = split_alpha(pil_img)

        curr_d = int(d) if d is not None else self.d
        curr_color = float(sigma_color) if sigma_color is not None else self.sigma_color
        curr_space = float(sigma_space) if sigma_space is not None else self.sigma_space

        rgb_arr = np.array(rgb_img)
        bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        denoised_bgr = cv2.bilateralFilter(bgr, curr_d, curr_color, curr_space)
        denoised_rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
        out_img = Image.fromarray(denoised_rgb)
        return merge_alpha(out_img, alpha_mask)


@register_denoiser("skimage_bilateral", aliases=["sk_bilateral"])
class SkimageBilateralDenoiser(BaseDenoiser):
    """
    scikit-image Bilateral Filter (skimage.restoration.denoise_bilateral).
    Matches the gentle pre-filter settings used in benchmark evaluations.
    """

    def __init__(
        self,
        sigma_color: float = 0.03,
        sigma_spatial: float = 5.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        if not _SKIMAGE_AVAILABLE:
            raise ImportError("scikit-image is required for skimage_bilateral.")
        self.sigma_color = float(sigma_color)
        self.sigma_spatial = float(sigma_spatial)

    @property
    def name(self) -> str:
        return "skimage_bilateral"

    def denoise(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        sigma_color: Optional[float] = None,
        sigma_spatial: Optional[float] = None,
        **kwargs
    ) -> Image.Image:
        pil_img = load_image(image, preserve_alpha=True)
        rgb_img, alpha_mask = split_alpha(pil_img)

        curr_color = float(sigma_color) if sigma_color is not None else self.sigma_color
        curr_spatial = float(sigma_spatial) if sigma_spatial is not None else self.sigma_spatial

        img_float = np.array(rgb_img, dtype=np.float32) / 255.0
        denoised = sk_denoise_bilateral(
            img_float,
            sigma_color=curr_color,
            sigma_spatial=curr_spatial,
            channel_axis=-1
        )
        denoised_uint8 = np.clip(denoised * 255.0, 0, 255).astype(np.uint8)
        out_img = Image.fromarray(denoised_uint8)
        return merge_alpha(out_img, alpha_mask)


@register_denoiser("tv_chambolle", aliases=["tv", "chambolle"])
class TvChambolleDenoiser(BaseDenoiser):
    """
    Total Variation Chambolle Denoising (skimage.restoration.denoise_tv_chambolle).
    Minimizes total variation norm. Effective on cartoon/flat imagery.
    """

    def __init__(self, weight: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        if not _SKIMAGE_AVAILABLE:
            raise ImportError("scikit-image is required for tv_chambolle.")
        self.weight = float(weight)

    @property
    def name(self) -> str:
        return "tv_chambolle"

    def denoise(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        weight: Optional[float] = None,
        **kwargs
    ) -> Image.Image:
        pil_img = load_image(image, preserve_alpha=True)
        rgb_img, alpha_mask = split_alpha(pil_img)

        curr_w = float(weight) if weight is not None else self.weight
        img_float = np.array(rgb_img, dtype=np.float32) / 255.0
        denoised = sk_denoise_tv_chambolle(img_float, weight=curr_w, channel_axis=-1)
        denoised_uint8 = np.clip(denoised * 255.0, 0, 255).astype(np.uint8)
        out_img = Image.fromarray(denoised_uint8)
        return merge_alpha(out_img, alpha_mask)


@register_denoiser("wavelet", aliases=["pywavelet", "wavelets"])
class WaveletDenoiser(BaseDenoiser):
    """
    Wavelet Denoising (skimage.restoration.denoise_wavelet).
    Multiscale frequency decomposition, BayesShrink thresholding.
    """

    def __init__(
        self,
        method: str = "BayesShrink",
        mode: str = "soft",
        wavelet: str = "db1",
        **kwargs
    ):
        super().__init__(**kwargs)
        if not _SKIMAGE_AVAILABLE:
            raise ImportError("scikit-image and PyWavelets are required for wavelet denoiser.")
        self.method = method
        self.mode = mode
        self.wavelet = wavelet

    @property
    def name(self) -> str:
        return "wavelet"

    def denoise(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        method: Optional[str] = None,
        mode: Optional[str] = None,
        **kwargs
    ) -> Image.Image:
        pil_img = load_image(image, preserve_alpha=True)
        rgb_img, alpha_mask = split_alpha(pil_img)

        m = method or self.method
        md = mode or self.mode

        img_float = np.array(rgb_img, dtype=np.float32) / 255.0
        denoised = sk_denoise_wavelet(
            img_float,
            channel_axis=-1,
            convert2ycbcr=True,
            method=m,
            mode=md,
            wavelet=self.wavelet,
            rescale_sigma=True
        )
        denoised = np.nan_to_num(denoised, nan=0.0)
        denoised_uint8 = np.clip(denoised * 255.0, 0, 255).astype(np.uint8)
        out_img = Image.fromarray(denoised_uint8)
        return merge_alpha(out_img, alpha_mask)


@register_denoiser("median", aliases=["cv2_median", "median_filter"])
class MedianDenoiser(BaseDenoiser):
    """
    Median Blur Filter (cv2.medianBlur).
    Ideal for salt-and-pepper noise removal.
    """

    def __init__(self, ksize: int = 3, **kwargs):
        super().__init__(**kwargs)
        if ksize % 2 == 0 or ksize < 3:
            raise ValueError(f"Median filter ksize must be an odd integer >= 3, got {ksize}")
        self.ksize = int(ksize)

    @property
    def name(self) -> str:
        return "median"

    def denoise(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        ksize: Optional[int] = None,
        **kwargs
    ) -> Image.Image:
        pil_img = load_image(image, preserve_alpha=True)
        rgb_img, alpha_mask = split_alpha(pil_img)

        k = int(ksize) if ksize is not None else self.ksize
        if k % 2 == 0 or k < 3:
            raise ValueError(f"ksize must be an odd integer >= 3, got {k}")

        rgb_arr = np.array(rgb_img)
        denoised_rgb = cv2.medianBlur(rgb_arr, k)
        out_img = Image.fromarray(denoised_rgb)
        return merge_alpha(out_img, alpha_mask)


def _adaptive_median_channel(channel: np.ndarray, s_max: int = 7) -> np.ndarray:
    """Vectorized Adaptive Median Filter on a 2D uint8 channel."""
    from numpy.lib.stride_tricks import sliding_window_view
    H, W = channel.shape
    output = channel.copy()
    unprocessed = np.ones((H, W), dtype=bool)

    pad_max = s_max // 2
    padded = np.pad(channel, pad_max, mode="reflect")

    for s in range(3, s_max + 1, 2):
        r = s // 2
        offset = pad_max - r
        sub_padded = padded[offset:H + 2 * pad_max - offset, offset:W + 2 * pad_max - offset]
        windows = sliding_window_view(sub_padded, (s, s))

        z_min = np.min(windows, axis=(2, 3))
        z_max = np.max(windows, axis=(2, 3))
        z_med = np.median(windows, axis=(2, 3)).astype(channel.dtype)
        z_xy = channel

        med_not_impulse = (z_med > z_min) & (z_med < z_max)
        xy_not_impulse = (z_xy > z_min) & (z_xy < z_max)

        keep_xy = unprocessed & med_not_impulse & xy_not_impulse
        output[keep_xy] = z_xy[keep_xy]
        unprocessed[keep_xy] = False

        replace_med = unprocessed & med_not_impulse & (~xy_not_impulse)
        output[replace_med] = z_med[replace_med]
        unprocessed[replace_med] = False

        if not np.any(unprocessed):
            break

    if np.any(unprocessed):
        output[unprocessed] = z_med[unprocessed]

    return output


@register_denoiser("adaptive_median", aliases=["adaptive", "adp_median"])
class AdaptiveMedianDenoiser(BaseDenoiser):
    """
    Adaptive Median Filter.
    Superior for images with high salt-and-pepper noise density.
    Dynamically expands filter window from 3 to s_max, leaving non-noisy pixels untouched.
    """

    def __init__(self, s_max: int = 7, **kwargs):
        super().__init__(**kwargs)
        if s_max % 2 == 0 or s_max < 3:
            raise ValueError(f"s_max must be an odd integer >= 3, got {s_max}")
        self.s_max = int(s_max)

    @property
    def name(self) -> str:
        return "adaptive_median"

    def denoise(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        s_max: Optional[int] = None,
        **kwargs
    ) -> Image.Image:
        pil_img = load_image(image, preserve_alpha=True)
        rgb_img, alpha_mask = split_alpha(pil_img)

        max_s = int(s_max) if s_max is not None else self.s_max
        if max_s % 2 == 0 or max_s < 3:
            raise ValueError(f"s_max must be an odd integer >= 3, got {max_s}")

        rgb_arr = np.array(rgb_img)
        denoised_channels = []
        for c in range(3):
            denoised_channels.append(_adaptive_median_channel(rgb_arr[:, :, c], s_max=max_s))

        denoised_rgb = np.stack(denoised_channels, axis=-1)
        out_img = Image.fromarray(denoised_rgb)
        return merge_alpha(out_img, alpha_mask)


@register_denoiser("gaussian", aliases=["gauss", "gaussian_blur"])
class GaussianDenoiser(BaseDenoiser):
    """
    Gaussian Blur Filter (cv2.GaussianBlur).
    General noise reduction while preserving soft transitions.
    """

    def __init__(self, sigma: float = 0.8, ksize: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.sigma = float(sigma)
        self.ksize = int(ksize)

    @property
    def name(self) -> str:
        return "gaussian"

    def denoise(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        sigma: Optional[float] = None,
        ksize: Optional[int] = None,
        **kwargs
    ) -> Image.Image:
        pil_img = load_image(image, preserve_alpha=True)
        rgb_img, alpha_mask = split_alpha(pil_img)

        curr_sigma = float(sigma) if sigma is not None else self.sigma
        curr_k = int(ksize) if ksize is not None else self.ksize
        if curr_k > 0 and curr_k % 2 == 0:
            curr_k += 1

        rgb_arr = np.array(rgb_img)
        denoised_rgb = cv2.GaussianBlur(rgb_arr, (curr_k, curr_k), sigmaX=curr_sigma, sigmaY=curr_sigma)
        out_img = Image.fromarray(denoised_rgb)
        return merge_alpha(out_img, alpha_mask)


@register_denoiser("mean", aliases=["box", "box_filter", "average"])
class MeanDenoiser(BaseDenoiser):
    """
    Mean / Box Filter (cv2.blur).
    Fast general spatial smoothing.
    """

    def __init__(self, ksize: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.ksize = int(ksize)

    @property
    def name(self) -> str:
        return "mean"

    def denoise(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        ksize: Optional[int] = None,
        **kwargs
    ) -> Image.Image:
        pil_img = load_image(image, preserve_alpha=True)
        rgb_img, alpha_mask = split_alpha(pil_img)

        k = int(ksize) if ksize is not None else self.ksize
        rgb_arr = np.array(rgb_img)
        denoised_rgb = cv2.blur(rgb_arr, (k, k))
        out_img = Image.fromarray(denoised_rgb)
        return merge_alpha(out_img, alpha_mask)


@register_denoiser("skimage_nlm", aliases=["sk_nlm", "nl_means"])
class SkimageNlmDenoiser(BaseDenoiser):
    """
    scikit-image Non-Local Means Denoising (skimage.restoration.denoise_nl_means).
    Mathematically thorough NLM implementation.
    """

    def __init__(
        self,
        patch_size: int = 7,
        patch_distance: int = 11,
        h: float = 0.1,
        fast_mode: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        if not _SKIMAGE_AVAILABLE:
            raise ImportError("scikit-image is required for skimage_nlm.")
        self.patch_size = int(patch_size)
        self.patch_distance = int(patch_distance)
        self.h = float(h)
        self.fast_mode = bool(fast_mode)

    @property
    def name(self) -> str:
        return "skimage_nlm"

    def denoise(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        h: Optional[float] = None,
        patch_size: Optional[int] = None,
        patch_distance: Optional[int] = None,
        **kwargs
    ) -> Image.Image:
        pil_img = load_image(image, preserve_alpha=True)
        rgb_img, alpha_mask = split_alpha(pil_img)

        curr_h = float(h) if h is not None else self.h
        ps = int(patch_size) if patch_size is not None else self.patch_size
        pd = int(patch_distance) if patch_distance is not None else self.patch_distance

        img_float = np.array(rgb_img, dtype=np.float32) / 255.0
        sigma_est = np.mean(estimate_sigma(img_float, channel_axis=-1))

        denoised = denoise_nl_means(
            img_float,
            h=curr_h * sigma_est if sigma_est > 0 else curr_h,
            patch_size=ps,
            patch_distance=pd,
            fast_mode=self.fast_mode,
            channel_axis=-1
        )
        denoised_uint8 = np.clip(denoised * 255.0, 0, 255).astype(np.uint8)
        out_img = Image.fromarray(denoised_uint8)
        return merge_alpha(out_img, alpha_mask)
