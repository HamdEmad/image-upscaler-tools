import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import requests
from tqdm import tqdm

MODEL_REGISTRY_INFO: Dict[str, Dict] = {
    "span": {
        "filename": "4xPurePhoto-span.pth",
        "alt_filenames": ["4xPurePhoto-Span.pth"],
        "urls": [
            "https://github.com/starinspace/StarinspaceUpscale/releases/download/Models/4xPurePhoto-Span.pth"
        ],
        "sha256": "c689eec59771ed3eaffc10eea933c44fdb9131f83251c51c5bab4cae7c4d3bf2",
        "size": 9016490,
        "description": "SPAN 4x PurePhoto (Parameter-free Attention, CVPR NTIRE winner)",
        "license": "Apache 2.0"
    },
    "hat": {
        "filename": "HAT_SRx4.pth",
        "alt_filenames": ["hat_srx4.pth"],
        "urls": [
            "https://huggingface.co/jaideepsingh/upscale_models/resolve/main/HAT/HAT_SRx4.pth",
            "https://huggingface.co/Actus/HAT/resolve/main/HAT_SRx4.pth"
        ],
        "sha256": "4ee053c42461187846dc0e93aa5abd34591c0725a8e044a59000e92ee215e833",
        "size": 85137601,
        "description": "HAT 4x (Hybrid Attention Transformer)",
        "license": "CC BY-NC-SA 4.0"
    },
    "realesrgan": {
        "filename": "RealESRGAN_x4plus.pth",
        "alt_filenames": ["realesrgan_x4plus.pth"],
        "urls": [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        ],
        "sha256": "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
        "size": 67040989,
        "description": "Real-ESRGAN 4x Plus",
        "license": "BSD 3-Clause"
    },
    "realesrnet": {
        "filename": "RealESRNet_x4plus.pth",
        "alt_filenames": ["realesrnet_x4plus.pth"],
        "urls": [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
        ],
        "sha256": "a820b9bde89a874d7599d545567308ce6c128fc8754a53208eda016d40aa81df",
        "size": 67040989,
        "description": "Real-ESRNet 4x Plus",
        "license": "BSD 3-Clause"
    },
    "ultrasharp": {
        "filename": "4x-UltraSharp.pth",
        "alt_filenames": ["4x_ultrasharp.pth", "4x-ultrasharp.pth"],
        "urls": [
            "https://huggingface.co/lokcx/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth"
        ],
        "sha256": "a5812231fc936b42af08a5edba784195495d303d5b3248c24489ef0c4021fe01",
        "size": 66961958,
        "description": "4x-UltraSharp ESRGAN",
        "license": "Non-commercial / Community"
    },
    "scunet": {
        "filename": "scunet_color_real_psnr.pth",
        "alt_filenames": ["scunet_real_psnr.pth"],
        "urls": [
            "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth"
        ],
        "sha256": "fa78899ba2caec9d235a900e91d96c689da71c42029230c2028b00f09f809c2e",
        "size": 71982841,
        "description": "SCUNet Real PSNR Denoiser",
        "license": "Apache 2.0"
    },
    "edsr": {
        "filename": "EDSR_x4.pb",
        "alt_filenames": ["edsr_x4.pb"],
        "urls": [
            "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb"
        ],
        "sha256": "dd35ce3cae53ecee2d16045e08a932c3e7242d641bb65cb971d123e06904347f",
        "size": 38573255,
        "description": "EDSR 4x (OpenCV dnn_superres)",
        "license": "BSD 3-Clause"
    }
}


class WeightsNotFoundError(RuntimeError):
    """Raised when required model weights cannot be found or downloaded."""
    pass


class WeightsVerificationError(RuntimeError):
    """Raised when model weights checksum verification fails."""
    pass


def compute_sha256(filepath: Path) -> str:
    """Compute sha256 checksum for a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def get_default_cache_dir() -> Path:
    """Return the default system cache directory for model weights."""
    cache_base = os.environ.get("IMAGE_UPSCALER_CACHE")
    if cache_base:
        p = Path(cache_base)
    else:
        p = Path.home() / ".cache" / "image_upscaler_tools" / "weights"
    p.mkdir(parents=True, exist_ok=True)
    return p


def download_file(
    urls: List[str],
    dest_path: Path,
    expected_sha256: Optional[str] = None,
    desc: str = "Downloading model"
) -> None:
    """Download a file with fallback across multiple mirror URLs and verify checksum."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")

    last_error = None
    for url in urls:
        try:
            print(f"[{desc}] Fetching from: {url}")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            chunk_size = 1024 * 1024

            with open(temp_path, "wb") as f, tqdm(
                total=total_size, unit="B", unit_scale=True, desc=dest_path.name
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            if expected_sha256:
                actual_sha = compute_sha256(temp_path)
                if actual_sha.lower() != expected_sha256.lower():
                    temp_path.unlink()
                    raise WeightsVerificationError(
                        f"Checksum mismatch for {dest_path.name}: expected {expected_sha256}, got {actual_sha}"
                    )

            temp_path.replace(dest_path)
            print(f"Saved verified checkpoint to: {dest_path}")
            return
        except Exception as e:
            last_error = e
            if temp_path.exists():
                temp_path.unlink()
            print(f"Warning: Failed to download from {url} ({e}). Trying next mirror...")

    raise WeightsNotFoundError(
        f"Failed to download model weights for {dest_path.name} from sources: {urls}. "
        f"Last error: {last_error}\n"
        f"Manual fix: Download '{dest_path.name}' manually and place it in '{dest_path.parent}' "
        f"or pass custom path via model_path='path/to/{dest_path.name}'."
    )


class ModelManager:
    """Handles discovery, local searching, and auto-downloading of model weights."""

    @staticmethod
    def get_model_path(
        model_key: str,
        custom_path: Optional[str] = None,
        verify_checksum: bool = False
    ) -> str:
        """
        Locate the model checkpoint file.
        Search order:
        1. Explicit custom_path if provided
        2. Current working directory (./<filename>)
        3. Local models or weights directory (./models/<filename>, ./weights/<filename>)
        4. Environment variable IMAGE_UPSCALER_WEIGHTS
        5. User cache directory (~/.cache/image_upscaler_tools/weights/<filename>)
        6. Automatic download from official verified URLs
        """
        key = model_key.lower().strip()
        if key not in MODEL_REGISTRY_INFO:
            if custom_path and os.path.exists(custom_path):
                return custom_path
            raise ValueError(f"Unknown model key '{model_key}'. Available: {list(MODEL_REGISTRY_INFO.keys())}")

        info = MODEL_REGISTRY_INFO[key]
        expected_sha = info.get("sha256")
        expected_names = [info["filename"]] + info.get("alt_filenames", [])

        # 1. Custom path override
        if custom_path:
            p = Path(custom_path)
            if p.exists() and p.is_file():
                if verify_checksum and expected_sha:
                    actual = compute_sha256(p)
                    if actual.lower() != expected_sha.lower():
                        raise WeightsVerificationError(
                            f"Model checkpoint at '{p}' failed checksum verification. "
                            f"Expected {expected_sha}, got {actual}."
                        )
                return str(p.resolve())
            raise FileNotFoundError(f"Specified custom model path does not exist: {custom_path}")

        # 2. Candidate search directories
        candidate_dirs = [
            Path.cwd(),
            Path.cwd() / "models",
            Path.cwd() / "weights",
        ]
        env_weights = os.environ.get("IMAGE_UPSCALER_WEIGHTS")
        if env_weights:
            candidate_dirs.append(Path(env_weights))

        candidate_dirs.append(get_default_cache_dir())

        for directory in candidate_dirs:
            for fname in expected_names:
                p = directory / fname
                if p.exists() and p.is_file() and p.stat().st_size > 1024:
                    if verify_checksum and expected_sha:
                        actual = compute_sha256(p)
                        if actual.lower() != expected_sha.lower():
                            print(f"Warning: Corrupt weights found at {p}. Re-downloading...")
                            continue
                    return str(p.resolve())

        # 3. Not found locally, attempt verified auto-download
        target_path = get_default_cache_dir() / info["filename"]
        urls = info.get("urls", [])
        if not urls:
            raise WeightsNotFoundError(
                f"No automatic download URLs configured for model '{key}' ({info['filename']}).\n"
                f"Please manually download the weights and place them in '{get_default_cache_dir()}' "
                f"or pass model_path='path/to/{info['filename']}'."
            )

        download_file(
            urls=urls,
            dest_path=target_path,
            expected_sha256=expected_sha,
            desc=info.get("description", key)
        )
        return str(target_path.resolve())


    @staticmethod
    def has_weights(model_key: str) -> bool:
        """Check if weights file exists locally without triggering network download."""
        key = model_key.lower().strip()
        info = MODEL_REGISTRY_INFO.get(key)
        if not info:
            return False
        expected_names = [info["filename"]] + info.get("alt_filenames", [])
        candidate_dirs = [
            Path.cwd(),
            Path.cwd() / "models",
            Path.cwd() / "weights",
        ]
        env_weights = os.environ.get("IMAGE_UPSCALER_WEIGHTS")
        if env_weights:
            candidate_dirs.append(Path(env_weights))
        candidate_dirs.append(get_default_cache_dir())

        for directory in candidate_dirs:
            for fname in expected_names:
                p = directory / fname
                if p.exists() and p.is_file() and p.stat().st_size > 1024:
                    return True
        return False

    @staticmethod
    def get_model_info(model_key: str) -> Optional[Dict]:
        """Return metadata for a model key."""
        return MODEL_REGISTRY_INFO.get(model_key.lower().strip())
