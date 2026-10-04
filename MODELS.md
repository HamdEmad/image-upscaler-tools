# Model Checkpoints, Licenses & Verification

This document provides a comprehensive inventory of all neural model checkpoints supported by `image-upscaler-tools`, including exact SHA-256 verification hashes, verified download mirrors, architectural details, and license terms.

---

## 1. Checkpoint Verification Matrix

| Model Key | Filename | Size | SHA-256 Checksum | License | Commercial Use |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `span` | `4xPurePhoto-span.pth` | 9,016,490 B | `c689eec59771ed3eaffc10eea933c44fdb9131f83251c51c5bab4cae7c4d3bf2` | Apache 2.0 | **Yes** |
| `hat` | `HAT_SRx4.pth` | 85,137,601 B | `02dabea478aa5902a7170ad89350124e691bd89c91356f24b3267022622dc030` | CC BY-NC-SA 4.0 | **No (Research only)** |
| `realesrgan` | `RealESRGAN_x4plus.pth` | 67,040,989 B | `4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1` | BSD 3-Clause | **Yes** |
| `realesrnet` | `RealESRNet_x4plus.pth` | 67,040,989 B | `a820b9bde89a874d7599d545567308ce6c128fc8754a53208eda016d40aa81df` | BSD 3-Clause | **Yes** |
| `ultrasharp` | `4x-UltraSharp.pth` | 66,961,958 B | `a5812231fc936b42af08a5edba784195495d303d5b3248c24489ef0c4021fe01` | Community / CC-BY-NC | **No (Attribution / Non-commercial)** |
| `scunet` | `scunet_color_real_psnr.pth` | 71,982,841 B | `fa78899ba2caec9d235a900e91d96c689da71c42029230c2028b00f09f809c2e` | Apache 2.0 | **Yes** |
| `edsr` | `EDSR_x4.pb` | 38,573,255 B | `dd35ce3cae53ecee2d16045e08a932c3e7242d641bb65cb971d123e06904347f` | BSD 3-Clause | **Yes** |

---

## 2. Model Details & Sources

### SPAN: Swift Parameter-free Attention Network (`span`)
- **Task**: 4x Super-Resolution
- **Architecture**: SPAN (CVPR 2024 NTIRE Challenge winner)
- **Primary Source**: [StarinspaceUpscale Releases](https://github.com/starinspace/StarinspaceUpscale/releases/download/Models/4xPurePhoto-Span.pth)
- **License**: Apache 2.0
- **Characteristics**: Sub-second execution on modern CPU/GPU, high edge stability, zero halluncinated artifacts.

### HAT: Hybrid Attention Transformer (`hat`)
- **Task**: 4x Super-Resolution
- **Architecture**: Hybrid Transformer combining window self-attention and channel attention (CVPR 2023)
- **Primary Source**: [Hugging Face jaideepsingh/upscale_models](https://huggingface.co/jaideepsingh/upscale_models/resolve/main/HAT/HAT_SRx4.pth)
- **Mirror**: [Hugging Face Actus/HAT](https://huggingface.co/jaideepsingh/upscale_models/resolve/main/HAT/HAT_SRx4.pth)
- **License**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
- **Characteristics**: State-of-the-art detail reconstruction. High compute cost (~20-30s on CPU).

### Real-ESRGAN & Real-ESRNet (`realesrgan`, `realesrnet`)
- **Task**: 4x General Real-World Image Restoration
- **Architecture**: RRDBNet (16.7M parameters)
- **Primary Sources**:
  - Real-ESRGAN: [xinntao/Real-ESRGAN v0.1.0](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)
  - Real-ESRNet: [xinntao/Real-ESRGAN v0.1.1](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth)
- **License**: BSD 3-Clause
- **Characteristics**: Robust against heavy JPEG compression and camera noise.

### 4x-UltraSharp (`ultrasharp`)
- **Task**: 4x Super-Resolution fine-tuned for high sharpness and text
- **Architecture**: ESRGAN RRDBNet fine-tuned by Kim2091
- **Primary Source**: [Hugging Face lokcx/4x-Ultrasharp](https://huggingface.co/lokcx/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth)
- **License**: Community Non-Commercial License
- **Characteristics**: Popular benchmark for crisp text rendering and sharp graphic edges.

### SCUNet Real PSNR (`scunet`)
- **Task**: Blind Color Image Denoising (1x resolution)
- **Architecture**: Swin-Conv-UNet (Zhang et al.)
- **Primary Source**: [cszn/KAIR v1.0](https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth)
- **License**: Apache 2.0
- **Characteristics**: Deep neural denoiser removing complex non-Gaussian real sensor noise without washing out textures.

### EDSR: Enhanced Deep Residual Networks (`edsr`)
- **Task**: 4x Super-Resolution (TensorFlow Frozen Graph)
- **Architecture**: ResNet-based deep SR network executed via OpenCV `dnn_superres`
- **Primary Source**: [Saafke/EDSR_Tensorflow](https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb)
- **License**: BSD 3-Clause

---

## 3. Manual Weights Placement

If your environment has restricted or offline internet access, you can download any of the checkpoints above and place them in any of the following locations:
1. Current working directory (`./<filename>`)
2. Local models folder (`./models/<filename>`)
3. Local weights folder (`./weights/<filename>`)
4. Directory specified by `IMAGE_UPSCALER_WEIGHTS` environment variable
5. User cache directory:
   - Linux/macOS: `~/.cache/image_upscaler_tools/weights/<filename>`
   - Windows: `C:\Users\<User>\.cache\image_upscaler_tools\weights\<filename>`
