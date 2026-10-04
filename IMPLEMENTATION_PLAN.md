# Implementation Plan: Modular Denoisers + Upscalers + Composable Pipeline

**Repository**: `C:\Users\157336\Python Scripts\image-upscaler-tools`
**Status**: Plan only. No code has been changed. Implementation starts when the owner replies **"proceed"**.
**Current version**: 0.1.0 (6 upscale engines). **Target version**: 0.2.0.

---

## 1. Goal

Split the package into two independent, first-class families and let users combine them freely:

| Mode | Behavior |
| :--- | :--- |
| **Denoise only** | Clean noise. Resolution unchanged. |
| **Upscale only** | Super-resolve directly from the raw input. |
| **Denoise then upscale** | Remove noise first so neural upscalers do not amplify it. |
| **Upscale then denoise** | Super-resolve first, then smooth residual artifacts. |
| **Arbitrary chain** | Any number of steps in any order, e.g. `median > bilateral > span`. |

## 2. Decisions (confirmed by the owner)

| Topic | Decision |
| :--- | :--- |
| Weights that cannot be auto-downloaded | Look for working mirrors in Phase 1. If none, raise a clear error with manual download steps. `--weights` / `model_path=` always override. |
| Non-square images | `fit="stretch" \| "contain" \| "cover"`. Default `stretch`, matching the results produced so far. |
| Performance | Tiling for neural engines. `numba` is optional for `adaptive_median` (NumPy fallback). |
| Package name / CLI | Unchanged: `image-upscaler-tools`, commands `image-upscaler` and `upscale`. |
| Main usage style | **`ImagePipeline` chaining.** Steps run in the order they are added. Denoise only, upscale only, or any mix, in any order. All other entry points (`process_image`, `steps="a>b"`, presets, CLI) are thin wrappers that build an `ImagePipeline`. |

## 3. Inventory

### 3.1 Denoisers (11 keys)

| Key | Class | Implementation | Main parameters | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `scunet` | `ScunetDenoiser` | spandrel, `scunet_color_real_psnr.pth` | none | Neural. Slow on CPU. |
| `fast_nlm` | `FastNlmDenoiser` | `cv2.fastNlMeansDenoisingColored` | `h=3, h_color=3, template=7, search=21` | Best for white-background JPEG cleanup. |
| `skimage_nlm` | `SkimageNlmDenoiser` | `skimage.restoration.denoise_nl_means` | `patch_size, patch_distance, h` | About 100x slower than OpenCV. |
| `bilateral` | `BilateralDenoiser` | `cv2.bilateralFilter` | `d=5, sigma_color, sigma_space` | OpenCV flavor. |
| `skimage_bilateral` | `SkimageBilateralDenoiser` | `skimage.restoration.denoise_bilateral` | `sigma_color=0.03, sigma_spatial=5` | Settings used in the earlier "bilateral then HAT" test. |
| `tv_chambolle` | `TvChambolleDenoiser` | `skimage.restoration.denoise_tv_chambolle` | `weight` | Earlier tests showed it can erase small text at high weight. |
| `wavelet` | `WaveletDenoiser` | `skimage.restoration.denoise_wavelet` | `method, mode` | Requires `PyWavelets`. |
| `median` | `MedianDenoiser` | `cv2.medianBlur` | `ksize=3` (odd; above 5 requires uint8) | Impulse noise. |
| `adaptive_median` | `AdaptiveMedianDenoiser` | custom, per channel | `s_max=7` | Leaves non-noisy pixels untouched. |
| `gaussian` | `GaussianDenoiser` | `cv2.GaussianBlur` | `sigma=0.8, ksize=0 (auto)` | |
| `mean` | `MeanDenoiser` | `cv2.blur` | `ksize=3` | Blurs edges. Use a small kernel only. |

### 3.2 Upscalers (7)

| Key | Class | Weights / backend | Native scale |
| :--- | :--- | :--- | :--- |
| `span` | `SpanUpscaler` | `4xPurePhoto-span.pth` (spandrel) | 4x |
| `hat` | `HatUpscaler` | `HAT_SRx4.pth` (spandrel) | 4x |
| `realesrgan` | `RealEsrganUpscaler` | `RealESRGAN_x4plus.pth` (spandrel) | 4x |
| `realesrnet` | `RealEsrnetUpscaler` | `RealESRNet_x4plus.pth` (spandrel) | 4x |
| `ultrasharp` | `UltraSharpUpscaler` | `4x-UltraSharp.pth` (spandrel) | 4x |
| `edsr` | `EdsrUpscaler` | `EDSR_x4.pb` (OpenCV `dnn_superres`, CPU only) | 4x |
| `pil` | `PilUpscaler` | none (Lanczos) | any |

### 3.3 Backward-compatible aliases

| Old name | Becomes |
| :--- | :--- |
| `scunet_before_hat` | pipeline `scunet > hat` (tensor hand-off preserved) |
| `get_upscaler`, `list_upscalers`, `upscale_image` | unchanged signatures |
| CLI `-m/--model` | alias of `--upscaler` |
| `image_upscaler_tools.engines.*` | re-export shim pointing at `upscalers` |

## 4. Target structure

```
image-upscaler-tools/
├── pyproject.toml
├── README.md
├── MODELS.md                    # NEW: source, license, sha256 per weight
├── IMPLEMENTATION_PLAN.md
├── LICENSE
├── .gitignore
├── assets/
├── src/image_upscaler_tools/
│   ├── __init__.py              # process_image, ImagePipeline, get_*/list_* exports
│   ├── __main__.py              # NEW: python -m image_upscaler_tools
│   ├── cli.py
│   ├── presets.py               # NEW: named recipes
│   ├── core/
│   │   ├── base.py              # BaseProcessor: input normalization, device, fit/resize
│   │   ├── imageio.py           # NEW: load/save, RGBA, grayscale, 16-bit, EXIF orientation
│   │   ├── model_manager.py     # URL list + sha256 + clear errors (no parent-dir search)
│   │   ├── registry.py          # separate registries: denoisers, upscalers
│   │   ├── tiling.py            # NEW: tiled inference with overlap blending
│   │   └── cache.py             # NEW: engine cache (LRU) so weights load once
│   ├── denoisers/
│   │   ├── base.py
│   │   ├── neural.py            # ScunetDenoiser
│   │   └── classical.py         # all classical denoisers
│   ├── upscalers/
│   │   ├── base.py
│   │   ├── span_engine.py  hat_engine.py  realesrgan_engine.py
│   │   ├── ultrasharp_engine.py  edsr_engine.py  pil_engine.py
│   ├── engines/                 # compatibility shim (re-exports upscalers)
│   └── pipeline/
│       ├── pipeline.py          # ImagePipeline
│       ├── orchestrator.py      # process_image()
│       └── parser.py            # NEW: "median>span>bilateral" step parser
├── tests/
│   ├── conftest.py              # weights fixture, `neural` marker, auto-skip
│   ├── test_denoisers.py  test_upscalers.py  test_pipeline.py
│   ├── test_imageio.py  test_model_manager.py  test_cli.py
├── examples/
│   ├── denoise_only.py  upscale_only.py  pipeline_mix.py
└── benchmarks/
    └── run_benchmark.py         # regenerates README timings and images
```

## 5. Public API

### 5.0 Primary API: `ImagePipeline` (call order = execution order)

```python
from image_upscaler_tools import ImagePipeline, get_denoiser, get_upscaler

# Mix: denoise, upscale, then smooth the result
pipeline = ImagePipeline()
pipeline.add(get_denoiser("fast_nlm", h=3))
pipeline.add(get_upscaler("hat"))
pipeline.add(get_denoiser("bilateral", d=5))
output = pipeline.run("raw.jpg", target_size=(500, 500))
output.save("out.jpg")

# Denoise only (resolution unchanged)
ImagePipeline().add(get_denoiser("median", ksize=3)).run("raw.jpg")

# Upscale only
ImagePipeline().add(get_upscaler("span")).run("raw.jpg", target_size=(500, 500))

# Reverse order: upscale first, then denoise
ImagePipeline().add(get_upscaler("hat")).add(get_denoiser("bilateral")).run("raw.jpg", target_size=(500, 500))
```

Shorter forms of the same thing (still `ImagePipeline` underneath):

```python
# add() also accepts a name plus parameters; the family (denoiser or upscaler) is detected automatically
ImagePipeline().add("fast_nlm", h=3).add("hat").add("bilateral", d=5).run("raw.jpg", target_size=(500, 500))

# constructor accepts a list of names or (name, params) tuples
ImagePipeline(["fast_nlm", "span"]).run("raw.jpg", target_size=(500, 500))
ImagePipeline([("fast_nlm", {"h": 3}), "hat"]).run("raw.jpg", target_size=(500, 500))
```

Behavior rules:
- Execution order is exactly the order of `add()` calls. A pipeline may contain only denoisers, only upscalers, or both, in any order and any length.
- `add()` returns the pipeline, so calls can be chained. Names are unique across both families, so a bare string is unambiguous (`"span"` is an upscaler, `"median"` is a denoiser). An unknown name raises an error that lists valid names.
- `run(image, target_size=None, fit="stretch", scale=None, save=None)` returns a `PIL.Image`. `image` may be a path, a PIL image, or a NumPy array. If `save="out.jpg"` is given, the result is also written to disk.
- `target_size` (or `scale`) is applied **once, after the last step**. If the pipeline has no upscaler, it is a plain Lanczos resize and a warning is logged.
- A pipeline with no steps raises `ValueError`. Two upscalers in a row are allowed (e.g. 16x), with a warning.
- After `run()`, `pipeline.last_timings` holds seconds per step and in total, for example `[("fast_nlm", 0.03), ("hat", 21.5), ("bilateral", 0.01)]`.
- Engines are created lazily on the first `run()` and reused for later runs and for folders, so weights load only once.
- `pipeline.run_folder("./raw", "./out", target_size=(500, 500))` processes every image with the same loaded engines.
- `ImagePipeline.from_preset("catalog_fast")` and `ImagePipeline.from_string("median > bilateral > span")` build pipelines from the wrappers below.

### 5.1 Functional wrapper

```python
from image_upscaler_tools import process_image

process_image("in.jpg", denoiser="fast_nlm")                         # denoise only
process_image("in.jpg", upscaler="span", target_size=(500, 500))     # upscale only
process_image("in.jpg", denoiser="fast_nlm", upscaler="span",
              order="denoise_first", target_size=(500, 500),
              denoiser_params={"h": 3})                              # denoise then upscale
process_image("in.jpg", upscaler="hat", denoiser="bilateral",
              order="upscale_first", target_size=(500, 500),
              denoiser_params={"d": 5, "sigma_color": 30})           # upscale then denoise
process_image("in.jpg", steps="median>bilateral>span", target_size=(500, 500))
process_image("in.jpg", preset="catalog_fast", target_size=(500, 500))
```

Rules:
- Per-stage parameters go in `denoiser_params` / `upscaler_params`. Bare `**kwargs` are not accepted (prevents collisions).
- `order` defaults to `denoise_first` when both engines are given. `order` with a single engine raises `ValueError`.
- `steps` is mutually exclusive with `denoiser` / `upscaler` / `order`.
- `target_size` is applied **once, at the end** of the pipeline, using `fit` (`stretch` default, `contain`, `cover`). `scale=` is an alternative to `target_size` (mutually exclusive).
- Common options: `device`, `tile`, `tile_overlap`, `fit`.

### 5.2 Object-oriented

The object form is `ImagePipeline` itself, described in 5.0. `process_image()` in 5.1 builds an `ImagePipeline` internally and calls `run()`, so both forms behave identically.

### 5.3 Discovery

`list_denoisers()`, `list_upscalers()`, `list_presets()`, `describe(name)` (returns weights file, license, device support).

### 5.4 Presets (from measured results)

| Preset | Steps | Intent |
| :--- | :--- | :--- |
| `catalog_fast` | `fast_nlm(h=3) > span` | High-volume, sub-second on CPU. |
| `catalog_premium` | `fast_nlm(h=3) > hat` | Best measured quality ("Pre-Denoised HAT"). |
| `restore_heavy` | `scunet > hat` | Dirty, JPEG-blocky supplier photos. |
| `gentle_bilateral_hat` | `skimage_bilateral(0.03, 5) > hat` | Clean output from the earlier bilateral test. |
| `impulse_clean_fast` | `adaptive_median > span` | Dust and dead-pixel noise. |

## 6. CLI

```bash
image-upscaler in.jpg --denoiser fast_nlm -o clean.jpg
image-upscaler in.jpg --upscaler span --size 500 500 -o up.jpg
image-upscaler in.jpg --denoiser fast_nlm --upscaler span --order denoise_first --size 500 500
image-upscaler in.jpg --upscaler hat --denoiser bilateral --order upscale_first --size 500 500
image-upscaler in.jpg --steps "median>bilateral>span" --size 500 500
image-upscaler in.jpg --preset catalog_fast --size 500 500
image-upscaler ./raw/ --preset catalog_fast --size 500 500 -o ./out/
image-upscaler in.jpg --upscaler hat --tile 256 --tile-overlap 16 --fit contain
image-upscaler --list-denoisers | --list-upscalers | --list-presets | --list-all
image-upscaler --denoiser-param h=3 --upscaler-param tile=128 ...
python -m image_upscaler_tools ...        # works without PATH changes
```

Backward compatible: `-m/--model` is an alias of `--upscaler`; `--weights` still overrides a weights path.

## 7. Phases

### Phase 1: Foundations (dependencies, weights, I/O)
1. `pyproject.toml`: add `scikit-image>=0.20`, `PyWavelets`, optional extra `fast = ["numba"]`, `dev = ["pytest"]`. Bump version to 0.2.0. Keep `opencv-python-headless` (EDSR needs `dnn_superres`; raise a clear error if unavailable).
2. `model_manager.py`:
   - Replace broken URLs. Known good: RealESRGAN (`xinntao/Real-ESRGAN` v0.1.0), RealESRNet (v0.1.1), 4x-UltraSharp (`lokcx/4x-Ultrasharp`), SCUNet (`cszn/KAIR` v1.0), EDSR (`Saafke/EDSR_Tensorflow`).
   - **Find working sources** for SPAN (`4xPurePhoto-span.pth`) and HAT (`HAT_SRx4.pth`). Candidates: OpenModelDB mirrors and the official HAT release. Each URL must be tested before it is added.
   - Add `sha256` per weight (computed from the files already in the ETL project), verified after download and on first use. Re-download on mismatch.
   - Remove the parent-directory search. Search order becomes: explicit path, `./`, `./models`, `./weights`, `IMAGE_UPSCALER_WEIGHTS` env var, user cache.
   - If a weight cannot be obtained: raise `WeightsNotFoundError` with the filename, the tried sources, and manual placement instructions.
3. `imageio.py`: unified loader. RGB, grayscale to RGB, RGBA (keep alpha, process RGB, re-attach resized alpha, flatten only when saving to JPEG), 16-bit to 8-bit with warning, EXIF orientation applied.
4. `core/cache.py`: LRU cache keyed by `(kind, name, device, params-that-affect-loading)`.

**Exit criteria**: every URL returns a real file whose sha256 matches; `python -m image_upscaler_tools --list-all` works.

### Phase 2: Denoiser subsystem
1. `denoisers/base.py`: `BaseDenoiser` with `denoise(image, **params) -> PIL.Image`, same size out as in.
2. `classical.py`: implement all 9 classical denoisers.
   - Validate parameters (odd `ksize`, uint8 requirement for `medianBlur` above 5, positive sigma).
   - `adaptive_median`: per-channel, window grows from 3 to `s_max`; vectorized NumPy implementation; `numba` JIT used when installed.
3. `neural.py`: `ScunetDenoiser` via spandrel; supports `tile`.
4. Registry: `get_denoiser`, `list_denoisers`, `register_denoiser`, aliases (`nlm` -> `fast_nlm`, `nl_means` -> `skimage_nlm`).

**Exit criteria**: each denoiser runs offline (classical) and returns the same size; impulse-noise test shows `adaptive_median` PSNR >= `median` PSNR on a synthetic image.

### Phase 3: Upscaler subsystem
1. Move engines into `upscalers/`; leave `engines/` as a re-export shim.
2. Add `RealEsrnetUpscaler`, `EdsrUpscaler` (`cv2.dnn_superres`, `setModel("edsr", 4)`, CPU only).
3. Remove the ad-hoc `pre_denoise` flag from SPAN (superseded by the pipeline).
4. `tiling.py`: tiled inference with configurable `tile` and `tile_overlap` and feathered blending; used by all neural upscalers and SCUNet. Default off.
5. Resize logic moves to the shared base (`fit`, `target_size`, `scale`).

**Exit criteria**: all 7 upscalers produce correct 4x output; tiled output matches untiled output within a small tolerance.

### Phase 4: Pipeline and orchestrator
1. `ImagePipeline` (the primary API, see 5.0): ordered steps; `add(instance)` and `add(name, **params)`; constructor accepting a list; `run()` with `target_size`, `fit`, `scale`, `save`; `run_folder()`; `last_timings`; `from_preset()` and `from_string()`; validation (at least one step, unknown names, `target_size` and `scale` mutually exclusive).
2. Neural-to-neural hand-off stays as tensors (avoids uint8 quantization between stages); classical steps convert as needed.
3. `process_image()`: modes in section 5.1; engines come from the cache.
4. `parser.py`: `"a>b>c"` and `"a(h=3)>b"` step syntax with parameter parsing.
5. `presets.py`: recipes from section 5.4.
6. `scunet_before_hat` alias implemented as the preset `restore_heavy`.

**Exit criteria**: all four modes and a 3-step chain run end to end; a second call with the same engines does not reload weights (verified by cache hit).

### Phase 5: CLI
1. Add `--denoiser`, `--upscaler`, `--order`, `--steps`, `--preset`, `--fit`, `--scale`, `--tile`, `--tile-overlap`, `--denoiser-param`, `--upscaler-param`, `--list-*`.
2. Keep `-m` alias and `--weights`.
3. Add `__main__.py`. Batch mode reuses one cached engine set for the whole folder and reports per-image and average time.
4. Clear errors and non-zero exit codes for invalid combinations.

### Phase 6: Tests and validation
- `conftest.py`: `neural` marker; tests auto-skip with a clear reason when weights are missing (no network needed for the default run).
- `test_denoisers.py`: shape preserved, dtype, parameter validation, impulse-noise behavior.
- `test_upscalers.py`: PIL always; neural ones under `neural`.
- `test_pipeline.py`: each mode; order matters (outputs differ); invalid combinations raise; `steps` parsing; presets resolve.
- `test_imageio.py`: RGBA, grayscale, 16-bit, EXIF.
- `test_model_manager.py`: sha256 mismatch triggers re-download (mocked); clear error when unavailable.
- `test_cli.py`: subprocess smoke tests for each mode.
- Real-image validation on `ptma401120.jpg` and `mlcc.jpg`:
  - Re-run the 6 original methods and confirm the pipeline reproduces the earlier outputs within tolerance (SPAN direct, HAT, `scunet > hat`).
  - Run each new preset on both images and record timings.
- Re-run `pip install -e .` from the new location.

### Phase 7: Documentation, licenses, release
1. README: leads with `ImagePipeline` chaining (denoise only, upscale only, mix, reverse order), then new overview, denoiser and upscaler tables, preset table, pipeline examples, CLI reference, tiling and `fit` explanation, and a **model license column**.
2. Verify the license of each weight file before publishing (notably 4x-UltraSharp, which I believe is non-commercial; this must be confirmed from its source page, not assumed). Add `MODELS.md` with source, license and sha256 per weight.
3. Regenerate README timings and the 2x3 comparison from `benchmarks/run_benchmark.py`. Add a second figure showing denoise-then-upscale vs upscale-then-denoise.
4. Remove the "verified mirrors" wording unless Phase 1 actually verified every URL.
5. Commit in logical steps on `main`. Push is blocked until the empty GitHub repo `HamdEmad/image-upscaler-tools` is created (owner action).

## 8. Risks

| Risk | Mitigation |
| :--- | :--- |
| SPAN or HAT weights have no stable direct URL | Mirror search in Phase 1; clear `WeightsNotFoundError`; documented manual placement; optional GitHub Release assets later. |
| Weight licenses restrict commercial use | License column and `MODELS.md`; verify each before publishing. |
| CPU time of HAT / SCUNet (20-30 s per image) | Document; tiling; presets steer volume work to `span`. |
| Pure-Python `adaptive_median` too slow | Vectorized NumPy plus optional `numba`. |
| `denoise_first` parameters tuned at 150px behave differently after upscaling | Parameters are per stage; README states they are resolution dependent. |
| `dnn_superres` missing on some OpenCV builds | EDSR raises a clear, actionable error; other engines unaffected. |
| Breaking existing imports | `engines/` shim, kept function signatures, kept CLI aliases. |

## 9. Definition of done

- All 11 denoisers and 7 upscalers selectable by name and runnable alone.
- Any denoiser before or after any upscaler works via API, `steps`, presets and CLI.
- All weight URLs verified, with sha256 checks.
- Offline test run passes; neural tests pass when weights are present.
- Earlier results (SPAN, HAT, `scunet > hat`) reproduced through the new pipeline.
- README and `MODELS.md` updated, committed on `main`.
