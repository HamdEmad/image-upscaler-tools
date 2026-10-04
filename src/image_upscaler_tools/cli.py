import argparse
import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from image_upscaler_tools.core.registry import (
    list_denoisers,
    list_upscalers,
    describe,
    is_denoiser,
    is_upscaler
)
from image_upscaler_tools.pipeline.pipeline import ImagePipeline
from image_upscaler_tools.pipeline.orchestrator import process_image
from image_upscaler_tools.pipeline.presets import list_presets


def _parse_key_value_args(arg_list: Optional[List[str]]) -> Dict[str, any]:
    params = {}
    if not arg_list:
        return params
    for item in arg_list:
        if "=" not in item:
            raise ValueError(f"Parameter must be in key=value format, got '{item}'")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        try:
            val = ast.literal_eval(v)
        except Exception:
            val = v
        params[k] = val
    return params


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="image-upscaler",
        description="Unified CLI for Image Denoising (11 methods) and Super-Resolution Upscaling (7 engines)."
    )

    # Positional input
    parser.add_argument(
        "input",
        nargs="?",
        type=str,
        help="Path to input image file or directory of images."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Path to output image file or output directory (when input is a directory)."
    )

    # Processing selectors
    parser.add_argument(
        "--denoiser",
        type=str,
        help=f"Denoiser method: {list_denoisers()}"
    )
    parser.add_argument(
        "-m", "--model", "--upscaler",
        dest="upscaler",
        type=str,
        help=f"Super-resolution upscaler engine: {list_upscalers()}"
    )
    parser.add_argument(
        "--order",
        type=str,
        default="denoise_first",
        choices=["denoise_first", "upscale_first"],
        help="Execution order when both denoiser and upscaler are given (default: denoise_first)."
    )
    parser.add_argument(
        "--steps",
        type=str,
        help="Custom pipeline step chain, e.g. 'fast_nlm(h=3) > span > bilateral(d=5)'."
    )
    parser.add_argument(
        "--preset",
        type=str,
        help=f"Named optimization recipe: {list_presets()}"
    )

    # Resolution & Fit
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Target output resolution (width height)."
    )
    parser.add_argument(
        "--scale",
        type=float,
        help="Multiplier scale factor for output resolution (e.g. 2.0, 4.0)."
    )
    parser.add_argument(
        "--fit",
        type=str,
        default="stretch",
        choices=["stretch", "contain", "cover"],
        help="Aspect ratio fit strategy (default: stretch)."
    )

    # Performance & Inference
    parser.add_argument(
        "--tile",
        type=int,
        help="Tile size for tiled neural inference (e.g. 256) to reduce memory usage."
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=16,
        help="Tile overlap in pixels for seamless blending (default: 16)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device ('auto', 'cpu', 'cuda', 'cuda:0')."
    )
    parser.add_argument(
        "--weights",
        type=str,
        help="Explicit path to model checkpoint file (overrides automatic search and download)."
    )

    # Parameter passing
    parser.add_argument(
        "--denoiser-param",
        action="append",
        metavar="KEY=VALUE",
        help="Pass parameter to denoiser stage (can be used multiple times, e.g. --denoiser-param h=3)."
    )
    parser.add_argument(
        "--upscaler-param",
        action="append",
        metavar="KEY=VALUE",
        help="Pass parameter to upscaler stage (can be used multiple times, e.g. --upscaler-param tile=128)."
    )

    # Discovery queries
    parser.add_argument("--list-denoisers", action="store_true", help="Print all available denoisers.")
    parser.add_argument("--list-upscalers", action="store_true", help="Print all available upscalers.")
    parser.add_argument("--list-presets", action="store_true", help="Print all available presets.")
    parser.add_argument("--list-all", action="store_true", help="Print all available engines, denoisers, and presets.")
    parser.add_argument("--describe", type=str, metavar="NAME", help="Show details and license for a given tool.")

    args = parser.parse_args()

    # Handle queries
    if args.list_denoisers:
        print("Available Denoisers:")
        for d in list_denoisers():
            print(f"  - {d}")
        return 0

    if args.list_upscalers:
        print("Available Upscalers:")
        for u in list_upscalers():
            print(f"  - {u}")
        return 0

    if args.list_presets:
        print("Available Presets:")
        for p in list_presets():
            print(f"  - {p}")
        return 0

    if args.list_all:
        print("==============================")
        print("image-upscaler-tools Toolkit")
        print("==============================")
        print("\n[Denoisers]")
        for d in list_denoisers():
            print(f"  - {d}")
        print("\n[Upscalers]")
        for u in list_upscalers():
            print(f"  - {u}")
        print("\n[Presets]")
        for p in list_presets():
            print(f"  - {p}")
        return 0

    if args.describe:
        try:
            info = describe(args.describe)
            print(f"\nTool: {info['name']}")
            print(f"Category: {info['category']}")
            print(f"Class: {info['class']}")
            print(f"License: {info['license']}")
            if info['weights']:
                print(f"Weights file: {info['weights']}")
            print(f"Description: {info['description'].strip()}")
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    # Require input for processing
    if not args.input:
        parser.print_help()
        return 1

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Error: Input path '{args.input}' does not exist.", file=sys.stderr)
        return 1

    # Parse key=value stage params
    try:
        den_params = _parse_key_value_args(args.denoiser_param)
        up_params = _parse_key_value_args(args.upscaler_param)
    except Exception as e:
        print(f"Error parsing parameter arguments: {e}", file=sys.stderr)
        return 1

    if args.weights:
        up_params["model_path"] = args.weights

    target_size = tuple(args.size) if args.size else None

    # Folder mode
    if in_path.is_dir():
        out_dir = Path(args.output) if args.output else in_path / "upscaled"
        out_dir.mkdir(parents=True, exist_ok=True)

        if args.preset:
            pipe = ImagePipeline.from_preset(args.preset, device=args.device)
        elif args.steps:
            pipe = ImagePipeline.from_string(args.steps, device=args.device)
        elif args.denoiser and args.upscaler:
            pipe = ImagePipeline(device=args.device)
            if args.order == "denoise_first":
                pipe.add(args.denoiser, **den_params)
                pipe.add(args.upscaler, **up_params)
            else:
                pipe.add(args.upscaler, **up_params)
                pipe.add(args.denoiser, **den_params)
        elif args.denoiser:
            pipe = ImagePipeline(device=args.device).add(args.denoiser, **den_params)
        elif args.upscaler:
            pipe = ImagePipeline(device=args.device).add(args.upscaler, **up_params)
        else:
            # Default fallback: SPAN
            pipe = ImagePipeline(device=args.device).add("span", **up_params)

        pipe.run_folder(
            input_dir=in_path,
            output_dir=out_dir,
            target_size=target_size,
            scale=args.scale,
            fit=args.fit,
            tile=args.tile,
            tile_overlap=args.tile_overlap
        )
        return 0

    # Single file mode
    if not args.output:
        stem = in_path.stem
        suffix = in_path.suffix
        tool_label = args.preset or (f"{args.denoiser}_{args.upscaler}" if (args.denoiser and args.upscaler) else (args.denoiser or args.upscaler or "enhanced"))
        out_path = in_path.with_name(f"{stem}_{tool_label}{suffix}")
    else:
        out_path = Path(args.output)

    # Default to span if no tool specified
    den = args.denoiser
    up = args.upscaler
    if not den and not up and not args.steps and not args.preset:
        up = "span"

    try:
        out_img = process_image(
            image=in_path,
            denoiser=den,
            upscaler=up,
            order=args.order,
            steps=args.steps,
            preset=args.preset,
            target_size=target_size,
            scale=args.scale,
            fit=args.fit,
            denoiser_params=den_params,
            upscaler_params=up_params,
            device=args.device,
            tile=args.tile,
            tile_overlap=args.tile_overlap,
            save=out_path
        )
        print(f"Successfully processed: {in_path} -> {out_path} (Output size: {out_img.size})")
        return 0
    except Exception as e:
        print(f"Processing failed: {e}", file=sys.stderr)
        return 1
