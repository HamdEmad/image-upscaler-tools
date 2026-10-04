import math
from typing import Callable, Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image


def _create_weight_map(h: int, w: int, overlap: int, device: torch.device) -> torch.Tensor:
    """Create a 2D feathered weight mask with linear ramps on overlapping boundaries."""
    h_weight = torch.ones(h, device=device)
    w_weight = torch.ones(w, device=device)

    if overlap > 0:
        ramp = torch.linspace(0.01, 1.0, overlap, device=device)
        if h > 2 * overlap:
            h_weight[:overlap] = ramp
            h_weight[-overlap:] = torch.flip(ramp, dims=[0])
        if w > 2 * overlap:
            w_weight[:overlap] = ramp
            w_weight[-overlap:] = torch.flip(ramp, dims=[0])

    return (h_weight.unsqueeze(1) * w_weight.unsqueeze(0)).unsqueeze(0).unsqueeze(0)


def tile_process_tensor(
    tensor: torch.Tensor,
    process_fn: Callable[[torch.Tensor], torch.Tensor],
    tile_size: int = 256,
    tile_overlap: int = 16,
    scale: int = 4
) -> torch.Tensor:
    """
    Process a 4D torch.Tensor [1, C, H, W] using overlapping tiles and linear feathering.
    
    Parameters
    ----------
    tensor : torch.Tensor
        Input image tensor [1, C, H, W].
    process_fn : Callable
        Function taking a tile [1, C, th, tw] and returning [1, C, th*scale, tw*scale].
    tile_size : int
        Size of square tiles (default 256).
    tile_overlap : int
        Overlap in pixels between adjacent tiles (default 16).
    scale : int
        Spatial scale factor produced by process_fn (default 4 for upscalers, 1 for denoisers).
    """
    _, c, h, w = tensor.shape
    device = tensor.device

    # If image fits within a single tile, process directly without tiling
    if h <= tile_size and w <= tile_size:
        return process_fn(tensor)

    stride = tile_size - tile_overlap
    h_tiles = max(1, math.ceil((h - tile_overlap) / stride))
    w_tiles = max(1, math.ceil((w - tile_overlap) / stride))

    out_h = h * scale
    out_w = w * scale

    output_canvas = torch.zeros((1, c, out_h, out_w), dtype=tensor.dtype, device=device)
    weight_canvas = torch.zeros((1, 1, out_h, out_w), dtype=tensor.dtype, device=device)

    for i in range(h_tiles):
        y1 = i * stride
        y2 = min(y1 + tile_size, h)
        if y2 - y1 < tile_size and y2 == h:
            y1 = max(0, h - tile_size)
        th = y2 - y1

        for j in range(w_tiles):
            x1 = j * stride
            x2 = min(x1 + tile_size, w)
            if x2 - x1 < tile_size and x2 == w:
                x1 = max(0, w - tile_size)
            tw = x2 - x1

            tile_in = tensor[:, :, y1:y2, x1:x2]
            tile_out = process_fn(tile_in)

            out_y1 = y1 * scale
            out_y2 = y2 * scale
            out_x1 = x1 * scale
            out_x2 = x2 * scale

            tile_out_h = out_y2 - out_y1
            tile_out_w = out_x2 - out_x1
            out_overlap = tile_overlap * scale

            weight = _create_weight_map(tile_out_h, tile_out_w, out_overlap, device)

            output_canvas[:, :, out_y1:out_y2, out_x1:out_x2] += tile_out * weight
            weight_canvas[:, :, out_y1:out_y2, out_x1:out_x2] += weight

    return output_canvas / (weight_canvas + 1e-8)
