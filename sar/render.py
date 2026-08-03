"""Crop-evidence rendering: the EM image with GT and predicted-point overlays.

Two rules govern this module, because the dashboard is an audit instrument:

1. Nothing is invented. When a record's EM crop is not in the public release,
   the tile is drawn as an explicit blank grid, never as synthetic EM
   texture. The point geometry is still shown, because the normalised
   coordinates are released even when the pixels are not.
2. Overlay positions come only from the released coordinates. Green crosses are
   the label-derived ground-truth centroids; red rings are the model's parsed
   predicted points. Neither is jittered, snapped, or re-ordered.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from . import theme as theme_mod

TILE = 320  # rendered tile edge in px; source crops are 256x256


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _draw_overlays(canvas: Image.Image, gt: list, pred: list, palette) -> None:
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    w, h = canvas.size
    r_halo, r_arm, r_ring = 0.072 * w, 0.032 * w, 0.028 * w
    gt_stroke = _rgb(palette.reliable)
    pred_stroke = _rgb(palette.silent)

    for point in gt:
        x, y = float(point[0]) * w, float(point[1]) * h
        draw.ellipse([x - r_halo, y - r_halo, x + r_halo, y + r_halo], fill=gt_stroke + (78,))
        draw.line([x - r_arm, y - r_arm, x + r_arm, y + r_arm], fill=gt_stroke, width=2)
        draw.line([x + r_arm, y - r_arm, x - r_arm, y + r_arm], fill=gt_stroke, width=2)

    for point in pred:
        x, y = float(point[0]) * w, float(point[1]) * h
        draw.ellipse([x - r_ring, y - r_ring, x + r_ring, y + r_ring],
                     outline=pred_stroke, width=3)
        draw.line([x - 2, y, x + 2, y], fill=pred_stroke, width=1)

    canvas.alpha_composite(layer)


def _placeholder(size: int, palette) -> Image.Image:
    """Blank grid used when the crop image is not available.

    Deliberately not EM-like. An audit tool that draws plausible microscopy
    texture where it has none is worse than one that shows an obvious blank.
    """
    grid_bg, grid_line = _rgb(palette.grid_bg), _rgb(palette.grid_line)
    canvas = Image.new("RGBA", (size, size), grid_bg + (255,))
    draw = ImageDraw.Draw(canvas)
    step = size // 8
    for i in range(1, 8):
        draw.line([i * step, 0, i * step, size], fill=grid_line, width=1)
        draw.line([0, i * step, size, i * step], fill=grid_line, width=1)
    draw.rectangle([0, 0, size - 1, size - 1], outline=grid_line, width=1)
    return canvas


def crop_tile(
    image_path: Path | None,
    gt_centroids: list,
    pred_points: list,
    size: int = TILE,
    palette_name: str = theme_mod.DEFAULT,
) -> tuple[Image.Image, bool]:
    """Return (tile, image_available).

    `image_available` is False when the tile is the blank grid, so the caller
    can label the panel honestly instead of letting a placeholder pass as data.
    """
    palette = theme_mod.get(palette_name)
    available = image_path is not None and Path(image_path).exists()
    if available:
        # NEAREST keeps the 256 -> 320 upscale free of interpolated pixels that
        # do not exist in the acquired image.
        base = Image.open(image_path).convert("RGBA").resize((size, size), Image.NEAREST)
    else:
        base = _placeholder(size, palette)

    _draw_overlays(base, gt_centroids or [], pred_points or [], palette)
    return base.convert("RGB"), available
