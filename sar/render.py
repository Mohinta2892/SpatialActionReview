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

# Overlay marks are scaled up for print legibility. The legend in sar/ui.py draws
# the same marks at the same relative proportions, so the two stay in step.
MARKER_SCALE = 1.6


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _draw_overlays(canvas: Image.Image, gt: list, pred: list, palette,
                   live: list | None = None) -> None:
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    w, h = canvas.size
    # Sized for print. At two-column width a 256 px crop is reproduced around
    # 40 mm across, and the previous markers (ring radius 0.028w, stroke 3 px)
    # closed up to almost nothing. Every dimension below is the earlier value
    # scaled by MARKER_SCALE.
    r_halo = MARKER_SCALE * 0.072 * w
    r_arm = MARKER_SCALE * 0.032 * w
    r_ring = MARKER_SCALE * 0.028 * w
    gt_width = max(1, round(MARKER_SCALE * 2))
    pred_width = max(1, round(MARKER_SCALE * 3))
    tick = max(1, round(MARKER_SCALE * 2))
    gt_stroke = _rgb(palette.reliable)
    pred_stroke = _rgb(palette.silent)
    live_stroke = _rgb(palette.phosphor)

    for point in gt:
        x, y = float(point[0]) * w, float(point[1]) * h
        draw.ellipse([x - r_halo, y - r_halo, x + r_halo, y + r_halo], fill=gt_stroke + (78,))
        draw.line([x - r_arm, y - r_arm, x + r_arm, y + r_arm], fill=gt_stroke, width=gt_width)
        draw.line([x + r_arm, y - r_arm, x - r_arm, y + r_arm], fill=gt_stroke, width=gt_width)

    for point in pred:
        x, y = float(point[0]) * w, float(point[1]) * h
        draw.ellipse([x - r_ring, y - r_ring, x + r_ring, y + r_ring],
                     outline=pred_stroke, width=pred_width)
        draw.line([x - tick, y, x + tick, y], fill=pred_stroke, width=1)

    # Live re-ask points get their own mark — a diamond in the interactive colour —
    # so a fresh answer can never be mistaken for the recorded one it sits beside.
    for point in live or []:
        x, y = float(point[0]) * w, float(point[1]) * h
        draw.polygon([(x, y - r_ring), (x + r_ring, y), (x, y + r_ring), (x - r_ring, y)],
                     outline=live_stroke, width=pred_width)

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
    live_points: list | None = None,
) -> tuple[Image.Image, bool]:
    """Return (tile, image_available).

    `image_available` is False when the tile is the blank grid, so the caller
    can label the panel honestly instead of letting a placeholder pass as data.

    `live_points` are drawn only when a live re-ask has been made for the record;
    they are a separate mark and are never mixed into `pred_points`.
    """
    palette = theme_mod.get(palette_name)
    available = image_path is not None and Path(image_path).exists()
    if available:
        # NEAREST keeps the 256 -> 320 upscale free of interpolated pixels that
        # do not exist in the acquired image.
        base = Image.open(image_path).convert("RGBA").resize((size, size), Image.NEAREST)
    else:
        base = _placeholder(size, palette)

    _draw_overlays(base, gt_centroids or [], pred_points or [], palette, live_points or [])
    return base.convert("RGB"), available
