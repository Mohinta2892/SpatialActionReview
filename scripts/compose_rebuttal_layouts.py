#!/usr/bin/env python3
"""Compose draft paper figures from captured dashboard panels.

The raw screenshots stay in `figures/`. This script creates two layout drafts:

  - `layout_dashboard_audit_workflow`: ledger, image-region audit, and routing.
  - `layout_revision_and_regression`: side-by-side model revision plus aggregate
    regression/churn plots from the matched 753-record audit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

APP_DIR = Path(__file__).resolve().parent.parent


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def load(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def fit_width(img: Image.Image, width: int) -> Image.Image:
    h = round(img.height * width / img.width)
    return img.resize((width, h), Image.LANCZOS)


def crop_height(img: Image.Image, max_height: int) -> Image.Image:
    return img.crop((0, 0, img.width, min(img.height, max_height)))


def label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], letter: str, text: str) -> None:
    draw.text(xy, f"{letter}", fill=(20, 27, 36), font=font(54, True))
    draw.text((xy[0] + 64, xy[1] + 10), text, fill=(47, 60, 77), font=font(30, True))


def paste(canvas: Image.Image, img: Image.Image, xy: tuple[int, int]) -> None:
    canvas.paste(img, xy)
    draw = ImageDraw.Draw(canvas)
    x, y = xy
    draw.rectangle((x, y, x + img.width - 1, y + img.height - 1), outline=(210, 219, 229), width=3)


def save(canvas: Image.Image, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(stem.with_suffix(".png"))
    canvas.save(stem.with_suffix(".pdf"), "PDF", resolution=300.0)


def audit_workflow(figures: Path, out: Path) -> None:
    panel_a = fit_width(load(figures / "panel_a.png"), 1450)
    panel_c = fit_width(load(figures / "panel_c.png"), 1450)
    panel_b = fit_width(crop_height(load(figures / "panel_b.png"), 4550), 1700)

    pad = 90
    title_h = 150
    gap = 70
    width = pad * 2 + panel_a.width + gap + panel_b.width
    left_h = panel_a.height + gap + panel_c.height
    height = title_h + pad + max(left_h, panel_b.height) + pad
    canvas = Image.new("RGB", (width, height), (246, 248, 251))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 50), "Dashboard audit workflow", fill=(20, 27, 36), font=font(56, True))
    draw.text(
        (pad, 112),
        "The supervisor moves from aggregate state, to image evidence, to a recorded routing decision.",
        fill=(88, 103, 121),
        font=font(30),
    )

    x0, y0 = pad, title_h
    label(draw, (x0, y0), "a", "Run-level answer-action ledger")
    paste(canvas, panel_a, (x0, y0 + 80))
    y_c = y0 + 80 + panel_a.height + gap
    label(draw, (x0, y_c), "c", "Routing decision")
    paste(canvas, panel_c, (x0, y_c + 80))

    x1 = x0 + panel_a.width + gap
    label(draw, (x1, y0), "b", "Image-region evidence")
    paste(canvas, panel_b, (x1, y0 + 80))
    save(canvas, out / "layout_dashboard_audit_workflow")


def revision_and_regression(figures: Path, out: Path) -> None:
    add = figures / "rebuttal_additions"
    revision = fit_width(crop_height(load(figures / "panel_revision.png"), 4300), 1750)
    trans = fit_width(load(add / "version_transition_matrices.png"), 1500)
    churn = fit_width(load(add / "audit_queue_churn_heatmap.png"), 720)
    workload = fit_width(load(add / "gate_workload_sweep.png"), 720)

    pad = 90
    title_h = 155
    gap = 70
    right_w = max(trans.width, churn.width + gap + workload.width)
    width = pad * 2 + revision.width + gap + right_w
    right_h = trans.height + gap + max(churn.height, workload.height)
    height = title_h + pad + max(revision.height, right_h) + pad
    canvas = Image.new("RGB", (width, height), (246, 248, 251))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 50), "Model revision audit", fill=(20, 27, 36), font=font(56, True))
    draw.text(
        (pad, 112),
        "Matched records reveal which failures are resolved, which persist, and which appear after an update.",
        fill=(88, 103, 121),
        font=font(30),
    )

    x0, y0 = pad, title_h
    label(draw, (x0, y0), "a", "One record before and after revision")
    paste(canvas, revision, (x0, y0 + 80))

    x1 = x0 + revision.width + gap
    label(draw, (x1, y0), "b", "State transitions across versions")
    paste(canvas, trans, (x1, y0 + 80))
    y2 = y0 + 80 + trans.height + gap
    label(draw, (x1, y2), "c", "Review-queue reuse")
    paste(canvas, churn, (x1, y2 + 80))
    x3 = x1 + churn.width + gap
    label(draw, (x3, y2), "d", "Workload as the gate changes")
    paste(canvas, workload, (x3, y2 + 80))
    save(canvas, out / "layout_revision_and_regression")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figures", type=Path, default=APP_DIR / "figures")
    ap.add_argument("--out", type=Path, default=APP_DIR / "figures" / "rebuttal_layouts")
    args = ap.parse_args()
    audit_workflow(args.figures, args.out)
    revision_and_regression(args.figures, args.out)
    for path in sorted(args.out.glob("layout_*.*")):
        print(path)


if __name__ == "__main__":
    main()
