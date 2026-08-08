"""The two dashboard palettes.

`night` is the graphite palette carried over unchanged from
`plan/oversight_inspector.html`, so the live app and the paper's dashboard figure
read as one system.

`day` is the light counterpart. Its four state colours are the ones
`plan/04_reliability_analysis.py` uses for the printed figures, which ties the
light dashboard to the quadrant and risk-map figures in the paper rather than
inventing a third palette.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    name: str
    ground: str      # page background
    panel: str       # card background
    panel2: str      # inset background
    hair: str        # hairline borders
    ink: str         # primary text
    muted: str       # secondary text
    dim: str         # tertiary text / keys
    reliable: str    # aligned pass, GT evidence
    silent: str      # silent failure, predicted points
    lucky: str       # action-only pass
    honest: str      # blocked
    phosphor: str    # interactive / selected
    grid_bg: str     # placeholder tile background
    grid_line: str   # placeholder tile grid
    heat_floor: str  # risk-map 0% colour
    on_accent: str   # text drawn on top of an accent fill
    warn: str        # caution accent, used only by notice banners


NIGHT = Palette(
    name="night",
    ground="#10151b", panel="#171e26", panel2="#1c242e", hair="#283442",
    ink="#cdd8e2", muted="#7c8b99", dim="#586675",
    reliable="#35c08a", silent="#e0526a", lucky="#4c7ab0", honest="#6b7885",
    phosphor="#4fd0e0",
    grid_bg="#12181f", grid_line="#283442",
    heat_floor="#141b23", on_accent="#10151b", warn="#d9a441",
)

DAY = Palette(
    name="day",
    ground="#f4f6f8", panel="#ffffff", panel2="#eef2f5", hair="#ccd5df",
    ink="#18212b", muted="#5a6773", dim="#78848f",
    # Figure palette from 04_reliability_analysis.py.
    reliable="#1f9e6b", silent="#c9364f", lucky="#3d5a80", honest="#6b7885",
    phosphor="#116b8f",
    grid_bg="#e7ecf1", grid_line="#c3ced9",
    heat_floor="#f3f6f8", on_accent="#ffffff", warn="#a4661a",
)

PALETTES = {p.name: p for p in (NIGHT, DAY)}
DEFAULT = NIGHT.name

LABELS = {"night": "Night", "day": "Day"}


def get(name: str) -> Palette:
    return PALETTES.get(name, NIGHT)
