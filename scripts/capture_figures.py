#!/usr/bin/env python3
"""Capture the camera-ready figure panels from the running dashboard.

Each panel is reached by URL, not by clicking, and captured as an element
screenshot rather than a page screenshot, so the crop is tight and the output is
byte-stable across runs and machines. The URL that produced each panel is printed
alongside it and can be cited in the paper.

    python scripts/capture_figures.py --record crop_aff63b305ab2

Requires the dev extras:

    pip install -r requirements-dev.txt
    playwright install chromium

The app must already be running (`streamlit run app.py`); pass --base-url if it is
not on http://localhost:8501.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import urlencode

APP_DIR = Path(__file__).resolve().parent.parent

# Defaults for the camera-ready run. Panel width is set by the dashboard's column
# width, so it scales with the viewport: widen the viewport to widen the panels.
VIEWPORT = {"width": 1600, "height": 1200}
DEVICE_SCALE = 3
MIN_WIDTH_PX = 1800

# Panels, in the order they appear in the figure. Each names the selector of the
# container it captures; those containers are created with `st.container(key=...)`
# in app.py, which Streamlit renders as `.st-key-<key>` — a stable hook that does
# not depend on the surrounding layout.
# `with_record` marks the panels whose content is a specific region. Panel a is
# aggregate — the ledger and the queue size — so it carries no record and instead
# selects the state whose cell should read as active. That split is necessary
# rather than cosmetic: a record that passes the gate cannot also appear in the
# silent-failure queue, so one record cannot satisfy both a silent-failure ledger
# selection and an aligned-pass crop.
PANELS = [
    {
        "name": "panel_a",
        "selector": ".st-key-panel_a",
        "state": "silent_failure",
        "with_record": False,
        "shows": "answer-action ledger with the selected cell, plus the audit queue caption",
    },
    {
        "name": "panel_b",
        "selector": ".st-key-panel_b",
        "state": None,          # taken from the record's own state
        "with_record": True,
        "shows": "crop tile, stray-actions panel, and the spatial-metrics block",
    },
    {
        "name": "panel_c",
        "selector": ".st-key-panel_c",
        "state": None,
        "with_record": True,
        "shows": "routing decision: four buttons, reason field, sign-off count",
    },
]

BASE_QUERY = {
    "source": "case_study",
    "condition": "staged_grpo",
    "tau": "0.50",
    "theme": "day",
}


def record_state(record: str, release_dir: Path) -> str:
    """The URL `state` whose queue contains this record, read from the release."""
    sys.path.insert(0, str(APP_DIR))
    from sar import urlstate
    from sar.data import build_records, load_release, with_gate

    df, _ = load_release(release_dir)
    scored = with_gate(build_records(df, "case_study", "Staged GRPO"),
                       float(BASE_QUERY["tau"]))
    match = scored[scored["crop_id"] == record]
    if match.empty:
        sys.exit(f"record {record} is not in the case-study record set")
    quadrant = str(match.iloc[0]["quadrant"])
    return urlstate.QUADRANT_TO_STATE[quadrant]


def build_url(base_url: str, record: str | None, state: str) -> str:
    query = dict(BASE_QUERY)
    query["state"] = state
    if record:
        query["record"] = record
    return f"{base_url.rstrip('/')}/?{urlencode(query)}"


def capture(base_url: str, record: str, out_dir: Path, settle_ms: int,
            state_for_record: str, viewport: dict, device_scale: int) -> list[dict]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit(
            "playwright is not installed. It is a development dependency:\n"
            "  pip install -r requirements-dev.txt\n"
            "  playwright install chromium"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=viewport, device_scale_factor=device_scale)
        try:
            for panel in PANELS:
                wants_record = panel["with_record"]
                state = panel["state"] or state_for_record
                url = build_url(base_url, record if wants_record else None, state)
                page.goto(url, wait_until="networkidle", timeout=120_000)
                page.wait_for_timeout(settle_ms)
                if panel.get("tab"):
                    page.get_by_role("tab", name=panel["tab"]).click()
                    page.wait_for_timeout(settle_ms)

                errors = [
                    (el.inner_text() or "")[:400]
                    for el in page.query_selector_all('[data-testid="stException"]')
                ]
                if errors:
                    sys.exit(f"{panel['name']}: the app raised an exception:\n{errors[0]}")

                # Confirm the URL actually opened the record asked for, so a panel
                # can never silently show a different region.
                shown = page.evaluate(
                    """() => {
                        const r = [...document.querySelectorAll('.sar-row')]
                            .find(x => x.innerText.startsWith('image region'));
                        return r ? r.innerText.split('\\n').pop().trim() : null;
                    }"""
                )
                if wants_record and shown != record:
                    sys.exit(
                        f"{panel['name']}: asked for record {record} but the page shows "
                        f"{shown!r}. Check that the record is in the {state} queue."
                    )

                locator = page.locator(panel["selector"])
                if locator.count() == 0:
                    sys.exit(f"{panel['name']}: selector {panel['selector']} matched nothing")
                locator.first.scroll_into_view_if_needed()
                page.wait_for_timeout(400)
                # Park the cursor away from the panel so no hover state or tooltip
                # is baked into the figure.
                page.mouse.move(viewport["width"] - 4, viewport["height"] - 4)
                page.wait_for_timeout(500)

                path = out_dir / f"{panel['name']}.png"
                locator.first.screenshot(path=str(path))

                from PIL import Image

                with Image.open(path) as img:
                    width, height = img.size
                results.append({
                    "name": panel["name"], "path": path, "url": url,
                    "width": width, "height": height, "shows": panel["shows"],
                })
        finally:
            browser.close()
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", required=True,
                    help="crop_id to show in every panel; see scripts/panel_b_candidates.py")
    ap.add_argument("--base-url", default="http://localhost:8501")
    ap.add_argument("--out", type=Path, default=APP_DIR / "figures")
    ap.add_argument("--release", type=Path, default=APP_DIR / "release")
    ap.add_argument("--viewport-width", type=int, default=VIEWPORT["width"],
                    help="panel width follows the dashboard column width, so widen "
                         "this to widen the panels")
    ap.add_argument("--viewport-height", type=int, default=VIEWPORT["height"])
    ap.add_argument("--device-scale", type=int, default=DEVICE_SCALE)
    ap.add_argument("--settle-ms", type=int, default=9000,
                    help="wait after load, for Streamlit to finish rendering")
    args = ap.parse_args()

    viewport = {"width": args.viewport_width, "height": args.viewport_height}
    state_for_record = record_state(args.record, args.release)
    results = capture(args.base_url, args.record, args.out, args.settle_ms,
                      state_for_record, viewport, args.device_scale)

    print(f"record  : {args.record}  (state: {state_for_record})")
    print(f"viewport: {viewport['width']}x{viewport['height']} @ {args.device_scale}x\n")
    warnings: list[str] = []
    for r in results:
        rel = r["path"].relative_to(APP_DIR) if r["path"].is_relative_to(APP_DIR) else r["path"]
        print(f"{r['name']}  {r['width']} x {r['height']} px  {rel}")
        print(f"          {r['shows']}")
        print(f"          {r['url']}")
        if r["width"] < MIN_WIDTH_PX:
            warnings.append(
                f"{r['name']} is {r['width']} px wide, under the {MIN_WIDTH_PX} px "
                "needed for a full-width two-column figure"
            )
    if warnings:
        print("\nwarnings")
        for w in warnings:
            print(f"  ! {w}")
        narrowest = min(r["width"] for r in results)
        # The stylesheet caps .block-container at 1340 px, so widening the viewport
        # stops helping once the columns reach that cap. Device scale keeps scaling.
        scale = -(-MIN_WIDTH_PX * args.device_scale // narrowest)  # ceil
        print(f"  Panel width is capped by the 1340 px content column, so widening "
              f"--viewport-width past ~1700 stops helping.")
        print(f"  Re-run with --device-scale {scale} to clear {MIN_WIDTH_PX} px.")


if __name__ == "__main__":
    main()
