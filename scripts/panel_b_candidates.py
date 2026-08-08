#!/usr/bin/env python3
"""Candidate records for panel b of the camera-ready figure.

Panel b has to support one specific claim: a region can pass the action gate and
still hand the workflow points that land on nothing. So the record shown must be
an aligned pass — answer correct, action over the gate — that nevertheless emits
several stray points, on a crop with more than one ground-truth object so the coverage
is non-trivial.

Selection criteria, all read from the shipped release:

    obj_recall      >= 0.5   passes the gate at the published tau
    points_stray    >= 3     the strays are visually obvious in print
    n_gt            >= 2     coverage is not a one-of-one special case
    answer_correct  == 1     the region is an aligned pass
    image_file      != ""    a crop image exists to show

Ranked by points_stray descending, then n_gt descending. This prints the top ten
and stops; choosing the record is a judgement about the figure, not a computation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

from sar.data import build_records, load_release, with_gate  # noqa: E402

TAU = 0.5
BUILD = "case_study"
CONDITION = "Staged GRPO"

COLUMNS = [
    "crop_id", "question_id", "task", "dataset", "n_gt", "n_pred",
    "points_on_target", "points_stray", "obj_recall", "point_f1", "pim_prec",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--release", type=Path, default=APP_DIR / "release")
    args = ap.parse_args()

    df, manifest = load_release(args.release)
    scored = with_gate(build_records(df, BUILD, CONDITION), TAU)

    passes = scored["obj_recall"] >= TAU
    strays = scored["points_stray"] >= 3
    objects = scored["n_gt"] >= 2
    correct = scored["answer_correct"] == 1
    has_image = scored["image_file"] != ""

    hits = scored[passes & strays & objects & correct & has_image]
    ranked = hits.sort_values(
        ["points_stray", "n_gt", "question_id"], ascending=[False, False, True]
    ).reset_index(drop=True)

    print(f"record set : {manifest['builds'][BUILD]['short_label']} "
          f"({len(scored)} records)")
    print(f"condition  : {manifest['model']} / {CONDITION}")
    print(f"gate       : tau = {TAU}")
    print("criteria   : obj_recall >= 0.5, points_stray >= 3, n_gt >= 2, "
          "answer_correct == 1, crop image present")
    print(f"matching   : {len(ranked)} of {len(scored)} records\n")

    # How the pool narrows, so a thin result is explained rather than mysterious.
    print("how the criteria narrow the pool")
    cumulative = [
        ("all records in this set", scored.index == scored.index),
        ("+ answer_correct == 1", correct),
        ("+ obj_recall >= 0.5 (aligned pass)", correct & passes),
        ("+ crop image present", correct & passes & has_image),
        ("+ n_gt >= 2", correct & passes & has_image & objects),
        ("+ points_stray >= 3", correct & passes & has_image & objects & strays),
    ]
    for label, mask in cumulative:
        print(f"  {label:38s} {int(mask.sum()):4d}")

    base = correct & passes & has_image & objects
    spread = scored[base]["points_stray"].value_counts().sort_index()
    print(f"\nstray-point spread across the {int(base.sum())} aligned passes with n_gt >= 2")
    for value, count in spread.items():
        print(f"  {int(value)} stray point(s): {int(count)}")
    print()

    if ranked.empty:
        print("No record satisfies all five criteria. Relax points_stray or n_gt.")
        return

    widths = {c: max(len(c), *(len(f"{v}") for v in ranked[c].head(args.top))) for c in COLUMNS}
    header = "  ".join(c.ljust(widths[c]) for c in COLUMNS)
    print(header)
    print("-" * len(header))
    for _, row in ranked.head(args.top).iterrows():
        cells = []
        for c in COLUMNS:
            v = row[c]
            if c in ("obj_recall", "point_f1", "pim_prec"):
                v = "—" if v != v else f"{float(v):.3f}"
            cells.append(str(v).ljust(widths[c]))
        print("  ".join(cells))

    if len(ranked) < args.top:
        print(f"\nOnly {len(ranked)} record(s) satisfy all five criteria. Two relaxations, "
              "for comparison:")
        for label, mask in [
            ("points_stray >= 2 instead of 3", base & (scored["points_stray"] >= 2)),
            ("n_gt >= 1 instead of 2", correct & passes & has_image & strays),
        ]:
            alt = scored[mask].sort_values(
                ["points_stray", "n_gt", "question_id"], ascending=[False, False, True])
            print(f"\n  {label} -> {len(alt)} record(s)")
            for _, row in alt.head(args.top).iterrows():
                print(f"    {row['crop_id']}  {row['task']:9s} {str(row['dataset']):7s} "
                      f"n_gt={int(row['n_gt'])} n_pred={int(row['n_pred'])} "
                      f"on_target={int(row['points_on_target'])} "
                      f"stray={int(row['points_stray'])} recall={row['obj_recall']:.3f}")

    print("\nChoose one and pass it to scripts/capture_figures.py --record <crop_id>.")


if __name__ == "__main__":
    main()
