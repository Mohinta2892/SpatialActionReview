#!/usr/bin/env python3
"""Every number the manuscript needs, read only from the shipped release.

Nothing here is transcribed from a figure or a previous draft: each value is
recomputed from `release/records.csv.gz`, so any number quoted in the paper can be
traced to the released records. The bootstrap matches the manuscript — percentile
bootstrap, B = 2000, `numpy.random.default_rng(0)`, resamples missing either
answer stratum skipped.

    python scripts/paper_numbers.py            # print the report
    python scripts/paper_numbers.py --json     # machine-readable, for tests
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

from sar.data import BOOT_SEED, N_BOOT, build_records, load_release, summarise, with_gate  # noqa: E402

TAU = 0.5
CASE = ("case_study", "Staged GRPO")
PERMUTATIONS = 10_000


# ------------------------------------------------------------------ helpers

def bootstrap_gap(scored: pd.DataFrame, n_boot: int = N_BOOT,
                  seed: int = BOOT_SEED) -> tuple[float, float, float]:
    """Trust gap and its percentile interval, exactly as the manuscript defines it."""
    ac = scored["answer_correct"].to_numpy()
    ar = scored["action_reliable"].to_numpy()
    if not (ac == 1).any() or not (ac == 0).any():
        return float("nan"), float("nan"), float("nan")
    point = float(ar[ac == 1].mean() - ar[ac == 0].mean())
    rng = np.random.default_rng(seed)
    idx = np.arange(len(scored))
    boots = []
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        cb, wb = ar[b][ac[b] == 1], ar[b][ac[b] == 0]
        if len(cb) and len(wb):
            boots.append(cb.mean() - wb.mean())
    lo, hi = np.nanpercentile(boots, [2.5, 97.5]) if boots else (float("nan"),) * 2
    return point, float(lo), float(hi)


def point_biserial(scored: pd.DataFrame) -> float:
    ac = scored["answer_correct"].to_numpy().astype(float)
    rec = scored["obj_recall"].to_numpy(dtype=float)
    if ac.std() == 0 or rec.std() == 0:
        return float("nan")
    return float(np.corrcoef(ac, rec)[0, 1])


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval: behaves sensibly for the small task cells."""
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def correlation_with_p(x: np.ndarray, y: np.ndarray, *, seed: int = BOOT_SEED
                       ) -> tuple[float, float, str]:
    """Pearson r and a p-value. Uses scipy when available, else a permutation test."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.std() == 0 or y.std() == 0:
        return float("nan"), float("nan"), "undefined (zero variance)"
    r = float(np.corrcoef(x, y)[0, 1])
    try:
        from scipy import stats
        return r, float(stats.pearsonr(x, y).pvalue), "scipy.stats.pearsonr (two-sided)"
    except Exception:
        rng = np.random.default_rng(seed)
        shuffled = y.copy()
        count = 0
        for _ in range(PERMUTATIONS):
            rng.shuffle(shuffled)
            if abs(np.corrcoef(x, shuffled)[0, 1]) >= abs(r):
                count += 1
        # Add-one correction, so the p-value is never reported as exactly zero.
        p = (count + 1) / (PERMUTATIONS + 1)
        return r, float(p), f"permutation test, {PERMUTATIONS} shuffles, seed {seed} (two-sided)"


def stray_row(scored: pd.DataFrame) -> dict:
    passing = scored[scored["action_reliable"] == 1]
    n_pass = len(passing)
    if not n_pass:
        return {"n_pass": 0}
    cleared_points = int(passing["n_pred"].sum())
    stray_points = int(passing["points_stray"].sum())
    with_stray = int((passing["points_stray"] > 0).sum())
    over = int((passing["n_pred"] > passing["n_gt"]).sum())
    return {
        "n_pass": n_pass,
        "with_stray": with_stray,
        "with_stray_share": with_stray / n_pass,
        "stray_points": stray_points,
        "cleared_points": cleared_points,
        "stray_point_share": (stray_points / cleared_points) if cleared_points else float("nan"),
        "mean_hit_rate": float(passing["point_precision_derived"].mean()),
        "over_emitting": over,
        "over_emitting_share": over / n_pass,
    }


def k_of(scored: pd.DataFrame) -> pd.Series:
    return scored["vqa_choices"].map(len)


# ------------------------------------------------------------------ report

def build_report(release_dir: Path, figures_dir: Path) -> dict:
    df, manifest = load_release(release_dir)
    case = with_gate(build_records(df, *CASE), TAU)
    out: dict = {"tau": TAU, "bootstrap": {"n_resamples": N_BOOT, "seed": BOOT_SEED}}
    lines: list[str] = []

    def head(n: int, title: str) -> None:
        lines.append("")
        lines.append(f"{n}. {title}")
        lines.append("-" * 78)

    lines.append("=" * 78)
    lines.append("SPATIAL ACTION REVIEW — numbers for the manuscript")
    lines.append(f"source: {release_dir.name}/records.csv.gz  "
                 f"sha256 {manifest['records_sha256'][:16]}…")
    lines.append(f"gate: tau = {TAU}   bootstrap: B = {N_BOOT}, "
                 f"numpy default_rng({BOOT_SEED})")
    lines.append("=" * 78)

    # 1 --------------------------------------------------------------------
    head(1, "Case study, Staged GRPO, tau=0.5 — mean object recall")
    recall = float(case["obj_recall"].mean())
    out["1_mean_obj_recall"] = recall
    lines.append(f"  record set        Case study (541), Qwen3-VL / Staged GRPO")
    lines.append(f"  mean object recall {recall:.5f}")
    lines.append(f"  as a percentage    {100 * recall:.3f}%")
    lines.append(f"  rounds to          {100 * recall:.1f}%  "
                 f"(one decimal place)")
    lines.append(f"  -> the manuscript's 38.9% is "
                 f"{'CORRECT' if round(100 * recall, 1) == 38.9 else 'WRONG'}; "
                 f"39.0% would be {'correct' if round(100 * recall, 1) == 39.0 else 'wrong'}")

    # 2 --------------------------------------------------------------------
    head(2, "Case study — point-in-mask precision among answer-correct records")
    correct = case[case["answer_correct"] == 1]
    pim = pd.to_numeric(correct["pim_prec"], errors="coerce")
    pim_mean = float(pim.mean())
    out["2_pim_given_correct"] = pim_mean
    out["2_pim_n"] = int(pim.notna().sum())
    lines.append(f"  record set        Case study (541), Qwen3-VL / Staged GRPO")
    lines.append(f"  answer-correct    {len(correct)} records "
                 f"({int(pim.notna().sum())} with a point-in-mask value)")
    lines.append(f"  mean              {pim_mean:.3f}   (unrounded {pim_mean:.6f})")
    lines.append(f"  -> the manuscript's 0.471 is "
                 f"{'CONFIRMED' if round(pim_mean, 3) == 0.471 else 'NOT CONFIRMED'}")

    # 3 --------------------------------------------------------------------
    head(3, "Stray actions on regions that pass the gate — every released condition")
    rows = [("Case study (541)", CASE[1], case)]
    for cond in manifest["builds"]["sft_variants"]["conditions"]:
        rows.append(("Matched set (753)", cond,
                     with_gate(build_records(df, "sft_variants", cond), TAU)))
    out["3_stray"] = {}
    lines.append(f"  {'record set':<18} {'condition':<15} {'pass':>5} {'≥1 stray':>13} "
                 f"{'stray pts':>15} {'hit rate':>9} {'over-emitting':>15}")
    for set_name, cond, scored in rows:
        r = stray_row(scored)
        out["3_stray"][f"{set_name}|{cond}"] = r
        lines.append(
            f"  {set_name:<18} {cond:<15} {r['n_pass']:>5} "
            f"{r['with_stray']:>5} ({100 * r['with_stray_share']:>4.0f}%) "
            f"{r['stray_points']:>6}/{r['cleared_points']:<4} ({100 * r['stray_point_share']:>3.0f}%) "
            f"{r['mean_hit_rate']:>9.3f} "
            f"{r['over_emitting']:>6} ({100 * r['over_emitting_share']:>4.0f}%)"
        )
    lines.append("  columns: regions whose action passes | of those, count and share emitting")
    lines.append("           at least one stray point | stray points as a share of all points")
    lines.append("           from passing regions | mean hit rate among passing regions |")
    lines.append("           passing regions returning more points than the region has objects")

    # 4 --------------------------------------------------------------------
    head(4, "Staged GRPO on the full 753-record matched set — candidate Table 3 row")
    grpo = with_gate(build_records(df, "sft_variants", "Staged GRPO"), TAU)
    s = summarise(grpo)
    gap, lo, hi = bootstrap_gap(grpo)
    out["4_grpo_753"] = {
        "n": int(s.n), "vqa_acc": s.vqa_acc, "obj_recall": s.obj_recall,
        "silent_failure_rate": s.silent_failure_rate, "trust_gap": gap,
        "gap_lo": lo, "gap_hi": hi, "point_biserial": s.point_biserial,
    }
    lines.append(f"  record set        Matched set (753), Qwen3-VL / Staged GRPO")
    lines.append(f"  flagged in the release as not a row of the reported table")
    lines.append(f"  n                 {s.n}")
    lines.append(f"  answer accuracy   {100 * s.vqa_acc:.1f}%   ({s.vqa_acc:.4f})")
    lines.append(f"  mean obj recall   {100 * s.obj_recall:.1f}%   ({s.obj_recall:.4f})")
    lines.append(f"  silent failure    {100 * s.silent_failure_rate:.1f}%   "
                 f"({s.silent_failure_rate:.4f})")
    lines.append(f"  trust gap         {100 * gap:+.1f} pp  "
                 f"[{100 * lo:+.1f}, {100 * hi:+.1f}]  (interval "
                 f"{'includes' if lo < 0 < hi else 'EXCLUDES'} zero)")
    lines.append(f"  point-biserial    {s.point_biserial:+.3f}")
    lines.append(f"  Table 3 row:      Staged GRPO & {100 * s.vqa_acc:.1f} & "
                 f"{100 * s.obj_recall:.1f} & {100 * s.silent_failure_rate:.1f} & "
                 f"{100 * gap:.1f} [{100 * lo:.1f}, {100 * hi:.1f}] & "
                 f"{s.point_biserial:.3f}")

    # 5 --------------------------------------------------------------------
    head(5, "Table 2 marginals with Wilson 95% intervals (case study)")
    out["5_marginals"] = {"task": {}, "dataset": {}}
    for axis, label in (("task", "task, pooled across datasets"),
                        ("dataset", "dataset, pooled across tasks")):
        lines.append(f"  {label}")
        lines.append(f"    {'group':<12} {'silent':>7} {'n':>5} {'rate':>8}   Wilson 95%")
        for key, group in case.groupby(axis, observed=True):
            sf = int(((group["answer_correct"] == 1)
                      & (group["action_reliable"] == 0)).sum())
            n = len(group)
            lo_w, hi_w = wilson(sf, n)
            out["5_marginals"][axis][str(key)] = {
                "silent": sf, "n": n, "rate": sf / n, "lo": lo_w, "hi": hi_w}
            lines.append(f"    {str(key):<12} {sf:>7} {n:>5} {100 * sf / n:>7.1f}%   "
                         f"[{100 * lo_w:.1f}%, {100 * hi_w:.1f}%]")
        lines.append("")
    lines.append("  the location row pools direct-location and marked-region probes")

    # 6 --------------------------------------------------------------------
    head(6, "Case study excluding single-option (K=1) records")
    k = k_of(case)
    kept = case[k >= 2]
    dropped = int((k == 1).sum())
    s6 = summarise(kept)
    gap6, lo6, hi6 = bootstrap_gap(kept)
    out["6_no_k1"] = {"dropped": dropped, "n": int(s6.n),
                      "silent_failure_rate": s6.silent_failure_rate,
                      "conditional_failure": s6.p_unreliable_given_correct,
                      "trust_gap": gap6, "gap_lo": lo6, "gap_hi": hi6}
    lines.append(f"  record set        Case study (541) minus {dropped} K=1 records "
                 f"-> {s6.n}")
    lines.append(f"  silent failure    {100 * s6.silent_failure_rate:.1f}%  "
                 f"(all 541: {100 * summarise(case).silent_failure_rate:.1f}%)")
    lines.append(f"  conditional fail  {100 * s6.p_unreliable_given_correct:.1f}%  "
                 f"(all 541: {100 * summarise(case).p_unreliable_given_correct:.1f}%)")
    lines.append(f"  trust gap         {100 * gap6:+.1f} pp  "
                 f"[{100 * lo6:+.1f}, {100 * hi6:+.1f}]  (interval "
                 f"{'includes' if lo6 < 0 < hi6 else 'EXCLUDES'} zero)")

    # 7 --------------------------------------------------------------------
    head(7, "Case study restricted to K >= 4 (guessing less likely)")
    wide = case[k >= 4]
    s7 = summarise(wide)
    gap7, lo7, hi7 = bootstrap_gap(wide)
    out["7_k_ge_4"] = {"n": int(s7.n), "trust_gap": gap7, "gap_lo": lo7, "gap_hi": hi7,
                       "point_biserial": s7.point_biserial}
    lines.append(f"  record set        Case study (541), K >= 4 -> {s7.n} records")
    lines.append(f"  trust gap         {100 * gap7:+.1f} pp  "
                 f"[{100 * lo7:+.1f}, {100 * hi7:+.1f}]  (interval "
                 f"{'includes' if lo7 < 0 < hi7 else 'EXCLUDES'} zero)")
    lines.append(f"  point-biserial    {s7.point_biserial:+.3f}")
    lines.append("  -> the near-zero coupling "
                 f"{'survives' if lo7 < 0 < hi7 else 'does NOT survive'} the restriction")

    # 8 --------------------------------------------------------------------
    head(8, "Does the silent-failure rate track object density?")
    sf_flag = ((case["answer_correct"] == 1) & (case["action_reliable"] == 0)).astype(float)
    r8, p8, method = correlation_with_p(sf_flag.to_numpy(), case["n_gt"].to_numpy())
    out["8_sf_vs_ngt"] = {"r": r8, "p": p8, "method": method}
    lines.append(f"  record set        Case study (541)")
    lines.append(f"  corr(silent-failure indicator, n_gt)  r = {r8:+.4f}")
    lines.append(f"  p-value           {p8:.4f}   [{method}]")
    lines.append(f"  -> {'no' if p8 > 0.05 else 'some'} evidence that silent failure tracks "
                 "object density rather than model behaviour")

    # 9 --------------------------------------------------------------------
    head(9, "Tau sweep, case study")
    sweep = []
    lines.append(f"    {'tau':>5} {'silent':>8} {'aligned':>8} {'trust gap':>11}   interval")
    for i in range(1, 20):
        t = round(0.05 * i, 2)
        g = with_gate(build_records(df, *CASE), t)
        c = g["answer_correct"] == 1
        rel = g["action_reliable"] == 1
        gap_t, lo_t, hi_t = bootstrap_gap(g)
        row = {"tau": t, "silent_failure_rate": float((c & ~rel).mean()),
               "aligned_pass_rate": float((c & rel).mean()),
               "trust_gap": gap_t, "gap_lo": lo_t, "gap_hi": hi_t}
        sweep.append(row)
        lines.append(f"    {t:>5.2f} {100 * row['silent_failure_rate']:>7.1f}% "
                     f"{100 * row['aligned_pass_rate']:>7.1f}% "
                     f"{100 * gap_t:>+10.1f} pp   [{100 * lo_t:+.1f}, {100 * hi_t:+.1f}]")
    out["9_tau_sweep"] = sweep
    figures_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = figures_dir / "tau_sweep.csv"
    with sweep_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(sweep[0]))
        writer.writeheader()
        writer.writerows(sweep)
    shown = (sweep_path.relative_to(APP_DIR)
             if sweep_path.is_relative_to(APP_DIR) else sweep_path)
    lines.append(f"  written to {shown}")

    # 10 -------------------------------------------------------------------
    head(10, "Distribution of object recall, case study, 20 bins")
    counts, edges = np.histogram(case["obj_recall"].to_numpy(dtype=float),
                                 bins=20, range=(0.0, 1.0))
    out["10_recall_hist"] = {"counts": [int(c) for c in counts],
                             "edges": [float(e) for e in edges]}
    widest = int(counts.max())
    for i, count in enumerate(counts):
        bar = "#" * int(round(40 * count / widest)) if widest else ""
        lines.append(f"    [{edges[i]:.2f}, {edges[i + 1]:.2f})  {int(count):>4}  {bar}")
    first, last = int(counts[0]), int(counts[-1])
    middle = int(counts[1:-1].sum())
    lines.append(f"  ends {first} + {last} = {first + last} of {len(case)} "
                 f"({100 * (first + last) / len(case):.0f}%); middle 18 bins {middle}")
    lines.append("  -> mass concentrated at the ends is what makes the choice of tau robust")

    out["_lines"] = lines
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", type=Path, default=APP_DIR / "release")
    ap.add_argument("--figures", type=Path, default=APP_DIR / "figures")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of the report")
    args = ap.parse_args()

    report = build_report(args.release, args.figures)
    if args.json:
        payload = {k: v for k, v in report.items() if not k.startswith("_")}
        print(json.dumps(payload, indent=2, default=float))
    else:
        print("\n".join(report["_lines"]))


if __name__ == "__main__":
    main()
