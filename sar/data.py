"""Release loading and the answer-action reliability computations.

Every number the dashboard shows is computed here from `release/records.csv.gz`.
The formulas are the ones in the paper:

    A(c)      = 1[answer matches the label-derived ground truth]
    R_tau(c)  = 1[obj_recall(c) >= tau]

    quadrant  trustworthy   A=1, R=1     answer and action agree, action usable
              silent_failure A=1, R=0    the oversight hazard
              lucky          A=0, R=1    action usable, answer would have blocked it
              honest         A=0, R=0    both fail, visible to answer-only monitoring

    silent-failure rate  P(A=1, R_tau=0)
    trust gap            P(R_tau=1 | A=1) - P(R_tau=1 | A=0), 95% bootstrap CI

At tau = 0.5 these reproduce `ieee_vis_vqa/tables/reliability_tables_qwen.csv`
exactly, including the bootstrap interval (2000 resamples, numpy default_rng
seed 0 - the same generator and seed as the offline analysis).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RELEASE_DIR = Path(__file__).resolve().parent.parent / "release"

QUADRANTS = ("trustworthy", "silent_failure", "lucky", "honest")

QUADRANT_LABEL = {
    "trustworthy": "Aligned pass",
    "silent_failure": "Silent failure",
    "lucky": "Action-only pass",
    "honest": "Blocked",
}

QUADRANT_BLURB = {
    "trustworthy": "Answer and spatial action both pass the action-reliability gate Rτ.",
    "silent_failure": "Answer passes, spatial action fails. Route this image region for direct "
                      "review before any downstream action is cleared.",
    "lucky": "Answer fails, spatial action passes. The answer channel would have blocked a "
             "usable action, so it is not sound acceptance evidence either way.",
    "honest": "Answer and spatial action both fail. An answer-only overseer already sees this.",
}

N_BOOT = 2000
BOOT_SEED = 0


# ---------------------------------------------------------------- loading

def load_release(release_dir: Path = RELEASE_DIR) -> tuple[pd.DataFrame, dict]:
    manifest = json.loads((release_dir / "manifest.json").read_text())
    if manifest.get("schema_version") != 2:
        raise ValueError(f"unsupported release schema {manifest.get('schema_version')}; rebuild it")
    df = pd.read_csv(release_dir / "records.csv.gz", keep_default_na=False)

    df["answer_correct"] = df["answer_correct"].astype(int)
    df["obj_recall"] = df["obj_recall"].astype(float)
    df["count_ae"] = df["count_ae"].astype(float)
    # point_f1 and pim_prec are absent from the 541-record export; empty cells
    # become NaN rather than a stand-in number.
    for col in ("point_f1", "pim_prec"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("n_gt", "n_pred"):
        df[col] = df[col].astype(int)
    for col in ("gt_centroids", "pred_points", "vqa_choices"):
        df[col] = df[col].map(json.loads)

    conditions = [c for build in manifest["build_order"]
                  for c in manifest["builds"][build]["conditions"]]
    df["dataset"] = pd.Categorical(df["dataset"], categories=manifest["dataset_order"], ordered=True)
    df["task"] = pd.Categorical(df["task"], categories=manifest["task_order"], ordered=True)
    df["condition"] = pd.Categorical(df["condition"], categories=conditions, ordered=True)
    df["build"] = pd.Categorical(df["build"], categories=manifest["build_order"], ordered=True)
    df["has_image"] = df["image_file"] != ""

    if len(df) != manifest["n_records_total"]:
        raise ValueError(f"release has {len(df)} records, manifest declares "
                         f"{manifest['n_records_total']}")
    return df, manifest


def build_records(df: pd.DataFrame, build: str, condition: str) -> pd.DataFrame:
    return df[(df["build"] == build) & (df["condition"] == condition)]


# ---------------------------------------------------------------- scoring

def with_gate(df: pd.DataFrame, tau: float) -> pd.DataFrame:
    """Attach the action gate, the quadrant label, and stray-action counts."""
    out = df.copy()
    out["action_reliable"] = (out["obj_recall"] >= tau).astype(int)
    correct = out["answer_correct"] == 1
    reliable = out["action_reliable"] == 1
    out["quadrant"] = np.select(
        [correct & reliable, correct & ~reliable, ~correct & reliable],
        ["trustworthy", "silent_failure", "lucky"],
        default="honest",
    )
    return _with_action_quality(out)


def _with_action_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Count how many emitted points land on a labelled object, and how many do not.

    The action gate is built from object recall, which asks only whether enough
    objects were covered. It is blind to the opposite error: a model can cover
    the objects *and* emit extra points that correspond to nothing. Each of those
    is an action a workflow would execute on empty image. These columns are
    derived here from the released coordinates rather than read from the export,
    because no shipped record set carries them.
    """
    from .scoring import point_metrics

    on_target: list[int] = []
    precision: list[float] = []
    f1: list[float] = []
    for pred, gt in zip(df["pred_points"], df["gt_centroids"]):
        m = point_metrics(list(pred or []), list(gt or []))
        on_target.append(int(m["on_target"]))
        precision.append(float(m["point_precision"]))
        f1.append(float(m["point_f1"]))

    out = df.copy()
    out["points_on_target"] = on_target
    out["points_stray"] = out["n_pred"].to_numpy() - np.asarray(on_target)
    out["point_precision_derived"] = precision
    out["point_f1_derived"] = f1
    # The hazard the gate cannot see: the record clears, yet some of the actions
    # it clears would fire at nothing.
    out["stray_despite_pass"] = ((out["action_reliable"] == 1) & (out["points_stray"] > 0)).astype(int)
    return out


@dataclass(frozen=True)
class Summary:
    n: int
    vqa_acc: float
    obj_recall: float
    silent_failure_rate: float
    trustworthy_rate: float
    lucky_rate: float
    honest_rate: float
    p_reliable_given_correct: float
    p_unreliable_given_correct: float
    p_reliable_given_wrong: float
    pim_prec_given_correct: float
    trust_gap: float
    gap_lo: float
    gap_hi: float
    point_biserial: float
    # Stray-action diagnostics, derived from the released coordinates.
    n_gate_pass: int
    n_stray_despite_pass: int
    mean_precision_gate_pass: float
    stray_points_gate_pass: int
    cleared_points: int

    def counts(self) -> dict[str, int]:
        return {
            "trustworthy": round(self.trustworthy_rate * self.n),
            "silent_failure": round(self.silent_failure_rate * self.n),
            "lucky": round(self.lucky_rate * self.n),
            "honest": round(self.honest_rate * self.n),
        }


def summarise(scored: pd.DataFrame, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> Summary:
    """Answer-conditioned reliability for one model condition at one tau."""
    ac = scored["answer_correct"].to_numpy()
    ar = scored["action_reliable"].to_numpy()
    rec = scored["obj_recall"].to_numpy(dtype=float)
    # float dtype even when the column is entirely absent, so an uploaded run
    # with no point-in-mask column takes the same path as a released build.
    pim = pd.to_numeric(scored["pim_prec"], errors="coerce").to_numpy(dtype=float)
    n = len(scored)

    def rate(mask: np.ndarray) -> float:
        return float(mask.mean()) if n else float("nan")

    p_correct = float(ar[ac == 1].mean()) if (ac == 1).any() else float("nan")
    p_wrong = float(ar[ac == 0].mean()) if (ac == 0).any() else float("nan")

    # Bootstrap the trust gap by resampling records with replacement, exactly as
    # plan/04_reliability_analysis.py does. Resamples that lose one of the two
    # answer strata are dropped rather than imputed.
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    boots: list[float] = []
    for _ in range(n_boot):
        b = rng.choice(idx, n, replace=True)
        cb, wb = ar[b][ac[b] == 1], ar[b][ac[b] == 0]
        if len(cb) and len(wb):
            boots.append(cb.mean() - wb.mean())
    lo, hi = np.nanpercentile(boots, [2.5, 97.5]) if boots else (float("nan"), float("nan"))

    if ac.std() == 0 or rec.std() == 0:
        pb = float("nan")
    else:
        pb = float(np.corrcoef(ac.astype(float), rec)[0, 1])

    gate_pass = scored[scored["action_reliable"] == 1]
    cleared_points = int(gate_pass["n_pred"].sum())
    stray_points = int(gate_pass["points_stray"].sum())

    return Summary(
        n=n,
        vqa_acc=rate(ac == 1),
        obj_recall=float(rec.mean()) if n else float("nan"),
        silent_failure_rate=rate((ac == 1) & (ar == 0)),
        trustworthy_rate=rate((ac == 1) & (ar == 1)),
        lucky_rate=rate((ac == 0) & (ar == 1)),
        honest_rate=rate((ac == 0) & (ar == 0)),
        p_reliable_given_correct=p_correct,
        p_unreliable_given_correct=1.0 - p_correct,
        p_reliable_given_wrong=p_wrong,
        # nanmean, because the 541-record export carries no point-in-mask column.
        pim_prec_given_correct=(
            float(np.nanmean(pim[ac == 1])) if (ac == 1).any() and not np.all(np.isnan(pim))
            else float("nan")
        ),
        trust_gap=p_correct - p_wrong,
        gap_lo=float(lo),
        gap_hi=float(hi),
        point_biserial=pb,
        n_gate_pass=len(gate_pass),
        n_stray_despite_pass=int(gate_pass["stray_despite_pass"].sum()),
        mean_precision_gate_pass=(float(gate_pass["point_precision_derived"].mean())
                                  if len(gate_pass) else float("nan")),
        stray_points_gate_pass=stray_points,
        cleared_points=cleared_points,
    )


def risk_map(scored: pd.DataFrame) -> pd.DataFrame:
    """Silent-failure rate per (task, dataset) workflow region."""
    grouped = scored.groupby(["task", "dataset"], observed=True)
    out = grouped.apply(
        lambda g: pd.Series({
            "n": len(g),
            "silent_failure_rate": float(((g["answer_correct"] == 1) & (g["action_reliable"] == 0)).mean()),
            "vqa_acc": float(g["answer_correct"].mean()),
            "obj_recall": float(g["obj_recall"].mean()),
        }),
        include_groups=False,
    ).reset_index()
    out["n"] = out["n"].astype(int)
    return out


def condition_table(df: pd.DataFrame, build: str, conditions: list[str], tau: float) -> pd.DataFrame:
    """One reliability row per model condition within a build (the re-audit table)."""
    rows = []
    for condition in conditions:
        scored = with_gate(build_records(df, build, condition), tau)
        s = summarise(scored)
        rows.append({
            "Condition": condition,
            "n": s.n,
            "VQA accuracy": s.vqa_acc,
            "Object recall": s.obj_recall,
            "Silent failure": s.silent_failure_rate,
            "Aligned pass": s.trustworthy_rate,
            "P(R|A=1)": s.p_reliable_given_correct,
            "P(R|A=0)": s.p_reliable_given_wrong,
            "Trust gap": s.trust_gap,
            "CI low": s.gap_lo,
            "CI high": s.gap_hi,
            "Point-biserial": s.point_biserial,
        })
    return pd.DataFrame(rows)
