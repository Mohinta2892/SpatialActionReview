"""Accepted point-action gate for paired answer-action records.

The camera-ready dashboard gate is coverage-sensitive:

    R_coverage(c) = 1[obj_recall(c) >= tau_r]

This is the definition used by the accepted manuscript numbers. The module also
derives stray-point diagnostics for display, but those diagnostics do not change
the action verdict.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .scoring import point_metrics


@dataclass(frozen=True)
class ActionContract:
    key: str
    label: str
    short_label: str
    blurb: str


CONTRACTS: dict[str, ActionContract] = {
    "coverage": ActionContract(
        "coverage",
        "Cover enough objects",
        "cover enough objects",
        "Pass when the predicted points cover enough labelled mitochondria.",
    ),
}

DEFAULT_CONTRACT = "coverage"


def ensure_action_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Attach point-hit-rate diagnostics if the frame does not already have them."""
    if {"points_on_target", "points_stray", "point_precision_derived",
            "point_f1_derived"}.issubset(df.columns):
        return df.copy()

    out = df.copy()
    on_target: list[int] = []
    precision: list[float] = []
    f1: list[float] = []
    for pred, gt in zip(out["pred_points"], out["gt_centroids"]):
        m = point_metrics(list(pred or []), list(gt or []))
        on_target.append(int(m["on_target"]))
        precision.append(float(m["point_precision"]))
        f1.append(float(m["point_f1"]))

    out["points_on_target"] = on_target
    out["points_stray"] = out["n_pred"].to_numpy() - np.asarray(on_target)
    out["point_precision_derived"] = precision
    out["point_f1_derived"] = f1
    return out


def action_pass_mask(
    df: pd.DataFrame,
    *,
    contract: str = DEFAULT_CONTRACT,
    recall_tau: float = 0.5,
    precision_tau: float | None = None,
    count_tolerance: int = 0,
) -> pd.Series:
    """Return the boolean action verdict under the accepted object-recall gate."""
    if contract not in CONTRACTS:
        raise ValueError(
            f"unknown point-action gate {contract!r}; expected the accepted object-coverage gate"
        )

    recall = pd.to_numeric(df["obj_recall"], errors="coerce")
    return (recall >= recall_tau).fillna(False)


def apply_action_contract(
    df: pd.DataFrame,
    *,
    contract: str = DEFAULT_CONTRACT,
    recall_tau: float = 0.5,
    precision_tau: float | None = None,
    count_tolerance: int = 0,
) -> pd.DataFrame:
    """Attach the action verdict, four-state quadrant, and gate metadata."""
    out = ensure_action_quality(df)
    reliable = action_pass_mask(
        out,
        contract=contract,
        recall_tau=recall_tau,
    )
    out["action_reliable"] = reliable.astype(int)
    correct = out["answer_correct"] == 1
    out["quadrant"] = np.select(
        [correct & reliable, correct & ~reliable, ~correct & reliable],
        ["trustworthy", "silent_failure", "lucky"],
        default="honest",
    )
    out["stray_despite_pass"] = ((out["action_reliable"] == 1)
                                 & (out["points_stray"] > 0)).astype(int)
    out["recall_tau"] = float(recall_tau)
    return out
