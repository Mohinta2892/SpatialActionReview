#!/usr/bin/env python3
"""Extract VIS rebuttal additions from the shipped Spatial Action Review release.

The analyses here require no model training. They read the matched 753-record
SFT-variant export in `release/records.csv.gz` and emit:

  - version-to-version four-state transition audits;
  - silent-failure audit-queue churn and reuse tables;
  - point-action pass-rule sensitivity tables;
  - gate-versus-workload sweeps;
  - side-by-side model-revision candidate records and dashboard URLs.

Examples:

  python scripts/vis_rebuttal_additions.py
  python scripts/vis_rebuttal_additions.py --contract composite --tau 0.50 --precision-tau 0.75
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

from sar import urlstate  # noqa: E402
from sar.contracts import CONTRACTS, contract_readout  # noqa: E402
from sar.data import QUADRANT_LABEL, QUADRANTS, build_records, load_release, with_gate  # noqa: E402

STATE_ORDER = list(QUADRANTS)
BUILD = "sft_variants"
RULE_LABEL = {
    "coverage": "cover enough objects",
    "precision": "avoid stray points",
    "composite": "cover objects and avoid strays",
    "count": "match object count",
}


def _slug(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text)).strip("_")


def _scored(df: pd.DataFrame, condition: str, args) -> pd.DataFrame:
    return with_gate(
        build_records(df, BUILD, condition),
        args.tau,
        contract=args.contract,
        precision_tau=args.precision_tau,
        count_tolerance=args.count_tolerance,
    ).set_index("question_id")


def _joined(df: pd.DataFrame, before: str, after: str, args) -> pd.DataFrame:
    left = _scored(df, before, args)
    right = _scored(df, after, args)
    out = left.join(right, lsuffix="_before", rsuffix="_after", how="inner")
    out["before_condition"] = before
    out["after_condition"] = after
    out["transition"] = out["quadrant_before"] + " -> " + out["quadrant_after"]
    out["answer_delta"] = out["answer_correct_after"] - out["answer_correct_before"]
    out["action_delta"] = out["action_reliable_after"] - out["action_reliable_before"]
    out["recall_delta"] = out["obj_recall_after"] - out["obj_recall_before"]
    out["hit_rate_delta"] = (
        out["point_precision_derived_after"] - out["point_precision_derived_before"]
    )
    out["count_error_delta"] = out["count_ae_after"] - out["count_ae_before"]
    return out.reset_index()


def transition_audit(df: pd.DataFrame, conditions: list[str], args, out_dir: Path) -> dict:
    rows = []
    matrices = []
    adjacent = list(zip(conditions[:-1], conditions[1:]))
    for before, after in adjacent:
        j = _joined(df, before, after, args)
        mat = pd.crosstab(
            j["quadrant_before"], j["quadrant_after"],
        ).reindex(index=STATE_ORDER, columns=STATE_ORDER, fill_value=0)
        mat.insert(0, "before_state", mat.index)
        mat.insert(0, "after_condition", after)
        mat.insert(0, "before_condition", before)
        matrices.append(mat.reset_index(drop=True))

        rows.append({
            "before_condition": before,
            "after_condition": after,
            "n_common_records": int(len(j)),
            "resolved_silent_failures": int(((j.quadrant_before == "silent_failure")
                                             & (j.quadrant_after != "silent_failure")).sum()),
            "persistent_silent_failures": int(((j.quadrant_before == "silent_failure")
                                               & (j.quadrant_after == "silent_failure")).sum()),
            "newly_introduced_silent_failures": int(((j.quadrant_before != "silent_failure")
                                                     & (j.quadrant_after == "silent_failure")).sum()),
            "stable_aligned_passes": int(((j.quadrant_before == "trustworthy")
                                          & (j.quadrant_after == "trustworthy")).sum()),
            "answer_improvements_with_action_regressions": int(((j.answer_delta > 0)
                                                                & (j.action_delta < 0)).sum()),
            "action_improvements_with_answer_regressions": int(((j.action_delta > 0)
                                                                & (j.answer_delta < 0)).sum()),
            "answer_improvements_with_recall_drop": int(((j.answer_delta > 0)
                                                         & (j.recall_delta < 0)).sum()),
            "recall_improvements_with_answer_regression": int(((j.recall_delta > 0)
                                                               & (j.answer_delta < 0)).sum()),
        })

    summary = pd.DataFrame(rows)
    matrix = pd.concat(matrices, ignore_index=True)
    summary.to_csv(out_dir / "version_transition_summary.csv", index=False)
    matrix.to_csv(out_dir / "version_transition_matrices.csv", index=False)
    _plot_transition_matrices(matrix, out_dir)
    return {"transition_pairs": rows}


def queue_churn(df: pd.DataFrame, conditions: list[str], args, out_dir: Path) -> dict:
    queues = {
        cond: set(_scored(df, cond, args).query("quadrant == 'silent_failure'").index)
        for cond in conditions
    }
    rows = []
    for before, after in combinations(conditions, 2):
        qi, qj = queues[before], queues[after]
        inter = qi & qj
        union = qi | qj
        rows.append({
            "condition_i": before,
            "condition_j": after,
            "n_i": len(qi),
            "n_j": len(qj),
            "intersection": len(inter),
            "union": len(union),
            "jaccard_intersection_over_union": len(inter) / len(union) if union else np.nan,
            "union_over_intersection_as_requested": len(union) / len(inter) if inter else np.inf,
            "previous_queue_stale_pct": (len(qi - qj) / len(qi)) if qi else np.nan,
            "new_queue_records_pct": (len(qj - qi) / len(qj)) if qj else np.nan,
        })

    membership = []
    all_ids = sorted(set().union(*queues.values()))
    common = set.intersection(*queues.values()) if queues else set()
    for qid in all_ids:
        present = [cond for cond in conditions if qid in queues[cond]]
        membership.append({
            "question_id": qid,
            "n_queues": len(present),
            "conditions": "|".join(present),
            **{f"in_{_slug(cond)}": int(qid in queues[cond]) for cond in conditions},
        })

    unique_rows = [
        {"condition": cond, "question_id": qid}
        for cond in conditions
        for qid in sorted(queues[cond] - set().union(*(queues[o] for o in conditions if o != cond)))
    ]

    churn = pd.DataFrame(rows)
    churn.to_csv(out_dir / "audit_queue_churn.csv", index=False)
    pd.DataFrame(membership).to_csv(out_dir / "silent_failure_queue_membership.csv", index=False)
    pd.DataFrame({"question_id": sorted(common)}).to_csv(
        out_dir / "records_common_to_every_silent_failure_queue.csv", index=False)
    pd.DataFrame(unique_rows).to_csv(
        out_dir / "records_unique_to_one_silent_failure_queue.csv", index=False)
    _plot_queue_churn(churn, conditions, out_dir)
    return {
        "queue_churn_pairs": rows,
        "records_common_to_every_queue": len(common),
        "records_unique_to_one_version": len(unique_rows),
    }


def contract_sensitivity(df: pd.DataFrame, conditions: list[str], args, out_dir: Path) -> dict:
    rows = []
    contracts_to_run = ["coverage", "precision", "composite", "count"]
    for condition in conditions:
        base = build_records(df, BUILD, condition)
        for contract in contracts_to_run:
            scored = with_gate(
                base,
                args.tau,
                contract=contract,
                precision_tau=args.precision_tau,
                count_tolerance=args.count_tolerance,
            )
            counts = scored["quadrant"].value_counts()
            reliable = scored["action_reliable"] == 1
            row = {
                "condition": condition,
                "contract": contract,
                "pass_rule": RULE_LABEL[contract],
                "contract_formula": contract_readout(
                    contract, args.tau, args.precision_tau, args.count_tolerance),
                "pass_rule_formula": contract_readout(
                    contract, args.tau, args.precision_tau, args.count_tolerance),
                "n": len(scored),
                "actions_cleared": int(reliable.sum()),
                "held_for_review": int((~reliable).sum()),
            }
            for q in STATE_ORDER:
                row[f"{q}_n"] = int(counts.get(q, 0))
                row[f"{q}_rate"] = int(counts.get(q, 0)) / len(scored)
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "point_action_pass_rule_sensitivity.csv", index=False)
    table.to_csv(out_dir / "action_contract_sensitivity.csv", index=False)
    _plot_contract_sensitivity(table, conditions, out_dir)
    return {"action_contract_rows": rows}


def workload_sweep(df: pd.DataFrame, conditions: list[str], args, out_dir: Path) -> dict:
    rows = []
    comp_rows = []
    thresholds = [round(0.05 * i, 2) for i in range(1, 20)]
    for condition in conditions:
        base = build_records(df, BUILD, condition)
        for tau in thresholds:
            ptau = tau if args.contract in {"precision", "composite"} else args.precision_tau
            scored = with_gate(
                base,
                tau,
                contract=args.contract,
                precision_tau=ptau,
                count_tolerance=args.count_tolerance,
            )
            reliable = scored["action_reliable"] == 1
            correct = scored["answer_correct"] == 1
            silent = correct & ~reliable
            rows.append({
                "condition": condition,
                "contract": args.contract,
                "pass_rule": RULE_LABEL[args.contract],
                "tau": tau,
                "precision_tau": ptau,
                "count_tolerance": args.count_tolerance,
                "actions_cleared": int(reliable.sum()),
                "held_for_review": int((~reliable).sum()),
                "silent_failure_queue": int(silent.sum()),
                "answer_correct_records_requiring_audit": int(silent.sum()),
                "aligned_pass": int((correct & reliable).sum()),
                "action_only_pass": int(((~correct) & reliable).sum()),
                "blocked": int(((~correct) & (~reliable)).sum()),
            })
            queue = scored[silent]
            grouped = queue.groupby(["task", "dataset"], observed=True).size().reset_index(name="n")
            for r in grouped.itertuples():
                comp_rows.append({
                    "condition": condition,
                    "contract": args.contract,
                    "pass_rule": RULE_LABEL[args.contract],
                    "tau": tau,
                    "task": r.task,
                    "dataset": r.dataset,
                    "silent_failure_queue_n": int(r.n),
                })
    sweep = pd.DataFrame(rows)
    comp = pd.DataFrame(comp_rows)
    sweep.to_csv(out_dir / "gate_workload_sweep.csv", index=False)
    comp.to_csv(out_dir / "gate_workload_queue_composition.csv", index=False)
    _plot_workload_sweep(sweep, args.focal_condition, out_dir)
    return {"workload_rows": len(rows), "queue_composition_rows": len(comp_rows)}


def revision_examples(df: pd.DataFrame, conditions: list[str], args, out_dir: Path) -> dict:
    before = args.before_condition or ("Joint SFT" if "Joint SFT" in conditions else conditions[-2])
    after = args.after_condition or ("Staged GRPO" if "Staged GRPO" in conditions else conditions[-1])
    j = _joined(df, before, after, args)
    j["priority"] = (
        ((j.quadrant_before == "silent_failure") & (j.quadrant_after != "silent_failure")).astype(int) * 5
        + ((j.quadrant_before != "silent_failure") & (j.quadrant_after == "silent_failure")).astype(int) * 5
        + ((j.answer_delta > 0) & (j.action_delta < 0)).astype(int) * 3
        + ((j.action_delta > 0) & (j.answer_delta < 0)).astype(int) * 3
        + j.recall_delta.abs()
    )

    rows = []
    for r in j.sort_values(["priority", "transition"], ascending=[False, True]).head(args.n_examples).itertuples():
        state_slug = urlstate.QUADRANT_TO_STATE[r.quadrant_after]
        cond_slug = urlstate.LABEL_TO_CONDITION.get(after, _slug(after))
        url = "/?" + urlencode({
            "source": "sft_variant",
            "condition": cond_slug,
            "tau": f"{args.tau:.2f}",
            "contract": args.contract,
            "pass_rule": RULE_LABEL[args.contract],
            "precision_tau": f"{args.precision_tau:.2f}",
            "count_tolerance": str(args.count_tolerance),
            "state": state_slug,
            "record": r.crop_id_after,
            "theme": "day",
        })
        rows.append({
            "question_id": r.question_id,
            "crop_id": r.crop_id_after,
            "dataset": r.dataset_after,
            "task": r.task_after,
            "before_condition": before,
            "after_condition": after,
            "transition": r.transition,
            "before_answer": r.pred_answer_snippet_before,
            "after_answer": r.pred_answer_snippet_after,
            "answer_delta": int(r.answer_delta),
            "action_delta": int(r.action_delta),
            "recall_before": float(r.obj_recall_before),
            "recall_after": float(r.obj_recall_after),
            "recall_delta": float(r.recall_delta),
            "hit_rate_before": float(r.point_precision_derived_before),
            "hit_rate_after": float(r.point_precision_derived_after),
            "count_error_before": float(r.count_ae_before),
            "count_error_after": float(r.count_ae_after),
            "dashboard_url_path": url,
        })
    pd.DataFrame(rows).to_csv(out_dir / "side_by_side_revision_examples.csv", index=False)
    return {"revision_pair": [before, after], "revision_examples": rows}


def _plot_transition_matrices(matrix: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    pairs = matrix[["before_condition", "after_condition"]].drop_duplicates().to_records(index=False)
    fig, axes = plt.subplots(1, len(pairs), figsize=(3.0 * len(pairs), 3.2), squeeze=False)
    vmax = matrix[STATE_ORDER].to_numpy().max()
    for ax, (before, after) in zip(axes[0], pairs):
        sub = matrix[(matrix.before_condition == before) & (matrix.after_condition == after)]
        data = sub[STATE_ORDER].to_numpy()
        ax.imshow(data, cmap="magma_r", vmin=0, vmax=vmax)
        ax.set_title(f"{before}\n-> {after}", fontsize=8)
        ax.set_xticks(range(len(STATE_ORDER)))
        ax.set_xticklabels([QUADRANT_LABEL[s] for s in STATE_ORDER], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(STATE_ORDER)))
        ax.set_yticklabels([QUADRANT_LABEL[s] for s in STATE_ORDER], fontsize=7)
        for i in range(data.shape[0]):
            for k in range(data.shape[1]):
                ax.text(k, i, str(int(data[i, k])), ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "version_transition_matrices.png", dpi=220)
    fig.savefig(out_dir / "version_transition_matrices.pdf")
    plt.close(fig)


def _plot_queue_churn(churn: pd.DataFrame, conditions: list[str], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    mat = pd.DataFrame(np.nan, index=conditions, columns=conditions)
    for r in churn.itertuples():
        mat.loc[r.condition_i, r.condition_j] = r.jaccard_intersection_over_union
        mat.loc[r.condition_j, r.condition_i] = r.jaccard_intersection_over_union
    np.fill_diagonal(mat.values, 1.0)
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    im = ax.imshow(mat.to_numpy(), cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels(conditions, fontsize=8)
    for i in range(len(conditions)):
        for j in range(len(conditions)):
            ax.text(j, i, f"{mat.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if mat.iloc[i, j] < 0.45 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Jaccard overlap")
    ax.set_title("Silent-failure queue reuse after model revision", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "audit_queue_churn_heatmap.png", dpi=220)
    fig.savefig(out_dir / "audit_queue_churn_heatmap.pdf")
    plt.close(fig)


def _plot_contract_sensitivity(table: pd.DataFrame, conditions: list[str], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    sub = table[table["condition"].isin(conditions)]
    labels = [f"{r.condition}\n{r.pass_rule}" for r in sub.itertuples()]
    x = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(max(7, 0.42 * len(sub)), 3.6))
    bottom = np.zeros(len(sub))
    colours = {
        "trustworthy": "#1f9e6b",
        "silent_failure": "#d1495b",
        "lucky": "#3d5a80",
        "honest": "#8d99ae",
    }
    for state in STATE_ORDER:
        vals = sub[f"{state}_n"].to_numpy()
        ax.bar(x, vals, bottom=bottom, color=colours[state], label=QUADRANT_LABEL[state])
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("records")
    ax.set_title("Ledger changes under different point-action pass rules", fontsize=9)
    ax.legend(fontsize=7, ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "point_action_pass_rule_sensitivity.png", dpi=220)
    fig.savefig(out_dir / "point_action_pass_rule_sensitivity.pdf")
    fig.savefig(out_dir / "action_contract_sensitivity.png", dpi=220)
    fig.savefig(out_dir / "action_contract_sensitivity.pdf")
    plt.close(fig)


def _plot_workload_sweep(sweep: pd.DataFrame, focal: str, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    sub = sweep[sweep["condition"] == focal]
    if sub.empty:
        sub = sweep[sweep["condition"] == sweep["condition"].iloc[-1]]
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.plot(sub["tau"], sub["actions_cleared"], label="actions cleared", color="#1f9e6b", lw=2)
    ax.plot(sub["tau"], sub["held_for_review"], label="held for review", color="#8d99ae", lw=2)
    ax.plot(sub["tau"], sub["silent_failure_queue"], label="silent-failure queue", color="#d1495b", lw=2)
    ax.set_xlabel("gate threshold")
    ax.set_ylabel("records")
    ax.set_title(f"Gate-versus-workload sweep: {sub['condition'].iloc[0]}", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "gate_workload_sweep.png", dpi=220)
    fig.savefig(out_dir / "gate_workload_sweep.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", type=Path, default=APP_DIR / "release")
    ap.add_argument("--out", type=Path, default=APP_DIR / "figures" / "rebuttal_additions")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--contract", choices=sorted(CONTRACTS), default="coverage")
    ap.add_argument("--precision-tau", type=float, default=0.5)
    ap.add_argument("--count-tolerance", type=int, default=0)
    ap.add_argument("--focal-condition", default="Staged GRPO")
    ap.add_argument("--before-condition")
    ap.add_argument("--after-condition")
    ap.add_argument("--n-examples", type=int, default=25)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    df, manifest = load_release(args.release)
    conditions = manifest["builds"][BUILD]["conditions"]

    summary = {
        "source": str(args.release / "records.csv.gz"),
        "build": BUILD,
        "conditions": conditions,
        "contract": args.contract,
        "pass_rule": RULE_LABEL[args.contract],
        "contract_formula": contract_readout(
            args.contract, args.tau, args.precision_tau, args.count_tolerance),
        "pass_rule_formula": contract_readout(
            args.contract, args.tau, args.precision_tau, args.count_tolerance),
        "tau": args.tau,
        "precision_tau": args.precision_tau,
        "count_tolerance": args.count_tolerance,
    }
    summary.update(transition_audit(df, conditions, args, args.out))
    summary.update(queue_churn(df, conditions, args, args.out))
    summary.update(contract_sensitivity(df, conditions, args, args.out))
    summary.update(workload_sweep(df, conditions, args, args.out))
    summary.update(revision_examples(df, conditions, args, args.out))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"wrote VIS rebuttal additions to {args.out}")
    for path in sorted(args.out.iterdir()):
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
