#!/usr/bin/env python3
"""Build the deployable Spatial Action Review release from the analysis artifacts.

The manuscript reports two audit builds that share one scoring definition and
differ only in their exported record sets. The release carries both, and the
dashboard loads either into the same views:

  case_study    Qwen3-VL Staged GRPO, 541 paired image regions.
                The run traced end to end in the paper: the teaser figure, the
                answer-action ledger table, and the task-by-dataset risk-map
                table. Source: plan/inspector_data.json (the file the static
                prototype already serves), enriched with nothing that would
                change a reported value.

  sft_variants  Qwen3-VL zero-shot / Perception SFT / Grounding SFT /
                Staged SFT / Joint SFT, 753 paired image regions each.
                The re-audit table. Sources: outputs/percrop.csv,
                outputs/paired_set.jsonl, outputs/preds_qwen3vl_*.jsonl.

Staged GRPO deliberately does NOT borrow per-record point-F1 or point-in-mask
precision from percrop.csv: those rows belong to other conditions and joining
them would fabricate action metrics for this run. Fields the 541-record export
does not carry are marked absent and the dashboard says so.

Image coverage: all 541 case-study crops are released. The 753-record build
draws on the same 541 crops, so 212 of its records have no pixels; they keep
every numeric and textual field and the app draws their point geometry on a
labelled neutral grid. No image is ever synthesised.

Usage:
    python tools/build_release.py                  # writes ./release/
    python tools/build_release.py --check          # verify an existing release
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

APP_DIR = Path(__file__).resolve().parent.parent
WORKSPACE = APP_DIR.parent

MODEL = "Qwen3-VL"
TAU = 0.5
N_BOOT = 2000
BOOT_SEED = 0

CASE_STUDY_CONDITION = "Staged GRPO"
CASE_STUDY_N = 541

# Per-record scores for the case-study condition, produced by re-running
# 03_score_and_join.py on the acquisition machine. Used only to add the two
# diagnostic columns the 541-record export omits; every score already in that
# export is verified against this file rather than replaced by it.
CASE_STUDY_SCORES = "percrop_staged_grpo.csv"

# (condition label as used in the manuscript, prediction file stem, percrop label)
SFT_CONDITIONS = [
    ("Zero-shot", "preds_qwen3vl_zero_shot", "Zero-shot"),
    ("Perception SFT", "preds_qwen3vl_perception_sft", "Perception SFT"),
    ("Grounding SFT", "preds_qwen3vl_grounding_only", "Grounding-only"),
    ("Staged SFT", "preds_qwen3vl_staged_sft", "Staged SFT"),
    ("Joint SFT", "preds_qwen3vl_joint_sft", "Joint SFT"),
]
SFT_N = 753

# Staged GRPO also has a complete 753-record scoring on the matched audit set.
# It is released as a sixth condition so its numbers are traceable to the shipped
# release, and flagged, because it is not a row of the manuscript's Table 3.
EXTRA_SFT_CONDITION = "Staged GRPO"
EXTRA_SFT_SOURCE = "percrop_staged_grpo.csv"

# Paper-facing dataset names; the raw id stays on every record for provenance.
DATASET_LABEL = {
    "lucchi_plus": "Lucchi",
    "vnc_drosophila": "VNC",
    "mitoem_human": "EM-H",
}
DATASET_ORDER = ["Lucchi", "VNC", "EM-H"]
# The Location row folds direct-location and marked-region probes together, as
# the manuscript's risk-map table does.
TASK_ORDER = ["presence", "counting", "location", "relation", "attribute"]

FIELDS = [
    "build", "model", "condition", "dataset_id", "dataset", "task", "probe_id",
    "crop_id", "question_id",
    "vqa_question", "vqa_choices", "expected_letter", "expected_answer",
    "pred_answer_snippet", "prediction_error", "answer_correct",
    "n_gt", "n_pred", "obj_recall", "point_f1", "pim_prec", "count_ae",
    "gt_centroids", "pred_points",
    "image_file", "source_image_path",
]
# Per-record diagnostic columns. Reported as absent only when genuinely empty for
# every record of a build, rather than assumed absent.
OPTIONAL_RECORD_FIELDS = ["point_f1", "pim_prec"]


def absent_fields(records: list[dict]) -> list[str]:
    return [f for f in OPTIONAL_RECORD_FIELDS
            if all(r.get(f) in ("", None) for r in records)]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def require(path: Path) -> Path:
    if not path.exists():
        sys.exit(f"missing required artifact: {path}")
    return path


def quadrant(answer_correct: int, obj_recall: float, tau: float = TAU) -> str:
    reliable = obj_recall >= tau
    if answer_correct:
        return "trustworthy" if reliable else "silent_failure"
    return "lucky" if reliable else "honest"


def points_to_pairs(points) -> list[list[float]]:
    out = []
    for p in points or []:
        if isinstance(p, dict):
            out.append([float(p["x"]), float(p["y"])])
        else:
            out.append([float(p[0]), float(p[1])])
    return out


# ------------------------------------------------------- case study (541)

def build_case_study(workspace: Path, available_images: set[str]) -> list[dict]:
    data = json.loads(require(workspace / "plan" / "inspector_data.json").read_text())
    meta = data.get("meta", {})
    if meta.get("condition") != CASE_STUDY_CONDITION:
        sys.exit(
            f"inspector_data.json declares condition {meta.get('condition')!r}, "
            f"expected {CASE_STUDY_CONDITION!r}"
        )
    crops = data["crops"]
    if len(crops) != CASE_STUDY_N:
        sys.exit(f"inspector_data.json has {len(crops)} records, expected {CASE_STUDY_N}")

    paired = {r["question_id"]: r for r in read_jsonl(workspace / "outputs" / "paired_set.jsonl")}

    # Optional per-record diagnostics for this condition. Absent -> the columns
    # stay empty, exactly as before.
    scores: dict[str, dict] = {}
    scores_path = workspace / "outputs" / CASE_STUDY_SCORES
    if scores_path.exists():
        with scores_path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("condition") == CASE_STUDY_CONDITION:
                    scores[row["question_id"]] = row

    records: list[dict] = []
    for crop in crops:
        qid = crop["question_id"]
        pair = paired.get(qid)
        if pair is None:
            sys.exit(f"case study: {qid} not found in paired_set.jsonl")

        # Only ever *add* columns. If the score file disagrees with the export on
        # any value the export already carries, it is a different run and must not
        # contribute.
        extra = scores.get(qid)
        if extra is not None:
            for field, exported in (("answer_correct", int(crop["answer_correct"])),
                                    ("n_gt", int(crop["n_gt"])),
                                    ("n_pred", int(crop["n_pred"]))):
                if int(extra[field]) != exported:
                    sys.exit(
                        f"case study/{qid}: {CASE_STUDY_SCORES} has {field}={extra[field]} but "
                        f"the export has {exported}. Refusing to mix runs."
                    )
            if abs(float(extra["obj_recall"]) - float(crop["obj_recall"])) > 5e-4:
                sys.exit(
                    f"case study/{qid}: {CASE_STUDY_SCORES} has obj_recall="
                    f"{extra['obj_recall']} but the export has {crop['obj_recall']}. "
                    "Refusing to mix runs."
                )
        # Ground-truth object counts are label-derived and condition-independent,
        # so they must agree between the export and the paired set.
        if int(pair.get("n_gt", -1)) != int(crop["n_gt"]):
            sys.exit(f"case study/{qid}: n_gt {crop['n_gt']} vs paired_set {pair.get('n_gt')}")

        gt = points_to_pairs(crop.get("gt_centroids") or pair.get("gt_centroids"))
        pred = points_to_pairs(crop.get("pred_points"))
        if len(gt) != int(crop["n_gt"]):
            sys.exit(f"case study/{qid}: {len(gt)} GT centroids but n_gt={crop['n_gt']}")
        if len(pred) != int(crop["n_pred"]):
            sys.exit(f"case study/{qid}: {len(pred)} points but n_pred={crop['n_pred']}")
        if crop["quadrant"] != quadrant(int(crop["answer_correct"]), float(crop["obj_recall"])):
            sys.exit(f"case study/{qid}: exported quadrant disagrees with the tau=0.5 gate")

        crop_id = crop["crop_id"]
        records.append({
            "build": "case_study",
            "model": MODEL,
            "condition": CASE_STUDY_CONDITION,
            "dataset_id": crop["dataset"],
            "dataset": DATASET_LABEL.get(crop["dataset"], crop["dataset"]),
            "task": crop["task"],
            "probe_id": pair.get("probe_id", ""),
            "crop_id": crop_id,
            "question_id": qid,
            "vqa_question": crop.get("vqa_question") or pair.get("vqa_question", ""),
            "vqa_choices": json.dumps(crop.get("vqa_choices") or pair.get("vqa_choices") or []),
            "expected_letter": crop.get("expected_letter", ""),
            "expected_answer": crop.get("expected_answer", ""),
            "pred_answer_snippet": crop.get("pred_answer_snippet", ""),
            "prediction_error": "",
            "answer_correct": int(crop["answer_correct"]),
            "n_gt": int(crop["n_gt"]),
            "n_pred": int(crop["n_pred"]),
            "obj_recall": float(crop["obj_recall"]),
            "point_f1": (float(extra["point_f1"]) if extra and extra.get("point_f1") else ""),
            "pim_prec": (float(extra["pim_prec"]) if extra and extra.get("pim_prec") else ""),
            "count_ae": float(abs(int(crop["n_pred"]) - int(crop["n_gt"]))),
            "gt_centroids": json.dumps(gt),
            "pred_points": json.dumps(pred),
            "image_file": f"{crop_id}.png" if crop_id in available_images else "",
            "source_image_path": crop.get("image_path", ""),
        })
    return records


# ------------------------------------------------------- SFT variants (753)

def build_sft_variants(workspace: Path, available_images: set[str]) -> list[dict]:
    with require(workspace / "outputs" / "percrop.csv").open(encoding="utf-8", newline="") as fh:
        percrop = list(csv.DictReader(fh))
    paired = {r["question_id"]: r for r in read_jsonl(workspace / "outputs" / "paired_set.jsonl")}

    records: list[dict] = []
    for label, pred_stem, percrop_label in SFT_CONDITIONS:
        preds = {r["question_id"]: r
                 for r in read_jsonl(require(workspace / "outputs" / f"{pred_stem}.jsonl"))}
        rows = [r for r in percrop if r["model"] == MODEL and r["condition"] == percrop_label]
        if len(rows) != SFT_N:
            sys.exit(
                f"{MODEL} / {percrop_label}: percrop.csv has {len(rows)} rows, expected {SFT_N}. "
                "Refusing to release an incomplete condition."
            )

        for row in rows:
            qid = row["question_id"]
            pair = paired.get(qid)
            pred = preds.get(qid)
            if pair is None:
                sys.exit(f"{label}: {qid} not found in paired_set.jsonl")
            if pred is None:
                sys.exit(f"{label}: {qid} not found in {pred_stem}.jsonl")

            pred_points = points_to_pairs(pred.get("pred_points"))
            gt = points_to_pairs(pair.get("gt_centroids"))
            if len(pred_points) != int(row["n_pred"]):
                sys.exit(f"{label}/{qid}: n_pred={row['n_pred']} but {len(pred_points)} parsed points")
            if len(gt) != int(row["n_gt"]):
                sys.exit(f"{label}/{qid}: n_gt={row['n_gt']} but {len(gt)} GT centroids")

            crop_id = row["crop_id"]
            records.append({
                "build": "sft_variants",
                "model": MODEL,
                "condition": label,
                "dataset_id": row["dataset"],
                "dataset": DATASET_LABEL.get(row["dataset"], row["dataset"]),
                "task": row["task"],
                "probe_id": pair.get("probe_id", ""),
                "crop_id": crop_id,
                "question_id": qid,
                "vqa_question": pair.get("vqa_question", ""),
                "vqa_choices": json.dumps(pair.get("vqa_choices") or []),
                "expected_letter": row.get("expected_letter", ""),
                "expected_answer": row.get("expected_answer", ""),
                "pred_answer_snippet": row.get("pred_answer_snippet", ""),
                "prediction_error": row.get("prediction_error", ""),
                "answer_correct": int(row["answer_correct"]),
                "n_gt": int(row["n_gt"]),
                "n_pred": int(row["n_pred"]),
                "obj_recall": float(row["obj_recall"]),
                "point_f1": float(row["point_f1"]),
                "pim_prec": float(row["pim_prec"]),
                "count_ae": float(row["count_ae"]),
                "gt_centroids": json.dumps(gt),
                "pred_points": json.dumps(pred_points),
                "image_file": f"{crop_id}.png" if crop_id in available_images else "",
                "source_image_path": pair.get("image_path", ""),
            })
    return records


def build_extra_sft(workspace: Path, available_images: set[str]) -> list[dict]:
    """Staged GRPO over the 753-record matched set, from its own score file."""
    path = workspace / "outputs" / EXTRA_SFT_SOURCE
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("condition") == EXTRA_SFT_CONDITION]
    if len(rows) != SFT_N:
        print(f"note: {EXTRA_SFT_SOURCE} has {len(rows)} rows for "
              f"{EXTRA_SFT_CONDITION}, expected {SFT_N}; not released")
        return []

    paired = {r["question_id"]: r for r in read_jsonl(workspace / "outputs" / "paired_set.jsonl")}
    records: list[dict] = []
    for row in rows:
        qid = row["question_id"]
        pair = paired.get(qid)
        if pair is None:
            sys.exit(f"{EXTRA_SFT_CONDITION}: {qid} not found in paired_set.jsonl")
        gt = points_to_pairs(json.loads(row["gt_centroids"]) if row.get("gt_centroids") else [])
        pred = points_to_pairs(json.loads(row["pred_points"]) if row.get("pred_points") else [])
        if len(gt) != int(row["n_gt"]) or len(pred) != int(row["n_pred"]):
            sys.exit(f"{EXTRA_SFT_CONDITION}/{qid}: coordinate counts disagree with the scores")
        crop_id = row["crop_id"]
        records.append({
            "build": "sft_variants",
            "model": MODEL,
            "condition": EXTRA_SFT_CONDITION,
            "dataset_id": row["dataset"],
            "dataset": DATASET_LABEL.get(row["dataset"], row["dataset"]),
            "task": row["task"],
            "probe_id": pair.get("probe_id", ""),
            "crop_id": crop_id,
            "question_id": qid,
            "vqa_question": row.get("vqa_question") or pair.get("vqa_question", ""),
            "vqa_choices": json.dumps(json.loads(row["vqa_choices"])
                                      if row.get("vqa_choices") else
                                      (pair.get("vqa_choices") or [])),
            "expected_letter": row.get("expected_letter", ""),
            "expected_answer": row.get("expected_answer", ""),
            "pred_answer_snippet": row.get("pred_answer_snippet", ""),
            "prediction_error": row.get("prediction_error", ""),
            "answer_correct": int(row["answer_correct"]),
            "n_gt": int(row["n_gt"]),
            "n_pred": int(row["n_pred"]),
            "obj_recall": float(row["obj_recall"]),
            "point_f1": float(row["point_f1"]) if row.get("point_f1") else "",
            "pim_prec": float(row["pim_prec"]) if row.get("pim_prec") else "",
            "count_ae": float(row["count_ae"]) if row.get("count_ae") else
                        float(abs(int(row["n_pred"]) - int(row["n_gt"]))),
            "gt_centroids": json.dumps(gt),
            "pred_points": json.dumps(pred),
            "image_file": f"{crop_id}.png" if crop_id in available_images else "",
            "source_image_path": row.get("image_path") or pair.get("image_path", ""),
        })
    return records


# ------------------------------------------------------- summaries

def summarise(records: list[dict], tau: float = TAU) -> dict:
    ac = np.array([r["answer_correct"] for r in records])
    rec = np.array([r["obj_recall"] for r in records])
    ar = (rec >= tau).astype(int)
    pim_vals = [r["pim_prec"] for r in records if r["pim_prec"] != ""]
    n = len(records)

    rng = np.random.default_rng(BOOT_SEED)
    idx = np.arange(n)
    boots = []
    for _ in range(N_BOOT):
        b = rng.choice(idx, n, replace=True)
        cb, wb = ar[b][ac[b] == 1], ar[b][ac[b] == 0]
        if len(cb) and len(wb):
            boots.append(cb.mean() - wb.mean())
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])

    p_correct = float(ar[ac == 1].mean())
    p_wrong = float(ar[ac == 0].mean())
    out = {
        "n": n,
        "vqa_acc": round(float(ac.mean()), 3),
        "obj_recall": round(float(rec.mean()), 3),
        "silent_failure_rate": round(float(((ac == 1) & (ar == 0)).mean()), 3),
        "trustworthy_rate": round(float(((ac == 1) & (ar == 1)).mean()), 3),
        "lucky_rate": round(float(((ac == 0) & (ar == 1)).mean()), 3),
        "honest_rate": round(float(((ac == 0) & (ar == 0)).mean()), 3),
        "P_reliable_given_correct": round(p_correct, 3),
        "P_unreliable_given_correct": round(1 - p_correct, 3),
        "trust_gap": round(p_correct - p_wrong, 3),
        "gap_lo": round(float(lo), 3),
        "gap_hi": round(float(hi), 3),
        "point_biserial": round(float(np.corrcoef(ac.astype(float), rec)[0, 1]), 3),
    }
    out["pim_prec_given_correct"] = (
        round(float(np.array(pim_vals)[ac == 1].mean()), 3) if len(pim_vals) == n else None
    )
    return out


def option_stats(records: list[dict]) -> dict:
    """Option-count distribution and record-wise uniform-choice baseline."""
    counts: dict[int, int] = {}
    inverses = []
    for r in records:
        k = len(json.loads(r["vqa_choices"]))
        counts[k] = counts.get(k, 0) + 1
        if k:
            inverses.append(1.0 / k)
    return {
        "k_distribution": {str(k): counts[k] for k in sorted(counts)},
        "uniform_choice_baseline": round(float(np.mean(inverses)), 3),
        "n_single_option": counts.get(1, 0),
    }


N_DEMO_PER_STATE = 5

# How each behavioural state is ranked when picking worked examples, and what the
# state teaches a supervisor. Ranking keys are explicit so the selection is
# reviewable and stable rather than hand-picked.
DEMO_RULES = {
    "silent_failure": (
        # Prefer presence probes with many objects and many emitted points that
        # still miss everything: the starkest form of the hazard. Then spread
        # across tasks so the examples are not five copies of one probe family.
        lambda r: (r["obj_recall"], -r["n_pred"], -r["n_gt"], r["question_id"]),
        "Correct answer, unreliable action. An answer-only overseer would clear this hand-off.",
    ),
    "trustworthy": (
        lambda r: (-r["obj_recall"], -r["n_gt"], r["question_id"]),
        "Answer and action agree and the action clears the gate: a hand-off that can proceed.",
    ),
    "lucky": (
        lambda r: (-r["obj_recall"], -r["n_gt"], r["question_id"]),
        "Wrong answer, reliable action. The answer channel would have blocked a usable action.",
    ),
    "honest": (
        lambda r: (r["obj_recall"], -r["n_gt"], r["question_id"]),
        "Both channels fail, so answer-only monitoring already surfaces this region.",
    ),
}


def pick_demo_cases(records: list[dict], per_state: int = N_DEMO_PER_STATE) -> dict:
    """Pin several worked examples per behavioural state.

    Within a state, examples are drawn round-robin across task families before
    falling back to the plain ranking, so a supervisor clicking through them
    sees the state in more than one kind of probe.
    """
    # A worked example has to have something to look at, so image regions with no
    # labelled mitochondrion (n_gt = 0) are excluded even though they are valid
    # audit records.
    pool = [r for r in records if r["image_file"] and r["n_gt"] > 0]
    out: dict[str, dict] = {}

    for quad, (key, note) in DEMO_RULES.items():
        cand = sorted(
            (r for r in pool if quadrant(r["answer_correct"], r["obj_recall"]) == quad),
            key=key,
        )
        by_task: dict[str, list[dict]] = {}
        for r in cand:
            by_task.setdefault(r["task"], []).append(r)

        picked: list[dict] = []
        # Round-robin over task families, best-ranked first within each family.
        while len(picked) < per_state and any(by_task.values()):
            for task in sorted(by_task, key=lambda t: TASK_ORDER.index(t) if t in TASK_ORDER else 99):
                if len(picked) >= per_state:
                    break
                if by_task[task]:
                    picked.append(by_task[task].pop(0))

        out[quad] = {
            "note": note,
            "cases": [
                {
                    "question_id": r["question_id"],
                    "crop_id": r["crop_id"],
                    "task": r["task"],
                    "dataset": r["dataset"],
                    "obj_recall": round(r["obj_recall"], 3),
                    "n_gt": r["n_gt"],
                    "n_pred": r["n_pred"],
                    "label": f"{r['task']} · {r['dataset']} · recall {r['obj_recall']:.2f}",
                }
                for r in picked
            ],
        }
    return out


# ------------------------------------------------------- assembly

def copy_images(workspace: Path, out_dir: Path, records: list[dict]) -> dict:
    """Copy released crops as optimised 8-bit PNGs (pixel-lossless: sources are grey)."""
    src_dir = workspace / "plan" / "inspector_assets" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = sorted({r["image_file"] for r in records if r["image_file"]})

    n_bytes = 0
    for name in wanted:
        img = Image.open(src_dir / name)
        arr = np.asarray(img.convert("RGB"))
        if not (np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 1], arr[..., 2])):
            sys.exit(f"{name} is not greyscale; refusing lossy channel collapse")
        dst = out_dir / name
        img.convert("L").save(dst, "PNG", optimize=True)
        n_bytes += dst.stat().st_size
    return {"n_images": len(wanted), "bytes": n_bytes}


def write_release(workspace: Path, out_dir: Path) -> dict:
    image_dir = workspace / "plan" / "inspector_assets" / "images"
    available = {p.stem for p in image_dir.glob("*.png")} if image_dir.exists() else set()

    case_study = build_case_study(workspace, available)
    sft = build_sft_variants(workspace, available)
    extra = build_extra_sft(workspace, available)
    sft = sft + extra
    records = case_study + sft

    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.csv.gz"
    with gzip.open(records_path, "wt", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)

    image_info = copy_images(workspace, out_dir / "images", records)

    sft_labels = [label for label, _, _ in SFT_CONDITIONS]
    reported_sft = list(sft_labels)
    if extra:
        sft_labels.append(EXTRA_SFT_CONDITION)
    builds = {
        "case_study": {
            "id": "case_study",
            "label": f"Case study · {MODEL} {CASE_STUDY_CONDITION} ({CASE_STUDY_N} records)",
            "short_label": "Case study (541)",
            "role": "The traced case study: one adapted checkpoint audited end to end, from "
                    "run-level state to task-dataset risk to image-region evidence.",
            "model": MODEL,
            "conditions": [CASE_STUDY_CONDITION],
            "n_records": len(case_study),
            "images_released": sum(1 for r in case_study if r["image_file"]),
            "absent_record_fields": absent_fields(case_study),
            "source": "plan/inspector_data.json",
            "reference_summary": {CASE_STUDY_CONDITION: summarise(case_study)},
            "option_stats": option_stats(case_study),
            "demo_cases": pick_demo_cases(case_study),
        },
        "sft_variants": {
            "id": "sft_variants",
            "label": f"SFT-variant audit · {MODEL} ({SFT_N} records × {len(sft_labels)} conditions)",
            "short_label": "SFT-variant audit (753)",
            "role": "The re-audit set: the same reliability measures recomputed after each "
                    "supervised adaptation, on one matched record set.",
            "model": MODEL,
            "conditions": sft_labels,
            "n_records": SFT_N,
            "images_released": len({r["image_file"] for r in sft if r["image_file"]}),
            "absent_record_fields": absent_fields(sft),
            "source": "outputs/percrop.csv + outputs/paired_set.jsonl + outputs/preds_qwen3vl_*.jsonl",
            "reference_summary": {
                label: summarise([r for r in sft if r["condition"] == label])
                for label in sft_labels
            },
            "reported_in_table": reported_sft,
            "not_in_reported_table": [c for c in sft_labels if c not in reported_sft],
            "option_stats": option_stats([r for r in sft if r["condition"] == sft_labels[0]]),
            "demo_cases": pick_demo_cases([r for r in sft if r["condition"] == "Joint SFT"]),
        },
    }

    manifest = {
        "schema_version": 2,
        "model": MODEL,
        "tau_default": TAU,
        "bootstrap": {"n_resamples": N_BOOT, "seed": BOOT_SEED,
                      "generator": "numpy.random.default_rng"},
        "default_build": "case_study",
        "build_order": ["case_study", "sft_variants"],
        "builds": builds,
        "dataset_order": DATASET_ORDER,
        "task_order": TASK_ORDER,
        "dataset_labels": DATASET_LABEL,
        "n_records_total": len(records),
        "images_released": image_info["n_images"],
        "images_bytes": image_info["bytes"],
        "records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "notes": {
            "location_row": "The Location task folds direct-location and marked-region probes "
                            "together; both require resolving spatial evidence in the region.",
            "case_study_metrics": "Point F1 and point-in-mask precision are not in the 541-record "
                                  "export. They are left empty rather than joined from another "
                                  "model condition.",
            "image_coverage": "All 541 case-study crops are released. 212 of the 753 re-audit "
                              "records have no released pixels; their point geometry is drawn on "
                              "a labelled neutral grid and the tile says so.",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def check_release(out_dir: Path) -> None:
    manifest = json.loads((out_dir / "manifest.json").read_text())
    digest = hashlib.sha256((out_dir / "records.csv.gz").read_bytes()).hexdigest()
    ok = digest == manifest["records_sha256"]
    n_images = len(list((out_dir / "images").glob("*.png")))
    print(f"records.csv.gz sha256 {'OK' if ok else 'MISMATCH'}")
    print(f"images on disk {n_images} / manifest {manifest['images_released']}")
    for name, build in manifest["builds"].items():
        print(f"  {name}: {build['n_records']} records, conditions {build['conditions']}")
    if not ok or n_images != manifest["images_released"]:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", type=Path, default=WORKSPACE)
    ap.add_argument("--out", type=Path, default=APP_DIR / "release")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.check:
        check_release(args.out)
        return

    manifest = write_release(args.workspace, args.out)
    print(json.dumps({
        "out": str(args.out),
        "records": manifest["n_records_total"],
        "builds": {k: {"n": v["n_records"], "conditions": v["conditions"],
                       "images": v["images_released"]}
                   for k, v in manifest["builds"].items()},
        "images_mb": round(manifest["images_bytes"] / 1e6, 1),
    }, indent=2))


if __name__ == "__main__":
    main()
