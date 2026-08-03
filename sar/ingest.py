"""Load an arbitrary `inspector_data.json` into the dashboard's record schema.

The paper claims the dashboard "can audit any vision-language run that exports
the paired audit record" (Sec. 3.3). This module is what makes that claim
demonstrable rather than asserted: point it at any `inspector_data.json` emitted
by `04_reliability_analysis.py` — a different model, a different adapter, a
different dataset mix — and the ledger, risk map, and image-region audit work
unchanged.

Nothing is inferred. Every field the file omits is reported back to the caller as
a named gap and rendered as "not exported in this build", and the scores are
recomputed from `answer_correct` and `obj_recall` with the same gate used for the
released builds rather than trusting whatever the file's `quadrant` field says.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# Fields the file must provide for the dashboard to mean anything.
REQUIRED = ("crop_id", "dataset", "task", "answer_correct", "obj_recall", "n_gt", "n_pred")

# Fields the dashboard uses when present and labels as absent when not.
OPTIONAL = (
    "question_id", "vqa_question", "vqa_choices", "expected_letter", "expected_answer",
    "pred_answer_snippet", "gt_centroids", "pred_points", "image_path", "local_image_path",
    "point_f1", "pim_prec", "count_ae", "probe_id", "prediction_error",
)

MAX_RECORDS = 20_000
MAX_IMAGES = 4_000

# Raw dataset ids used by this project's exports, shown under the paper's names.
# Anything else is displayed exactly as it appears in the file.
DATASET_LABEL = {
    "lucchi_plus": "Lucchi",
    "vnc_drosophila": "VNC",
    "mitoem_human": "EM-H",
}


class IngestError(ValueError):
    """The upload cannot be audited, with a reason a user can act on."""


@dataclass
class Ingested:
    records: pd.DataFrame
    model: str
    condition: str
    tau_declared: float | None
    image_dir: Path | None
    n_images: int
    missing_fields: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _as_points(value) -> list[list[float]]:
    """Accept [{'x':..,'y':..}], [[x,y]], or a JSON string of either."""
    if value is None or value == "":
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    out: list[list[float]] = []
    for p in value:
        try:
            if isinstance(p, dict):
                out.append([float(p["x"]), float(p["y"])])
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                out.append([float(p[0]), float(p[1])])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _as_list(value) -> list:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return value if isinstance(value, list) else []


def _num(value):
    """Return a float, or None when the field is absent/unparseable."""
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def extract_images(zip_bytes: bytes, dest: Path) -> tuple[int, list[str]]:
    """Extract `<crop_id>.png` images from an uploaded archive.

    Entries are flattened to their basename and anything that is not a plain
    image file is skipped, so a zip made with any directory layout works and a
    crafted archive cannot write outside `dest`.
    """
    dest.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    written = 0
    with zipfile.ZipFile(io_bytes(zip_bytes)) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        if len(members) > MAX_IMAGES:
            raise IngestError(f"archive holds {len(members)} entries; the limit is {MAX_IMAGES}")
        for member in members:
            name = Path(member.filename).name
            if name.startswith(".") or Path(name).suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            with zf.open(member) as src, (dest / name).open("wb") as out:
                out.write(src.read())
            written += 1
    if not written:
        warnings.append("No .png/.jpg entries found in the archive.")
    return written, warnings


def io_bytes(data: bytes):
    import io
    return io.BytesIO(data)


def ingest(payload: dict, image_dir: Path | None = None) -> Ingested:
    """Validate and normalise an `inspector_data.json` payload."""
    if not isinstance(payload, dict):
        raise IngestError("Top level of the file is not a JSON object.")
    crops = payload.get("crops")
    if not isinstance(crops, list) or not crops:
        raise IngestError("No 'crops' array found. Expected the inspector_data.json schema "
                          "emitted by 04_reliability_analysis.py.")
    if len(crops) > MAX_RECORDS:
        raise IngestError(f"{len(crops)} records exceeds the {MAX_RECORDS} limit for an upload.")

    first = crops[0]
    missing_required = [f for f in REQUIRED if f not in first]
    if missing_required:
        raise IngestError("Records are missing required field(s): " + ", ".join(missing_required))

    meta = payload.get("meta") or {}
    model = str(meta.get("model") or "uploaded model")
    condition = str(meta.get("condition") or "uploaded run")
    tau_declared = _num(meta.get("tau"))

    # crop_id -> actual filename on disk, so a .jpg upload works as well as .png.
    available_stems: dict[str, str] = {}
    if image_dir and image_dir.exists():
        for p in sorted(image_dir.glob("*")):
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                available_stems.setdefault(p.stem, p.name)

    warnings: list[str] = []
    rows = []
    bad_gate = 0
    for crop in crops:
        try:
            answer_correct = int(crop["answer_correct"])
            obj_recall = float(crop["obj_recall"])
            n_gt, n_pred = int(crop["n_gt"]), int(crop["n_pred"])
        except (KeyError, TypeError, ValueError) as exc:
            raise IngestError(f"Record {crop.get('crop_id', '?')} has unreadable scores: {exc}")

        gt = _as_points(crop.get("gt_centroids"))
        pred = _as_points(crop.get("pred_points"))
        # Overlays must never claim more evidence than the record was scored on.
        if gt and len(gt) != n_gt:
            bad_gate += 1
            gt = gt[:n_gt]
        if pred and len(pred) != n_pred:
            bad_gate += 1
            pred = pred[:n_pred]

        crop_id = str(crop["crop_id"])
        image_file = available_stems.get(crop_id, "")

        rows.append({
            "build": "uploaded",
            "model": model,
            "condition": condition,
            "dataset_id": str(crop["dataset"]),
            "dataset": DATASET_LABEL.get(str(crop["dataset"]), str(crop["dataset"])),
            "task": str(crop["task"]),
            "probe_id": str(crop.get("probe_id") or ""),
            "crop_id": crop_id,
            "question_id": str(crop.get("question_id") or crop_id),
            "vqa_question": str(crop.get("vqa_question") or ""),
            "vqa_choices": _as_list(crop.get("vqa_choices")),
            "expected_letter": str(crop.get("expected_letter") or ""),
            "expected_answer": str(crop.get("expected_answer") or ""),
            "pred_answer_snippet": str(crop.get("pred_answer_snippet") or ""),
            "prediction_error": str(crop.get("prediction_error") or ""),
            "answer_correct": answer_correct,
            "n_gt": n_gt,
            "n_pred": n_pred,
            "obj_recall": obj_recall,
            "point_f1": _num(crop.get("point_f1")),
            "pim_prec": _num(crop.get("pim_prec")),
            "count_ae": _num(crop.get("count_ae")) if crop.get("count_ae") not in (None, "")
                        else float(abs(n_pred - n_gt)),
            "gt_centroids": gt,
            "pred_points": pred,
            "image_file": image_file,
            "source_image_path": str(crop.get("image_path") or ""),
        })

    if bad_gate:
        warnings.append(
            f"{bad_gate} record(s) listed a different number of coordinates than n_gt/n_pred; "
            "the extra coordinates were dropped so the overlay cannot overstate the evidence."
        )

    df = pd.DataFrame(rows)
    # Match the released schema: absent diagnostics are NaN floats, not None
    # objects, so the scoring code sees one dtype regardless of source.
    for col in ("point_f1", "pim_prec", "count_ae"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["has_image"] = df["image_file"] != ""

    # Report which optional fields the file simply does not carry, so the UI can
    # say so instead of showing a blank that looks like a zero.
    missing_fields = [
        name for name in ("vqa_question", "vqa_choices", "expected_answer",
                          "pred_answer_snippet", "gt_centroids", "pred_points",
                          "point_f1", "pim_prec")
        if name not in first or first.get(name) in (None, "", [])
    ]
    # A field is only "missing" if it is absent for every record.
    truly_missing = []
    for name in missing_fields:
        col = name if name in df.columns else None
        if col is None:
            truly_missing.append(name)
        elif name in ("vqa_choices", "gt_centroids", "pred_points"):
            if df[col].map(len).sum() == 0:
                truly_missing.append(name)
        elif name in ("point_f1", "pim_prec"):
            if df[col].isna().all():
                truly_missing.append(name)
        elif (df[col].astype(str).str.strip() == "").all():
            truly_missing.append(name)

    if tau_declared is not None and abs(tau_declared - 0.5) > 1e-9:
        warnings.append(
            f"The file declares tau={tau_declared:g}. The dashboard recomputes every state from "
            "obj_recall at the tau you select, so the file's own quadrant labels are ignored."
        )

    n_images = int(df["has_image"].sum())
    if image_dir is not None and n_images == 0 and available_stems:
        warnings.append(
            "None of the uploaded image filenames match a crop_id. Images must be named "
            "<crop_id>.png."
        )

    return Ingested(
        records=df, model=model, condition=condition, tau_declared=tau_declared,
        image_dir=image_dir, n_images=n_images,
        missing_fields=truly_missing, warnings=warnings,
    )


def ingest_file(path: Path, image_dir: Path | None = None) -> Ingested:
    return ingest(json.loads(Path(path).read_text()), image_dir)
