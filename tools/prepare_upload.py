#!/usr/bin/env python3
"""Normalise an arbitrary model export into an `inspector_data.json` the
dashboard's **Your own run** upload will load cleanly.

Why this exists: the upload loader (`sar/ingest.py`) already accepts a few
equivalent encodings of the same field — `vqa_choices`, `gt_centroids` and
`pred_points` may each arrive as a native JSON list or as a JSON-encoded
*string* of one, because different export pipelines produce different forms
(this project's own release build serialises `vqa_choices` as a string; the
case-study export in `plan/inspector_data.json` writes it as a list). The
dashboard treats both the same way. Anything that consumes the raw file
*without* going through that normalisation — a notebook, a one-off analysis
script — does not get that for free, and will silently disagree with the
dashboard about something as basic as how many options a record's multiple
choice question had.

This script runs an export through the exact same normalisation the dashboard
applies, then writes out a canonical file where every record uses the native
list form. There is then only one representation to reason about, on disk and
in the dashboard.

It also validates the file the same way the upload loader does — the required
fields, per record — and reports the option-count (K) distribution and the
mean(1/K) uniform-choice baseline so you can sanity-check a run before
uploading it, the same numbers the dashboard's case-study read-out shows.

Usage:
    python tools/prepare_upload.py RAW_EXPORT.json -o inspector_data.json
    python tools/prepare_upload.py RAW_EXPORT.json -o inspector_data.json \\
        --images crops/ --images-zip crops.zip
    python tools/prepare_upload.py RAW_EXPORT.json -o inspector_data.json \\
        --rename id=crop_id --rename is_correct=answer_correct \\
        --model "My-VLM" --condition "zero-shot" --drop-invalid

Input shapes accepted:
    {"meta": {...}, "crops": [...]}      the dashboard's own schema
    [...]                                a bare list of records
    {"records": [...]} / {"data": [...]} / {"results": [...]}
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

from sar.ingest import (  # noqa: E402
    REQUIRED,
    IngestError,
    as_list,
    as_points,
    ingest,
)

# Fields that must round-trip as a native list, never a JSON-encoded string.
LIST_FIELDS = ("vqa_choices",)
POINT_FIELDS = ("gt_centroids", "pred_points")

# Common container keys other export pipelines use instead of "crops".
CONTAINER_KEYS = ("crops", "records", "data", "results", "rows")


class PrepareError(ValueError):
    """The input cannot be turned into an upload, with a reason to act on."""


def _load_records(payload) -> tuple[list[dict], dict]:
    """Return (records, meta) from any of the accepted top-level shapes."""
    if isinstance(payload, list):
        return payload, {}
    if isinstance(payload, dict):
        for key in CONTAINER_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
                return value, meta
        raise PrepareError(
            "No record array found. Expected a bare JSON list, or an object with one of "
            f"these keys holding a list: {', '.join(CONTAINER_KEYS)}."
        )
    raise PrepareError("Top level of the file is neither a JSON list nor a JSON object.")


def _rename(record: dict, mapping: dict[str, str]) -> dict:
    if not mapping:
        return record
    out = dict(record)
    for old, new in mapping.items():
        if old in out:
            out[new] = out.pop(old)
    return out


def _canonicalise(record: dict) -> dict:
    """Rewrite the ambiguous fields to their one native-list form in place."""
    out = dict(record)
    for field in LIST_FIELDS:
        if field in out:
            out[field] = as_list(out[field])
    for field in POINT_FIELDS:
        if field in out:
            out[field] = as_points(out[field])
    return out


def _option_stats(records: list[dict]) -> dict:
    counts: Counter[int] = Counter()
    inverses = []
    for r in records:
        k = len(r.get("vqa_choices") or [])
        if k:
            counts[k] += 1
            inverses.append(1.0 / k)
    return {
        "k_distribution": dict(sorted(counts.items())),
        "n_no_options": sum(1 for r in records if not r.get("vqa_choices")),
        "uniform_choice_baseline": round(sum(inverses) / len(inverses), 3) if inverses else None,
    }


def prepare(payload, rename: dict[str, str], drop_invalid: bool,
            model: str | None, condition: str | None, tau: float | None) -> tuple[dict, list[str]]:
    """Validate and canonicalise `payload`. Returns (inspector_data.json body, report lines)."""
    raw_records, meta = _load_records(payload)
    if not raw_records:
        raise PrepareError("The record array is empty.")

    report: list[str] = [f"{len(raw_records)} record(s) in the input."]

    records = [_canonicalise(_rename(r, rename)) for r in raw_records]

    missing_by_field: Counter[str] = Counter()
    valid, dropped = [], []
    for r in records:
        missing = [f for f in REQUIRED if f not in r or r[f] in (None, "")]
        if missing:
            for f in missing:
                missing_by_field[f] += 1
            dropped.append(r)
            continue
        valid.append(r)

    if missing_by_field:
        detail = ", ".join(f"{field} (missing on {n})" for field, n in missing_by_field.items())
        if drop_invalid:
            report.append(f"Dropped {len(dropped)} record(s) missing required field(s): {detail}.")
        else:
            raise PrepareError(
                f"{len(dropped)} record(s) are missing required field(s): {detail}. "
                "Use --rename to map differently-named source fields onto these, or "
                "--drop-invalid to skip the offending records instead of failing."
            )
    if not valid:
        raise PrepareError("No record has every required field: " + ", ".join(REQUIRED))

    out_meta = dict(meta)
    if model is not None:
        out_meta["model"] = model
    if condition is not None:
        out_meta["condition"] = condition
    if tau is not None:
        out_meta["tau"] = tau

    payload_out = {"meta": out_meta, "crops": valid}

    # Round-trip it through the exact loader the dashboard uses, so a file this
    # script accepts is, by construction, a file the upload tab accepts too.
    ingested = ingest(payload_out)
    report.append(f"{len(valid)} record(s) pass validation for upload.")
    if ingested.missing_fields:
        report.append(
            "Optional fields not present in this export (shown as absent in the dashboard, "
            "not as zero): " + ", ".join(ingested.missing_fields) + "."
        )
    report.extend(ingested.warnings)

    opt = _option_stats(valid)
    if opt["k_distribution"]:
        dist = ", ".join(f"K={k}: {n}" for k, n in opt["k_distribution"].items())
        report.append(f"Option-count (K) distribution — {dist}.")
        report.append(f"mean(1/K) uniform-choice baseline: {opt['uniform_choice_baseline']}.")
    if opt["n_no_options"]:
        report.append(
            f"{opt['n_no_options']} record(s) have no vqa_choices — excluded from the baseline."
        )

    datasets = Counter(str(r["dataset"]) for r in valid)
    tasks = Counter(str(r["task"]) for r in valid)
    report.append("By dataset: " + ", ".join(f"{k}: {v}" for k, v in sorted(datasets.items())) + ".")
    report.append("By task: " + ", ".join(f"{k}: {v}" for k, v in sorted(tasks.items())) + ".")

    return payload_out, report


def bundle_images(records: list[dict], images_dir: Path, out_zip: Path) -> list[str]:
    """Zip <crop_id>.(png|jpg|jpeg) files from `images_dir`, named for upload."""
    warnings: list[str] = []
    available = {
        p.stem: p for p in images_dir.glob("*")
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    }
    wanted = [str(r["crop_id"]) for r in records]
    matched = [cid for cid in wanted if cid in available]
    if not matched:
        warnings.append(f"No file under {images_dir} matches a crop_id; the zip will be empty.")
    else:
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for cid in matched:
                zf.write(available[cid], arcname=available[cid].name)
        if len(matched) < len(wanted):
            warnings.append(
                f"{len(wanted) - len(matched)} of {len(wanted)} record(s) have no matching image "
                f"file in {images_dir}."
            )
    return warnings


def _parse_rename(pairs: list[str]) -> dict[str, str]:
    mapping = {}
    for pair in pairs:
        if "=" not in pair:
            raise PrepareError(f"--rename expects old=new, got {pair!r}")
        old, new = pair.split("=", 1)
        mapping[old.strip()] = new.strip()
    return mapping


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="raw export to normalise (JSON)")
    ap.add_argument("-o", "--output", type=Path, default=Path("inspector_data.json"),
                     help="where to write the canonical inspector_data.json (default: ./inspector_data.json)")
    ap.add_argument("--images", type=Path, default=None,
                     help="directory of crop images to bundle, matched to crop_id by filename stem")
    ap.add_argument("--images-zip", type=Path, default=None,
                     help="output zip path for --images (default: <output stem>_images.zip)")
    ap.add_argument("--rename", action="append", default=[], metavar="OLD=NEW",
                     help="rename a source field before validation; repeatable")
    ap.add_argument("--drop-invalid", action="store_true",
                     help="skip records missing a required field instead of failing")
    ap.add_argument("--model", default=None, help="override meta.model in the output")
    ap.add_argument("--condition", default=None, help="override meta.condition in the output")
    ap.add_argument("--tau", type=float, default=None, help="override meta.tau in the output")
    args = ap.parse_args(argv)

    try:
        payload = json.loads(args.input.read_text())
        rename = _parse_rename(args.rename)
        out_payload, report = prepare(
            payload, rename, args.drop_invalid, args.model, args.condition, args.tau
        )
    except (PrepareError, IngestError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    args.output.write_text(json.dumps(out_payload, indent=2))
    report.append(f"Wrote {args.output}.")

    if args.images is not None:
        out_zip = args.images_zip or args.output.with_name(args.output.stem + "_images.zip")
        try:
            report.extend(bundle_images(out_payload["crops"], args.images, out_zip))
            report.append(f"Wrote {out_zip}.")
        except OSError as exc:
            print(f"error bundling images: {exc}", file=sys.stderr)
            return 1

    print("\n".join(f"- {line}" for line in report))
    print(
        f"\nUpload {args.output.name}"
        + (f" and {out_zip.name}" if args.images is not None else "")
        + " in the dashboard's 'Your own run' tab."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
