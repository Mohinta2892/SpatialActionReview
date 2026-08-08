"""tools/prepare_upload.py must produce exactly what the upload loader expects.

The two encodings of `vqa_choices` (native list vs. JSON-encoded string) that
this project's own artifacts mix are the motivating case: a file the script
writes must always carry the native-list form, and a file built from either
input encoding must produce the same K distribution and baseline.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

from sar.ingest import ingest  # noqa: E402
from tools.prepare_upload import (  # noqa: E402
    PrepareError,
    bundle_images,
    prepare,
)

RECORD = {
    "crop_id": "c1", "dataset": "lucchi_plus", "task": "location",
    "answer_correct": 1, "obj_recall": 0.8, "n_gt": 2, "n_pred": 2,
    "vqa_choices": ["a", "b", "c"],
}


def test_bare_list_and_container_keys_all_accepted():
    for payload in ([RECORD], {"crops": [RECORD]}, {"records": [RECORD]}, {"data": [RECORD]}):
        out, _ = prepare(payload, {}, drop_invalid=False, model=None, condition=None, tau=None)
        assert len(out["crops"]) == 1


def test_string_and_list_vqa_choices_normalise_to_the_same_k():
    as_string = {**RECORD, "crop_id": "c1", "vqa_choices": json.dumps(["a", "b", "c"])}
    as_list = {**RECORD, "crop_id": "c2", "vqa_choices": ["a", "b", "c"]}
    out, report = prepare({"crops": [as_string, as_list]}, {}, False, None, None, None)
    for crop in out["crops"]:
        assert isinstance(crop["vqa_choices"], list)
        assert len(crop["vqa_choices"]) == 3
    assert any("K=3: 2" in line for line in report)


def test_output_round_trips_through_the_real_upload_loader():
    out, _ = prepare({"crops": [RECORD]}, {}, False, None, None, None)
    ingested = ingest(out)
    assert len(ingested.records) == 1
    assert ingested.records.iloc[0]["vqa_choices"] == ["a", "b", "c"]


def test_rename_maps_differently_named_source_fields():
    raw = {"id": "c1", "dataset": "lucchi_plus", "task": "location", "is_correct": 1,
           "obj_recall": 0.8, "n_gt": 2, "n_pred": 2}
    out, _ = prepare(
        {"crops": [raw]}, {"id": "crop_id", "is_correct": "answer_correct"},
        drop_invalid=False, model=None, condition=None, tau=None,
    )
    assert out["crops"][0]["crop_id"] == "c1"
    assert out["crops"][0]["answer_correct"] == 1


def test_missing_required_field_fails_without_drop_invalid():
    bad = {"dataset": "D", "task": "t", "answer_correct": 1, "obj_recall": 0.5,
           "n_gt": 1, "n_pred": 1}  # no crop_id
    with pytest.raises(PrepareError, match="crop_id"):
        prepare({"crops": [bad]}, {}, drop_invalid=False, model=None, condition=None, tau=None)


def test_drop_invalid_skips_bad_records_and_keeps_the_rest():
    bad = {"dataset": "D", "task": "t", "answer_correct": 1, "obj_recall": 0.5,
           "n_gt": 1, "n_pred": 1}
    out, report = prepare({"crops": [bad, RECORD]}, {}, drop_invalid=True,
                           model=None, condition=None, tau=None)
    assert len(out["crops"]) == 1
    assert any("Dropped 1" in line for line in report)


def test_empty_record_array_is_refused():
    with pytest.raises(PrepareError):
        prepare({"crops": []}, {}, False, None, None, None)


def test_meta_overrides_are_applied():
    out, _ = prepare({"crops": [RECORD]}, {}, False, model="M", condition="C", tau=0.7)
    assert out["meta"]["model"] == "M"
    assert out["meta"]["condition"] == "C"
    assert out["meta"]["tau"] == 0.7


def test_bundle_images_matches_by_crop_id_stem(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "c1.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (images / "unrelated.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    out_zip = tmp_path / "out.zip"
    warnings = bundle_images([RECORD], images, out_zip)
    assert not warnings
    with zipfile.ZipFile(out_zip) as zf:
        assert zf.namelist() == ["c1.png"]


def test_bundle_images_warns_on_unmatched_records(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    out_zip = tmp_path / "out.zip"
    warnings = bundle_images([RECORD], images, out_zip)
    assert any("no file" in w.lower() for w in warnings)
