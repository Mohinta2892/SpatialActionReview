"""The upload path must add no assumptions of its own.

The paper claims the dashboard "can audit any vision-language run that exports
the paired audit record". The strongest available check on that claim is that
both published exports, fed through the *upload* loader rather than the release
loader, still reproduce their published rows exactly. If the loader guessed at
anything, these numbers would move.

The tests also pin the refusals: what the loader rejects, and what it reports as
absent instead of silently rendering as a zero.
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent
WORKSPACE = APP_DIR.parent
sys.path.insert(0, str(APP_DIR))

from sar.data import risk_map, summarise, with_gate  # noqa: E402
from sar.ingest import IngestError, extract_images, ingest, ingest_file  # noqa: E402

PAPER_TAU = 0.5
IMAGES = WORKSPACE / "plan" / "inspector_assets" / "images"

CASE_STUDY_JSON = WORKSPACE / "plan" / "inspector_data.json"
SFT_JOINT_JSON = WORKSPACE / "outputs" / "figs" / "inspector_data.json"

# (vqa_acc, obj_recall, silent_failure, trust_gap, gap_lo, gap_hi, point_biserial)
PUBLISHED = {
    # Section 4.1 / Tables 1-2.
    "case_study": (CASE_STUDY_JSON, "Staged GRPO", 541,
                   (0.503, 0.389, 0.274, 0.058, -0.029, 0.147, 0.061)),
    # Table 3, Joint SFT row.
    "joint_sft": (SFT_JOINT_JSON, "Joint SFT", 753,
                  (0.529, 0.325, 0.335, 0.026, -0.043, 0.094, 0.028)),
}


def _tuple(s):
    return (round(s.vqa_acc, 3), round(s.obj_recall, 3), round(s.silent_failure_rate, 3),
            round(s.trust_gap, 3), round(s.gap_lo, 3), round(s.gap_hi, 3),
            round(s.point_biserial, 3))


@pytest.mark.parametrize("key", list(PUBLISHED))
def test_upload_path_reproduces_published_row(key):
    path, condition, n, expected = PUBLISHED[key]
    if not path.exists():
        pytest.skip(f"{path} is not in this workspace")
    ing = ingest_file(path, image_dir=IMAGES)
    assert ing.condition == condition
    assert len(ing.records) == n
    assert _tuple(summarise(with_gate(ing.records, PAPER_TAU))) == expected


def test_upload_path_reproduces_the_risk_map_table():
    """Table 2, via the upload loader rather than the release."""
    if not CASE_STUDY_JSON.exists():
        pytest.skip("case-study export is not in this workspace")
    rm = risk_map(with_gate(ingest_file(CASE_STUDY_JSON, IMAGES).records, PAPER_TAU))
    expected = {
        ("presence", "Lucchi"): (0.471, 17), ("presence", "VNC"): (0.667, 6),
        ("presence", "EM-H"): (0.737, 19),
        ("counting", "Lucchi"): (0.071, 28), ("counting", "VNC"): (0.091, 11),
        ("counting", "EM-H"): (0.227, 22),
        ("location", "Lucchi"): (0.241, 145), ("location", "VNC"): (0.306, 49),
        ("location", "EM-H"): (0.329, 155),
        ("relation", "Lucchi"): (0.000, 16), ("relation", "VNC"): (0.250, 4),
        ("relation", "EM-H"): (0.000, 10),
        ("attribute", "Lucchi"): (0.111, 27), ("attribute", "VNC"): (0.222, 9),
        ("attribute", "EM-H"): (0.304, 23),
    }
    seen = {}
    for row in rm.itertuples():
        seen[(str(row.task), str(row.dataset))] = (round(row.silent_failure_rate, 3), row.n)
    assert set(seen) == set(expected)
    for key, (rate, n) in expected.items():
        got_rate, got_n = seen[key]
        assert got_n == n, key
        assert got_rate == pytest.approx(rate, abs=0.0006), key


def test_dataset_ids_are_shown_under_the_papers_names():
    ing = ingest_file(CASE_STUDY_JSON, IMAGES)
    assert set(ing.records["dataset"]) == {"Lucchi", "VNC", "EM-H"}
    # The raw id is preserved for provenance rather than replaced.
    assert set(ing.records["dataset_id"]) == {"lucchi_plus", "vnc_drosophila", "mitoem_human"}


def test_absent_diagnostics_are_reported_not_zeroed():
    ing = ingest_file(CASE_STUDY_JSON, IMAGES)
    assert "point_f1" in ing.missing_fields and "pim_prec" in ing.missing_fields
    assert ing.records["point_f1"].isna().all()
    assert ing.records["pim_prec"].isna().all()


def test_quadrants_are_recomputed_not_trusted():
    """A file whose own quadrant label is wrong must still be scored correctly."""
    payload = {
        "meta": {"model": "M", "condition": "C", "tau": 0.5},
        "crops": [
            # Correct answer, recall below the gate: a silent failure, whatever
            # the file says.
            {"crop_id": "a", "dataset": "D", "task": "presence", "answer_correct": 1,
             "obj_recall": 0.1, "n_gt": 1, "n_pred": 1, "quadrant": "trustworthy"},
            {"crop_id": "b", "dataset": "D", "task": "presence", "answer_correct": 0,
             "obj_recall": 0.9, "n_gt": 1, "n_pred": 1, "quadrant": "honest"},
        ],
    }
    scored = with_gate(ingest(payload).records, PAPER_TAU)
    assert list(scored["quadrant"]) == ["silent_failure", "lucky"]


def test_tau_is_applied_at_read_time_not_baked_in():
    payload = {
        "meta": {"model": "M", "condition": "C"},
        "crops": [{"crop_id": "a", "dataset": "D", "task": "presence", "answer_correct": 1,
                   "obj_recall": 0.4, "n_gt": 1, "n_pred": 1}],
    }
    records = ingest(payload).records
    assert with_gate(records, 0.3)["quadrant"].iloc[0] == "trustworthy"
    assert with_gate(records, 0.5)["quadrant"].iloc[0] == "silent_failure"


def test_point_formats_are_both_accepted():
    """Points may arrive as {'x':..,'y':..} dicts or [x, y] pairs."""
    base = {"crop_id": "a", "dataset": "D", "task": "presence", "answer_correct": 1,
            "obj_recall": 1.0, "n_gt": 1, "n_pred": 1}
    dicts = ingest({"crops": [{**base, "gt_centroids": [{"x": 0.25, "y": 0.75}],
                               "pred_points": [{"x": 0.5, "y": 0.5}]}]}).records
    pairs = ingest({"crops": [{**base, "gt_centroids": [[0.25, 0.75]],
                               "pred_points": [[0.5, 0.5]]}]}).records
    assert dicts["gt_centroids"].iloc[0] == pairs["gt_centroids"].iloc[0] == [[0.25, 0.75]]
    assert dicts["pred_points"].iloc[0] == pairs["pred_points"].iloc[0] == [[0.5, 0.5]]


def test_overlay_cannot_overstate_the_evidence():
    """More coordinates than n_gt/n_pred must be trimmed, and reported."""
    payload = {"crops": [{"crop_id": "a", "dataset": "D", "task": "presence",
                          "answer_correct": 1, "obj_recall": 1.0, "n_gt": 1, "n_pred": 1,
                          "gt_centroids": [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]],
                          "pred_points": [[0.4, 0.4], [0.5, 0.5]]}]}
    ing = ingest(payload)
    assert len(ing.records["gt_centroids"].iloc[0]) == 1
    assert len(ing.records["pred_points"].iloc[0]) == 1
    assert any("overstate" in w for w in ing.warnings)


@pytest.mark.parametrize("payload, fragment", [
    ({}, "crops"),
    ({"crops": []}, "crops"),
    ({"crops": [{"crop_id": "a"}]}, "missing required field"),
    ({"crops": [{"crop_id": "a", "dataset": "D", "task": "t", "answer_correct": 1,
                 "obj_recall": "not a number", "n_gt": 1, "n_pred": 1}]}, "unreadable"),
])
def test_bad_payloads_are_refused_with_a_reason(payload, fragment):
    with pytest.raises(IngestError) as err:
        ingest(payload)
    assert fragment in str(err.value)


def test_non_object_payload_is_refused():
    with pytest.raises(IngestError):
        ingest([1, 2, 3])  # type: ignore[arg-type]


def test_zip_entries_are_flattened_and_filtered(tmp_path):
    """Any directory layout works, and non-images are skipped."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("deep/nested/dir/crop_a.png", b"\x89PNG\r\n\x1a\n")
        zf.writestr("crop_b.PNG", b"\x89PNG\r\n\x1a\n")
        zf.writestr("notes.txt", b"ignore me")
        zf.writestr("__MACOSX/._crop_c.png", b"ignore me")
    written, _ = extract_images(buf.getvalue(), tmp_path)
    assert written == 2
    assert {p.name for p in tmp_path.iterdir()} == {"crop_a.png", "crop_b.PNG"}


def test_zip_cannot_escape_the_destination(tmp_path):
    """A path-traversal entry must land inside dest, not above it."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../../escaped.png", b"\x89PNG\r\n\x1a\n")
    dest = tmp_path / "images"
    extract_images(buf.getvalue(), dest)
    assert (dest / "escaped.png").exists()
    assert not (tmp_path.parent / "escaped.png").exists()


def test_images_are_matched_by_crop_id(tmp_path):
    (tmp_path / "crop_a.png").write_bytes(b"x")
    payload = {"crops": [
        {"crop_id": "crop_a", "dataset": "D", "task": "t", "answer_correct": 1,
         "obj_recall": 1.0, "n_gt": 1, "n_pred": 1},
        {"crop_id": "crop_missing", "dataset": "D", "task": "t", "answer_correct": 0,
         "obj_recall": 0.0, "n_gt": 1, "n_pred": 1},
    ]}
    ing = ingest(payload, tmp_path)
    assert ing.n_images == 1
    assert list(ing.records["has_image"]) == [True, False]


def test_unmatched_image_names_are_flagged(tmp_path):
    (tmp_path / "unrelated.png").write_bytes(b"x")
    payload = {"crops": [{"crop_id": "crop_a", "dataset": "D", "task": "t",
                          "answer_correct": 1, "obj_recall": 1.0, "n_gt": 1, "n_pred": 1}]}
    ing = ingest(payload, tmp_path)
    assert ing.n_images == 0
    assert any("crop_id" in w for w in ing.warnings)


def test_record_limit_is_enforced():
    payload = {"crops": [{"crop_id": str(i), "dataset": "D", "task": "t", "answer_correct": 1,
                          "obj_recall": 1.0, "n_gt": 1, "n_pred": 1} for i in range(3)]}
    from sar import ingest as mod
    original = mod.MAX_RECORDS
    try:
        mod.MAX_RECORDS = 2
        with pytest.raises(IngestError) as err:
            mod.ingest(payload)
        assert "exceeds" in str(err.value)
    finally:
        mod.MAX_RECORDS = original
