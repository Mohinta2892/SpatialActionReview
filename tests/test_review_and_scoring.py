"""Sign-off bookkeeping and the live-scoring port.

A routing decision is only meaningful with the state it was taken under, so these
tests pin that the log records the gate and the behavioural state, that
re-deciding replaces rather than duplicates, and that a decision taken at one
gate is reported as stale rather than silently re-evaluated when the gate moves.

The scoring tests pin the port of the offline scorer: the same edge cases, and
the same greedy fallback when optimal assignment is unavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

from sar import review, scoring  # noqa: E402


def _row(question_id="q1", quadrant="silent_failure", recall=0.1, correct=1):
    return pd.Series({
        "question_id": question_id, "crop_id": question_id.split(":")[0],
        "condition": "Staged GRPO", "task": "presence", "dataset": "Lucchi",
        "quadrant": quadrant, "obj_recall": recall, "answer_correct": correct,
    })


# ------------------------------------------------------------------ routing

def test_routes_are_the_four_dispositions():
    assert [r.key for r in review.ROUTES] == [
        "accept", "human_audit", "stricter_gate", "model_revision"
    ]
    for route in review.ROUTES:
        assert route.label and route.blurb


def test_a_decision_stores_the_state_it_was_taken_under():
    log = review.record({}, row=_row(), source="Case study", tau=0.5, route="accept",
                        note="looks fine")
    d = log["q1"]
    assert d.route == "accept"
    assert d.tau == 0.5
    assert d.state == "silent_failure"
    assert d.obj_recall == 0.1
    assert d.answer_correct == 1
    assert d.condition == "Staged GRPO"
    assert d.source == "Case study"
    assert d.note == "looks fine"
    assert d.decided_at.endswith("+00:00")


def test_re_deciding_replaces_rather_than_duplicates():
    log = review.record({}, row=_row(), source="s", tau=0.5, route="accept")
    log = review.record(log, row=_row(), source="s", tau=0.5, route="human_audit")
    assert len(log) == 1
    assert log["q1"].route == "human_audit"


def test_unknown_route_is_refused():
    with pytest.raises(ValueError):
        review.record({}, row=_row(), source="s", tau=0.5, route="looks_ok_to_me")


def test_clear_removes_only_that_record():
    log = review.record({}, row=_row("q1"), source="s", tau=0.5, route="accept")
    log = review.record(log, row=_row("q2"), source="s", tau=0.5, route="accept")
    review.clear(log, "q1")
    assert set(log) == {"q2"}
    review.clear(log, "not-there")  # must not raise
    assert set(log) == {"q2"}


def test_counts_cover_every_route():
    log = {}
    for i, route in enumerate(r.key for r in review.ROUTES):
        log = review.record(log, row=_row(f"q{i}"), source="s", tau=0.5, route=route)
    assert review.counts(log) == {r.key: 1 for r in review.ROUTES}
    assert review.counts({}) == {r.key: 0 for r in review.ROUTES}


def test_decisions_taken_at_another_gate_are_reported_stale():
    """Clearing at tau = 0.3 is not clearing at tau = 0.7."""
    log = review.record({}, row=_row(), source="s", tau=0.3, route="accept")
    assert review.stale(log, 0.3) == []
    assert len(review.stale(log, 0.7)) == 1


def test_coverage_counts_only_the_current_queue():
    log = review.record({}, row=_row("q1"), source="s", tau=0.5, route="accept")
    assert review.coverage(log, ["q1", "q2", "q3"]) == (1, 3)
    assert review.coverage(log, []) == (0, 0)


def test_frame_is_exportable_and_empty_frame_has_the_columns():
    empty = review.to_frame({})
    assert list(empty.columns)
    assert empty.empty
    log = review.record({}, row=_row(), source="s", tau=0.5, route="model_revision")
    frame = review.to_frame(log)
    assert frame.loc[0, "route_label"] == "Flag for model revision"
    assert "note" in frame.columns
    assert frame.to_csv(index=False)


# ------------------------------------------------------------------ scoring

def test_point_metrics_edge_cases_match_the_offline_scorer():
    assert scoring.point_metrics([], []) == {"point_precision": 1.0, "obj_recall": 1.0,
                                             "point_f1": 1.0, "on_target": 0}
    assert scoring.point_metrics([[0.5, 0.5]], [])["obj_recall"] == 0.0
    assert scoring.point_metrics([], [[0.5, 0.5]])["obj_recall"] == 0.0


def test_a_point_inside_the_threshold_counts_and_outside_does_not():
    gt = [[0.5, 0.5]]
    assert scoring.point_metrics([[0.54, 0.5]], gt)["obj_recall"] == 1.0   # dist 0.04
    assert scoring.point_metrics([[0.65, 0.5]], gt)["obj_recall"] == 0.0   # dist 0.15


def test_recall_is_over_ground_truth_and_precision_over_predictions():
    gt = [[0.2, 0.2], [0.8, 0.8]]
    pred = [[0.2, 0.2], [0.5, 0.5], [0.55, 0.5]]
    m = scoring.point_metrics(pred, gt)
    assert m["obj_recall"] == pytest.approx(0.5)
    assert m["point_precision"] == pytest.approx(1 / 3)


def test_on_target_counts_hits_and_the_remainder_are_stray():
    """The count the recall-only gate cannot see: points that hit nothing."""
    gt = [[0.2, 0.2], [0.8, 0.8]]
    pred = [[0.2, 0.2], [0.8, 0.8], [0.5, 0.1], [0.1, 0.9]]
    m = scoring.point_metrics(pred, gt)
    assert m["on_target"] == 2
    assert len(pred) - m["on_target"] == 2      # two actions on empty image
    assert m["obj_recall"] == 1.0               # yet coverage is perfect
    assert m["point_precision"] == pytest.approx(0.5)


def test_each_object_can_only_be_matched_once():
    """Two predictions on one object must not count as two hits."""
    m = scoring.point_metrics([[0.5, 0.5], [0.51, 0.5]], [[0.5, 0.5], [0.9, 0.9]])
    assert m["obj_recall"] == pytest.approx(0.5)


def test_point_tags_are_parsed_and_out_of_range_dropped():
    text = ('<point x="0.10" y="0.20" alt="mitochondrion"/> '
            '<point x="0.9" y="1.4"/> '
            "<point x='0.5' y='0.5' />")
    assert scoring.parse_points(text) == [[0.10, 0.20], [0.5, 0.5]]
    assert scoring.parse_points("") == []
    assert scoring.parse_points("no points here") == []


@pytest.mark.parametrize("text, expected", [
    ("B", "B"), ("B.", "B"), ("b) no", "B"), ("  C : center", "C"),
    ("The answer is D", "D"), ("", ""), ("zzz", ""),
])
def test_option_letter_recovery(text, expected):
    assert scoring.parse_option_letter(text, 4) == expected


def test_option_letter_respects_the_number_of_options():
    # With two options, 'E' is not a valid choice.
    assert scoring.parse_option_letter("E. center", 2) == ""
    assert scoring.parse_option_letter("E. center", 9) == "E"


# ------------------------------------------------- stray-action diagnostics

def test_stray_columns_are_attached_and_consistent():
    """Every record must account for each emitted point as on-target or stray."""
    from sar.data import build_records, load_release, with_gate
    df, manifest = load_release(APP_DIR / "release")
    for build in manifest["build_order"]:
        for condition in manifest["builds"][build]["conditions"]:
            scored = with_gate(build_records(df, build, condition), 0.5)
            assert (scored["points_on_target"] + scored["points_stray"]
                    == scored["n_pred"]).all(), (build, condition)
            assert (scored["points_on_target"] <= scored["n_gt"]).all()
            assert (scored["points_stray"] >= 0).all()
            flagged = scored[scored["stray_despite_pass"] == 1]
            assert (flagged["action_reliable"] == 1).all()
            assert (flagged["points_stray"] > 0).all()


def test_stray_diagnostics_survive_in_the_summary():
    from sar.data import build_records, load_release, summarise, with_gate
    df, _ = load_release(APP_DIR / "release")
    scored = with_gate(build_records(df, "case_study", "Staged GRPO"), 0.5)
    s = summarise(scored)
    assert s.n_gate_pass == int((scored["action_reliable"] == 1).sum())
    assert 0 <= s.n_stray_despite_pass <= s.n_gate_pass
    assert 0.0 <= s.mean_precision_gate_pass <= 1.0
    assert 0 <= s.stray_points_gate_pass <= s.cleared_points
    # The finding this diagnostic exists for: a large share of the actions the
    # gate clears land on nothing.
    assert s.n_stray_despite_pass > 0


def test_a_perfect_action_raises_no_flag():
    from sar.data import with_gate
    frame = pd.DataFrame([{
        "answer_correct": 1, "obj_recall": 1.0, "n_gt": 2, "n_pred": 2,
        "gt_centroids": [[0.2, 0.2], [0.8, 0.8]],
        "pred_points": [[0.2, 0.2], [0.8, 0.8]],
    }])
    scored = with_gate(frame, 0.5)
    assert scored.loc[0, "points_stray"] == 0
    assert scored.loc[0, "stray_despite_pass"] == 0


def test_json_document_is_self_describing():
    """A downstream workflow must be able to read the export without this app."""
    import json
    log = {}
    for i, route in enumerate(r.key for r in review.ROUTES):
        log = review.record(log, row=_row(f"q{i}"), source="Case study", tau=0.5, route=route)
    doc = review.to_document(log, context={"model": "Qwen3-VL", "condition": "Staged GRPO"})

    assert doc["schema"] == "spatial-action-review/routing-decisions"
    assert doc["schema_version"] == review.SCHEMA_VERSION
    assert doc["context"]["condition"] == "Staged GRPO"
    # The route vocabulary travels with the data.
    assert {r["key"] for r in doc["routes"]} == {r.key for r in review.ROUTES}
    assert all(r["label"] and r["meaning"] for r in doc["routes"])
    # Totals are stated so a reader can check their own aggregation.
    assert doc["totals"]["decisions"] == len(log)
    assert doc["totals"]["by_route"] == review.counts(log)
    # Every decision carries the pass mark and state it was taken under.
    for d in doc["decisions"]:
        assert {"question_id", "route", "tau", "state", "obj_recall", "decided_at"} <= set(d)
    assert [d["decided_at"] for d in doc["decisions"]] == sorted(
        d["decided_at"] for d in doc["decisions"]
    )
    assert set(doc["field_notes"]) >= {"tau", "state", "obj_recall", "decided_at"}
    json.dumps(doc)          # must be serialisable as-is


def test_empty_document_is_still_valid():
    doc = review.to_document({})
    assert doc["totals"]["decisions"] == 0
    assert doc["decisions"] == []
    assert doc["routes"]


# --- live re-ask overlay ---------------------------------------------------
# Live points are new inference drawn beside a record, never merged into it, so
# the layer must be additive and must not exist unless a re-ask was made.

def test_live_points_are_a_separate_layer():
    from sar.render import crop_tile

    gt, pred, live = [[0.5, 0.3]], [[0.2, 0.2]], [[0.8, 0.8]]
    plain, _ = crop_tile(None, gt, pred)
    with_live, _ = crop_tile(None, gt, pred, live_points=live)
    default_is_empty, _ = crop_tile(None, gt, pred, live_points=[])

    assert plain.tobytes() != with_live.tobytes(), "live points were not drawn"
    assert plain.tobytes() == default_is_empty.tobytes(), (
        "an empty live layer must render exactly as no live layer"
    )


def test_live_layer_does_not_disturb_the_recorded_marks():
    from sar.render import crop_tile

    gt = [[0.5, 0.3]]
    only_recorded, _ = crop_tile(None, gt, [[0.2, 0.2]])
    recorded_plus_live, _ = crop_tile(None, gt, [[0.2, 0.2]], live_points=[[0.8, 0.8]])
    only_live, _ = crop_tile(None, gt, [], live_points=[[0.8, 0.8]])

    # The three tiles are mutually distinct: the recorded and the live marks are
    # independently present or absent, so neither can be read as the other.
    assert len({only_recorded.tobytes(), recorded_plus_live.tobytes(),
                only_live.tobytes()}) == 3


def test_live_scoring_never_writes_into_the_record():
    """`reask` returns its own result object; it must not mutate the record."""
    from sar import serve

    record = {
        "vqa_question": "q", "vqa_choices": ["a", "b"], "expected_letter": "A",
        "gt_centroids": [[0.5, 0.5]],
    }
    before = dict(record)
    endpoint = serve.Endpoint()          # unconfigured, so every call fails fast
    result = serve.reask(endpoint, record, None, 0.5)
    assert record == before
    assert result.errors and result.obj_recall is None
