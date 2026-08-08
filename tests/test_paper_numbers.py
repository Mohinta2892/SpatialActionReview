"""Camera-ready numbers in the report are pinned to this release.

These are the values destined for the manuscript text. Pinning them means a
future change to the release, the scoring code or the builder cannot move a
published number without a test saying so.

Values were produced by `python scripts/paper_numbers.py` against
release/records.csv.gz and cross-checked against the manuscript where it already
quotes them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(APP_DIR / "scripts"))

import paper_numbers  # noqa: E402


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    figures = tmp_path_factory.mktemp("figures")
    return paper_numbers.build_report(APP_DIR / "release", figures)


# --- item 1: mean object recall, to the precision that decides the rounding ---

def test_item1_mean_object_recall(report):
    value = report["1_mean_obj_recall"]
    assert round(value, 5) == 0.38948
    # The question the manuscript needed answered: 38.9%, not 39.0%.
    assert round(100 * value, 1) == 38.9


# --- item 2: point-in-mask precision among answer-correct records ------------

def test_item2_point_in_mask_precision(report):
    assert round(report["2_pim_given_correct"], 3) == 0.471
    assert report["2_pim_n"] == 272


# --- item 3: the stray-action table, every released condition ----------------

EXPECTED_STRAY = {
    "Case study (541)|Staged GRPO":
        dict(n_pass=231, with_stray=172, stray_points=213, cleared_points=510, over=103),
    "Matched set (753)|Zero-shot":
        dict(n_pass=163, with_stray=65, stray_points=85, cleared_points=257, over=26),
    "Matched set (753)|Perception SFT":
        dict(n_pass=195, with_stray=96, stray_points=129, cleared_points=333, over=40),
    "Matched set (753)|Grounding SFT":
        dict(n_pass=186, with_stray=122, stray_points=142, cleared_points=383, over=59),
    "Matched set (753)|Staged SFT":
        dict(n_pass=297, with_stray=214, stray_points=263, cleared_points=643, over=125),
    "Matched set (753)|Joint SFT":
        dict(n_pass=267, with_stray=197, stray_points=243, cleared_points=581, over=106),
}


def test_item3_covers_every_released_condition(report):
    assert set(report["3_stray"]) == set(EXPECTED_STRAY)


@pytest.mark.parametrize("key", list(EXPECTED_STRAY))
def test_item3_stray_counts(report, key):
    got, want = report["3_stray"][key], EXPECTED_STRAY[key]
    assert got["n_pass"] == want["n_pass"]
    assert got["with_stray"] == want["with_stray"]
    assert got["stray_points"] == want["stray_points"]
    assert got["cleared_points"] == want["cleared_points"]
    assert got["over_emitting"] == want["over"]
    # The shares are derived, so check they agree with the counts they came from.
    assert got["with_stray_share"] == pytest.approx(want["with_stray"] / want["n_pass"])
    assert got["stray_point_share"] == pytest.approx(
        want["stray_points"] / want["cleared_points"])


def test_item3_hit_rate_is_a_proportion(report):
    for key, row in report["3_stray"].items():
        assert 0.0 <= row["mean_hit_rate"] <= 1.0, key


def test_case_study_values_stay_pinned(report):
    """The 541-record case study stays distinct from the 753-record SFT audit."""
    from sar.data import build_records, load_release, summarise, with_gate

    df, _ = load_release(APP_DIR / "release")
    s = summarise(with_gate(build_records(df, "case_study", "Staged GRPO"), 0.5))
    assert s.n == 541
    assert round(s.vqa_acc, 3) == 0.503
    assert round(s.obj_recall, 3) == 0.389
    assert round(s.silent_failure_rate, 3) == 0.274
    assert round(s.trust_gap, 3) == 0.058
    assert round(s.gap_lo, 3) == -0.029
    assert round(s.gap_hi, 3) == 0.147
    assert round(s.point_biserial, 3) == 0.061


# --- the report itself ------------------------------------------------------

def test_report_is_reproducible(report):
    """Two runs must agree: every random step is seeded."""
    again = paper_numbers.build_report(APP_DIR / "release", APP_DIR / "figures")
    for key in ("1_mean_obj_recall", "2_pim_given_correct"):
        assert again[key] == report[key]
    assert again["7_sf_vs_ngt"]["p"] == report["7_sf_vs_ngt"]["p"]


def test_tau_sweep_covers_the_documented_range(report):
    sweep = report["8_tau_sweep"]
    assert [r["tau"] for r in sweep] == [round(0.05 * i, 2) for i in range(1, 20)]
    # At the published gate the sweep must agree with the reported ledger.
    at_half = next(r for r in sweep if r["tau"] == 0.5)
    assert round(at_half["silent_failure_rate"], 3) == 0.274
    assert round(at_half["aligned_pass_rate"], 3) == 0.229
    assert round(at_half["trust_gap"], 3) == 0.058


def test_recall_histogram_sums_to_the_record_set(report):
    hist = report["9_recall_hist"]
    assert len(hist["counts"]) == 20
    assert sum(hist["counts"]) == 541
