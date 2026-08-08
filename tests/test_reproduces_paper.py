"""The dashboard must reproduce the manuscript. These tests are the contract.

Every value asserted here is transcribed from
`rebuttal/uploaded_manuscript_v1/main.tex`. If a change to the release or to the
scoring code moves any of them, these tests fail rather than letting the live
dashboard drift away from the submitted numbers.

The manuscript reports two audit builds:

  case_study    Qwen3-VL Staged GRPO, 541 records. Section 4.1 global signals,
                Table 1 (ledger), Table 2 (risk map), and the teaser figure.
  sft_variants  Qwen3-VL, 753 records × 5 conditions. Table 3 (re-audit).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

from sar.data import (  # noqa: E402
    build_records,
    condition_table,
    load_release,
    risk_map,
    summarise,
    with_gate,
)

PAPER_TAU = 0.5

# --- Section 4.1, "Global Signals Reveal Weak Answer-Action Coupling" ---------
# "the dashboard reports 50.3% VQA accuracy and 39.0% mean object recall over
# 541 paired image regions"; "P(A=1,R_tau=0)=27.4%"; "P(R_tau=0|A=1)=54.4%";
# "The trust gap is Delta=0.058 ... with bootstrap interval [-2.9, 14.7]
# percentage points"; "the point-biserial association is 0.061".
CASE_STUDY_SIGNALS = dict(
    n=541, vqa_acc=0.503, obj_recall=0.389, silent_failure_rate=0.274,
    p_unreliable_given_correct=0.544, gap=0.058, lo=-0.029, hi=0.147, pb=0.061,
)

# --- Table 1, run-level answer-action ledger ---------------------------------
CASE_STUDY_LEDGER = {
    "trustworthy": (124, 0.229),
    "silent_failure": (148, 0.274),
    "lucky": (107, 0.198),
    "honest": (162, 0.299),
}

# --- Table 2, silent-failure rate by task and dataset (541 records) ----------
CASE_STUDY_RISK_MAP = {
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

# --- Table 3, Qwen3-VL SFT-variant audit (753 records per condition) ---------
SFT_TABLE = {
    "Zero-shot":      dict(vqa=0.437, recall=0.214, silent=0.336, gap=0.026, lo=-0.032, hi=0.086, pb=0.023),
    "Perception SFT": dict(vqa=0.506, recall=0.256, silent=0.381, gap=-0.025, lo=-0.087, hi=0.041, pb=-0.009),
    "Grounding SFT":  dict(vqa=0.432, recall=0.231, silent=0.316, gap=0.036, lo=-0.026, hi=0.103, pb=0.033),
    "Staged SFT":     dict(vqa=0.506, recall=0.371, silent=0.297, gap=0.036, lo=-0.034, hi=0.108, pb=0.043),
    "Joint SFT":      dict(vqa=0.529, recall=0.325, silent=0.335, gap=0.026, lo=-0.043, hi=0.094, pb=0.028),
}

# --- Sections 4.1 and 4.5, probe-width descriptions --------------------------
CASE_STUDY_OPTIONS = dict(baseline=0.263, n_single=26)
SFT_OPTIONS = dict(
    baseline=0.264, n_single=32,
    k_distribution={"1": 32, "2": 105, "3": 30, "4": 156, "5": 144, "6": 80, "9": 206},
)


@pytest.fixture(scope="module")
def release():
    return load_release(APP_DIR / "release")


def _case_study(df, manifest, tau=PAPER_TAU):
    condition = manifest["builds"]["case_study"]["conditions"][0]
    return with_gate(build_records(df, "case_study", condition), tau)


# ------------------------------------------------------------- release shape

def test_release_declares_both_builds(release):
    _, manifest = release
    assert manifest["schema_version"] == 2
    assert manifest["build_order"] == ["case_study", "sft_variants"]
    assert manifest["default_build"] == "case_study"
    assert manifest["builds"]["case_study"]["conditions"] == ["Staged GRPO"]
    sft = manifest["builds"]["sft_variants"]
    assert sft["conditions"] == list(SFT_TABLE)
    assert sft["reported_in_table"] == list(SFT_TABLE)
    assert sft["not_in_reported_table"] == []


def test_release_record_counts(release):
    df, manifest = release
    assert (df["build"] == "case_study").sum() == 541
    for condition in manifest["builds"]["sft_variants"]["conditions"]:
        assert len(build_records(df, "sft_variants", condition)) == 753
    n_sft = len(manifest["builds"]["sft_variants"]["conditions"])
    assert len(df) == manifest["n_records_total"] == 541 + n_sft * 753


def test_manifest_hash_matches_records(release):
    _, manifest = release
    digest = hashlib.sha256((APP_DIR / "release" / "records.csv.gz").read_bytes()).hexdigest()
    assert digest == manifest["records_sha256"]


def test_bootstrap_protocol_is_the_one_the_paper_describes(release):
    """"B=2000 samples with replacement using seed 0" (Section 3.5)."""
    _, manifest = release
    assert manifest["bootstrap"] == {
        "n_resamples": 2000, "seed": 0, "generator": "numpy.random.default_rng",
    }
    assert manifest["tau_default"] == PAPER_TAU


# ------------------------------------------------- case study (541 records)

def test_case_study_global_signals(release):
    df, manifest = release
    s = summarise(_case_study(df, manifest))
    e = CASE_STUDY_SIGNALS
    assert s.n == e["n"]
    assert round(s.vqa_acc, 3) == e["vqa_acc"]
    assert round(s.obj_recall, 3) == e["obj_recall"]
    assert round(s.silent_failure_rate, 3) == e["silent_failure_rate"]
    assert round(s.p_unreliable_given_correct, 3) == e["p_unreliable_given_correct"]
    assert round(s.trust_gap, 3) == e["gap"]
    assert round(s.gap_lo, 3) == e["lo"], "bootstrap CI drifted (seed 0, 2000 resamples)"
    assert round(s.gap_hi, 3) == e["hi"], "bootstrap CI drifted (seed 0, 2000 resamples)"
    assert round(s.point_biserial, 3) == e["pb"]


def test_case_study_trust_gap_interval_spans_zero(release):
    """"The interval spans zero" — the paper's central negative result."""
    df, manifest = release
    s = summarise(_case_study(df, manifest))
    assert s.gap_lo < 0 < s.gap_hi


def test_case_study_ledger_matches_table_1(release):
    df, manifest = release
    scored = _case_study(df, manifest)
    counts = scored["quadrant"].value_counts()
    for quadrant, (count, rate) in CASE_STUDY_LEDGER.items():
        assert int(counts[quadrant]) == count, quadrant
        assert round(count / len(scored), 3) == rate, quadrant
    assert sum(c for c, _ in CASE_STUDY_LEDGER.values()) == 541


def test_case_study_risk_map_matches_table_2(release):
    df, manifest = release
    rm = risk_map(_case_study(df, manifest))
    seen = set()
    for row in rm.itertuples():
        key = (str(row.task), str(row.dataset))
        expected_rate, expected_n = CASE_STUDY_RISK_MAP[key]
        assert row.n == expected_n, key
        assert round(row.silent_failure_rate, 3) == pytest.approx(expected_rate, abs=0.0006), key
        seen.add(key)
    assert seen == set(CASE_STUDY_RISK_MAP), "risk map is missing or inventing cells"


def test_case_study_presence_is_the_highest_risk_task(release):
    """"Presence questions have the largest rates across all datasets." (Sec 4.3)"""
    df, manifest = release
    rm = risk_map(_case_study(df, manifest))
    by_task = rm.groupby("task", observed=True).apply(
        lambda g: (g["silent_failure_rate"] * g["n"]).sum() / g["n"].sum(), include_groups=False
    )
    assert by_task.idxmax() == "presence"
    presence = rm[rm["task"].astype(str) == "presence"]
    assert (presence["silent_failure_rate"] >= 0.47).all()


def test_case_study_location_row_range(release):
    """"Location and marked-region questions ... ranging from 24.1% to 32.9%.\""""
    df, manifest = release
    rm = risk_map(_case_study(df, manifest))
    location = rm[rm["task"].astype(str) == "location"]
    assert round(location["silent_failure_rate"].min(), 3) == 0.241
    assert round(location["silent_failure_rate"].max(), 3) == 0.329


def test_case_study_all_crops_are_released(release):
    """The traced case study must never fall back to a placeholder tile."""
    df, manifest = release
    scored = _case_study(df, manifest)
    assert scored["has_image"].all()
    assert manifest["builds"]["case_study"]["images_released"] == 541
    image_dir = APP_DIR / "release" / "images"
    missing = [n for n in sorted(set(scored["image_file"])) if not (image_dir / n).exists()]
    assert not missing


def test_case_study_diagnostics_come_from_the_matching_run(release):
    """Point F1 and point-in-mask precision must belong to this condition.

    They are absent from the 541-record export itself and are joined from
    `outputs/percrop_staged_grpo.csv`. The builder refuses the join unless that
    file agrees with the export on answer_correct, obj_recall, n_gt and n_pred for
    every record, so the check here is that the joined values reproduce the one
    figure the manuscript reports for them: point-in-mask precision among
    answer-correct records, 0.471 (Section 4.1 / Table 1 run).

    If the columns are absent the release simply predates the join, and the
    dashboard reports them as not yet present rather than as zero.
    """
    df, manifest = release
    scored = _case_study(df, manifest)
    absent = manifest["builds"]["case_study"]["absent_record_fields"]

    if "pim_prec" in absent:
        assert scored["pim_prec"].isna().all()
        assert scored["point_f1"].isna().all()
        return

    assert scored["pim_prec"].notna().all()
    assert scored["point_f1"].notna().all()
    assert round(summarise(scored).pim_prec_given_correct, 3) == 0.471
    assert ((scored["pim_prec"] >= 0) & (scored["pim_prec"] <= 1)).all()
    assert ((scored["point_f1"] >= 0) & (scored["point_f1"] <= 1)).all()
    # A point-in-mask hit implies at least one point was emitted.
    assert (scored.loc[scored["pim_prec"] > 0, "n_pred"] > 0).all()


def test_case_study_option_stats_match_section_4_1(release):
    """"the number of answer options ranges from K=1 to K=9, with a record-wise
    uniform-choice baseline of 26.3% ... There are 26 such records.\""""
    _, manifest = release
    opt = manifest["builds"]["case_study"]["option_stats"]
    assert opt["uniform_choice_baseline"] == CASE_STUDY_OPTIONS["baseline"]
    assert opt["n_single_option"] == CASE_STUDY_OPTIONS["n_single"]
    assert min(int(k) for k in opt["k_distribution"]) == 1
    assert max(int(k) for k in opt["k_distribution"]) == 9
    assert sum(opt["k_distribution"].values()) == 541
    # "26 such records, or 4.8% of the dashboard run"
    assert round(100 * opt["n_single_option"] / 541, 1) == 4.8


# ---------------------------------------------- SFT variants (753 records)

@pytest.mark.parametrize("condition", list(SFT_TABLE))
def test_sft_variant_row_matches_table_3(release, condition):
    df, _ = release
    e = SFT_TABLE[condition]
    s = summarise(with_gate(build_records(df, "sft_variants", condition), PAPER_TAU))
    assert s.n == 753
    assert round(s.vqa_acc, 3) == e["vqa"]
    assert round(s.obj_recall, 3) == e["recall"]
    assert round(s.silent_failure_rate, 3) == e["silent"]
    assert round(s.trust_gap, 3) == e["gap"]
    assert round(s.gap_lo, 3) == e["lo"], "bootstrap CI drifted (seed 0, 2000 resamples)"
    assert round(s.gap_hi, 3) == e["hi"], "bootstrap CI drifted (seed 0, 2000 resamples)"
    assert round(s.point_biserial, 3) == e["pb"]


def test_sft_ranges_quoted_in_section_4_5(release):
    """"VQA accuracy ranges from 43.2% to 52.9%, and mean object recall ranges
    from 21.4% to 37.1% ... trust gap ... from -2.5 to 3.6 percentage points ...
    Point-biserial ... from -0.009 to 0.043. Silent-failure rates range from
    29.7% to 38.1%.\""""
    df, manifest = release
    stats = [
        summarise(with_gate(build_records(df, "sft_variants", c), PAPER_TAU))
        for c in manifest["builds"]["sft_variants"]["reported_in_table"]
    ]
    assert (round(min(s.vqa_acc for s in stats), 3),
            round(max(s.vqa_acc for s in stats), 3)) == (0.432, 0.529)
    assert (round(min(s.obj_recall for s in stats), 3),
            round(max(s.obj_recall for s in stats), 3)) == (0.214, 0.371)
    assert (round(min(s.trust_gap for s in stats), 3),
            round(max(s.trust_gap for s in stats), 3)) == (-0.025, 0.036)
    assert (round(min(s.point_biserial for s in stats), 3),
            round(max(s.point_biserial for s in stats), 3)) == (-0.009, 0.043)
    assert (round(min(s.silent_failure_rate for s in stats), 3),
            round(max(s.silent_failure_rate for s in stats), 3)) == (0.297, 0.381)


def test_every_sft_trust_gap_interval_includes_zero(release):
    """"every trust-gap interval in Table 3 includes zero.\""""
    df, manifest = release
    for condition in manifest["builds"]["sft_variants"]["reported_in_table"]:
        s = summarise(with_gate(build_records(df, "sft_variants", condition), PAPER_TAU))
        assert s.gap_lo < 0 < s.gap_hi, condition


def test_sft_conditions_share_one_question_set(release):
    """"All five conditions use the same question set." (Section 4.5)"""
    df, manifest = release
    sets = [
        frozenset(build_records(df, "sft_variants", c)["question_id"])
        for c in manifest["builds"]["sft_variants"]["conditions"]
    ]
    assert len(set(sets)) == 1
    assert len(sets[0]) == 753


def test_sft_option_stats_match_section_4_5(release):
    _, manifest = release
    opt = manifest["builds"]["sft_variants"]["option_stats"]
    assert opt["k_distribution"] == SFT_OPTIONS["k_distribution"]
    assert opt["uniform_choice_baseline"] == SFT_OPTIONS["baseline"]
    assert opt["n_single_option"] == SFT_OPTIONS["n_single"]


def test_condition_table_rows_match_individual_summaries(release):
    df, manifest = release
    conditions = manifest["builds"]["sft_variants"]["conditions"]
    table = condition_table(df, "sft_variants", conditions, PAPER_TAU).set_index("Condition")
    for condition in conditions:
        s = summarise(with_gate(build_records(df, "sft_variants", condition), PAPER_TAU))
        assert round(table.loc[condition, "Silent failure"], 6) == round(s.silent_failure_rate, 6)
        assert round(table.loc[condition, "Trust gap"], 6) == round(s.trust_gap, 6)


# ------------------------------------------------------- structural invariants

def test_quadrants_partition_every_build(release):
    df, manifest = release
    for build in manifest["build_order"]:
        for condition in manifest["builds"][build]["conditions"]:
            scored = with_gate(build_records(df, build, condition), PAPER_TAU)
            s = summarise(scored)
            total = s.trustworthy_rate + s.silent_failure_rate + s.lucky_rate + s.honest_rate
            assert round(total, 9) == 1.0, (build, condition)


def test_risk_map_regions_sum_to_the_record_set(release):
    df, manifest = release
    for build in manifest["build_order"]:
        for condition in manifest["builds"][build]["conditions"]:
            scored = with_gate(build_records(df, build, condition), PAPER_TAU)
            assert risk_map(scored)["n"].sum() == len(scored), (build, condition)


def test_gate_monotonic_in_tau(release):
    """Raising tau can only move records out of the 'reliable' column."""
    df, manifest = release
    sub = build_records(df, "case_study", manifest["builds"]["case_study"]["conditions"][0])
    previous = None
    for tau in [0.1, 0.3, 0.5, 0.7, 0.9]:
        reliable = int(with_gate(sub, tau)["action_reliable"].sum())
        if previous is not None:
            assert reliable <= previous
        previous = reliable


def test_overlay_counts_equal_the_scored_counts(release):
    """A tile must never draw more or fewer points than the record was scored on."""
    df, _ = release
    assert (df["gt_centroids"].map(len) == df["n_gt"]).all()
    assert (df["pred_points"].map(len) == df["n_pred"]).all()


def test_coordinates_are_normalised(release):
    df, _ = release
    for column in ("gt_centroids", "pred_points"):
        flat = [v for seq in df[column] for point in seq for v in point]
        assert flat, f"{column} is empty"
        assert min(flat) >= 0.0 and max(flat) <= 1.0, f"{column} leaves the unit square"


def test_demo_cases_land_in_their_advertised_state(release):
    df, manifest = release
    for build in manifest["build_order"]:
        meta = manifest["builds"][build]
        demo = meta["demo_cases"]
        assert set(demo) == {"trustworthy", "silent_failure", "lucky", "honest"}, build

        # Demo cases are pinned against the build's first (or focal) condition.
        condition = "Joint SFT" if build == "sft_variants" else meta["conditions"][0]
        by_qid = with_gate(build_records(df, build, condition), PAPER_TAU).set_index("question_id")

        for quadrant, group in demo.items():
            assert group["note"].strip(), (build, quadrant)
            assert 3 <= len(group["cases"]) <= 5, (build, quadrant, len(group["cases"]))
            assert len({c["question_id"] for c in group["cases"]}) == len(group["cases"])
            for case in group["cases"]:
                record = by_qid.loc[case["question_id"]]
                assert record["quadrant"] == quadrant, (build, case["question_id"])
                assert record["image_file"] != "", "a worked example must show real EM pixels"
                # An example with no ground-truth object would render an empty crop.
                assert record["n_gt"] > 0, (build, case["question_id"])
                # The button label must describe the record it actually opens.
                assert str(record["task"]) == case["task"]
                assert str(record["dataset"]) == case["dataset"]
                assert round(record["obj_recall"], 3) == case["obj_recall"]


def test_demo_cases_span_task_families(release):
    """Clicking through a state should not show five copies of one probe family."""
    _, manifest = release
    for build in manifest["build_order"]:
        for quadrant, group in manifest["builds"][build]["demo_cases"].items():
            tasks = {c["task"] for c in group["cases"]}
            assert len(tasks) >= 3, (build, quadrant, tasks)


def test_dataset_and_task_vocabularies_match_the_paper(release):
    df, manifest = release
    assert manifest["dataset_order"] == ["Lucchi", "VNC", "EM-H"]
    assert manifest["task_order"] == ["presence", "counting", "location", "relation", "attribute"]
    assert set(df["dataset"].astype(str)) == {"Lucchi", "VNC", "EM-H"}
    assert set(df["task"].astype(str)) <= set(manifest["task_order"])
