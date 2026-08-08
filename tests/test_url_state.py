"""URL parameters must resolve to exactly the view they name.

These tests exercise the resolution layer — parsing a query string, mapping it to
the record set's own vocabulary, and deriving the ledger counts and audit queue
that the page then renders. They do not drive a browser; what they pin is that a
URL determines a view deterministically, which is the property the figure-capture
script and the paper's URL citations depend on.
"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

from sar import urlstate  # noqa: E402
from sar.data import QUADRANTS, build_records, load_release, with_gate  # noqa: E402

PAPER_TAU = 0.5

# Table 1 of the manuscript, as the ledger must report it for the case study.
CASE_STUDY_LEDGER = {
    "trustworthy": 124, "silent_failure": 148, "lucky": 107, "honest": 162,
}


@pytest.fixture(scope="module")
def release():
    return load_release(APP_DIR / "release")


def _params(url: str) -> dict[str, str]:
    """Query string -> flat mapping, as st.query_params presents it."""
    return {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}


def _resolve(release, url: str):
    """Everything the page derives from a URL: state, scored records, queue."""
    df, manifest = release
    default_build = manifest["default_build"]
    view = urlstate.parse(
        _params(url),
        default_build=default_build,
        default_condition=manifest["builds"][default_build]["conditions"][0],
        default_tau=PAPER_TAU,
    )
    offered = manifest["builds"].get(view.build, {}).get("conditions", [])
    condition = view.condition if view.condition in offered else (offered[0] if offered else None)
    scored = with_gate(
        build_records(df, view.build, condition),
        view.tau,
    )

    queue = scored[scored["quadrant"] == view.quadrant]
    if view.region is not None:
        task, dataset = view.region
        queue = queue[(queue["task"].astype(str) == task)
                      & (queue["dataset"].astype(str) == dataset)]
    queue = queue[queue["has_image"]]
    queue = queue.sort_values(["dataset", "task", "question_id"]).reset_index(drop=True)
    return view, condition, scored, queue


# ------------------------------------------------------------------ parsing

def test_every_documented_parameter_is_accepted():
    view = urlstate.parse(
        _params("/?source=sft_variant&condition=joint_sft&tau=0.35&state=action_only"
                "&risk_cell=location:EM-H&record=crop_abc&theme=day"),
        default_build="case_study", default_condition="Staged GRPO", default_tau=0.5,
    )
    assert view.ok, view.rejected
    assert view.build == "sft_variants"
    assert view.condition == "Joint SFT"
    assert view.tau == 0.35
    assert (view.contract, view.precision_tau, view.count_tolerance) == ("coverage", 0.5, 0)
    assert view.quadrant == "lucky"
    assert view.region == ("location", "EM-H")
    assert view.record == "crop_abc"
    assert view.palette == "day"


def test_absent_parameters_fall_back_to_the_defaults():
    view = urlstate.parse(
        {}, default_build="case_study", default_condition="Staged GRPO", default_tau=0.5)
    assert view.ok
    assert (view.build, view.condition, view.tau) == ("case_study", "Staged GRPO", 0.5)
    assert (view.contract, view.precision_tau, view.count_tolerance) == ("coverage", 0.5, 0)
    assert view.quadrant == "silent_failure"
    assert view.palette == "night"
    assert view.region is None and view.record is None


@pytest.mark.parametrize("query, bad", [
    ("?source=nope", "source"),
    ("?condition=not_a_condition", "condition"),
    ("?tau=abc", "tau"),
    ("?tau=1.4", "tau"),
    ("?tau=-0.1", "tau"),
    ("?contract=precision", "contract"),
    ("?precision_tau=abc", "precision_tau"),
    ("?precision_tau=1.4", "precision_tau"),
    ("?count_tolerance=abc", "count_tolerance"),
    ("?count_tolerance=8", "count_tolerance"),
    ("?state=maybe", "state"),
    ("?risk_cell=presence", "risk_cell"),
    ("?risk_cell=a:b:c", "risk_cell"),
    ("?theme=purple", "theme"),
])
def test_bad_values_are_reported_and_defaulted_not_raised(query, bad):
    view = urlstate.parse(
        _params("/" + query),
        default_build="case_study", default_condition="Staged GRPO", default_tau=0.5)
    assert bad in view.rejected, view.rejected
    # The view is still usable: every field holds a valid value.
    assert view.build in urlstate.SOURCE_TO_BUILD.values()
    assert view.quadrant in QUADRANTS
    assert urlstate.TAU_MIN <= view.tau <= urlstate.TAU_MAX
    assert view.palette in urlstate.THEMES


def test_tau_is_snapped_to_the_slider_step():
    for raw, expected in [("0.5", 0.5), ("0.37", 0.35), ("0.38", 0.40), ("0.05", 0.05),
                          ("0.95", 0.95)]:
        view = urlstate.parse({"tau": raw}, default_build="case_study",
                              default_condition=None, default_tau=0.5)
        assert view.tau == expected, raw


def test_round_trip_through_to_params_is_stable():
    url = ("/?source=sft_variant&condition=staged_sft&tau=0.65&state=blocked"
           "&risk_cell=counting:VNC&record=crop_xyz&theme=day")
    view = urlstate.parse(_params(url), default_build="case_study",
                          default_condition=None, default_tau=0.5)
    written = urlstate.to_params(
        build=view.build, condition=view.condition, tau=view.tau,
        quadrant=view.quadrant, region=view.region, record=view.record,
        palette=view.palette,
    )
    again = urlstate.parse(written, default_build="case_study",
                           default_condition=None, default_tau=0.5)
    assert (again.build, again.condition, again.tau, again.quadrant,
            again.region, again.record, again.palette) == (
        view.build, view.condition, view.tau, view.quadrant,
        view.region, view.record, view.palette)


def test_optional_parameters_are_omitted_when_unset():
    written = urlstate.to_params(
        build="case_study", condition="Staged GRPO", tau=0.5,
        quadrant="silent_failure", region=None, record=None, palette="night")
    assert "risk_cell" not in written and "record" not in written
    assert written == {"source": "case_study", "tau": "0.50",
                       "state": "silent_failure", "theme": "night",
                       "condition": "staged_grpo"}


def test_legacy_gate_parameters_do_not_change_the_ledger(release):
    base = _resolve(
        release, "/?source=case_study&condition=staged_grpo&tau=0.5")[2]
    legacy = _resolve(
        release,
        "/?source=case_study&condition=staged_grpo&tau=0.5"
        "&contract=precision&precision_tau=0.75&count_tolerance=1",
    )[2]
    assert int(base["action_reliable"].sum()) == int(legacy["action_reliable"].sum())


# --------------------------------------------------- URL -> view resolution

def test_case_study_url_gives_the_published_ledger_counts(release):
    url = ("/?source=case_study&condition=staged_grpo&tau=0.5"
           "&state=silent_failure&theme=day")
    view, condition, scored, queue = _resolve(release, url)
    assert view.ok and condition == "Staged GRPO"
    assert len(scored) == 541
    counts = scored["quadrant"].value_counts().to_dict()
    assert {k: int(counts[k]) for k in CASE_STUDY_LEDGER} == CASE_STUDY_LEDGER
    # The queue holds the silent failures, and all 541 crops are shipped.
    assert len(queue) == CASE_STUDY_LEDGER["silent_failure"]


def test_state_parameter_selects_the_matching_ledger_cell(release):
    for state, quadrant in urlstate.STATE_TO_QUADRANT.items():
        view, _, _, queue = _resolve(
            release, f"/?source=case_study&condition=staged_grpo&tau=0.5&state={state}")
        assert view.quadrant == quadrant
        assert len(queue) == CASE_STUDY_LEDGER[quadrant], state
        assert (queue["quadrant"] == quadrant).all()


def test_risk_cell_parameter_narrows_the_queue(release):
    url = ("/?source=case_study&condition=staged_grpo&tau=0.5"
           "&state=aligned_pass&risk_cell=presence:Lucchi")
    view, _, _, queue = _resolve(release, url)
    assert view.region == ("presence", "Lucchi")
    assert len(queue) == 7
    assert set(queue["task"].astype(str)) == {"presence"}
    assert set(queue["dataset"].astype(str)) == {"Lucchi"}


def test_record_parameter_selects_that_record_in_the_queue(release):
    """A URL naming a record must open that record, not merely contain it."""
    _, _, _, queue = _resolve(
        release, "/?source=case_study&condition=staged_grpo&tau=0.5&state=silent_failure")
    wanted = str(queue.iloc[5]["crop_id"])

    view, _, _, queue2 = _resolve(
        release,
        "/?source=case_study&condition=staged_grpo&tau=0.5&state=silent_failure"
        f"&record={wanted}")
    assert view.record == wanted
    # The app moves the cursor to this index; the queue ordering must be stable.
    match = queue2.index[queue2["crop_id"] == wanted]
    assert len(match) == 1
    assert int(match[0]) == 5
    assert str(queue2.iloc[int(match[0])]["crop_id"]) == wanted


def test_tau_parameter_changes_the_ledger(release):
    """A different gate must give a different, and correctly ordered, ledger."""
    strict = _resolve(release, "/?source=case_study&condition=staged_grpo&tau=0.9")[2]
    loose = _resolve(release, "/?source=case_study&condition=staged_grpo&tau=0.1")[2]
    assert int(strict["action_reliable"].sum()) < int(loose["action_reliable"].sum())
    # Raising the gate can only move answer-correct records from pass to failure.
    assert int((strict["quadrant"] == "silent_failure").sum()) > \
        int((loose["quadrant"] == "silent_failure").sum())


def test_sft_variant_source_resolves_each_condition(release):
    _, manifest = release
    for slug, label in urlstate.CONDITION_TO_LABEL.items():
        if label not in manifest["builds"]["sft_variants"]["conditions"]:
            continue
        view, condition, scored, _ = _resolve(
            release, f"/?source=sft_variant&condition={slug}&tau=0.5")
        assert condition == label
        assert len(scored) == 753, label


def test_condition_not_offered_by_the_source_falls_back(release):
    """A case-study URL asking for an SFT condition must not produce an empty view."""
    view, condition, scored, _ = _resolve(
        release, "/?source=case_study&condition=joint_sft&tau=0.5")
    assert view.condition == "Joint SFT"          # parsed as asked
    assert condition == "Staged GRPO"             # but resolved to what the source has
    assert len(scored) == 541
