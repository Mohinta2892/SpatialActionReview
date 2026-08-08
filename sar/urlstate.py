"""URL query parameters for the dashboard's view state.

Any state reachable by clicking is also reachable by URL, and any state reached by
clicking writes itself back into the URL. That makes a figure in the paper
reproducible from its address rather than from a description of which cells to
select, which is the same provenance argument the paper makes about its numbers.

The vocabulary is deliberately URL-shaped (`staged_grpo`, `silent_failure`) and is
mapped here to the labels the record set uses (`Staged GRPO`, `silent_failure`),
so neither side has to know about the other. Parsing is total: an unknown or
malformed value falls back to the default and is reported, so a mistyped link
still opens a working dashboard instead of an error.

    ?source=case_study&condition=staged_grpo&tau=0.5&state=silent_failure
    &risk_cell=presence:EM-H&record=crop_5ca713fb225e&theme=day
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .contracts import DEFAULT_CONTRACT

# --- source ---------------------------------------------------------------
SOURCE_TO_BUILD = {
    "case_study": "case_study",
    "sft_variant": "sft_variants",
    "upload": "uploaded",
}
BUILD_TO_SOURCE = {v: k for k, v in SOURCE_TO_BUILD.items()}

# --- condition ------------------------------------------------------------
CONDITION_TO_LABEL = {
    "zero_shot": "Zero-shot",
    "perception_sft": "Perception SFT",
    "grounding_sft": "Grounding SFT",
    "staged_sft": "Staged SFT",
    "joint_sft": "Joint SFT",
    "staged_grpo": "Staged GRPO",
}
LABEL_TO_CONDITION = {v: k for k, v in CONDITION_TO_LABEL.items()}

# --- behavioural state ----------------------------------------------------
STATE_TO_QUADRANT = {
    "aligned_pass": "trustworthy",
    "silent_failure": "silent_failure",
    "action_only": "lucky",
    "blocked": "honest",
}
QUADRANT_TO_STATE = {v: k for k, v in STATE_TO_QUADRANT.items()}

# --- theme ----------------------------------------------------------------
THEMES = ("day", "night")

TAU_MIN, TAU_MAX, TAU_STEP = 0.05, 0.95, 0.05

PARAMS = (
    "source", "condition", "tau", "state", "risk_cell", "record", "theme",
)


@dataclass
class ViewState:
    """Dashboard state, in the vocabulary the rest of the app uses."""
    build: str
    condition: str | None
    tau: float
    contract: str
    precision_tau: float
    count_tolerance: int
    quadrant: str
    region: tuple[str, str] | None
    record: str | None
    palette: str
    rejected: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.rejected


def _snap_tau(value: float) -> float:
    """Clamp into range and snap to the slider's step, so URL and widget agree."""
    clamped = min(max(value, TAU_MIN), TAU_MAX)
    return round(round(clamped / TAU_STEP) * TAU_STEP, 2)


def parse(
    params,
    *,
    default_build: str,
    default_condition: str | None,
    default_tau: float,
    default_contract: str = DEFAULT_CONTRACT,
    default_precision_tau: float = 0.5,
    default_count_tolerance: int = 0,
    default_quadrant: str = "silent_failure",
    default_palette: str = "night",
) -> ViewState:
    """Read a query-parameter mapping into a ViewState.

    `params` is anything dict-like, including `st.query_params`. Every rejected
    value is recorded with a reason rather than raising, because a bad link should
    still open the dashboard.
    """
    rejected: dict[str, str] = {}

    def get(name: str) -> str | None:
        try:
            raw = params.get(name)
        except Exception:
            return None
        if raw is None:
            return None
        if isinstance(raw, (list, tuple)):          # some params APIs return lists
            raw = raw[0] if raw else None
        if raw is None:
            return None
        text = str(raw).strip()
        return text or None

    # source
    build = default_build
    raw = get("source")
    if raw is not None:
        if raw in SOURCE_TO_BUILD:
            build = SOURCE_TO_BUILD[raw]
        else:
            rejected["source"] = f"{raw!r} is not one of {sorted(SOURCE_TO_BUILD)}"

    # condition
    condition = default_condition
    raw = get("condition")
    if raw is not None:
        if raw in CONDITION_TO_LABEL:
            condition = CONDITION_TO_LABEL[raw]
        else:
            rejected["condition"] = f"{raw!r} is not one of {sorted(CONDITION_TO_LABEL)}"

    # tau
    tau = default_tau
    raw = get("tau")
    if raw is not None:
        try:
            value = float(raw)
        except ValueError:
            rejected["tau"] = f"{raw!r} is not a number"
        else:
            if not (TAU_MIN - 1e-9 <= value <= TAU_MAX + 1e-9):
                rejected["tau"] = f"{value} is outside [{TAU_MIN}, {TAU_MAX}]"
            else:
                tau = _snap_tau(value)

    # Accepted camera-ready gate: object recall only. Legacy links may still
    # contain old gate parameters, but they cannot change the action verdict.
    contract = default_contract
    raw = get("contract")
    if raw is not None:
        if raw != DEFAULT_CONTRACT:
            rejected["contract"] = f"{raw!r} is not part of the accepted dashboard gate"

    # precision_tau
    precision_tau = default_precision_tau
    raw = get("precision_tau")
    if raw is not None:
        try:
            value = float(raw)
        except ValueError:
            rejected["precision_tau"] = f"{raw!r} is not a number"
        else:
            if not (TAU_MIN - 1e-9 <= value <= TAU_MAX + 1e-9):
                rejected["precision_tau"] = f"{value} is outside [{TAU_MIN}, {TAU_MAX}]"
            else:
                if abs(_snap_tau(value) - default_precision_tau) > 1e-9:
                    rejected["precision_tau"] = (
                        "ignored; the accepted dashboard gate uses object recall only"
                    )

    # count_tolerance
    count_tolerance = default_count_tolerance
    raw = get("count_tolerance")
    if raw is not None:
        try:
            value = int(raw)
        except ValueError:
            rejected["count_tolerance"] = f"{raw!r} is not an integer"
        else:
            if not (0 <= value <= 4):
                rejected["count_tolerance"] = f"{value} is outside [0, 4]"
            else:
                if value != default_count_tolerance:
                    rejected["count_tolerance"] = (
                        "ignored; the accepted dashboard gate uses object recall only"
                    )

    # state
    quadrant = default_quadrant
    raw = get("state")
    if raw is not None:
        if raw in STATE_TO_QUADRANT:
            quadrant = STATE_TO_QUADRANT[raw]
        else:
            rejected["state"] = f"{raw!r} is not one of {sorted(STATE_TO_QUADRANT)}"

    # risk_cell = <task>:<dataset>
    region = None
    raw = get("risk_cell")
    if raw is not None:
        if raw.count(":") == 1 and all(part.strip() for part in raw.split(":")):
            task, dataset = (part.strip() for part in raw.split(":"))
            region = (task, dataset)
        else:
            rejected["risk_cell"] = f"{raw!r} is not '<task>:<dataset>'"

    # record = crop_id
    record = get("record")

    # theme
    palette = default_palette
    raw = get("theme")
    if raw is not None:
        if raw in THEMES:
            palette = raw
        else:
            rejected["theme"] = f"{raw!r} is not one of {sorted(THEMES)}"

    return ViewState(
        build=build, condition=condition, tau=tau, contract=contract,
        precision_tau=precision_tau, count_tolerance=count_tolerance, quadrant=quadrant,
        region=region, record=record, palette=palette, rejected=rejected,
    )


def to_params(
    *,
    build: str,
    condition: str | None,
    tau: float,
    quadrant: str,
    region: tuple[str, str] | None,
    record: str | None,
    palette: str,
) -> dict[str, str]:
    """Render the current state as query parameters.

    Optional parameters are omitted when unset rather than written empty, so a
    shared URL contains only what is actually selected.
    """
    out: dict[str, str] = {
        "source": BUILD_TO_SOURCE.get(build, build),
        "tau": f"{tau:.2f}",
        "state": QUADRANT_TO_STATE.get(quadrant, quadrant),
        "theme": palette,
    }
    if condition and condition in LABEL_TO_CONDITION:
        out["condition"] = LABEL_TO_CONDITION[condition]
    if region:
        out["risk_cell"] = f"{region[0]}:{region[1]}"
    if record:
        out["record"] = record
    return out


def describe_rejections(state: ViewState) -> str:
    return "; ".join(f"{k}: {v}" for k, v in sorted(state.rejected.items()))
