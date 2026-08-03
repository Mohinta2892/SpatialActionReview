"""Human sign-off: the routing decision a supervisor records for a record.

The dashboard's purpose is not to display a metric but to end in a decision. Four
dispositions are available, and they are the four the audit path defines: accept
the model output, send the image region to human audit, hold it behind a stricter
gate, or flag the model for revision.

A disposition is only meaningful alongside the state it was taken under, so each
one stores the record, the source and condition being audited, the threshold in
force, and the behavioural state at the moment of signing. That makes the log a
provenance trail rather than a list of opinions: a reader can tell that a record
was accepted at one gate value without assuming it would pass at another.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import pandas as pd

STATE_KEY = "review_log"


@dataclass(frozen=True)
class Route:
    key: str
    label: str
    short: str
    blurb: str


ROUTES: tuple[Route, ...] = (
    Route(
        "accept", "Accept action", "accepted",
        "The point action may proceed to the downstream workflow as it stands.",
    ),
    Route(
        "human_audit", "Send to human audit", "sent to audit",
        "The image region needs direct inspection before the action is used.",
    ),
    Route(
        "stricter_gate", "Hold for a stricter gate", "held",
        "Do not accept at this gate value; re-audit with a stricter one.",
    ),
    Route(
        "model_revision", "Flag for model revision", "flagged",
        "The failure is systematic enough to belong in prompt or model revision.",
    ),
)

ROUTE_BY_KEY = {r.key: r for r in ROUTES}


@dataclass(frozen=True)
class Disposition:
    question_id: str
    crop_id: str
    source: str
    condition: str
    task: str
    dataset: str
    state: str
    tau: float
    obj_recall: float
    answer_correct: int
    route: str
    note: str
    decided_at: str


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def record(
    log: dict[str, Disposition],
    *,
    row,
    source: str,
    tau: float,
    route: str,
    note: str = "",
) -> dict[str, Disposition]:
    """Sign off one record. Re-deciding replaces the earlier disposition."""
    if route not in ROUTE_BY_KEY:
        raise ValueError(f"unknown route {route!r}")
    log[str(row["question_id"])] = Disposition(
        question_id=str(row["question_id"]),
        crop_id=str(row["crop_id"]),
        source=source,
        condition=str(row["condition"]),
        task=str(row["task"]),
        dataset=str(row["dataset"]),
        state=str(row["quadrant"]),
        tau=float(tau),
        obj_recall=float(row["obj_recall"]),
        answer_correct=int(row["answer_correct"]),
        route=route,
        note=note.strip(),
        decided_at=_now(),
    )
    return log


def clear(log: dict[str, Disposition], question_id: str) -> dict[str, Disposition]:
    log.pop(str(question_id), None)
    return log


def counts(log: dict[str, Disposition]) -> dict[str, int]:
    out = {r.key: 0 for r in ROUTES}
    for d in log.values():
        out[d.route] = out.get(d.route, 0) + 1
    return out


def to_frame(log: dict[str, Disposition]) -> pd.DataFrame:
    if not log:
        return pd.DataFrame(columns=[f.name for f in Disposition.__dataclass_fields__.values()])
    frame = pd.DataFrame([asdict(d) for d in log.values()])
    frame["route_label"] = frame["route"].map(lambda k: ROUTE_BY_KEY[k].label)
    return frame.sort_values("decided_at", ascending=False).reset_index(drop=True)


def coverage(log: dict[str, Disposition], queue_ids) -> tuple[int, int]:
    """How much of the current queue has been signed off."""
    ids = [str(q) for q in queue_ids]
    done = sum(1 for q in ids if q in log)
    return done, len(ids)


def stale(log: dict[str, Disposition], tau: float) -> list[Disposition]:
    """Dispositions taken under a different gate value than the one now in force.

    Surfaced rather than silently rewritten: a record accepted at tau = 0.3 has
    not been accepted at tau = 0.7, and the log should not pretend otherwise.
    """
    return [d for d in log.values() if abs(d.tau - tau) > 1e-9]


SCHEMA_VERSION = 1


def to_document(log: dict[str, Disposition], *, context: dict | None = None) -> dict:
    """Serialise the review log as a self-describing JSON document.

    Written so a downstream workflow can consume it without knowing anything
    about the dashboard: the route vocabulary is included inline, each decision
    carries the gate value and answer-action state it was taken under, and the
    per-route totals are stated so a reader does not have to re-aggregate to
    check they got the same answer.
    """
    return {
        "schema": "spatial-action-review/routing-decisions",
        "schema_version": SCHEMA_VERSION,
        "context": context or {},
        "routes": [
            {"key": r.key, "label": r.label, "meaning": r.blurb} for r in ROUTES
        ],
        "totals": {
            "decisions": len(log),
            "by_route": counts(log),
        },
        "decisions": [asdict(d) for d in sorted(log.values(), key=lambda d: d.decided_at)],
        "field_notes": {
            "tau": "Action-reliability gate in force when the decision was taken. A region "
                   "passes when its point action covers at least this fraction of the "
                   "labelled objects.",
            "state": "Answer-action state at the moment of the decision: trustworthy "
                     "(answer correct, action passes), silent_failure (answer correct, action "
                     "fails), lucky (answer wrong, action passes), honest (both fail).",
            "obj_recall": "Fraction of labelled objects the point action covered.",
            "decided_at": "UTC, ISO 8601.",
        },
    }
