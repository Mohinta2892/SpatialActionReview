"""Point-action scoring for records produced live, ported from the offline scorer.

This is a faithful port of `point_metrics` / `hungarian_pairs` in
`03_score_and_join.py`, including its scipy-optional behaviour: optimal
assignment when `scipy` is installed, the same greedy nearest-pair fallback when
it is not. A live re-ask therefore gets scored by the same rule as the released
records, with one documented difference — the offline scorer prefers
point-in-mask hits when label masks are reachable, and only falls back to
centroid matching. Masks are not part of the release, so live scoring is always
the centroid path at a normalised distance threshold of 0.1.
"""

from __future__ import annotations

import math
import re

POINT_THRESHOLD = 0.1

# Matches the <point x="..." y="..." /> tags the grounding prompts ask for.
POINT_TAG = re.compile(
    r'<point\s+x=["\']?([\d.]+)["\']?\s+y=["\']?([\d.]+)["\']?[^>]*/?>',
    re.IGNORECASE,
)


def parse_points(text: str) -> list[list[float]]:
    """Extract normalised points from a model response."""
    out: list[list[float]] = []
    for match in POINT_TAG.finditer(text or ""):
        try:
            x, y = float(match.group(1)), float(match.group(2))
        except ValueError:
            continue
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            out.append([x, y])
    return out


def parse_option_letter(text: str, n_options: int) -> str:
    """Recover the chosen option letter from a free-form answer."""
    stripped = (text or "").strip()
    if not stripped:
        return ""
    valid = {chr(65 + i) for i in range(max(n_options, 0))}
    # A leading letter, optionally followed by punctuation: "B", "B.", "B)".
    head = re.match(r"\s*([A-Za-z])\s*[.):\-]?", stripped)
    if head:
        letter = head.group(1).upper()
        if not valid or letter in valid:
            return letter
    # Otherwise the first standalone letter that is a valid option.
    for token in re.findall(r"\b([A-Za-z])\b", stripped):
        if token.upper() in valid:
            return token.upper()
    return ""


def _pairs(pred: list, gt: list) -> list[tuple[int, int, float]]:
    distances = [
        [math.dist((float(p[0]), float(p[1])), (float(g[0]), float(g[1]))) for g in gt]
        for p in pred
    ]
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(np.asarray(distances))
        return [(int(r), int(c), float(distances[r][c])) for r, c in zip(rows, cols)]
    except Exception:
        # Same greedy fallback as the offline scorer: take the closest available
        # pair repeatedly, never reusing a prediction or a ground-truth object.
        chosen: list[tuple[int, int, float]] = []
        used_p: set[int] = set()
        used_g: set[int] = set()
        for dist, i, j in sorted(
            (distances[i][j], i, j) for i in range(len(pred)) for j in range(len(gt))
        ):
            if i not in used_p and j not in used_g:
                used_p.add(i)
                used_g.add(j)
                chosen.append((i, j, dist))
        return chosen


def point_metrics(pred: list, gt: list, threshold: float = POINT_THRESHOLD) -> dict[str, float]:
    """Object recall, point precision, point F1, and the on-target point count.

    `on_target` is how many of the emitted points land on a distinct labelled
    object. `len(pred) - on_target` is therefore the number of points the
    predicted point set that correspond to no ground-truth object at all — a
    quantity the recall-only action gate cannot see.
    """
    if not pred and not gt:
        return {"point_precision": 1.0, "obj_recall": 1.0, "point_f1": 1.0, "on_target": 0}
    if not pred or not gt:
        return {"point_precision": 0.0, "obj_recall": 0.0, "point_f1": 0.0, "on_target": 0}
    good = sum(1 for _, _, dist in _pairs(pred, gt) if dist <= threshold)
    precision = good / len(pred)
    recall = good / len(gt)
    return {
        "point_precision": precision,
        "obj_recall": recall,
        "point_f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "on_target": good,
    }


def optimal_assignment_available() -> bool:
    try:
        import scipy.optimize  # noqa: F401
        return True
    except Exception:
        return False
