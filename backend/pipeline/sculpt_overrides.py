"""Click-to-fix overrides of the Sculpt part order, and the parts payload (design decision 1).

**What a click means.** The sculptor clicks a part and picks an action:

- ``forward``: the part goes in front of every part it touches;
- ``back``: it goes behind every part it touches;
- ``reset``: the part's own override is removed;
- ``reset_all``: every override is removed (including ones that no longer hit a part).

"Touches" means a trusted seam (`sculpt_order.part_seams`). A part *resting on* the clicked part
(one whose own boundary is mostly shared with it, the containment cue: a palm on a body) is left
out of the clicked part's constraints, so it keeps its relation to the part and rises or falls
with it instead of being pushed behind it.

**Storage.** Overrides live in the project's ``meta.json`` as ``"sculpt_overrides"``: a list of
``{"anchor": [u, v], "relation": "front" | "back"}``, oldest first, with ``v`` measured downward.
The anchor is the clicked part's own anchor (the centre of its largest inscribed circle on the
analysis grid), the point most likely to land on the same part after a reprocess.

**Applying them.** Each override resolves to ``part_at(anchor)``; one whose anchor hits no part
(background or a line) is *dropped*: it adds no constraint and is counted in
``dropped_overrides``. Dropped overrides stay stored (a later reprocess with other settings may
bring the part back); ``reset_all`` clears them. Only the most recent override per part counts;
an older one on the same part (two parts a reprocess merged) is shadowed, and is counted in
``dropped_overrides`` too.
Each part's override becomes hard (front, back) constraints against its current neighbours,
listed in click order, and `sculpt_order.solve_layers` lets the later of two contradicting
constraints win (A over B after B over A), and breaks any remaining cycle of constraints by
dropping its earliest one. The last click's constraints come last in the list and all leave (or
all enter) the clicked part, so every cycle holds an earlier constraint: the last click is
always honoured, and the solver always gets an acyclic, feasible model.

**No-op.** Re-clicking the latest action on a part leaves the stored list unchanged; a click that
changes the list but not the solved layers changes no relief either.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from backend.pipeline.sculpt_order import CUE_CONTAINMENT, Vote, containment_votes, part_seams, solve_layers, tjunction_votes
from backend.pipeline.sculpt_parts import PartsAnalysis, part_at, part_near

RELATIONS = ("front", "back")
ACTIONS = ("forward", "back", "reset", "reset_all")
_ACTION_RELATION = {"forward": "front", "back": "back"}
META_KEY = "sculpt_overrides"


@dataclass
class OrderInputs:
    """Everything the order solve needs besides the overrides (cached on the analysis)."""

    n_parts: int
    votes: list[Vote]
    seams: dict[tuple[int, int], float]
    neighbours: dict[int, list[int]]  # trusted neighbours per part, sorted
    resting: set[tuple[int, int]]  # (x, p): part x rests on part p (containment cue)
    auto: np.ndarray  # the automatic order (no overrides), int32 per part, index 0 unused


def order_inputs(pa: PartsAnalysis) -> OrderInputs:
    """Votes, seams and the automatic order of ``pa`` (computed once, ~0.3 s, then cached on it)."""
    cached = pa.order
    if isinstance(cached, OrderInputs):
        return cached
    votes = tjunction_votes(None, pa) + containment_votes(pa)
    seams = part_seams(pa)
    n = int(pa.labels.max())
    nbrs: dict[int, set[int]] = {}
    for a, b in seams:
        nbrs.setdefault(a, set()).add(b)
        nbrs.setdefault(b, set()).add(a)
    inputs = OrderInputs(
        n_parts=n,
        votes=votes,
        seams=seams,
        neighbours={p: sorted(q) for p, q in nbrs.items()},
        resting={(int(v.front), int(v.back)) for v in votes if v.cue == CUE_CONTAINMENT},
        auto=solve_layers(n, votes, [], seams),
    )
    pa.order = inputs
    return inputs


# --------------------------------------------------------------------------- stored overrides


def _valid(o: Any) -> bool:
    if not isinstance(o, dict) or o.get("relation") not in RELATIONS:
        return False
    a = o.get("anchor")
    return (
        isinstance(a, (list, tuple))
        and len(a) == 2
        and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in a)
    )


def stored(meta: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The project's overrides from ``meta`` (malformed entries are ignored), oldest first."""
    raw = (meta or {}).get(META_KEY) or []
    if not isinstance(raw, list):
        return []
    return [{"anchor": [float(o["anchor"][0]), float(o["anchor"][1])], "relation": o["relation"]} for o in raw if _valid(o)]


def _resolve(pa: PartsAnalysis, overrides: list[dict[str, Any]]) -> tuple[list[tuple[int, str]], int]:
    """([(part, relation)] in click order, latest per part; number of dropped overrides).

    Dropped: an anchor that hits no part, and an override shadowed by a later one on the same
    part. `apply_action` keeps at most one override per part, so two resolving to one part mean
    a reprocess merged their parts (or moved the anchors); the older one adds no constraint and
    is counted, never silently lost (design decision 1).
    """
    latest: dict[int, str] = {}
    dropped = 0
    for o in overrides:
        p = part_at(pa, o["anchor"][0], o["anchor"][1])
        if p <= 0:
            dropped += 1
            continue
        if p in latest:
            dropped += 1  # shadowed: an earlier override on the same (merged) part
            del latest[p]  # re-insert: dict order is the order of each part's latest click
        latest[p] = o["relation"]
    return list(latest.items()), dropped


def constraints(pa: PartsAnalysis, overrides: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Hard (front, back) constraints from the overrides, in click order (later ones win)."""
    inp = order_inputs(pa)
    out: list[tuple[int, int]] = []
    for p, rel in _resolve(pa, overrides)[0]:
        for q in inp.neighbours.get(p, ()):
            if (q, p) in inp.resting:
                continue  # q rests on p: it keeps its relation and moves with p
            out.append((p, q) if rel == "front" else (q, p))
    return out


def solve(pa: PartsAnalysis, overrides: list[dict[str, Any]] | None) -> np.ndarray:
    """The part order (layer per part, int32, index 0 unused) with ``overrides`` applied."""
    inp = order_inputs(pa)
    cons = constraints(pa, overrides or [])
    if not cons:
        return inp.auto.copy()
    return solve_layers(inp.n_parts, inp.votes, cons, inp.seams)


def dropped(pa: PartsAnalysis, overrides: list[dict[str, Any]]) -> int:
    """How many stored overrides hit no part of ``pa`` (they add no constraint)."""
    return _resolve(pa, overrides)[1]


def click_part(pa: PartsAnalysis, u: float | None, v: float | None) -> int:
    """The part a click at normalised (u, v), v downward, selects: on a line pixel its nearest
    part; 0 on the background, outside the image or without a point."""
    if u is None or v is None or not (math.isfinite(u) and math.isfinite(v)):
        return 0
    return part_near(pa, float(u), float(v))


def apply_action(
    pa: PartsAnalysis, overrides: list[dict[str, Any]], part: int, action: str
) -> list[dict[str, Any]]:
    """The stored list after ``action`` on ``part`` (``part`` is ignored by ``reset_all``).

    ``forward`` / ``back`` replace the part's override with one at the end (the latest click);
    re-clicking the part's latest action when it is already the most recent click returns the
    list unchanged. ``reset`` removes the part's override(s).
    """
    if action == "reset_all":
        return []
    if action not in ACTIONS:
        raise ValueError(f"unknown action {action!r}")
    keep = [o for o in overrides if part_at(pa, o["anchor"][0], o["anchor"][1]) != part]
    if action == "reset":
        return keep
    u, v = (float(x) for x in pa.anchors[part])
    return keep + [{"anchor": [round(u, 6), round(v, 6)], "relation": _ACTION_RELATION[action]}]


# --------------------------------------------------------------------------- payload


def labels_on_preview(pa: PartsAnalysis, shape: tuple[int, int]) -> np.ndarray:
    """Analysis labels nearest-sampled onto ``shape`` (line and background pixels 0), uint16."""
    H, W = int(shape[0]), int(shape[1])
    h, w = pa.shape
    ys = np.minimum(((np.arange(H) + 0.5) * h / H).astype(np.int64), h - 1)
    xs = np.minimum(((np.arange(W) + 0.5) * w / W).astype(np.int64), w - 1)
    return np.clip(pa.labels[np.ix_(ys, xs)], 0, 65535).astype("<u2")


def parts_payload(
    pa: PartsAnalysis, layer: np.ndarray, overrides: list[dict[str, Any]], shape: tuple[int, int]
) -> dict[str, Any]:
    """The GET /parts body: the label map on the preview grid ``shape`` (base64 of little-endian
    uint16, row 0 = top) and, per part, its layer, automatic layer, override and anchor."""
    inp = order_inputs(pa)
    resolved, n_dropped = _resolve(pa, overrides)
    rel = dict(resolved)
    lab = labels_on_preview(pa, shape)
    parts = [
        {
            "id": p,
            "layer": int(layer[p]),
            "auto_layer": int(inp.auto[p]),
            "override": rel.get(p),
            "anchor": [float(pa.anchors[p, 0]), float(pa.anchors[p, 1])],
        }
        for p in range(1, inp.n_parts + 1)
    ]
    return {
        "width": int(lab.shape[1]),
        "height": int(lab.shape[0]),
        "labels": base64.b64encode(lab.tobytes()).decode("ascii"),
        "parts": parts,
        "dropped_overrides": int(n_dropped),
    }
