"""Tests for the automatic front/back order of Sculpt parts (backend.pipeline.sculpt_order)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.pipeline.sculpt_order import (
    CONTAINMENT_WEIGHT,
    CUE_CONTAINMENT,
    auto_layers,
    part_seams,
    Vote,
    containment_votes,
    solve_layers,
    tjunction_votes,
    trusted_pairs,
)
from backend.pipeline.sculpt_parts import ANALYSIS_RES, analyze_parts, analyze_parts_from_source, part_at

N = ANALYSIS_RES


def _analyse(white: np.ndarray, sil: np.ndarray):
    L = (white & sil).astype(np.float32)
    S = sil.astype(np.float32)
    return L, analyze_parts(L, S, L * S)


def _disc(cx, cy, r):
    yy, xx = np.mgrid[:N, :N]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def _ring(cx, cy, r, half):
    yy, xx = np.mgrid[:N, :N]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return np.abs(d - r) <= half


def _disc_over_rect(style: str):
    """A disc covering the left end of a rectangle, drawn as white-on-black line art.

    The disc's contour is a full dark circle; the rectangle's contour stops at it.
    ``bare``: the outer contour is the silhouette edge only (the artist's style);
    ``drawn``: a dark outline band runs around the whole figure as well.
    """
    disc = _disc(800, 1024, 380)
    rect = np.zeros((N, N), bool)
    rect[870:1180, 800:1750] = True
    union = disc | rect
    ring = _ring(800, 1024, 380, 2.0)
    if style == "bare":
        sil = union
    else:
        sil = cv2.dilate(union.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    white = union & ~ring
    return white, sil


def _sum(votes, front, back):
    return sum(v.weight for v in votes if v.front == front and v.back == back)


@pytest.mark.parametrize("style", ["bare", "drawn"])
def test_tjunction_simple(style):
    L, pa = _analyse(*_disc_over_rect(style))
    assert int(pa.labels.max()) == 2
    disc = part_at(pa, 800 / N, 1024 / N)
    rect = part_at(pa, 1600 / N, 1024 / N)
    assert {disc, rect} == {1, 2}
    votes = tjunction_votes(L, pa)
    assert all(isinstance(v, Vote) for v in votes)
    assert _sum(votes, disc, rect) > 0.5
    assert _sum(votes, rect, disc) < 0.5 * _sum(votes, disc, rect)
    layers = solve_layers(2, votes, [])
    assert layers.dtype == np.int32 and layers.shape == (3,)
    assert layers[disc] == 1 and layers[rect] == 0


def test_tjunction_votes_are_deterministic():
    L, pa = _analyse(*_disc_over_rect("drawn"))
    a, b = tjunction_votes(L, pa), tjunction_votes(L, pa)
    assert a == b
    assert a == sorted(a, key=lambda v: (v.front, v.back))


def test_tjunction_accepts_no_luminance():
    L, pa = _analyse(*_disc_over_rect("bare"))
    assert tjunction_votes(None, pa) == tjunction_votes(L, pa)


def test_tjunction_gap_open_stroke_casts_no_vote():
    """A stroke stopping just short of a disc's contour is found as a gap T-junction, but its
    two sides are the same part (they join through the gap), so it is an open-stroke stem and
    casts no vote (fix round 1 ruling)."""
    from backend.pipeline.sculpt_order import detect_tjunctions

    sil = np.zeros((N, N), bool)
    sil[400:1650, 300:1750] = True
    ring = _ring(1200, 1024, 300, 2.0)
    stroke = np.zeros((N, N), bool)
    stroke[1022:1026, 500:893] = True  # ends 5 px before the ring's outer edge at x = 898
    white = sil & ~ring & ~stroke
    L, pa = _analyse(white, sil)
    assert int(pa.labels.max()) == 2  # the stroke does not close off a region
    gaps = [t for t in detect_tjunctions(L, pa) if t.kind == "gap"]
    assert any(abs(t.pos[0] - 1024) < 6 and abs(t.pos[1] - 900) < 8 for t in gaps)
    assert all(t.front == 0 for t in gaps)
    assert tjunction_votes(L, pa) == []


def test_gap_must_be_white():
    """A faint dark bridge between the stem's end and the bar: not a gap T-junction."""
    from backend.pipeline.sculpt_order import detect_tjunctions

    sil = np.zeros((N, N), bool)
    sil[400:1650, 300:1750] = True
    ring = _ring(1200, 1024, 300, 2.0)
    stroke = np.zeros((N, N), bool)
    stroke[1022:1026, 500:893] = True
    white = sil & ~ring & ~stroke
    L = white.astype(np.float32)
    L[1023:1025, 893:898] = 0.7  # light grey bridge: white for the part mask, not for the gap test
    S = sil.astype(np.float32)
    pa = analyze_parts(L, S, L * S)
    near = lambda t: abs(t.pos[0] - 1024) < 6 and abs(t.pos[1] - 900) < 8  # noqa: E731
    Lg = L.copy()
    Lg[1022:1026, 895:898] = 0.3  # part of the gap dark (in the luminance only): not all white
    assert any(near(t) for t in detect_tjunctions(L, pa) if t.kind == "gap")
    assert not any(near(t) for t in detect_tjunctions(Lg, pa) if t.kind == "gap")


def test_plain_blob_has_no_votes():
    blob = _disc(1024, 1024, 600)
    L, pa = _analyse(blob, blob)
    assert tjunction_votes(L, pa) == [] and containment_votes(pa) == [] and trusted_pairs(pa) == []


def test_cycle_is_broken_by_weakest_vote():
    votes = [Vote(1, 2, 1.0), Vote(2, 3, 0.9), Vote(3, 1, 0.4)]
    layers = solve_layers(3, votes, [])
    # 3 > 1 is the weakest edge in the cycle and is dropped: 1 > 2 > 3
    assert layers.tolist() == [0, 2, 1, 0]


def test_votes_net_out_per_pair():
    # 1 > 2 with 0.8, 2 > 1 with 0.6: net 0.2 < 0.3, no edge
    assert solve_layers(2, [Vote(1, 2, 0.8), Vote(2, 1, 0.6)], []).tolist() == [0, 0, 0]
    # repeated votes for the same direction add up: 0.2 + 0.2 = 0.4 >= 0.3
    assert solve_layers(2, [Vote(1, 2, 0.2), Vote(1, 2, 0.2)], []).tolist() == [0, 1, 0]


def test_constraint_overrides_votes():
    layers = solve_layers(2, [Vote(1, 2, 5.0)], [(2, 1)])
    assert layers[2] > layers[1]
    # a constraint closing a cycle of votes breaks the cycle at a vote, never at itself
    layers = solve_layers(3, [Vote(1, 2, 1.0), Vote(2, 3, 2.0)], [(3, 1)])
    assert layers[3] > layers[1]
    assert layers[2] > layers[3]  # the weaker vote 1 > 2 was dropped, 2 > 3 kept
    # a constraint alone makes an edge
    assert solve_layers(3, [], [(1, 3)]).tolist() == [0, 1, 0, 0]


def test_contradicting_constraints_latest_wins():
    assert solve_layers(2, [], [(1, 2), (2, 1)]).tolist() == [0, 0, 1]


def test_no_votes_all_level():
    seams = {(1, 2): 100.0, (2, 3): 50.0, (3, 4): 10.0, (4, 5): 80.0}
    layers = solve_layers(5, [], [], seams)
    assert layers.dtype == np.int32 and len(set(layers[1:].tolist())) == 1
    # weak evidence below the edge threshold leaves parts on the same level
    layers = solve_layers(3, [Vote(1, 2, 0.29)], [], {(1, 2): 50.0, (2, 3): 50.0})
    assert len(set(layers[1:].tolist())) == 1


def test_edgeless_part_level_with_neighbours():
    # 1 > 2; part 3 has no evidence and shares its longest seam with 1
    votes = [Vote(1, 2, 1.0)]
    layers = solve_layers(3, votes, [], {(1, 2): 100.0, (1, 3): 200.0, (2, 3): 50.0})
    assert layers[1] > layers[2] and layers[3] == layers[1]
    # longest seam with 2: level with 2
    layers = solve_layers(3, votes, [], {(1, 2): 100.0, (1, 3): 50.0, (2, 3): 200.0})
    assert layers[3] == layers[2]
    # equal seams: the higher neighbour
    layers = solve_layers(3, votes, [], {(1, 2): 100.0, (1, 3): 80.0, (2, 3): 80.0})
    assert layers[3] == layers[1]
    # a chain of evidence-free parts follows the first placed neighbour
    layers = solve_layers(4, votes, [], {(1, 2): 100.0, (1, 3): 80.0, (3, 4): 60.0})
    assert layers[4] == layers[3] == layers[1]


def test_sink_not_below_its_unvoted_neighbours():
    # 1 > 3 > 4 and 1 > 2: the sink 2 shares an evidence-free seam with 3, so it rises to 3's
    # layer instead of sinking to the back as a pit
    votes = [Vote(1, 3, 1.0), Vote(3, 4, 1.0), Vote(1, 2, 1.0)]
    seams = {(1, 2): 100.0, (1, 3): 100.0, (3, 4): 100.0, (2, 3): 100.0}
    layers = solve_layers(4, votes, [], seams)
    assert layers.tolist() == [0, 2, 1, 1, 0]


def test_layers_are_compact():
    # two independent chains of different length share the same back and stay within D
    votes = [Vote(1, 2, 1.0), Vote(2, 3, 1.0), Vote(4, 5, 1.0)]
    seams = {(1, 2): 1.0, (2, 3): 1.0, (4, 5): 1.0, (3, 5): 1.0, (1, 4): 1.0}
    layers = solve_layers(5, votes, [], seams)
    # the longest chain needs 2 steps and nothing goes deeper; 4 > 5 stretches to sit level
    # with 1 and 3 across the two evidence-free seams
    assert layers.tolist() == [0, 2, 1, 0, 2, 0]


def test_containment_alone_makes_an_edge():
    c = Vote(1, 2, CONTAINMENT_WEIGHT, CUE_CONTAINMENT)
    assert solve_layers(2, [c], []).tolist() == [0, 1, 0]
    # with T-junction votes on the pair it only adds its weight: 0.25 - 0.2 < 0.3, no edge
    assert solve_layers(2, [c, Vote(2, 1, 0.2)], []).tolist() == [0, 0, 0]
    # ... and T evidence the other way wins
    assert solve_layers(2, [c, Vote(2, 1, 0.8)], []).tolist() == [0, 0, 1]


def test_milp_failure_retries_without_presolve(monkeypatch):
    """HiGHS presolve has wrongly reported a feasible placement infeasible (2 of 200
    contradictory override sequences). The solver retries once with presolve off
    before falling back to the longest-path layers."""
    import scipy.optimize as so

    votes = [Vote(1, 3, 1.0), Vote(3, 4, 1.0), Vote(1, 2, 1.0)]
    seams = {(1, 2): 100.0, (1, 3): 100.0, (3, 4): 100.0, (2, 3): 100.0}
    expected = solve_layers(4, votes, [], seams)  # [0, 2, 1, 1, 0]; longest path gives 2 -> 0
    real = so.milp
    calls = []

    class _Fail:
        status, x, message = 2, None, "presolve: infeasible (simulated)"

    def flaky(*args, **kwargs):
        calls.append(dict(kwargs.get("options") or {}))
        if len(calls) == 1:
            return _Fail()
        return real(*args, **kwargs)

    monkeypatch.setattr(so, "milp", flaky)
    layers = solve_layers(4, votes, [], seams)
    assert len(calls) == 2 and calls[1].get("presolve") is False, calls
    assert layers.tolist() == expected.tolist()


def test_milp_time_limit_falls_back_to_longest_path(monkeypatch):
    """Every placement MILP runs with a time limit (`MILP_TIME_LIMIT_S`). When the solver runs
    out of time (status 1: a pathological model or a slow machine) it is not retried: the
    longest-path layers are used at once, the same as after a failed solve. Here the real HiGHS
    runs with a limit too small to finish."""
    import scipy.optimize as so

    from backend.pipeline import sculpt_order

    votes = [Vote(1, 3, 1.0), Vote(3, 4, 1.0), Vote(1, 2, 1.0)]
    seams = {(1, 2): 100.0, (1, 3): 100.0, (3, 4): 100.0, (2, 3): 100.0}
    real = so.milp
    calls = []

    class _Fail:
        status, x, message = 4, None, "failed (simulated)"

    def failing(*args, **kwargs):
        calls.append(dict(kwargs.get("options") or {}))
        return _Fail()

    monkeypatch.setattr(so, "milp", failing)
    longest = solve_layers(4, votes, [], seams)  # the fallback: both calls fail
    assert len(calls) == 2 and all(c.get("time_limit") == sculpt_order.MILP_TIME_LIMIT_S for c in calls), calls
    calls.clear()

    def spy(*args, **kwargs):
        calls.append(dict(kwargs.get("options") or {}))
        res = real(*args, **kwargs)
        calls[-1]["status"] = res.status
        return res

    monkeypatch.setattr(so, "milp", spy)
    optimal = solve_layers(4, votes, [], seams)
    assert calls[-1]["status"] == 0 and calls[-1]["time_limit"] == sculpt_order.MILP_TIME_LIMIT_S
    assert optimal.tolist() != longest.tolist()  # the fallback is observable here
    calls.clear()
    monkeypatch.setattr(sculpt_order, "MILP_TIME_LIMIT_S", 1e-9)
    t0 = time.perf_counter()
    timed_out = solve_layers(4, votes, [], seams)
    if os.environ.get("MONOLITH_CHECK_TIMINGS") == "1":
        assert time.perf_counter() - t0 < 1.0
    assert [c["status"] for c in calls] == [1], calls  # out of time: no presolve-off retry
    assert timed_out.tolist() == longest.tolist()


def test_tjunction_joined_open_stroke_casts_no_vote():
    """A stroke that runs into a disc's contour (a *joined* T, the stroke touching the line) has
    the same part on both sides: a crease, not an occluded contour, so it casts no vote. Without
    the open-stroke rule the disc would be voted in front of the outer part."""
    from backend.pipeline.sculpt_order import detect_tjunctions

    sil = np.zeros((N, N), bool)
    sil[400:1650, 300:1750] = True
    ring = _ring(1200, 1024, 300, 2.0)
    stroke = np.zeros((N, N), bool)
    stroke[1022:1026, 500:901] = True  # runs into the ring (outer edge x = 898)
    # a second stroke on the far side splits the ring into two arcs, so each touch point is a
    # proper three-branch junction (a lone ring is one looping branch, which is skipped)
    stroke[1022:1026, 1499:1700] = True
    white = sil & ~ring & ~stroke
    L, pa = _analyse(white, sil)
    assert int(pa.labels.max()) == 2  # the strokes do not close off a region
    tjs = detect_tjunctions(L, pa)
    near = [t for t in tjs if abs(t.pos[0] - 1024) < 8 and (abs(t.pos[1] - 900) < 10 or abs(t.pos[1] - 1500) < 10)]
    joined = [t for t in near if t.kind == "joined"]
    assert len(joined) == 2, [(t.kind, t.pos) for t in tjs]
    assert all(t.front == 0 for t in joined)
    assert all(len(set(t.backs)) <= 1 for t in joined)
    assert tjunction_votes(L, pa) == []


def test_longest_path_depth():
    # 1 > 2 > 3 and 1 > 3: 3 is the back (0), 2 is 1, 1 is 2
    votes = [Vote(1, 2, 1.0), Vote(2, 3, 1.0), Vote(1, 3, 1.0)]
    assert solve_layers(4, votes, []).tolist() == [0, 2, 1, 0, 0]


def test_containment_vote():
    big = _disc(1024, 1024, 700)
    ring = _ring(1100, 1000, 150, 2.0)  # a closed contour inside: a hand on a shoulder
    white = big & ~ring
    L, pa = _analyse(white, big)
    assert int(pa.labels.max()) == 2
    s = part_at(pa, 1100 / N, 1000 / N)
    b = part_at(pa, 1024 / N, 1650 / N)
    votes = containment_votes(pa)
    assert votes == [Vote(s, b, CONTAINMENT_WEIGHT, CUE_CONTAINMENT)]
    assert tjunction_votes(L, pa) == []
    layers = auto_layers(pa)  # containment is the only evidence: it orders the pair
    assert layers[s] > layers[b]


def _fingers():
    """Two fingers with dark 2 px outlines, separated by a 2 px background gap."""
    sil = np.zeros((N, N), bool)
    sil[600:1400, 900:1000] = True
    sil[600:1400, 1002:1102] = True  # bg gap at x = 1000..1001
    white = np.zeros((N, N), bool)
    white[602:1398, 902:998] = True
    white[602:1398, 1004:1100] = True
    return white, sil


def test_thin_background_gap_is_not_trusted():
    L, pa = _analyse(*_fingers())
    assert int(pa.labels.max()) == 2
    # precondition: the Euclidean line roles see a (spurious) separating line between them
    assert (1, 2) in pa.adjacency
    assert (1, 2) not in trusted_pairs(pa)
    votes = tjunction_votes(L, pa) + containment_votes(pa)
    assert not [v for v in votes if {v.front, v.back} == {1, 2}]
    assert solve_layers(2, votes, []).tolist() == [0, 0, 0]


def test_island_across_moat_gets_no_containment():
    """A disc inside a ring part, separated by a thin background moat: no containment."""
    outer = _disc(1024, 1024, 700)
    moat = _ring(1024, 1024, 300, 1.0)
    sil = outer & ~moat
    lines = _ring(1024, 1024, 297.5, 1.0) | _ring(1024, 1024, 302.5, 1.0)  # dark outlines
    white = sil & ~lines
    L, pa = _analyse(white, sil)
    assert int(pa.labels.max()) == 2
    assert (1, 2) in pa.adjacency  # precondition: Euclidean roles pair them across the moat
    assert (1, 2) not in trusted_pairs(pa)
    assert containment_votes(pa) == []
    assert solve_layers(2, tjunction_votes(L, pa) + containment_votes(pa), []).tolist() == [0, 0, 0]


def test_real_seam_is_trusted():
    L, pa = _analyse(*_disc_over_rect("bare"))
    assert set(pa.adjacency) == set(trusted_pairs(pa)) == {(1, 2)}
