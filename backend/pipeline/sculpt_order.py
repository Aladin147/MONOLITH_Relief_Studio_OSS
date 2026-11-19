"""Automatic front/back order of Sculpt parts, plus the artist's override constraints.

Evidence comes from **T-junctions** in the line drawing: one stroke (the *stem*) ends against
another (the *bar*). The bar is the occluding contour, so the part across the bar from the stem
is in front, and the parts on either side of the stem are behind it.

Everything runs on the fixed analysis grid of `PartsAnalysis` (longest edge 2048), so the
order is identical at every working resolution.

**The line mask.** It is ``pa.line_role > 0`` plus the 1 px inner rim of the silhouette. In the
artist's white-on-black style the outer contour of a part is the silhouette edge itself, not a
drawn line, so the commonest T-junction is a separating line that ends at the background with
the front part's silhouette edge continuing it (a back leg's edge stopping at the front leg).
With the rim in the mask, that is an ordinary three-branch junction of the skeleton.

**Two kinds of T-junction** are detected on the skeleton (``cv2.ximgproc.thinning``):

- *gap*: a skeleton endpoint lies within ``2r`` of another skeleton branch that it does not
  reach along its own path (not within ``4r`` along the skeleton). That branch is the bar.
  Weight ``clamp(cos(stem direction, bar normal), 0, 1) * min(1, stem length / (8r))``.
- *joined*: a junction with exactly three branches. Each branch in turn is tried as the stem
  with the other two as the bar; the T-score is ``bar straightness * cos(stem, bar normal)``,
  where bar straightness is ``clamp(-cos(angle between the two bar branches), 0, 1)`` (1 for
  a straight bar). The best stem must beat the runner-up by ``T_MARGIN``, which rejects
  Y-junctions. Weight ``T-score * min(1, stem length / (8r))``. For a straight bar this is the
  same formula as the gap kind.

``r`` is `sculpt_parts.line_radius` (6 px at 2048). The front part is the first part met when
stepping from the junction across the bar, away from the stem; the background ends the probe
(no vote). The back parts are the first parts met stepping sideways from the stem. Votes are
only cast for adjacent part pairs that `trusted_pairs` accepts.

Two checks beyond the plain T rule:

- *Open-stroke stems* are ignored: a stem with the same part on both sides (a finger crease
  ending at the hand's own outline) is detail, not an occluded contour.
- *Valley between two bumps* (`_owned_share`): where the line between two fingers meets their
  rounded tips, the tips look like a bar, and the plain rule would put the part beyond them in
  front. When both bar halves bend back towards the stem by more than `BUMP_START_DEG`, the
  bumps own the contour (figure-ground convexity) and that share of the weight votes the
  stem's sides in front instead.

**Trusted pairs.** Where two parts are within about ``2r`` of each other across a thin strip of
*background* (adjacent fingers), their outline pixels are classed as separating and the pair
appears in ``pa.adjacency`` although nothing joins them. A pair is trusted only if at least
half of its separating pixels reach both parts *through the silhouette* (geodesic distance at
most ``r``, background blocks).

**Along-stem geometry.** Directions, side probes and bar bends are measured over pixels
reached *along* the branch from the junction (`_along`), so a branch that loops back near the
junction does not mix its far end in; side probes go across the stem's local direction.

**Containment** (`containment_votes`) is a weak vote (0.25) that a part whose own boundary is
mostly (>= 70 %) shared with one neighbour is in front of it (a palm resting on a body).

`solve_layers` turns votes plus hard (front, back) constraints into a layer index per part:
every kept edge holds strictly, the layers are compact (no deeper than the longest chain of
edges), steps across seams without evidence are minimised exactly (a small integer program),
and a part with no evidence sits level with its neighbours. `auto_layers` runs everything.
"""

from __future__ import annotations

import heapq
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree

from backend.pipeline.sculpt_parts import (
    ROLE_OUTLINE,
    ROLE_SEPARATING,
    WHITE_THRESHOLD,
    PartsAnalysis,
    line_radius,
)

EDGE_MIN_WEIGHT = 0.3  # net vote weight for a pair to become an edge
CONTAINMENT_WEIGHT = 0.25
CONTAINMENT_SHARE = 0.70  # shared boundary / the part's own boundary
TRUST_MIN_FRACTION = 0.5  # separating pixels of a pair that must join both parts through the silhouette
T_MARGIN = 0.2  # a joined junction's best stem must beat the runner-up by this T-score
GAP_REACH = 2  # gap junction: the bar lies within GAP_REACH * r of the stem's end
PATH_REACH = 4  # ...and is not reachable within PATH_REACH * r along the skeleton
DIR_REACH = 3  # branch directions from skeleton pixels within DIR_REACH * r
STEM_FULL = 8  # stems of STEM_FULL * r px or longer count fully
# Valley between two bumps (see `_owned_share`): bar halves measured out to BUMP_REACH * r.
BUMP_REACH = 5
BUMP_START_DEG = 15.0
BUMP_SPAN_DEG = 25.0

_K3 = np.ones((3, 3), np.uint8)
_CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
# 8-neighbours in circular order (dy, dx): N, NE, E, SE, S, SW, W, NW
_RING8 = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))


CUE_TJUNCTION = "tjunction"
CUE_CONTAINMENT = "containment"


@dataclass(frozen=True)
class Vote:
    front: int
    back: int
    weight: float
    cue: str = CUE_TJUNCTION  # or CUE_CONTAINMENT


@dataclass
class TJunction:
    """One detected T-junction (for inspection and debug overlays)."""

    kind: str  # "joined" or "gap"
    pos: tuple[float, float]  # (y, x) on the analysis grid: junction centre / bar point
    stem: np.ndarray  # (K, 2) int (y, x) skeleton pixels of the stem branch
    bar: np.ndarray  # (K, 2) int (y, x) skeleton pixels of the bar near the junction
    front: int  # 0 = background (no vote)
    backs: tuple[int, ...]
    weight: float  # T-score times the stem length factor
    owned: float  # share of the weight for front > backs (convexity); the rest votes backs > front


# --------------------------------------------------------------------------- helpers


def _silhouette(pa: PartsAnalysis) -> np.ndarray:
    return (pa.labels > 0) | (pa.line_role > 0)


def _unit(v: np.ndarray) -> np.ndarray | None:
    n = float(np.hypot(v[0], v[1]))
    return None if n < 1e-6 else v / n


def _perp(v: np.ndarray) -> np.ndarray:
    return np.array([v[1], -v[0]])


def _probe(labels: np.ndarray, sil: np.ndarray, p: np.ndarray, d: np.ndarray, max_d: int) -> int:
    """First part id met stepping from ``p`` along ``d``; 0 if the background comes first."""
    h, w = labels.shape
    for s in range(1, max_d + 1):
        y = int(round(p[0] + d[0] * s))
        x = int(round(p[1] + d[1] * s))
        if not (0 <= y < h and 0 <= x < w) or not sil[y, x]:
            return 0
        lab = int(labels[y, x])
        if lab > 0:
            return lab
    return 0


def _majority(ids: list[int]) -> int:
    ids = [i for i in ids if i > 0]
    if not ids:
        return 0
    c = Counter(ids)
    best = max(c.values())
    return min(i for i, k in c.items() if k == best)


def _along(pts: np.ndarray, origin: np.ndarray, reach: float) -> tuple[np.ndarray, np.ndarray]:
    """Pixels of a skeleton curve within ``reach`` of ``origin`` *along the curve*.

    Starts at the pixel nearest ``origin`` and walks the 8-connected pixels (steps 1 and
    sqrt 2) inside a window of radius ``reach``, so a branch that loops back near the junction
    does not mix its far end in. Returns (pixels, distance along the curve from ``origin``).
    """
    if len(pts) == 0:
        return pts, np.zeros(0)
    d = np.hypot(pts[:, 0] - origin[0], pts[:, 1] - origin[1])
    win = pts[d <= reach + 1]
    dwin = d[d <= reach + 1]
    if len(win) == 0:
        return win, dwin
    key = {(int(y), int(x)): i for i, (y, x) in enumerate(win)}
    geo = np.full(len(win), np.inf)
    s0 = int(np.argmin(dwin))
    geo[s0] = dwin[s0]
    heap = [(float(geo[s0]), s0)]
    while heap:
        g0, i = heapq.heappop(heap)
        if g0 > geo[i]:
            continue
        y, x = int(win[i, 0]), int(win[i, 1])
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                j = key.get((y + dy, x + dx))
                if j is None or j == i:
                    continue
                g1 = g0 + (1.4142135 if dy and dx else 1.0)
                if g1 < geo[j] and g1 <= reach:
                    geo[j] = g1
                    heapq.heappush(heap, (g1, j))
    ok = np.isfinite(geo)
    return win[ok], geo[ok]


def _direction(pts: np.ndarray, origin: np.ndarray, reach: float) -> np.ndarray | None:
    """Unit vector from ``origin`` towards the mean of the curve's first ``reach`` px."""
    sub, _ = _along(pts, origin, reach)
    if len(sub) < 2:
        d = np.hypot(pts[:, 0] - origin[0], pts[:, 1] - origin[1])
        sub = pts[d <= np.sort(d)[min(len(d) - 1, 3)]]
    return _unit(sub.mean(axis=0) - origin)


# --------------------------------------------------------------------------- trusted pairs


def trusted_pairs(pa: PartsAnalysis) -> list[tuple[int, int]]:
    """Adjacent pairs (a < b), sorted, that are really joined by line pixels.

    A separating pixel owned by (a, b) supports the pair when both a and b are reachable from
    it within ``r`` steps through the silhouette (background blocks the path). The pair is
    trusted when at least `TRUST_MIN_FRACTION` of its separating pixels support it.
    """
    if not pa.adjacency:
        return []
    labels = pa.labels
    sil = _silhouette(pa)
    r = line_radius(pa.shape)
    ys, xs = np.nonzero(sil)
    y0, y1 = max(0, ys.min() - 1), ys.max() + 2
    x0, x1 = max(0, xs.min() - 1), xs.max() + 2
    lab = labels[y0:y1, x0:x1]
    s = sil[y0:y1, x0:x1]
    n = int(labels.max())
    big = 65535 if n < 65535 else (1 << 24)
    dt = np.uint16 if big == 65535 else np.float32
    s_m = s.astype(dt)
    hi = lab.astype(dt)
    lo_inv = np.where(lab > 0, big - lab, 0).astype(dt)  # max of this = big - min label
    for _ in range(r):
        hi = cv2.dilate(hi, _K3) * s_m
        lo_inv = cv2.dilate(lo_inv, _K3) * s_m

    ly, lx = np.nonzero(pa.line_role > 0)
    own = pa.line_owner
    sep = (pa.line_role[ly, lx] == ROLE_SEPARATING) & (own[:, 0] > 0) & (own[:, 1] > own[:, 0])
    a = own[sep, 0].astype(np.int64)
    b = own[sep, 1].astype(np.int64)
    py, px = ly[sep] - y0, lx[sep] - x0
    g_hi = hi[py, px].astype(np.int64)
    g_lo_inv = lo_inv[py, px].astype(np.int64)
    support = (g_hi == b) & (g_lo_inv > 0) & (big - g_lo_inv == a)
    key = a << 32 | b
    keys, inv = np.unique(key, return_inverse=True)
    tot = np.bincount(inv, minlength=len(keys))
    ok = np.bincount(inv, weights=support.astype(np.float64), minlength=len(keys))
    good = {
        (int(k >> 32), int(k & 0xFFFFFFFF))
        for k, t_, o in zip(keys.tolist(), tot.tolist(), ok.tolist())
        if o >= TRUST_MIN_FRACTION * t_
    }
    return sorted(p for p in pa.adjacency if p in good)


# --------------------------------------------------------------------------- skeleton graph


@dataclass
class _Graph:
    sk: np.ndarray  # bool skeleton
    branch: np.ndarray  # int32 branch id per pixel (0 = none / junction region)
    pts: list[np.ndarray]  # branch id -> (K, 2) float (y, x); index 0 unused
    length: np.ndarray  # branch id -> pixel count
    free_ends: list[tuple[int, int, int]]  # (y, x, branch id) of free endpoints, raster order
    junctions: list[tuple[np.ndarray, list[int], bool]]  # (centre, incident branches, loop?)


def _skeleton_graph(mask: np.ndarray, r: int) -> _Graph:
    sk = cv2.ximgproc.thinning(mask.astype(np.uint8) * 255, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN) > 0
    h, w = sk.shape
    p = np.pad(sk.astype(np.uint8), 1)
    nb = [p[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w] for dy, dx in _RING8]
    cn = np.zeros((h, w), np.uint8)  # crossing number: 0->1 transitions around the pixel
    for i in range(8):
        cn += (nb[i] == 0) & (nb[(i + 1) % 8] == 1)
    ends = sk & (cn == 1)
    junc = sk & (cn >= 3)

    jzone = cv2.dilate(junc.astype(np.uint8), _K3) > 0  # merge junction pixels 2 px apart
    n_j, jlab = cv2.connectedComponents(jzone.astype(np.uint8), connectivity=8)
    jreg = jzone & sk
    br = sk & ~jreg
    n_b, blab = cv2.connectedComponents(br.astype(np.uint8), connectivity=8)

    ys, xs = np.nonzero(br)
    ids = blab[ys, xs]
    order = np.argsort(ids, kind="stable")
    counts = np.bincount(ids, minlength=n_b)
    split = np.split(np.column_stack([ys[order], xs[order]]).astype(np.float64), np.cumsum(counts)[:-1])
    pts = [np.zeros((0, 2))] + split[1:]

    ey, ex = np.nonzero(ends & br)
    free_ends = [(int(y), int(x), int(blab[y, x])) for y, x in zip(ey, ex)]
    has_free = np.zeros(n_b, bool)
    has_free[blab[ey, ex]] = True

    # branch ends touching a junction region: (cluster, branch) with the touching pixels
    touch = br & (cv2.dilate(jreg.astype(np.uint8), _K3) > 0)
    jl_reg = np.where(jreg, jlab, 0).astype(np.float32)
    tclu = cv2.dilate(jl_reg, _K3).astype(np.int64)
    ty, tx = np.nonzero(touch)
    tb = blab[ty, tx].astype(np.int64)
    tc = tclu[ty, tx]
    tkey = tc * n_b + tb
    o = np.lexsort((tx, ty, tkey))
    tkey, ty, tx, tb, tc = tkey[o], ty[o], tx[o], tb[o], tc[o]
    uk, start = np.unique(tkey, return_index=True)
    stop = np.append(start[1:], len(tkey))

    jy, jx = np.nonzero(jreg)
    jc = jlab[jy, jx]
    cnt = np.bincount(jc, minlength=n_j).astype(np.float64)
    cy = np.bincount(jc, weights=jy, minlength=n_j) / np.maximum(cnt, 1)
    cx = np.bincount(jc, weights=jx, minlength=n_j) / np.maximum(cnt, 1)

    spur = has_free & (counts < r)
    inc: dict[int, list[int]] = {}
    loop: dict[int, bool] = {}
    for k_, s0, s1 in zip(uk.tolist(), start.tolist(), stop.tolist()):
        c, b = int(tc[s0]), int(tb[s0])
        if c <= 0 or b <= 0:
            continue
        gy, gx = ty[s0:s1], tx[s0:s1]
        if np.hypot(gy - gy.mean(), gx - gx.mean()).max() > 2.5:
            loop[c] = True  # the branch leaves and re-enters the same junction
        if not spur[b]:
            inc.setdefault(c, []).append(b)
    junctions = [
        (np.array([cy[c], cx[c]]), inc[c], loop.get(c, False)) for c in sorted(inc)
    ]
    return _Graph(sk, blab, pts, counts, free_ends, junctions)


# --------------------------------------------------------------------------- T-junctions


def _side_parts(
    labels: np.ndarray, sil: np.ndarray, stem: np.ndarray, origin: np.ndarray, ds: np.ndarray, r: int
) -> list[int]:
    """Parts on either side of the stem, a little way from the junction.

    Probes go across the stem's *local* direction at each probe point, so a curved stem (a
    round head's contour) is still probed on both of its sides. Side 0 is to the left of
    ``ds`` at the junction, side 1 to the right.
    """
    stem, d = _along(stem, origin, 4.5 * r)
    if len(stem) == 0:
        return [0, 0]
    sides: tuple[list[int], list[int]] = ([], [])
    ts = [t for t in (1.5 * r, 2.5 * r, 3.5 * r) if d.max() >= t - 1] or [float(d.max())]
    for t in ts:
        q = stem[int(np.argmin(np.abs(d - t)))]
        # local direction: from the stem pixels just before q to those just after it
        before = stem[(d >= t - r) & (d < t)]
        after = stem[(d > t) & (d <= t + r)]
        tan = None
        if len(before) and len(after):
            tan = _unit(after.mean(axis=0) - before.mean(axis=0))
        if tan is None or float(tan @ ds) < 0:
            tan = ds if tan is None else -tan
        nrm = _perp(tan)
        sides[0].append(_probe(labels, sil, q, nrm, 2 * r))
        sides[1].append(_probe(labels, sil, q, -nrm, 2 * r))
    return [_majority(sides[0]), _majority(sides[1])]


def _turn_to_stem(half: np.ndarray, origin: np.ndarray, n_stem: np.ndarray, r: int) -> float | None:
    """Degrees by which one bar half bends back towards the stem's side (negative: away).

    The half's direction near the junction (pixels r..2.5r away) is compared with its course
    further out (2.5r..5r); None if the half is too short to tell.
    """
    half, d = _along(half, origin, BUMP_REACH * r)
    near = (d >= r) & (d <= 2.5 * r)
    far = (d > 2.5 * r) & (d <= BUMP_REACH * r)
    if not near.any() or not far.any():
        return None
    p1 = half[near].mean(axis=0)
    e1 = _unit(p1 - origin)
    if e1 is None:
        return None
    e2 = _unit(n_stem - float(n_stem @ e1) * e1)  # perpendicular to e1, on the stem's side
    if e2 is None:
        return None
    d2 = half[far].mean(axis=0) - p1
    return float(np.degrees(np.arctan2(float(d2 @ e2), float(d2 @ e1))))


def _owned_share(halves: list[np.ndarray], origin: np.ndarray, n_stem: np.ndarray, r: int) -> float:
    """Share of a junction's weight that says "the part across the bar is in front".

    An occluding contour is straight or bulges towards what it hides (a disc's rim into the
    rectangle behind it), so beyond the junction it bends away from the stem. When *both* bar
    halves bend back towards the stem's side, the junction is a valley between two rounded
    bumps (two fingertips side by side): by figure-ground convexity the bumps own the contour,
    so that share of the weight votes the stem's sides in front instead. Using the smaller of
    the two turns keeps a single bend (a jaw line) from flipping a vote. The share is 1 up to
    `BUMP_START_DEG`, falling linearly to 0 at `BUMP_START_DEG + BUMP_SPAN_DEG`.
    """
    turns = [_turn_to_stem(h, origin, n_stem, r) for h in halves]
    if any(t is None for t in turns):
        return 1.0
    turn = min(turns)
    return min(1.0, max(0.0, 1.0 - (turn - BUMP_START_DEG) / BUMP_SPAN_DEG))


def _make_tj(kind, pos, stem, bar, front, sides, weight, owned) -> TJunction:
    # Open-stroke stem: the same part on both sides (a crease ending at the part's own
    # outline) says nothing about occlusion: no vote.
    if sides[0] > 0 and sides[0] == sides[1]:
        front = 0
    backs = tuple(sorted({s for s in sides if s > 0 and s != front}))
    return TJunction(kind, (float(pos[0]), float(pos[1])), stem.astype(np.int32), bar.astype(np.int32),
                     int(front), backs, float(weight), float(owned))


def detect_tjunctions(L_clean_analysis: np.ndarray | None, pa: PartsAnalysis) -> list[TJunction]:
    """All T-junctions with their front / back parts, in a deterministic order.

    ``L_clean_analysis`` (luminance on the analysis grid, or None) only decides whether the gap
    between a stem's end and its bar is white; without it, part pixels count as white.
    """
    labels = pa.labels
    sil = _silhouette(pa)
    if L_clean_analysis is not None:
        L = np.asarray(L_clean_analysis, np.float32)
        if L.shape != pa.shape:
            raise ValueError(f"L_clean_analysis shape {L.shape} != analysis grid {pa.shape}")
        white = sil & (L >= WHITE_THRESHOLD)
    else:
        white = labels > 0
    r = line_radius(pa.shape)
    rim = sil & (cv2.erode(sil.astype(np.uint8), _K3, borderType=cv2.BORDER_CONSTANT, borderValue=0) == 0)
    g = _skeleton_graph((pa.line_role > 0) | rim, r)
    full = float(STEM_FULL * r)
    out: list[TJunction] = []

    # joined junctions: three branches meet
    for j, inc, loop in g.junctions:
        if loop or len(inc) != 3:
            continue
        dirs = [_direction(g.pts[b], j, DIR_REACH * r) for b in inc]
        if any(d is None for d in dirs):
            continue
        scores = []
        for s in range(3):
            a, b = [i for i in range(3) if i != s]
            straight = min(1.0, max(0.0, -float(dirs[a] @ dirs[b])))
            t = _unit(dirs[a] - dirs[b])
            if t is None:
                scores.append((0.0, s, None))
                continue
            n = _perp(t)
            cos = min(1.0, abs(float(dirs[s] @ n)))
            scores.append((straight * cos, s, n))
        ranked = sorted(scores, key=lambda z: (-z[0], z[1]))
        best, s, n = ranked[0]
        if best - ranked[1][0] < T_MARGIN or n is None:
            continue
        ds = dirs[s]
        away = -n if float(ds @ n) > 0 else n
        front = _probe(labels, sil, j, away, 3 * r)
        stem = g.pts[inc[s]]
        sides = _side_parts(labels, sil, stem, j, ds, r)
        bar = np.concatenate([g.pts[inc[i]] for i in range(3) if i != s])
        near = np.hypot(bar[:, 0] - j[0], bar[:, 1] - j[1]) <= DIR_REACH * r
        owned = _owned_share([g.pts[inc[i]] for i in range(3) if i != s], j, -away, r)
        w = best * min(1.0, g.length[inc[s]] / full)
        out.append(_make_tj("joined", j, stem, bar[near], front, sides, w, owned))

    # gap junctions: a stem ends just short of another branch
    sky, skx = np.nonzero(g.sk)
    sk_pts = np.column_stack([sky, skx]).astype(np.float64)
    tree = cKDTree(sk_pts)
    reach = PATH_REACH * r
    h, w_ = labels.shape
    for ey, ex, b in g.free_ends:
        if b <= 0 or g.length[b] < r:
            continue
        e = np.array([ey, ex], np.float64)
        cand = tree.query_ball_point(e, GAP_REACH * r)
        if not cand:
            continue
        # skeleton pixels reachable from e within PATH_REACH * r steps along the skeleton
        wy0, wy1 = max(0, ey - reach - 1), min(h, ey + reach + 2)
        wx0, wx1 = max(0, ex - reach - 1), min(w_, ex + reach + 2)
        sub = g.sk[wy0:wy1, wx0:wx1].astype(np.uint8)
        seen = np.zeros_like(sub)
        seen[ey - wy0, ex - wx0] = 1
        for _ in range(reach):
            seen = cv2.dilate(seen, _K3) & sub
        cand = np.array(sorted(cand))
        cy, cx = sky[cand], skx[cand]
        inwin = (cy >= wy0) & (cy < wy1) & (cx >= wx0) & (cx < wx1)
        reached = np.zeros(len(cand), bool)
        reached[inwin] = seen[cy[inwin] - wy0, cx[inwin] - wx0] > 0
        other = ~reached & (g.branch[cy, cx] != b)
        if not other.any():
            continue
        oc = cand[other]
        dist = np.hypot(sky[oc] - ey, skx[oc] - ex)
        qi = oc[np.lexsort((skx[oc], sky[oc], dist))[0]]
        q = sk_pts[qi]
        # the gap must be white (the stem really ends) and must not cross the background
        m = int(np.ceil(np.hypot(*(q - e)))) + 1
        sy = np.rint(np.linspace(e[0], q[0], m)).astype(int)
        sx = np.rint(np.linspace(e[1], q[1], m)).astype(int)
        if not sil[sy, sx].all():
            continue
        on_line = pa.line_role[sy, sx] > 0
        mid = np.nonzero(~on_line)[0]
        if len(mid) == 0:
            continue
        gap = slice(mid[0], mid[-1] + 1)  # between the stem's end and the bar's line pixels
        if not white[sy[gap], sx[gap]].all():
            continue
        # the bar: skeleton pixels around q that the stem does not reach along its path
        near = np.array(sorted(tree.query_ball_point(q, GAP_REACH * r)))
        ny, nx = sky[near], skx[near]
        inw = (ny >= wy0) & (ny < wy1) & (nx >= wx0) & (nx < wx1)
        rch = np.zeros(len(near), bool)
        rch[inw] = seen[ny[inw] - wy0, nx[inw] - wx0] > 0
        bar = sk_pts[near[~rch & (g.branch[ny, nx] != b)]]
        if len(bar) < 3:
            continue
        cov = np.cov((bar - bar.mean(axis=0)).T)
        evals, evecs = np.linalg.eigh(cov)
        t = evecs[:, int(np.argmax(evals))]
        n = _perp(t)
        ds = _direction(g.pts[b], e, DIR_REACH * r)
        if ds is None:
            continue
        cos = min(1.0, abs(float(ds @ n)))
        away = -n if float(ds @ n) > 0 else n
        front = _probe(labels, sil, q, away, 3 * r)
        sides = _side_parts(labels, sil, g.pts[b], e, ds, r)
        wgt = cos * min(1.0, g.length[b] / full)
        wide = np.array(sorted(tree.query_ball_point(q, BUMP_REACH * r)))
        wy, wx = sky[wide], skx[wide]
        inw = (wy >= wy0) & (wy < wy1) & (wx >= wx0) & (wx < wx1)
        rch = np.zeros(len(wide), bool)
        rch[inw] = seen[wy[inw] - wy0, wx[inw] - wx0] > 0
        wbar = sk_pts[wide[~rch & (g.branch[wy, wx] != b)]]
        side = (wbar - q) @ t
        owned = _owned_share([wbar[side > 0], wbar[side < 0]], q, -away, r)
        out.append(_make_tj("gap", q, g.pts[b], bar, front, sides, wgt, owned))
    return out


def tjunction_votes(L_clean_analysis: np.ndarray | None, pa: PartsAnalysis) -> list[Vote]:
    """Front/back votes from T-junctions, summed per (front, back), sorted by (front, back).

    ``L_clean_analysis``: the analysis-grid luminance (shape ``pa.shape``) or None; see
    `detect_tjunctions`. Only trusted adjacent pairs (`trusted_pairs`) get votes.
    """
    trusted = set(trusted_pairs(pa))
    acc: dict[tuple[int, int], float] = {}
    for tj in detect_tjunctions(L_clean_analysis, pa):
        if tj.front <= 0 or tj.weight <= 0:
            continue
        for b in tj.backs:
            if (min(tj.front, b), max(tj.front, b)) not in trusted:
                continue
            for key, w in (((tj.front, b), tj.weight * tj.owned), ((b, tj.front), tj.weight * (1.0 - tj.owned))):
                if w > 0:
                    acc[key] = acc.get(key, 0.0) + w
    return [Vote(f, b, float(acc[(f, b)])) for f, b in sorted(acc)]


# --------------------------------------------------------------------------- containment


def containment_votes(pa: PartsAnalysis) -> list[Vote]:
    """Weak votes: a part whose boundary is mostly (>= 70 %) shared with one neighbour is in front.

    A part's boundary counts its pixels next to outline or separating lines or the background
    (the sides of open strokes inside it do not count); the shared length is
    ``pa.adjacency``. Only trusted pairs vote. Sorted by (front, back).
    """
    trusted = trusted_pairs(pa)
    if not trusted:
        return []
    lab = pa.labels
    n = int(lab.max())
    role = pa.line_role
    sil = _silhouette(pa)
    r = line_radius(pa.shape)

    def touching(m: np.ndarray) -> np.ndarray:
        d = cv2.dilate(m.astype(np.uint8), _CROSS, borderType=cv2.BORDER_CONSTANT, borderValue=1)
        return (lab > 0) & (d > 0)

    at_sep = touching(role == ROLE_SEPARATING)
    edge = at_sep | touching((role == ROLE_OUTLINE) | ~sil)
    perim = np.bincount(lab[edge], minlength=n + 1)
    # the part across a separating line: the other id within r (highest, else lowest)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    lab_f = lab.astype(np.float32)
    hi = cv2.dilate(lab_f, k)
    lo = cv2.erode(np.where(lab > 0, lab_f, np.float32(1e9)), k)
    ey, ex = np.nonzero(at_sep)
    a_ = lab[ey, ex].astype(np.int64)
    h_, l_ = hi[ey, ex].astype(np.int64), lo[ey, ex].astype(np.int64)
    other = np.where(h_ != a_, h_, np.where(l_ != a_, l_, 0))
    sel = other > 0
    pair_key = a_[sel] * (n + 1) + other[sel]
    shared = np.bincount(pair_key, minlength=(n + 1) * (n + 1)).reshape(n + 1, n + 1)
    votes = []
    for a, b in trusted:
        if shared[a, b] >= CONTAINMENT_SHARE * perim[a]:
            votes.append(Vote(a, b, CONTAINMENT_WEIGHT, CUE_CONTAINMENT))
        if shared[b, a] >= CONTAINMENT_SHARE * perim[b]:
            votes.append(Vote(b, a, CONTAINMENT_WEIGHT, CUE_CONTAINMENT))
    return sorted(votes, key=lambda v: (v.front, v.back))


# --------------------------------------------------------------------------- solving


def _find_cycle(n_parts: int, edges: dict[tuple[int, int], float]) -> list[tuple[int, int]] | None:
    """One directed cycle as a list of edges, found deterministically, or None."""
    out: dict[int, list[int]] = {}
    for f, b in sorted(edges):
        out.setdefault(f, []).append(b)
    state = [0] * (n_parts + 1)  # 0 new, 1 on stack, 2 done
    for root in range(1, n_parts + 1):
        if state[root]:
            continue
        stack = [(root, 0)]
        path = [root]
        state[root] = 1
        while stack:
            v, i = stack[-1]
            nxt = out.get(v, [])
            if i < len(nxt):
                stack[-1] = (v, i + 1)
                u = nxt[i]
                if state[u] == 1:
                    cyc = path[path.index(u) :] + [u]
                    return list(zip(cyc[:-1], cyc[1:]))
                if state[u] == 0:
                    state[u] = 1
                    stack.append((u, 0))
                    path.append(u)
            else:
                state[v] = 2
                stack.pop()
                path.pop()
    return None


def _net_edges(n_parts: int, votes: list[Vote]) -> dict[tuple[int, int], float]:
    """Front -> back edges from votes, netted per pair (see `solve_layers`)."""
    t_net: dict[tuple[int, int], float] = {}
    c_net: dict[tuple[int, int], float] = {}
    for v in votes:
        f, b = int(v.front), int(v.back)
        if f == b or not (1 <= f <= n_parts and 1 <= b <= n_parts):
            continue
        key = (min(f, b), max(f, b))
        acc = c_net if v.cue == CUE_CONTAINMENT else t_net
        acc[key] = acc.get(key, 0.0) + (v.weight if f < b else -v.weight)
    edges: dict[tuple[int, int], float] = {}
    for key in sorted(set(t_net) | set(c_net)):
        a, b = key
        w = t_net.get(key, 0.0) + c_net.get(key, 0.0)
        # containment alone (no T-junction vote on the pair either way) always makes an edge
        need = EDGE_MIN_WEIGHT if key in t_net else 0.0
        if w > 0 and w >= need:
            edges[(a, b)] = w
        elif w < 0 and -w >= need:
            edges[(b, a)] = -w
    return edges


# This cap (per solver call) only guards a pathological model or a very slow machine, which then gets
# the longest-path layers (always feasible, less compact) instead of a hang.
MILP_TIME_LIMIT_S = 3.0
_MILP_TIME_LIMIT = 1  # scipy.optimize.milp status: iteration or time limit reached


def _solve_placement(
    n_parts: int,
    edged: list[int],
    edges: dict[tuple[int, int], float],
    level: dict[int, list[tuple[int, float]]],
    start: np.ndarray,
    top: int,
) -> np.ndarray:
    from scipy.optimize import Bounds, LinearConstraint, milp

    idx = {p: i for i, p in enumerate(edged)}
    m = len(edged)
    seam_list = sorted(
        {(min(p, q), max(p, q)): w for p in edged for q, w in level.get(p, ()) if q in idx}.items()
    )
    k = len(seam_list)
    e_list = sorted(edges)
    eps = 0.5 / (len(e_list) * top + 1)
    eps2 = 0.5 * eps / (m * top + 1)
    c = np.zeros(m + k)
    for f, b in e_list:  # edge stretch: sum over edges of (L_f - L_b)
        c[idx[f]] += eps
        c[idx[b]] -= eps
    c[:m] -= eps2
    for j in range(k):
        c[m + j] = 1.0  # every unevidenced seam counts the same, however long
    rows, lb = [], []
    for f, b in e_list:
        r_ = np.zeros(m + k)
        r_[idx[f]], r_[idx[b]] = 1.0, -1.0
        rows.append(r_)
        lb.append(1.0)
    for j, ((a, b), _) in enumerate(seam_list):
        for sgn in (1.0, -1.0):
            r_ = np.zeros(m + k)
            r_[m + j] = 1.0
            r_[idx[a]], r_[idx[b]] = -sgn, sgn
            rows.append(r_)
            lb.append(0.0)
    problem = dict(
        constraints=LinearConstraint(np.array(rows), np.array(lb), np.full(len(lb), np.inf)),
        bounds=Bounds(np.r_[np.zeros(m), np.zeros(k)], np.r_[np.full(m, float(top)), np.full(k, np.inf)]),
        integrality=np.r_[np.ones(m), np.zeros(k)],
    )
    limit = {"time_limit": float(MILP_TIME_LIMIT_S)}
    res = milp(c, **problem, options=limit)
    if res.status not in (0, _MILP_TIME_LIMIT) or (res.status == 0 and res.x is None):
        # The model is always feasible (the longest-path start satisfies it), but HiGHS presolve
        # has reported it infeasible.
        # Retry once without presolve before falling back to the longest-path layers. A solve
        # that ran out of time is not retried: on a slow machine the click must not hang.
        res = milp(c, **problem, options={"presolve": False, **limit})
    out = start.copy()
    if res.status != 0 or res.x is None:
        return out
    sol = np.rint(res.x[:m]).astype(np.int32)
    if any(sol[idx[f]] <= sol[idx[b]] for f, b in e_list):
        return out
    out[edged] = sol
    return out


def _place(
    n_parts: int, edges: dict[tuple[int, int], float], seams: dict[tuple[int, int], float]
) -> np.ndarray:
    """Integer layers satisfying every edge, compact and level across unevidenced seams."""
    layer = np.zeros(n_parts + 1, np.int32)
    backs: dict[int, list[int]] = {}
    fronts: dict[int, list[int]] = {}
    for f, b in sorted(edges):
        backs.setdefault(f, []).append(b)
        fronts.setdefault(b, []).append(f)
    edged = sorted(set(backs) | set(fronts))

    # 1. longest-path depth from the back: the lowest feasible layer; its maximum D is the
    #    smallest possible number of steps, and no part goes above it
    done = [False] * (n_parts + 1)
    for root in edged:  # iterative post-order DFS over the acyclic graph
        if done[root]:
            continue
        stack = [(root, 0)]
        while stack:
            v, i = stack[-1]
            bs = backs.get(v, [])
            if i < len(bs):
                stack[-1] = (v, i + 1)
                if not done[bs[i]]:
                    stack.append((bs[i], 0))
            else:
                layer[v] = 1 + max(int(layer[u]) for u in bs) if bs else 0
                done[v] = True
                stack.pop()
    top = int(layer.max())

    # seams without a kept edge, between parts that have edges: these should not step
    level: dict[int, list[tuple[int, float]]] = {}
    for (a, b), w in sorted(seams.items()):
        if (a, b) in edges or (b, a) in edges or not (1 <= a <= n_parts and 1 <= b <= n_parts):
            continue
        level.setdefault(a, []).append((b, float(w)))
        level.setdefault(b, []).append((a, float(w)))
    is_edged = np.zeros(n_parts + 1, bool)
    is_edged[edged] = True

    # 2. exact placement of the parts with edges (a small integer program, HiGHS via
    #    scipy.optimize.milp): within 0..D and every edge front >= back + 1, minimise, in
    #    lexicographic order (by scaling), the summed steps across unevidenced seams (every
    #    seam counts the same); then the stretch of edges beyond one step; then prefer higher layers (sinks
    #    rise to their neighbours instead of forming pits). The longest-path start stays the
    #    fallback if the solver fails.
    if edged and top > 0:
        layer[:] = _solve_placement(n_parts, edged, edges, level, layer, top)

    # 3. parts without edges take the layer of the neighbour they share the longest seam with
    #    (the higher one on ties), propagating through chains of such parts; a group with no
    #    placed neighbour at all stays at 0
    nbrs: dict[int, list[tuple[float, int]]] = {}
    for (a, b), w in sorted(seams.items()):
        if 1 <= a <= n_parts and 1 <= b <= n_parts:
            nbrs.setdefault(a, []).append((float(w), b))
            nbrs.setdefault(b, []).append((float(w), a))
    placed = is_edged.copy()
    todo = [p for p in range(1, n_parts + 1) if not placed[p] and p in nbrs]
    while todo:
        progress = False
        for strict in (True, False):
            for p in todo:
                cand = nbrs[p]
                if strict:
                    wmax = max(w for w, _ in cand)
                    cand = [(w, q) for w, q in cand if w == wmax]
                    if not all(placed[q] for _, q in cand):
                        continue
                cand = [(w, int(layer[q]), q) for w, q in cand if placed[q]]
                if not cand:
                    continue
                layer[p] = max(cand)[1]
                placed[p] = True
                progress = True
                if not strict:
                    break  # one relaxed placement, then back to strict ones
            todo = [p for p in todo if not placed[p]]
            if progress:
                break
        if not progress:
            break
    layer[0] = 0
    return layer


def solve_layers(
    n_parts: int,
    votes: list[Vote],
    constraints: list[tuple[int, int]],
    seams: dict[tuple[int, int], float] | None = None,
) -> np.ndarray:
    """Layer index per part (int32, length ``n_parts + 1``, index 0 unused); 0 = back.

    ``seams`` maps trusted adjacent pairs (a < b) to their shared length (see `part_seams`);
    without it no part has neighbours, so parts without edges stay at 0.

    Edges:
    - T-junction votes are netted per pair; a net weight >= `EDGE_MIN_WEIGHT` becomes a
      front -> back edge. Containment votes add their weight to that net; on a pair with no
      T-junction vote in either direction, containment alone makes the edge.
    - Hard ``constraints`` are (front, back) pairs from overrides. They replace any vote edge
      on the same pair and are never dropped to break a cycle of votes. Only when the
      constraints contradict each other (A > B and B > A, or a longer cycle of constraints) does
      the later one in the list win; the earliest constraint in the cycle is dropped.
    - Cycles are broken by repeatedly dropping the lowest-weight vote edge in a cycle (ties: the
      smallest (front, back)).

    Placement (design 2, "no evidence: same level"; the deepest layer stays small):
    1. Every kept edge holds strictly (front > back). Start from the longest-path depth from
       the back; its maximum ``D`` is the fewest steps the edges need, and no layer exceeds it.
    2. The parts with edges are then placed exactly (a small integer program, HiGHS via
       ``scipy.optimize.milp``; the longest-path start is the fallback): within 0..D and the
       edges, minimise the summed steps across seams that carry no edge (each seam counts the
       same), then edge stretch beyond one step, then prefer higher layers, so sinks rise to
       their neighbours instead of forming pits.
    3. A part with no edges takes the layer of its trusted neighbour with the longest seam (the
       highest on ties).
    """
    return _place(n_parts, _kept_edges(n_parts, votes, constraints), seams or {})


def _kept_edges(
    n_parts: int, votes: list[Vote], constraints: list[tuple[int, int]]
) -> dict[tuple[int, int], float]:
    """Acyclic front -> back edges: netted votes plus constraints, cycles broken."""
    edges = _net_edges(n_parts, votes)

    hard: dict[tuple[int, int], int] = {}  # (front, back) -> position in the list (recency)
    for i, (f, b) in enumerate(constraints):
        f, b = int(f), int(b)
        if f == b or not (1 <= f <= n_parts and 1 <= b <= n_parts):
            continue
        hard.pop((b, f), None)
        hard[(f, b)] = i
    for f, b in hard:
        edges.pop((b, f), None)
        edges[(f, b)] = float("inf")

    while (cyc := _find_cycle(n_parts, edges)) is not None:
        soft = [e for e in cyc if e not in hard]
        if soft:
            drop = min(soft, key=lambda e: (edges[e], e))
        else:
            drop = min(cyc, key=lambda e: hard[e])
            del hard[drop]
        del edges[drop]
    return edges


def part_seams(pa: PartsAnalysis) -> dict[tuple[int, int], float]:
    """Trusted adjacent pairs (a < b) -> shared seam length in px, for `solve_layers`."""
    return {p: float(pa.adjacency[p]) for p in trusted_pairs(pa)}


def auto_layers(
    pa: PartsAnalysis,
    constraints: list[tuple[int, int]] = (),
    L_clean_analysis: np.ndarray | None = None,
) -> np.ndarray:
    """The automatic order plus overrides: T-junction and containment votes, solved."""
    votes = tjunction_votes(L_clean_analysis, pa) + containment_votes(pa)
    return solve_layers(int(pa.labels.max()), votes, list(constraints), part_seams(pa))
