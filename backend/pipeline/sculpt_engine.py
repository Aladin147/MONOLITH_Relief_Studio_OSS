"""Sculpt relief style: cushion volume, shallow creases and layered seams.

**Height model** (per pixel, in "part heights", before normalisation)::

    Re  = max(smin(d_thick, R), R_lo)         # edge roundness, adapted to thin parts
    f(u)= w1 c(min(u/rho, 1)) + (1 - w1) c(min(u, 1)),  c(t) = 1 - (1 - t)^3   # two-scale C2 cushion
    c_w = f(d_wall / Re) f_s(d_seam / Re)     # the outline cushion times the front-seam cushion
    c_o = f(d_open / Re)                      # the outline only (background, drawn outline)
    g   = f(d_open / max(smin(Re, d_gate), D_lo))  # support gate: ~1 on own seams, 0 at the background
    k   = ink * max(1 - s(d_crease / Wc), (1 - L)(1 - s(d_crease_all / Wc))),  s = sin(pi/2 min(., 1))
    q   = fullness + (1 - fullness) * dome - crease_depth * k
    z   = L * support * g + (L c_w + (1 - L) c_o) * q

``L = layering`` times the part's step factor (`TOP_LAYER_LAMBDA` on the top-of-stack layers,
else 1; `_layer_lambda`); ``R = edge_roundness_pct % * min(H, W)`` is the outer shoulder scale and
``rho * R`` the steep foot's reach (`CUSHION_RHO`, `CUSHION_W1`); ``Wc = crease_width_pct % * min(H, W)``.
``R_lo``, ``D_lo`` and a lower bound on ``Wc`` come from the draft clamp (`_radii`). The result
is normalised into ``[h_min, 1]`` inside the silhouette; the background is exactly 0.

The one user slider, Edge roundness, is the outer shoulder ``R``; the steep foot follows it,
weighted `CUSHION_W1`, with the foot slope ``(pi/2) / 0.4`` per ``R`` of the former sine cushion
(it saturates at ``CUSHION_RHO * R``, 5.3 % of min(H, W) at the default 7 %). The cubic ``c`` is
C2 at both knots (``c'' (1) = 0``): a sine has a curvature jump there, which a lit render shows
as Mach-band lines parallel to every wall (final review I1).

- **Walls.** ``d_wall`` is the distance to the part's own walls: the background and drawn
  outline touching it, and the separating lines touching it where it is *in front*. Where it is
  behind, it carries on underneath. Parts on the same layer that touch form a *level group*
  sharing one wall field; their seam is a crease.
- **Creases** are open strokes and separating lines between parts on the same layer. Their depth
  is scaled by the stroke's ink weight (darkness x width in ``L_clean``), so faint strokes give
  faint creases; open strokes stay symmetric creases (design §3).
- **Thickness.** ``d_thick`` is `THICK_ALPHA` x the largest wall distance nearby (per level
  group on a `THICK_RES` grid, blended smoothly), so a forearm narrower than ``2R`` still
  reaches a full, flat top.
- **Support** (decision 4) is the stacked height of what a part rests on: the harmonic (Laplace)
  extension under each level group of the surfaces of the lower parts it touches, solved once
  per order at the default settings on a coarse grid and upsampled bicubically. Every recompute
  re-evaluates the surfaces behind *at the current settings* on the seam pixels only, layer by
  layer, and sets each seam pixel's support so that its actual height ``L S g + mix q`` equals
  the highest surface behind, extrapolated from the neighbour to the seam pixel itself (so the
  foot continues that surface's slope instead of a flat pixel pair along the label staircase,
  which a lit render shows as comb ticks; clipped at `SEAM_EXTRAP_CLIP`). The residual is spread smoothly: binned on the support
  grid, normalised-Gaussian blurred, sampled bicubically over a band (`SEAM_BAND_FRAC`), plus a
  short fade (`SEAM_FIX_FRAC`) of the exact seam value. So seams match at every slider value.
- **Distances** come from working-resolution geometry (lines from ``L_clean``, roles from the
  nearest analysis-grid line pixel) and are Gaussian-smoothed with a zero-preserving offset
  (sigma `DIST_SMOOTH_FRAC` x min(H, W)), which removes the streaks an exact EDT of a
  pixel-staircase outline fans across every wall. The gate's denominator is smoothed more
  (`GATE_SMOOTH_FRAC`, floor `GATE_MIN_FRAC`), so there is no 1/r fan at seam ends.
- **Tone** (`_tone`): the raw z is normalised to its `KNEE_PCT` percentile with a soft knee
  (monotone, C1) rather than to its maximum, so a few top-of-stack pixels do not push the rest
  of the relief down; the knee's slope gain (at most `_KNEE_GAIN`) is part of the draft budget.
  The knee alone would still flatten the crests of the parts above that percentile (the hands),
  so the topmost layers covering less than ``100 - KNEE_PCT`` % of the silhouette step up by
  only ``TOP_LAYER_LAMBDA x Layering`` (`_layer_lambda`): the stack under them is lower and the
  knee compresses less of them. The factor is a per-part lookup on the labels, applied only in
  the rows those parts occupy; the support and seam solves use the level's own step.
- **Walls**: ``c_w = f(d_wall / Re) f_s(d_seam / Re)``, the outline cushion times the steep-footed
  front-seam cushion (a product, not a min, so the two join without a crease).
- **Draft clamp** (decision 3, `_radii`): the slope budget ``cot(theta)`` is split between the
  walls and gate (``(FMAX + L FMAX_s) / Re``), creases (``cd pi/2 / Wc``), the dome and the support
  gradient (times the gate), which raises ``R_lo``, ``D_lo`` and ``Wc``. When the clamp binds,
  an exact interior slope limiter (`_limit_where_needed`) then removes what the bound does not
  cover (seam corrections), so no interior slope exceeds the draft angle; outer walls stay the
  shared tail's exact envelope.

Storage: labels uint16, distances / thickness / ink weight float16 (upcast per chunk with OpenCV), dome and
support float32: about 30 bytes per working pixel, plus fixed analysis-grid arrays (about 30 MB)
and seam caches (about 25 MB at 4K).

Threads: large recomputes run on a module pool (`_N_WORKERS`); images up to `_INLINE_PX` run
inline on the caller's thread so a preview never queues behind 4K work. Workers only call NumPy
and OpenCV element-wise / reduction functions, which OpenCV does not parallelise, so there is
no oversubscription. `sculpt_profile` must never be called from inside the pool.

Brightness, contrast and gamma do not apply to Sculpt. ``h_min`` and ``invert_depth`` do.
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse.linalg import spsolve

from backend.pipeline.sculpt_parts import (
    ROLE_OPEN,
    ROLE_OUTLINE,
    ROLE_SEPARATING,
    WHITE_THRESHOLD,
    PartsAnalysis,
    labels_on_grid,
    line_radius,
)

SCULPT_LATEST_VERSION = 1

SCULPT_DEFAULTS: dict[str, float] = {
    "edge_roundness_pct": 7.0,
    "fullness": 0.45,
    "crease_depth": 0.18,
    "crease_width_pct": 1.0,
    # the lowest of 0.3 / 0.35 / 0.4 that passes every gate (0.3 misses the thigh wall gate by
    # 0.008); lower than 0.5, it agrees better with the competitor (corr, part order) and does
    # not drop the whole lower body a step below the torso (final review I4)
    "layering": 0.35,
}

# two-scale cushion: f(u) = W1 c(min(u/RHO, 1)) + (1 - W1) c(min(u, 1)), u = d / R, with the C2
# curve c(t) = 1 - (1 - t)^3 (c'(0) = 3, c'(1) = c''(1) = 0). Each RHO is the former sine
# cushion's times 6/pi, which keeps the foot term's slope at the wall (3 / RHO = (pi/2) / 0.4)
_C2_FOOT = 6.0 / math.pi
CUSHION_W1 = 0.55
CUSHION_RHO = 0.4 * _C2_FOOT  # the steep foot reaches 0.76 R (5.3 % of min(H, W) at R = 7 %)
# the same cushion with a much steeper foot for the part's own front seams (occluding edges), so
# a part in front steps off the surface behind instead of ramping up over the outer foot
CUSHION_RHO_SEAM = 0.08 * _C2_FOOT
CUSHION_W1_SEAM = 0.75
_LUT_N = 16384
_FMAX = 3.0 * (CUSHION_W1 / CUSHION_RHO + (1.0 - CUSHION_W1))  # max slope of f (at u = 0)
_FMAX_S = 3.0 * (CUSHION_W1_SEAM / CUSHION_RHO_SEAM + (1.0 - CUSHION_W1_SEAM))  # ... of f_s
FULLNESS_MIN = 0.2  # below it the seam solve cannot keep the foot on the surface behind (G8)
# tone: heights are normalised to a high percentile with a soft knee instead of to the maximum, so
# a few top-of-stack pixels do not flatten the rest: x = (z - zmin) / s, s = z_p97 - zmin (at
# least span / KNEE_XMAX), y = x below KNEE_X, KNEE_X + (1 - KNEE_X) tanh((x - KNEE_X) / (1 -
# KNEE_X)) above (monotone, C1: seams stay exact), then renormalised to [h_min, 1]
KNEE_PCT = 97.0
KNEE_X = 0.8
KNEE_XMAX = 1.4
_KNEE_GAIN = KNEE_XMAX / (KNEE_X + (1.0 - KNEE_X) * math.tanh((KNEE_XMAX - KNEE_X) / (1.0 - KNEE_X)))
# per-layer step: the topmost layers that together cover less than (100 - KNEE_PCT) % of the
# silhouette (the top-of-stack parts the knee would otherwise compress, e.g. hands) step up by
# Layering x TOP_LAYER_LAMBDA instead of Layering, which lowers the stack under them
TOP_LAYER_LAMBDA = 0.3

ROLE_BG = 4  # role map value for pixels outside the silhouette
SUPPORT_RES = 384  # longest edge of the grid the support's Laplace problem is solved on
THICK_RES = 1024  # longest edge of the grid the thickness field is computed on
DOME_CAP_FRAC = 0.12  # dome: distance to the outline, capped at this fraction of min(H, W) ...
DOME_SIGMA_FRAC = 0.15  # ... and blurred with this sigma (fraction of min(H, W))
THICK_RHO_FRAC = 0.06  # thickness: largest wall distance within this radius ...
THICK_ALPHA = 0.85  # ... times this (< 1: thin crests saturate, no medial ridge) ...
# Narrow blending keeps neighbouring thick regions from flattening thin crests.
THICK_BLUR_FRAC = 0.006
DIST_SMOOTH_FRAC = 0.003  # distance-field smoothing sigma, fraction of min(H, W)
SEAM_SMOOTH_MUL = 0.3  # front-seam distances are smoothed less (crisp occluding steps)
SUBPIX_REACH_FRAC = 0.12  # sub-pixel distance corrections reach this far from a wall
SUBPIX_CONTOUR_FRAC = 0.006  # wall outlines are smoothed along their length with this sigma
GATE_SMOOTH_FRAC = 0.008  # gate denominator smoothing (no 1/r fan at seam ends)
GATE_MIN_FRAC = 0.01  # floor of the gate denominator, fraction of min(H, W)
SEAM_BAND_FRAC = 0.015  # seam residual band width, fraction of min(H, W)
SEAM_SPREAD_SG = 1.5  # residual spread: normalised Gaussian sigma on the support grid (px)
SEAM_FIX_FRAC = 0.004  # the spread's error on the seam fades out within this, fraction of min(H, W)
SEAM_TARGET_CAP = 3.0  # a seam's support target is at most this times the stack height behind
# a seam pixel's target is the surface behind *extrapolated to the seam pixel* (2 z(n) - z(n2), n2
# the next pixel of the same part beyond its neighbour n), not z(n): z(n) puts a flat pixel pair
# along the label staircase, which a lit render shows as comb ticks (review round 4). The
# extrapolation is clipped to this fraction of a part height (fullness + (1 - fullness) dome)
SEAM_EXTRAP_CLIP = 0.015
INK_SIGMA_FRAC = 0.003  # ink weight: Gaussian line integral of darkness at this sigma ...
INK_BLUR_FRAC = 0.01  # the per-stroke weight is spread and blurred (no jumps between strokes)
INK_REF_FRAC = 0.0025  # ... relative to a solid line of this width (fraction of min(H, W))
_GATE_FLOOR = 0.05  # floor of g when solving the foot equation
_NO_SEAM = np.float16(60000.0)  # d_gate of pixels whose part has no front seam (g = c_o)
_HALF_PI = np.float32(np.pi / 2.0)
_K5 = np.ones((5, 5), np.uint8)
# 4 workers: measured as fast as 8 at 4K (the kernel saturates memory bandwidth) while a preview
# computed meanwhile slows 1.8x instead of 3.8x (task-4 report, fix round 1)
_N_WORKERS = max(1, min(4, (os.cpu_count() or 2) - 1))
_INLINE_PX = 2_500_000  # images up to this many pixels are computed on the caller's thread
_POOL: ThreadPoolExecutor | None = None
_POOL_LOCK = threading.Lock()
_TLS = threading.local()
_REMAP_W = 1024  # scattered remap points per map row
_CHUNK_PX = 1 << 17  # pixels per sub-chunk (smaller: GIL-bound; larger: cache misses)
# draft budget share (of the allowed slope) reserved for creases
_CREASE_SHARE = 0.15


def _c2(t: np.ndarray) -> np.ndarray:
    """The cushion curve ``1 - (1 - t)^3`` on ``t`` clipped to [0, 1] (float64): C2 at t = 1."""
    return 1.0 - (1.0 - np.clip(t, 0.0, 1.0)) ** 3


def _make_lut(rho: float = CUSHION_RHO, w1: float = CUSHION_W1) -> np.ndarray:
    u = np.arange(_LUT_N + 1, dtype=np.float64) / _LUT_N
    return (w1 * _c2(u / rho) + (1 - w1) * _c2(u)).astype(np.float32)


_LUT = _make_lut()
_LUT_S = _make_lut(CUSHION_RHO_SEAM, CUSHION_W1_SEAM)
# smooth minimum (a p-norm) of the edge radius with R and of the gate denominator with Re: a hard
# min leaves a C0 contour along the switch locus (review round 4, P1). smin(a, b) =
# (a^-p + b^-p)^(-1/p) = m psi(m / M), m = min, M = max, psi(r) = (1 + r^p)^(-1/p) in [2^(-1/p), 1]
SMIN_P = 8.0
_SMIN_K = 2.0 ** (-1.0 / SMIN_P)  # smin(a, b) >= _SMIN_K min(a, b)
_LUT_PSI = ((1.0 + (np.arange(_LUT_N + 1, dtype=np.float64) / _LUT_N) ** SMIN_P) ** (-1.0 / SMIN_P)).astype(np.float32)


# --------------------------------------------------------------------------- small helpers


def cushion(u: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """``sin(pi/2 * clip(u, 0, 1))`` in float32: 0 at the wall, C1 at u = 1, slope pi/2 at u = 0.

    The single-scale curve used for creases; walls use the two-scale `cushion2`. ``out`` may be
    ``u`` itself (in place).
    """
    u = np.asarray(u, dtype=np.float32)
    if out is None:
        out = np.empty_like(u)
    np.clip(u, np.float32(0.0), np.float32(1.0), out=out)
    np.multiply(out, _HALF_PI, out=out)
    np.sin(out, out=out)
    return out


def cushion2(u: np.ndarray, rho: float = CUSHION_RHO, w1: float = CUSHION_W1) -> np.ndarray:
    """The two-scale wall cushion f(u) (see the module docstring), exact (no lookup table);
    ``rho, w1 = CUSHION_RHO_SEAM, CUSHION_W1_SEAM`` give the front-seam cushion f_s."""
    u = np.clip(np.asarray(u, np.float64), 0.0, None)
    return (w1 * _c2(u / rho) + (1 - w1) * _c2(u)).astype(np.float32)


def smin(a: np.ndarray, b, p: float = SMIN_P) -> np.ndarray:
    """The p-norm smooth minimum ``(a^-p + b^-p)^(-1/p)`` (exact, float64; a, b > 0): at most
    ``min(a, b)``, at least `_SMIN_K` times it. The kernel evaluates it through `_LUT_PSI`."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    m, M = np.minimum(a, b), np.maximum(a, b)
    return m * (1.0 + (m / M) ** p) ** (-1.0 / p)


def edge_radius(d_thick: np.ndarray, R: float, R_lo: float) -> np.ndarray:
    """The per-pixel edge radius ``Re = max(smin(d_thick, R), R_lo)`` (exact, float64)."""
    return np.maximum(smin(d_thick, R), min(R_lo, R))


def _get(params: Any, key: str, default: Any) -> Any:
    if params is None:
        return default
    if isinstance(params, Mapping):
        v = params.get(key, default)
    else:
        v = getattr(params, key, default)
    return default if v is None else v


def _shape_params(params: Any) -> tuple[float, float, float]:
    """(fullness, crease_depth, layering), clamped; fullness is floored at `FULLNESS_MIN`."""
    F = min(1.0, max(FULLNESS_MIN, float(_get(params, "fullness", SCULPT_DEFAULTS["fullness"]))))
    cd = max(0.0, float(_get(params, "crease_depth", SCULPT_DEFAULTS["crease_depth"])))
    L = min(1.0, max(0.0, float(_get(params, "layering", SCULPT_DEFAULTS["layering"]))))
    return F, cd, L


def sculpt_order_key(layers_np: np.ndarray) -> str:
    """Identifies the order a basis was built for (rebuild when it changes)."""
    return hashlib.sha1(np.ascontiguousarray(layers_np, dtype=np.int32).tobytes()).hexdigest()


def _pool() -> ThreadPoolExecutor:
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ThreadPoolExecutor(max_workers=_N_WORKERS, thread_name_prefix="sculpt")
        return _POOL


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def _smooth(d: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth a distance field in place, keeping its zero set at 0.

    A wall (zero half-plane) blurs to 0.4 sigma at its edge and a line a little less; the
    offset brings both back to 0 (a crease bottom reaches 0) and the rest shifts by 0.4 sigma.
    """
    np.minimum(d, np.float32(60000.0), out=d)  # no zero pixel at all: keep float16-safe
    if sigma > 0.3:
        cv2.GaussianBlur(d, (0, 0), sigma, dst=d, borderType=cv2.BORDER_REPLICATE)
        d -= np.float32(0.4 * sigma)
        np.maximum(d, np.float32(0.0), out=d)
    return d


def _edt(zero: np.ndarray, bbox, sil: np.ndarray, sigma: float, basis: dict | None = None, dtype=np.float16) -> np.ndarray:
    """Sub-pixel (`_subpixel`, from ``basis``), smoothed EDT to the ``zero`` pixels inside
    ``bbox`` (padded by 1); 0 outside sil."""
    H, W = zero.shape
    out = np.zeros((H, W), dtype)
    if bbox is None:
        return out
    r0, r1, c0, c1 = max(0, bbox[0] - 1), min(H, bbox[1] + 1), max(0, bbox[2] - 1), min(W, bbox[3] + 1)
    src = np.ascontiguousarray(~zero[r0:r1, c0:c1]).view(np.uint8)
    d = cv2.distanceTransform(src, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    _apply_subpixel(d, basis, slice(r0, r1), slice(c0, c1))
    out[r0:r1, c0:c1] = _smooth(d, sigma)
    out[~sil] = 0
    return out


def _smooth_closed(pts: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth a closed polyline (N, 2) along its index (wrap-around)."""
    n = len(pts)
    r = int(min(n // 2, math.ceil(3 * sigma)))
    if r < 1:
        return pts.astype(np.float32)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    ext = np.concatenate([pts[-r:], pts, pts[:r]]).astype(np.float64)
    out = np.empty((n, 2), np.float32)
    for j in range(2):
        out[:, j] = np.convolve(ext[:, j], k, mode="valid")
    return out


def _subpixel(L: np.ndarray | None, wall: np.ndarray, bbox, m: float) -> np.ndarray:
    """Sub-pixel correction of binary distances to walls: distances to the smoothed outline.

    An exact EDT measures to the centres of a pixel staircase, so along a slanted outline it
    ripples with the stair period, and a lit render shows that as streaks across every wall.
    (The anti-aliasing in ``L_clean`` is too irregular to place the edge per pixel on the
    artist's drawing, so the edge is recovered geometrically.) Every wall boundary is traced
    (``cv2.findContours``) and smoothed along its own arc length (sigma `SUBPIX_CONTOUR_FRAC`
    of min(H, W)), which straightens the staircase; each boundary pixel keeps its smoothed point
    and outward normal (two of each where a thin line is traced from both sides). For each pixel
    x within ``SUBPIX_REACH_FRAC`` of a wall, with p its nearest wall pixel, the distance is the
    projection ``(x - e_p) . n_p - 0.5`` (the edge lies half a pixel beyond the wall pixels'
    centres); along a straight edge every p gives the same value. Returned as an int8
    correction (1/64 px) to the exact binary distance to the walls (the ripple only, so it fades
    out at the reach without a step; the half pixel is dropped by every field). A field whose
    own nearest wall is a different one uses no correction (`_apply_subpixel`). ``L`` is unused
    (kept for the interface); without walls nothing is corrected.
    """
    H, W = wall.shape
    reach = max(3.0, SUBPIX_REACH_FRAC * m)
    corr = np.zeros((H, W), np.int8)
    if bbox is None or not wall.any() or wall.all():
        return corr
    pad = int(reach) + 3
    r0, r1, c0, c1 = max(0, bbox[0] - pad), min(H, bbox[1] + pad), max(0, bbox[2] - pad), min(W, bbox[3] + pad)
    z = np.ascontiguousarray(wall[r0:r1, c0:c1])
    hh, ww = z.shape
    sig_c = max(1.0, SUBPIX_CONTOUR_FRAC * m)
    contours, _ = cv2.findContours(z.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    idx_l, e_l, n_l = [], [], []
    for c in contours:
        pts = c[:, 0, :]  # (x, y)
        if len(pts) < 8:  # specks: no correction
            continue
        e = _smooth_closed(pts, sig_c)
        t = np.roll(e, -1, axis=0) - np.roll(e, 1, axis=0)
        nrm = np.stack([t[:, 1], -t[:, 0]], axis=1)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-6)
        # orient every normal towards the bright side
        px_, py_ = pts[:, 0], pts[:, 1]
        votes = np.zeros(len(pts), np.int32)
        for t_ in (1.0, 2.0, 3.0):  # walls along +n and bright along -n vote to flip
            for sgn in (1.0, -1.0):
                sx_ = np.clip(np.rint(px_ + sgn * t_ * nrm[:, 0]).astype(int), 0, ww - 1)
                sy_ = np.clip(np.rint(py_ + sgn * t_ * nrm[:, 1]).astype(int), 0, hh - 1)
                votes += np.where(z[sy_, sx_], 1, -1) * int(sgn)
        flip = votes > 0
        nrm[flip] *= -1.0
        idx_l.append(py_.astype(np.int64) * ww + px_)
        e_l.append(e)
        n_l.append(nrm)
    if not idx_l:
        return corr
    idx = np.concatenate(idx_l)
    e = np.concatenate(e_l)
    nr = np.concatenate(n_l)
    order = np.argsort(idx, kind="stable")
    idx, e, nr = idx[order], e[order], nr[order]
    uniq, first, cnt = np.unique(idx, return_index=True, return_counts=True)
    second = first + np.minimum(cnt, 2) - 1  # a second entry where a pixel is traced twice
    d_bin = cv2.distanceTransform(np.ascontiguousarray(~z).view(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    traced = np.zeros(hh * ww, bool)
    traced[uniq] = True
    traced = traced.reshape(hh, ww)
    _, lab = cv2.distanceTransformWithLabels(np.ascontiguousarray(~traced).view(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_5,
                                             labelType=cv2.DIST_LABEL_PIXEL)
    # the labels number the traced pixels in raster order, i.e. as ``uniq``: label - 1 is the slot
    yy, xx = np.nonzero(~z & (d_bin < reach))
    slot = lab[yy, xx] - 1
    db = d_bin[yy, xx]
    yf, xf = yy.astype(np.float32), xx.astype(np.float32)
    # per traced entry (n_x, n_y, e . n): the projection is x n_x + y n_y - e . n
    tab = np.empty((len(e), 3), np.float32)
    tab[:, :2] = nr
    tab[:, 2] = (e[:, 0] * nr[:, 0] + e[:, 1] * nr[:, 1]).astype(np.float32)
    t1 = tab[first[slot]]
    best = xf * t1[:, 0] + yf * t1[:, 1] - t1[:, 2]
    twice = np.flatnonzero(cnt[slot] > 1)  # pixels whose nearest wall pixel is traced twice
    if len(twice):
        t2 = tab[second[slot[twice]]]
        best[twice] = np.maximum(best[twice], xf[twice] * t2[:, 0] + yf[twice] * t2[:, 1] - t2[:, 2])
    # the ripple only: the projection on the smoothed wall-centre line minus the binary distance
    # (both measure to wall pixel centres; every field also drops the half pixel to the edge)
    c = best - db
    # a staircase is off by at most ~0.7 px; larger values mean the nearest pixel's edge is not
    # this pixel's (thin lines, tips): roll those back to no correction, smoothly
    c = c * np.clip((1.2 - np.abs(c)) / 0.5, 0.0, 1.0)
    c = np.maximum(c, 0.5 - db)
    c *= np.clip((reach - db) / (0.2 * reach), 0.0, 1.0)
    corr[yy + r0, xx + c0] = np.rint(c * 64).astype(np.int8)
    return corr


def _apply_subpixel(d: np.ndarray, basis: dict | None, sy: slice, sx: slice) -> np.ndarray:
    """Add the sub-pixel correction to a binary EDT window ``d`` (float32, in place) where the
    field's nearest wall is the globally nearest one (its distance equals the distance to all
    walls, recomputed here for the window)."""
    if basis is None or basis.get("_sub_corr") is None:
        d -= np.float32(0.5)  # wall pixel centres -> the edge half a pixel beyond
        np.maximum(d, np.float32(0.0), out=d)
        return d
    c = cv2.add(basis["_sub_corr"][sy, sx], 0, dtype=cv2.CV_32F)
    dg = basis.get("_tmp_dglob")  # the distance to all walls, kept during a build / rebuild
    if dg is None:
        walls = np.ascontiguousarray(basis["_role"][sy, sx] == 0).view(np.uint8)
        dg = cv2.distanceTransform(walls, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    else:
        dg = dg[sy, sx]
    # weight 1 where the field's nearest wall is the global one: w = clip(1 - (d - dg) / 1.5)
    w = cv2.addWeighted(d, -1.0 / 1.5, dg, 1.0 / 1.5, 1.0)
    cv2.min(w, 1.0, dst=w)
    cv2.max(w, 0.0, dst=w)
    cv2.multiply(c, w, dst=c, scale=1.0 / 64.0)
    cv2.subtract(d, 0.5, dst=d)  # wall pixel centres -> the edge half a pixel beyond
    cv2.max(d, 0.0, dst=d)
    cv2.add(d, c, dst=d)
    cv2.max(d, 0.0, dst=d)
    return d


def _layer_vec(layers_np: np.ndarray, n: int) -> np.ndarray:
    lay = np.zeros(n + 1, np.int32)
    src = np.asarray(layers_np, dtype=np.int32).ravel()
    k = min(len(src), n + 1)
    lay[:k] = src[:k]
    lay[0] = 0
    return lay


def _nearest_line_index(pa: PartsAnalysis) -> np.ndarray | None:
    """(h, w) int32: flat index of the nearest analysis-grid line pixel, or None without lines."""
    line = pa.line_role > 0
    if not line.any():
        return None
    src = np.ascontiguousarray(~line).view(np.uint8)
    _, lab = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, cv2.DIST_MASK_5, labelType=cv2.DIST_LABEL_PIXEL)
    flat = np.flatnonzero(line).astype(np.int32)
    return flat[np.maximum(lab, 1) - 1]  # labels number the zero pixels in raster order


def _layer_near(basis: dict, layer: np.ndarray):
    """Lowest / highest layer of the parts within ``r`` of each analysis pixel's nearest line pixel."""
    near = basis["_near"]
    if near is None:
        return None, None
    lab = basis["_pa_labels"]
    r = basis["_pa_r"]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    lv = layer.astype(np.float32)[lab]
    part = lab > 0
    mn = cv2.erode(np.where(part, lv, np.float32(1e6)), k)
    mx = cv2.dilate(np.where(part, lv, np.float32(-1e6)), k)
    return mn.ravel()[near], mx.ravel()[near]


def _groups(layer: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Level groups: parts on the same layer joined through touching parts. Root id per part."""
    n = len(layer) - 1
    parent = np.arange(n + 1)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    for a, b in pairs.tolist():
        if layer[a] == layer[b]:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(p) for p in range(n + 1)], np.int32)


def _label_pairs(labels: np.ndarray) -> np.ndarray:
    """(K, 2) int32 sorted unique (a < b) pairs of 4-adjacent different part labels."""
    keys = []
    for a, b in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        m = (a > 0) & (b > 0) & (a != b)
        if m.any():
            lo = np.minimum(a[m], b[m]).astype(np.int64)
            hi = np.maximum(a[m], b[m]).astype(np.int64)
            keys.append(np.unique(lo << 32 | hi))
    if not keys:
        return np.zeros((0, 2), np.int32)
    k = np.unique(np.concatenate(keys))
    return np.column_stack([k >> 32, k & 0xFFFFFFFF]).astype(np.int32)


def _nbr_max(src: np.ndarray) -> np.ndarray:
    """Per pixel, the maximum of ``src`` over its four neighbours (-inf outside)."""
    h, w = src.shape
    out = np.full((h, w), -np.inf, np.float32)
    P = np.pad(src, 1, constant_values=-np.inf)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        np.maximum(out, P[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w], out=out)
    return out


def _grid(H: int, W: int, res: int) -> tuple[int, int, np.ndarray, np.ndarray]:
    """A coarse grid (longest edge ``res``): shape and the working rows / columns of its centres."""
    s = min(1.0, res / float(max(H, W)))
    hs, ws = max(1, int(round(H * s))), max(1, int(round(W * s)))
    ys = np.minimum(((np.arange(hs) + 0.5) * H / hs).astype(np.int64), H - 1)
    xs = np.minimum(((np.arange(ws) + 0.5) * W / ws).astype(np.int64), W - 1)
    return hs, ws, ys, xs


def _upsample_smooth(a: np.ndarray, H: int, W: int) -> np.ndarray:
    """Bicubic upsample of a coarse-grid field to (H, W), then a Gaussian of half the factor.

    A bicubic upsample is C1, but its curvature repeats in every coarse cell, which a strongly
    lit 4K render shows as a square grid; the blur removes it.
    """
    if a.shape == (H, W):
        return a.astype(np.float32, copy=True)
    up = cv2.resize(np.ascontiguousarray(a, np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    f = max(H / a.shape[0], W / a.shape[1])
    if f > 1.5:
        cv2.GaussianBlur(up, (0, 0), 0.5 * f, dst=up, borderType=cv2.BORDER_REPLICATE)
    return up


def _sample_up(src: np.ndarray, H: int, W: int, sy: slice, sx: slice) -> np.ndarray:
    """Bicubic sample of a coarse-grid field at the working pixel centres of a window."""
    hs, ws = src.shape
    ax, ay = ws / W, hs / H
    M = np.array([[ax, 0.0, (sx.start + 0.5) * ax - 0.5], [0.0, ay, (sy.start + 0.5) * ay - 0.5]])
    return cv2.warpAffine(
        np.ascontiguousarray(src, np.float32), M, (sx.stop - sx.start, sy.stop - sy.start),
        flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE,
    )


# --------------------------------------------------------------------------- basis


def build_sculpt_basis(
    pa: PartsAnalysis,
    layers_np: np.ndarray,
    M_sil: np.ndarray,
    shape: tuple[int, int],
    *,
    L_clean: np.ndarray | None = None,
) -> dict[str, Any]:
    """Cached Sculpt arrays on the working grid ``shape`` (see the module docstring).

    ``L_clean`` (working-grid luminance) gives the working-resolution line mask and the ink
    weight of strokes; without it the analysis-grid lines are nearest-upsampled at full ink.

    Public keys: ``labels`` (uint16), ``d_wall_px``, ``d_open_px``, ``d_gate_px``,
    ``d_crease_px``, ``d_crease_all_px``, ``d_thick_px`` (float16), ``dome`` (float32, 0..1),
    ``support`` (float32, part heights), ``ink`` (float16, 1 = full crease), ``layer`` (int32 per part,
    index 0 unused) and ``_sculpt_order_key``. Other ``_`` keys are engine-internal.
    """
    H, W = int(shape[0]), int(shape[1])
    sil = np.asarray(M_sil) > 0.5
    if sil.shape != (H, W):
        raise ValueError(f"M_sil shape {sil.shape} != {(H, W)}")
    h, w = pa.shape
    n = int(pa.labels.max())
    m = float(min(H, W))
    ys = np.minimum(((np.arange(H) + 0.5) * h / H).astype(np.int32), h - 1)
    xs = np.minimum(((np.arange(W) + 0.5) * w / W).astype(np.int32), w - 1)
    labels = labels_on_grid(pa, (H, W), sil)
    labels = labels.astype(np.uint16 if n < 65535 else np.int32)

    # working-resolution lines, each with the role of the nearest analysis-grid line pixel
    if L_clean is not None:
        L = np.asarray(L_clean, dtype=np.float32)
        if L.shape != (H, W):
            raise ValueError(f"L_clean shape {L.shape} != {(H, W)}")
        line = sil & (L < np.float32(WHITE_THRESHOLD))
        dark = np.where(sil, np.float32(1.0) - L, np.float32(0.0))
    else:
        line = sil & (pa.line_role[np.ix_(ys, xs)] > 0)
        dark = line.astype(np.float32)
    near = _nearest_line_index(pa)
    role = np.zeros((H, W), np.uint8)
    if near is not None:
        role_near = pa.line_role.ravel()[near]
        role[line] = role_near[np.ix_(ys, xs)][line]
    else:
        role[line] = ROLE_OPEN
    role[~sil] = ROLE_BG
    sep_flat = np.flatnonzero(role == ROLE_SEPARATING)
    sep_ana = ys[sep_flat // W].astype(np.int64) * w + xs[sep_flat % W]

    fg_bbox = _bbox(sil)
    sig = DIST_SMOOTH_FRAC * m
    basis: dict[str, Any] = {
        "labels": labels,
        "layer": _layer_vec(layers_np, n),
        "_shape": (H, W),
        "_sil": sil,
        "_sil_u8": sil.view(np.uint8),
        "_fg_bbox": fg_bbox,
        "_role": role,
        "_ys": ys,
        "_xs": xs,
        "_near": near,
        "_pa_labels": pa.labels,
        "_pa_r": line_radius(pa.shape),
        "_sep_flat": sep_flat,
        "_sep_ana": sep_ana,
        "_pairs": _label_pairs(labels),
        "_n_parts": n,
        "_r_work": max(1.0, line_radius(pa.shape) * max(H / h, W / w)),
        "_part_slices": ndi.find_objects(labels.astype(np.int32, copy=False), max_label=n) if n else [],
        # part areas on the fixed analysis grid (`_layer_lambda`): every working resolution, the
        # 1024 preview included, then picks the same per-layer step factors
        "_part_px": np.bincount(np.asarray(pa.labels).ravel(), minlength=n + 1).astype(np.int64),
        "_row_ext": _row_extents(sil),
        "_lock": threading.Lock(),
    }
    # order-independent fields
    basis["_sub_corr"] = _subpixel(None if L_clean is None else L, role > 0, fg_bbox, m)
    basis["_tmp_dglob"] = cv2.distanceTransform(np.ascontiguousarray(role == 0).view(np.uint8), cv2.DIST_L2,
                                                cv2.DIST_MASK_PRECISE)
    basis["d_open_px"] = _edt((role == ROLE_BG) | (role == ROLE_OUTLINE), fg_bbox, sil, sig, basis)
    creasable = (role == ROLE_OPEN) | (role == ROLE_SEPARATING)
    basis["d_crease_all_px"] = _edt(creasable, fg_bbox, sil, sig, basis)
    basis["ink"] = _ink(dark, creasable, fg_bbox, sil, m)
    dome, basis["_g_dome"] = _dome(sil)
    basis["dome"] = dome
    _rebuild_order_fields(basis, basis["layer"], groups_to_redo=None, support_from=None)
    basis.pop("_tmp_dglob", None)
    return basis


def _row_extents(sil: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per row, the first and one-past-last silhouette column (0, 0 for empty rows)."""
    any_ = sil.any(axis=1)
    first = np.where(any_, np.argmax(sil, axis=1), 0).astype(np.int32)
    last = np.where(any_, sil.shape[1] - np.argmax(sil[:, ::-1], axis=1), 0).astype(np.int32)
    return first, last


def _ink(dark: np.ndarray, creasable: np.ndarray, bbox, sil: np.ndarray, m: float) -> np.ndarray:
    """float16 crease weight per pixel (0..1): the ink weight of the nearest crease-able stroke.

    A stroke's ink weight is its darkness integrated across it (Gaussian line integral at sigma
    `INK_SIGMA_FRAC`), relative to a solid line `INK_REF_FRAC` wide, clipped to 1; spread to
    every pixel from the nearest stroke and blurred (`INK_BLUR_FRAC`) so it has no jumps. Stored
    as float16, not uint8: the 1/255 steps of a uint8 weight showed as contour lines across the
    crease next to every line in a lit 4K render (review round 4, the hatching beside seams).
    """
    H, W = sil.shape
    out = np.zeros((H, W), np.float16)
    if bbox is None or not creasable.any():
        return out
    r0, r1, c0, c1 = max(0, bbox[0] - 1), min(H, bbox[1] + 1), max(0, bbox[2] - 1), min(W, bbox[3] + 1)
    s = max(0.5, INK_SIGMA_FRAC * m)
    integral = cv2.GaussianBlur(np.ascontiguousarray(dark[r0:r1, c0:c1]), (0, 0), s) * np.float32(math.sqrt(2 * math.pi) * s)
    weight = np.clip(integral / np.float32(max(INK_REF_FRAC * m, 1e-3)), 0.0, 1.0)
    cw = np.ascontiguousarray(creasable[r0:r1, c0:c1])
    _, lab = cv2.distanceTransformWithLabels(
        np.ascontiguousarray(~cw).view(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_5, labelType=cv2.DIST_LABEL_PIXEL
    )
    spread = weight[cw][np.maximum(lab, 1) - 1].astype(np.float32)
    cv2.GaussianBlur(spread, (0, 0), max(0.5, INK_BLUR_FRAC * m), dst=spread)
    out[r0:r1, c0:c1] = np.clip(spread, 0.0, 1.0)
    out[~sil] = 0
    return out


def _dome(sil: np.ndarray) -> tuple[np.ndarray, float]:
    """Smooth whole-figure curvature 0..1 (float32: float16 steps would show as contour lines in a
    lit 4K render) and its largest gradient per working px."""
    H, W = sil.shape
    hs, ws, ysg, xsg = _grid(H, W, SUPPORT_RES)
    sg = np.ascontiguousarray(sil[np.ix_(ysg, xsg)])
    if not sg.any():
        return np.zeros((H, W), np.float32), 0.0
    mm = float(min(hs, ws))
    d = cv2.distanceTransform(np.pad(sg, 1).view(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    d = np.minimum(d, np.float32(DOME_CAP_FRAC * mm))
    d = cv2.GaussianBlur(d, (0, 0), sigmaX=max(0.5, DOME_SIGMA_FRAC * mm))
    mx = float(d[sg].max())
    if mx > 1e-9:
        d /= np.float32(mx)
    gy, gx = np.gradient(d)
    g = float(np.hypot(gx, gy)[sg].max()) / max(H / hs, W / ws)
    up = _upsample_smooth(d, H, W)
    np.clip(up, 0.0, 1.0, out=up)
    up[~sil] = 0.0
    return up, g


def _window(basis: dict, parts: np.ndarray, pad: int) -> tuple[slice, slice] | None:
    sl = [basis["_part_slices"][p - 1] for p in parts if basis["_part_slices"][p - 1] is not None]
    if not sl:
        return None
    H, W = basis["_shape"]
    r0 = max(0, min(s[0].start for s in sl) - pad)
    r1 = min(H, max(s[0].stop for s in sl) + pad)
    c0 = max(0, min(s[1].start for s in sl) - pad)
    c1 = min(W, max(s[1].stop for s in sl) + pad)
    return slice(r0, r1), slice(c0, c1)


def _group_parts(groups: np.ndarray, g: int) -> np.ndarray:
    return np.flatnonzero(groups[1:] == g) + 1


def _rebuild_order_fields(
    basis: dict, layer: np.ndarray, groups_to_redo: set[int] | None, support_from: int | None
) -> None:
    """(Re)derive the order-dependent fields: walls, gate, creases, thickness, support, seams.

    ``groups_to_redo`` = None rebuilds every wall field; otherwise only those level groups' (by
    root id under the new order). ``support_from`` = None rebuilds all support and seam caches,
    otherwise every layer from that one up.
    """
    H, W = basis["_shape"]
    sil = basis["_sil"]
    role = basis["_role"]
    m = float(min(H, W))
    mn_near, mx_near = _layer_near(basis, layer)
    basis["layer"] = layer
    basis["_sculpt_order_key"] = sculpt_order_key(layer)
    groups = _groups(layer, basis["_pairs"])
    basis["_groups"] = groups

    # creases: open strokes plus separating lines between parts on the same layer
    crease = role == ROLE_OPEN
    if mn_near is not None and len(basis["_sep_flat"]):
        lvl = mn_near.ravel()[basis["_sep_ana"]] == mx_near.ravel()[basis["_sep_ana"]]
        crease.ravel()[basis["_sep_flat"][lvl]] = True
    sig = DIST_SMOOTH_FRAC * m
    win = None
    if groups_to_redo is not None and "d_crease_px" in basis:
        # a reorder only changes creases on the affected groups' seams: recompute a window
        # around them, with a margin beyond which the crease has no effect (crease width <= 5 %)
        parts = np.concatenate([_group_parts(groups, g) for g in groups_to_redo] or [np.zeros(0, int)])
        win = _window(basis, parts, 0) if len(parts) else None
    if win is None:
        basis["d_crease_px"] = _edt(crease, basis["_fg_bbox"], sil, sig, basis)
    else:
        H_, W_ = basis["_shape"]
        mg = int(math.ceil(0.05 * m + 4 * sig)) + 2
        sy, sx = win
        oy, ox = slice(max(0, sy.start - mg), min(H_, sy.stop + mg)), slice(max(0, sx.start - mg), min(W_, sx.stop + mg))
        src = np.ascontiguousarray(~crease[oy, ox]).view(np.uint8)
        d = cv2.distanceTransform(src, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        d = _smooth(_apply_subpixel(d, basis, oy, ox), sig)
        iy, ix = slice(sy.start - oy.start, sy.stop - oy.start), slice(sx.start - ox.start, sx.stop - ox.start)
        basis["d_crease_px"][sy, sx] = np.where(sil[sy, sx], d[iy, ix], 0)

    if groups_to_redo is None or "d_wall_px" not in basis:
        basis["d_wall_px"] = basis["d_open_px"].copy()  # pixels without a part: the outline only
        basis["d_seam_px"] = np.full(sil.shape, _NO_SEAM, np.float16)
        basis["d_gate_px"] = np.full(sil.shape, _NO_SEAM, np.float16)
        redo = sorted({int(g) for g in groups[1:]})
    else:
        redo = sorted(groups_to_redo)
    pad = int(math.ceil(2 * basis["_r_work"] + 3 * GATE_SMOOTH_FRAC * m)) + 2
    jobs = [pp for pp in (_group_parts(groups, g) for g in redo) if len(pp)]
    walls = lambda pp: _group_walls(basis, pp, int(layer[pp[0]]), mn_near, pad)  # noqa: E731
    if H * W > _INLINE_PX and _N_WORKERS > 1 and len(jobs) > 1:
        # groups own disjoint pixels (each writes only its members), so they run concurrently
        list(_pool().map(walls, jobs))
    else:
        for pp in jobs:
            walls(pp)

    _thickness(basis, groups, None if groups_to_redo is None else redo)
    basis["_lam_part"], basis["_lam_layer"], basis["_lam_rows"] = _layer_lambda(basis, layer)
    _support(basis, layer, groups, support_from)
    # the seam correction's slope term at the default settings, with a margin: `_prepare`'s first
    # guess for the draft budget (a guess at least as large needs no second seam solve)
    basis["_g_corr_est"] = 1.25 * _seam_correct(basis, _defaults_P(basis))
    _restore_base(basis)


def _layer_lambda(basis: dict, layer: np.ndarray) -> tuple[np.ndarray, dict, np.ndarray | None]:
    """The per-layer step factor: (per part, float32 with index 0 = 1; per layer; rows holding a
    part with a factor < 1, or None when there is none).

    The topmost layers that together cover less than ``100 - KNEE_PCT`` % of the silhouette
    (never the lowest layer) get `TOP_LAYER_LAMBDA`; the kernel uses ``L * lambda`` there and the
    support / seam solves the level's own ``L * lambda``, so their feet stay exact. Areas are
    counted on the analysis-grid labels (``_part_px``), so the rule does not depend on the
    working resolution.
    """
    n = basis["_n_parts"]
    lam = np.ones(n + 1, np.float32)
    lam_layer: dict[int, float] = {}
    if n == 0:
        return lam, lam_layer, None
    px = basis["_part_px"][1 : n + 1].astype(np.float64)
    lay = np.asarray(layer[1 : n + 1])
    levels = np.unique(lay[px > 0])
    total = float(px.sum())
    cum = 0.0
    for lv in levels[::-1][:-1]:  # top down, never the lowest layer
        cum += float(px[lay == lv].sum())
        if cum >= (100.0 - KNEE_PCT) / 100.0 * total:
            break
        lam_layer[int(lv)] = TOP_LAYER_LAMBDA
    if not lam_layer:
        return lam, lam_layer, None
    top = np.isin(lay, list(lam_layer))
    lam[1:][top] = np.float32(TOP_LAYER_LAMBDA)
    rows = np.zeros(basis["_shape"][0], bool)
    for p in np.flatnonzero(top) + 1:
        sl = basis["_part_slices"][p - 1]
        if sl is not None:
            rows[sl[0]] = True
    return lam, lam_layer, rows


def _group_walls(basis: dict, parts: np.ndarray, lvl: int, mn_near, pad: int) -> None:
    win = _window(basis, parts, pad)
    if win is None:
        return
    sy, sx = win
    H, W = basis["_shape"]
    m = float(min(H, W))
    gm = np.zeros(basis["_n_parts"] + 1, bool)
    gm[parts] = True
    memb = gm[basis["labels"][sy, sx]]
    role_w = basis["_role"][sy, sx]
    # the background and drawn outline touching the group (not the background beyond a part in
    # front of it: the group carries on underneath that part) ...
    near = cv2.dilate(memb.view(np.uint8), _K5) > 0
    zero = ((role_w == ROLE_BG) | (role_w == ROLE_OUTLINE)) & near
    # ... and separating lines touching the group where it is in front (the whole line may lie
    # on the other part's side of the label boundary)
    seam = np.zeros_like(zero)
    sep = (role_w == ROLE_SEPARATING) & near
    if mn_near is not None and sep.any():
        ysw, xsw = basis["_ys"][sy], basis["_xs"][sx]
        seam = sep & (mn_near[np.ix_(ysw, xsw)] < lvl)
    sig = DIST_SMOOTH_FRAC * m
    dist = cv2.distanceTransform(np.ascontiguousarray(~zero).view(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dist = _smooth(_apply_subpixel(dist, basis, sy, sx), sig)
    np.copyto(basis["d_wall_px"][sy, sx], dist, where=memb, casting="same_kind")
    if seam.any():
        d_seam = cv2.distanceTransform(np.ascontiguousarray(~seam).view(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        ds = _smooth(_apply_subpixel(d_seam.copy(), basis, sy, sx), SEAM_SMOOTH_MUL * sig)
        np.copyto(basis["d_seam_px"][sy, sx], ds, where=memb, casting="same_kind")
        d_seam += basis["d_open_px"][sy, sx].astype(np.float32)
        d_seam = _blur_coarse(d_seam, GATE_SMOOTH_FRAC * m)
        np.minimum(d_seam, np.float32(_NO_SEAM), out=d_seam)
        np.copyto(basis["d_gate_px"][sy, sx], d_seam, where=memb, casting="same_kind")
    else:
        np.copyto(basis["d_seam_px"][sy, sx], _NO_SEAM, where=memb)
        np.copyto(basis["d_gate_px"][sy, sx], _NO_SEAM, where=memb)


def _blur_coarse(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur with a large sigma, done at a reduced resolution (sigma / f >= 2 px)."""
    f = max(1, int(sigma // 4))
    if f == 1:
        return cv2.GaussianBlur(a, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    h, w = a.shape
    small = cv2.resize(a, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / f, borderType=cv2.BORDER_REPLICATE)
    up = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    # a bicubic upsample's curvature repeats in every coarse cell (a grid in a lit 4K render)
    cv2.GaussianBlur(up, (0, 0), 0.5 * f, dst=up, borderType=cv2.BORDER_REPLICATE)
    return up


def _thickness(basis: dict, groups: np.ndarray, redo: list[int] | None = None) -> None:
    """``d_thick_px``: THICK_ALPHA x the local thickness (largest wall distance nearby).

    Per level group the wall distance is grey-dilated within the group, so a thin part is not
    measured by the thick part it lies on; the result is blended across groups with a small
    normalised blur (no jumps at seams) and upsampled bicubically. The per-group field is cached
    on the thickness grid; ``redo`` (group roots) limits the dilation to the groups that changed.
    """
    H, W = basis["_shape"]
    hs, ws, ysg, xsg = _grid(H, W, THICK_RES)
    lab = basis["labels"][np.ix_(ysg, xsg)].astype(np.int32)
    dw = np.minimum(basis["d_wall_px"][np.ix_(ysg, xsg)], basis["d_seam_px"][np.ix_(ysg, xsg)]).astype(np.float32)
    gid = groups[lab]
    T = basis.get("_thick_raw")
    if T is None or redo is None or T.shape != (hs, ws):
        T = basis["_thick_raw"] = np.zeros((hs, ws), np.float16)
        redo = np.unique(gid[lab > 0])
    mm = float(min(hs, ws))
    rho = max(1, int(round(THICK_RHO_FRAC * mm)))
    k = np.ones((2 * rho + 1, 2 * rho + 1), np.uint8)  # square: separable, fast
    for g in redo:
        msk = (gid == g) & (lab > 0)
        if not msk.any():
            continue
        msk = (gid == g) & (lab > 0)
        yy, xx = np.nonzero(msk)
        y0, y1 = max(0, yy.min() - rho), min(hs, yy.max() + rho + 1)
        x0, x1 = max(0, xx.min() - rho), min(ws, xx.max() + rho + 1)
        mw = msk[y0:y1, x0:x1]
        t = cv2.dilate(np.where(mw, dw[y0:y1, x0:x1], np.float32(0)).astype(np.float32), k)
        T[y0:y1, x0:x1][mw] = t[mw]
    inside = (lab > 0).astype(np.float32)
    sb = max(0.5, THICK_BLUR_FRAC * mm)
    num = cv2.GaussianBlur(T.astype(np.float32) * inside, (0, 0), sb)
    den = cv2.GaussianBlur(inside, (0, 0), sb)
    Tb = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0).astype(np.float32)
    up = _upsample_smooth(Tb, H, W)
    np.maximum(up, np.float32(1.0), out=up)
    up *= np.float32(THICK_ALPHA)
    basis["d_thick_px"] = np.minimum(up, np.float32(60000.0)).astype(np.float16)
    vals = Tb[lab > 0]
    basis["_thick_min"] = float(up[basis["_sil"]].min()) if basis["_sil"].any() else 0.0
    basis["_thick_p05"] = float(np.percentile(np.maximum(vals, 1.0), 5)) * THICK_ALPHA if vals.size else 0.0


# --------------------------------------------------------------------------- the height kernel


def _up(x: np.ndarray, buf: np.ndarray) -> np.ndarray:
    """``x`` (float16 / uint8 / float32) as float32 in ``buf``."""
    if x.dtype == np.float32:
        np.copyto(buf, x)
    else:
        cv2.add(x, 0.0, dst=buf, dtype=cv2.CV_32F)
    return buf


def _lut(u_scaled: np.ndarray, ib: np.ndarray, table: np.ndarray | None = None) -> np.ndarray:
    """Two-scale cushion of u, given u * _LUT_N, in place (``table`` defaults to `_LUT`)."""
    np.minimum(u_scaled, np.float32(_LUT_N), out=u_scaled)
    np.copyto(ib, u_scaled, casting="unsafe")
    np.take(_LUT if table is None else table, ib, out=u_scaled, mode="clip")  # "clip": no bounds check, 3x faster
    return u_scaled


def _smin_lut(a: np.ndarray, b, tmp: np.ndarray, ib: np.ndarray) -> np.ndarray:
    """``a`` <- smin(a, b) in place (b an array or a scalar), through `_LUT_PSI` in m / M."""
    np.maximum(a, b, out=tmp)  # M
    np.minimum(a, b, out=a)  # m
    cv2.divide(a, tmp, dst=tmp, scale=float(_LUT_N))  # (m / M) N, in [0, N]
    np.copyto(ib, tmp, casting="unsafe")
    np.take(_LUT_PSI, ib, out=tmp, mode="clip")
    np.multiply(a, tmp, out=a)
    return a


def _kernel(f: dict, P: tuple, bufs: list, z_out: np.ndarray | None, parts: bool = False):
    """The height model on matching arrays ``f`` (2-D chunks or 1-D gathers).

    ``f`` keys: dw, ds, do, dg, dc, dca, T (float16), dome, S (float32), ink (float16), and
    optionally lab (labels) with lamp (the per-part step factor, `_layer_lambda`): then Layering
    is ``L * lamp[lab]`` per pixel. Writes z into ``z_out``; with ``parts`` returns copies of
    (mix, g, q) instead, mix = L c_w + (1 - L) c_o, so that z = L S g + mix q (for the seam solve).
    """
    R, Rlo, Dlo, Wc, F, cd, L = P
    A, B, C, D, E, Fb, G, X, ib = bufs
    lam = f.get("lab") is not None
    if lam:  # X = L * lambda per pixel (a lookup by label: no per-pixel field is stored)
        np.take(f["lamp"], f["lab"], out=X, mode="clip")
        np.multiply(X, np.float32(L), out=X)
    blend = lam or L < 1.0 - 1e-6
    N = np.float32(_LUT_N)
    _up(f["T"], A)
    _smin_lut(A, np.float32(R), D, ib)  # Re = max(smin(d_thick, R), R_lo): no contour at d_thick = R
    np.maximum(A, np.float32(min(Rlo, R)), out=A)
    _up(f["do"], C)
    np.divide(N, A, out=G)  # N / Re
    _up(f["dw"], B)
    np.multiply(B, G, out=B)
    _lut(B, ib)  # outline walls
    _up(f["ds"], D)
    np.multiply(D, G, out=D)
    _lut(D, ib, _LUT_S)  # own front seams: steep foot
    np.multiply(B, D, out=B)  # c_w = f f_s: C1 where both distances are smooth (a min creases)
    if blend:
        np.multiply(C, G, out=D)
        _lut(D, ib)  # c_o
        if lam:  # wall mix c_o + L (c_w - c_o)
            cv2.subtract(B, D, dst=B)
            cv2.multiply(B, X, dst=B)
            cv2.add(B, D, dst=B)
        else:
            cv2.addWeighted(B, L, D, 1.0 - L, 0.0, dst=B)  # wall mix
    # gate g = f(d_open / max(smin(Re, d_gate), D_lo)): no contour where d_gate = Re
    _up(f["dg"], E)
    _smin_lut(E, A, D, ib)
    np.maximum(E, np.float32(Dlo), out=E)
    np.divide(C, E, out=E)
    np.multiply(E, N, out=E)
    _lut(E, ib)
    # creases k = ink * max(1 - s(d_crease / Wc), (1 - L)(1 - s(d_crease_all / Wc)))
    sW = _HALF_PI / np.float32(Wc)
    _up(f["dc"], Fb)
    np.multiply(Fb, sW, out=Fb)
    np.minimum(Fb, _HALF_PI, out=Fb)
    np.sin(Fb, out=Fb)
    cv2.addWeighted(Fb, -1.0, Fb, 0.0, 1.0, dst=Fb)
    if blend:
        _up(f["dca"], G)
        np.multiply(G, sW, out=G)
        np.minimum(G, _HALF_PI, out=G)
        np.sin(G, out=G)
        if lam:  # (1 - L)(1 - s) = (1 - s) - L (1 - s)
            cv2.addWeighted(G, -1.0, G, 0.0, 1.0, dst=G)
            cv2.multiply(G, X, dst=D)
            cv2.subtract(G, D, dst=G)
        else:
            cv2.addWeighted(G, -(1.0 - L), G, 0.0, 1.0 - L, dst=G)
        np.maximum(Fb, G, out=Fb)
    _up(f["ink"], G)
    cv2.multiply(Fb, G, dst=Fb, scale=cd)  # cd * k
    _up(f["dome"], G)
    cv2.addWeighted(G, 1.0 - F, Fb, -1.0, F, dst=G)  # q
    if parts:
        return B.copy(), E.copy(), G.copy()
    np.multiply(G, B, out=G)  # mix * q
    np.multiply(f["S"], E, out=E)  # S * g
    if lam:
        np.multiply(E, X, out=E)
        cv2.add(E, G, dst=z_out)
    else:
        cv2.addWeighted(E, L, G, 1.0, 0.0, dst=z_out)
    return None


def _point_bufs(n: int) -> list:
    return [np.empty(n, np.float32) for _ in range(8)] + [np.empty(n, np.int32)]


# OpenCV's Python bindings read a 1-D array of <= 4 elements as a Scalar, so a kernel run on a
# gather of 1-4 points returns wrong values without an error (task-4 review round 4); point
# gathers are padded to at least this many points (repeating the first) and sliced back
_MIN_GATHER = 8


def _pad_idx(idx: np.ndarray) -> np.ndarray:
    idx = np.asarray(idx)
    if 0 < len(idx) < _MIN_GATHER:
        return np.concatenate([idx, np.full(_MIN_GATHER - len(idx), idx[0], idx.dtype)])
    return idx


def _fields_at(basis: dict, idx: np.ndarray) -> dict:
    g = lambda key: basis[key].ravel()[idx]  # noqa: E731
    f = {
        "dw": g("d_wall_px"), "ds": g("d_seam_px"), "do": g("d_open_px"), "dg": g("d_gate_px"), "dc": g("d_crease_px"),
        "dca": g("d_crease_all_px"), "dome": g("dome"), "T": g("d_thick_px"), "S": g("support"), "ink": g("ink"),
    }
    if basis.get("_lam_rows") is not None:
        f["lab"], f["lamp"] = g("labels"), basis["_lam_part"]
    return f


def _z_at(basis: dict, idx: np.ndarray, P: tuple) -> np.ndarray:
    """Raw z at flat pixel indices ``idx`` (any count, padded: `_MIN_GATHER`)."""
    n = len(idx)
    if not n:
        return np.empty(0, np.float32)
    ip = _pad_idx(idx)
    out = np.empty(len(ip), np.float32)
    _kernel(_fields_at(basis, ip), P, _point_bufs(len(ip)), out)
    return out[:n] if len(ip) != n else out


def _parts_at(basis: dict, idx: np.ndarray, P: tuple):
    """(mix, g, q) at flat pixel indices ``idx`` (any count, padded: `_MIN_GATHER`)."""
    n = len(idx)
    if not n:
        return tuple(np.empty(0, np.float32) for _ in range(3))
    ip = _pad_idx(idx)
    res = _kernel(_fields_at(basis, ip), P, _point_bufs(len(ip)), None, parts=True)
    return res if len(ip) == n else tuple(a[:n].copy() for a in res)


# --------------------------------------------------------------------------- support


def _defaults_P(basis: dict) -> tuple:
    return _radii(basis, SCULPT_DEFAULTS, 1.0, 0.0)[0]


def _support(basis: dict, layer: np.ndarray, groups: np.ndarray, from_layer: int | None) -> None:
    """Base support per level group (harmonic, at the default settings) and the seam caches."""
    H, W = basis["_shape"]
    labels = basis["labels"]
    if "support" not in basis or from_layer is None:
        basis["support"] = np.zeros((H, W), np.float32)
        basis["_seams"] = {}
        basis["_g_support"] = {}
        basis["_gs_px"] = {}
    sup = basis["support"]
    n = basis["_n_parts"]
    if n == 0:
        return
    _restore_base(basis)
    P0 = _defaults_P(basis)
    L0 = P0[6]
    roots = np.unique(groups[1:])
    glayer = {int(g): int(layer[g]) for g in roots}
    levels = sorted(set(glayer.values()))
    lo_layer = levels[0]
    if from_layer is not None:
        levels = [lv for lv in levels if lv >= from_layer]
    for store in ("_seams", "_g_support", "_gs_px"):
        for lv in list(basis[store]):
            if from_layer is None or lv >= from_layer:
                basis[store].pop(lv, None)

    hs, ws, ysg, xsg = _grid(H, W, SUPPORT_RES)
    flat_sg = ysg[:, None] * W + xsg[None, :]
    lab_sg = labels[np.ix_(ysg, xsg)].astype(np.int32)
    lay_px = layer[lab_sg]
    scale = max(H / hs, W / ws)
    for lv in levels:
        gs = [int(g) for g in roots if glayer[int(g)] == lv]
        lam_lv = float(basis["_lam_layer"].get(lv, 1.0))
        Llv = L0 * lam_lv  # the level's own step (`_layer_lambda`)
        if lv == lo_layer:
            for g in gs:
                parts = _group_parts(groups, g)
                win = _window(basis, parts, 0)
                if win is not None:
                    gm = np.zeros(n + 1, bool)
                    gm[parts] = True
                    np.copyto(sup[win], 0.0, where=gm[labels[win]])
            continue
        # 1. coarse Laplace solve for every group on this layer at once (they never touch); the
        #    region is dilated a little so the field continues smoothly past the group's edge
        gm_l = np.zeros(n + 1, bool)
        for g in gs:
            gm_l[_group_parts(groups, g)] = True
        region = gm_l[lab_sg] & (lab_sg > 0)
        lower = (lab_sg > 0) & (lay_px < lv)
        best = np.full((hs, ws), -np.inf, np.float32)
        if lower.any() and region.any():
            li = flat_sg[lower]
            z = _z_at(basis, li, P0)
            # the support that puts a front foot (c_w = 0) on this surface, taking c_o and q
            # from the lower side; a smooth first guess, the per-recompute seam solve is exact
            Pc = (*P0[:6], 0.0)  # with L = 0 the "mix" is c_o
            co, _g, q = _parts_at(basis, li, Pc)
            q_low = np.full((hs, ws), -np.inf, np.float32)
            q_low[lower] = (z - np.float32(1 - Llv) * co * q) / np.float32(max(Llv, 0.05))
            best = _nbr_max(q_low)
        fixed = region & np.isfinite(best)
        grown = cv2.dilate(region.view(np.uint8), np.ones((3, 3), np.uint8), iterations=3) > 0
        field = _laplace(grown, fixed, np.where(fixed, best, 0.0).astype(np.float64))
        ext = _extend(field, grown)
        # draft budget: the support's slope reaches the surface times the gate g, which is small
        # where S is steep (seam ends by the background); keep |grad S| per working px for a
        # bound max(g |grad S|) at the current settings, and a conservative one at the defaults
        # (the clamp only enlarges the radii, which lowers g)
        gy, gx = np.gradient(ext)
        inner = cv2.erode(region.view(np.uint8), np.ones((3, 3), np.uint8)) > 0
        if inner.any():
            # (times the level's step factor: its support term is L lambda S g)
            gmag = (np.hypot(gx, gy)[inner] * (lam_lv / scale)).astype(np.float32)
            gidx = flat_sg[inner]
            basis["_gs_px"][lv] = (gidx, gmag)
            g_in = _parts_at(basis, gidx, P0)[1]
            basis["_g_support"][lv] = float((gmag * np.minimum(g_in + 0.1, 1.0)).max())
        # 2. per group: bicubic upsample of the base, then the seam cache
        one = lambda g: _group_support(basis, _group_parts(groups, g), lv, ext)  # noqa: E731
        if H * W > _INLINE_PX and _N_WORKERS > 1 and len(gs) > 1:  # disjoint member pixels
            entries = [e for e in _pool().map(one, gs) if e is not None]
        else:
            entries = [e for e in map(one, gs) if e is not None]
        if entries:
            basis["_seams"][lv] = _merge_seams(entries, basis, (hs, ws))


def _laplace(region: np.ndarray, fixed: np.ndarray, val: np.ndarray) -> np.ndarray:
    """Harmonic function on ``region`` (4-neighbour graph Laplacian), Dirichlet on ``fixed``,
    natural (Neumann) elsewhere. Components without a fixed pixel rest on nothing (0)."""
    out = np.zeros(region.shape, np.float64)
    out[fixed] = val[fixed]
    unk = region & ~fixed
    if not unk.any():
        return out
    ncomp, comp = cv2.connectedComponents(np.ascontiguousarray(region).view(np.uint8), connectivity=4)
    has_fixed = np.zeros(ncomp, bool)
    has_fixed[np.unique(comp[fixed])] = True
    unk &= has_fixed[comp]
    m = int(unk.sum())
    if m == 0:
        return out
    hs, ws = region.shape
    idx = np.full((hs + 2, ws + 2), -1, np.int64)
    idx[1:-1, 1:-1][unk] = np.arange(m)
    Rg = np.pad(region, 1)
    Fx = np.pad(fixed, 1)
    V = np.pad(np.where(fixed, val, 0.0), 1)
    uy, ux = np.nonzero(unk)  # raster order == idx order
    pi = np.arange(m)
    diag = np.zeros(m)
    rhs = np.zeros(m)
    rows, cols = [], []
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        qy, qx = uy + 1 + dy, ux + 1 + dx
        inr = Rg[qy, qx]
        diag += inr
        fx = inr & Fx[qy, qx]
        rhs[fx] += V[qy[fx], qx[fx]]
        un = inr & ~Fx[qy, qx]
        rows.append(pi[un])
        cols.append(idx[qy[un], qx[un]])
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    A = sparse.csc_matrix((np.r_[diag, -np.ones(len(r))], (np.r_[pi, r], np.r_[pi, c])), shape=(m, m))
    out[unk] = spsolve(A, rhs)
    return out


def _extend(field: np.ndarray, region: np.ndarray) -> np.ndarray:
    """``field`` on ``region``, extended outward with the nearest region value (float32)."""
    if not region.any():
        return np.zeros(field.shape, np.float32)
    _, lab = cv2.distanceTransformWithLabels(
        np.ascontiguousarray(~region).view(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_5, labelType=cv2.DIST_LABEL_PIXEL
    )
    vals = field[region].astype(np.float32)
    return vals[np.maximum(lab, 1) - 1]


def _group_support(basis: dict, parts: np.ndarray, lv: int, ext: np.ndarray) -> dict | None:
    """Write the group's base support; return its seam cache (pixels, neighbours, band)."""
    H, W = basis["_shape"]
    m = float(min(H, W))
    labels = basis["labels"]
    layer = basis["layer"]
    win = _window(basis, parts, 1)
    if win is None:
        return None
    sy, sx = win
    hw, ww = sy.stop - sy.start, sx.stop - sx.start
    gm = np.zeros(basis["_n_parts"] + 1, bool)
    gm[parts] = True
    lab_w = labels[sy, sx]
    memb = gm[lab_w]
    if not memb.any():
        return None
    S_up = _sample_up(ext, H, W, sy, sx)
    f_up = max(H / ext.shape[0], W / ext.shape[1])
    if f_up > 1.5:  # no per-cell curvature pattern (see `_upsample_smooth`)
        cv2.GaussianBlur(S_up, (0, 0), 0.5 * f_up, dst=S_up, borderType=cv2.BORDER_REPLICATE)
    np.copyto(basis["support"][sy, sx], S_up, where=memb)
    lower = (lab_w > 0) & (layer[lab_w] < lv)
    if not lower.any():
        return None
    # seam pixels: members 4-adjacent to a lower part; their (up to 4) lower neighbours
    Pd = np.pad(lower, 1)
    nbs = [Pd[1 + dy : 1 + dy + hw, 1 + dx : 1 + dx + ww] for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))]
    seam = memb & (nbs[0] | nbs[1] | nbs[2] | nbs[3])
    if not seam.any():
        return None
    yy, xx = np.nonzero(seam)
    gy, gx = yy + sy.start, xx + sx.start
    ny = np.full((len(yy), 4), -1, np.int64)
    for j, (dy, dx) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
        ok = nbs[j][yy, xx]
        ny[ok, j] = (gy[ok] + dy).astype(np.int64) * W + (gx[ok] + dx)
    # the next pixel beyond each lower neighbour, where it is the same part (for the extrapolation)
    labels_f = labels.ravel()
    ny2 = np.full((len(yy), 4), -1, np.int64)
    for j, (dy, dx) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
        y2, x2 = gy + 2 * dy, gx + 2 * dx
        i = np.flatnonzero(nbs[j][yy, xx] & (y2 >= 0) & (y2 < H) & (x2 >= 0) & (x2 < W))
        n2 = y2[i].astype(np.int64) * W + x2[i]
        i_ok = labels_f[n2] == labels_f[ny[i, j]]
        ny2[i[i_ok], j] = n2[i_ok]
    # band: other members within SEAM_BAND_FRAC of the seam, with a weight fading to 0
    band_w = max(1.0, SEAM_BAND_FRAC * m)
    by0, by1 = max(0, int(yy.min() - band_w - 1)), min(hw, int(yy.max() + band_w + 2))
    bx0, bx1 = max(0, int(xx.min() - band_w - 1)), min(ww, int(xx.max() + band_w + 2))
    sub = np.ascontiguousarray(seam[by0:by1, bx0:bx1])
    d, lab = cv2.distanceTransformWithLabels(
        np.ascontiguousarray(~sub).view(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_5, labelType=cv2.DIST_LABEL_PIXEL
    )
    order = np.full(seam.shape, -1, np.int64)
    order[seam] = np.arange(len(yy))  # raster order of the window == raster order of the sub-window
    seam_ids = order[by0:by1, bx0:bx1][sub]
    band = memb[by0:by1, bx0:bx1] & ~sub & (d < band_w)
    bys, bxs = np.nonzero(band)
    fix_w = max(2.0, SEAM_FIX_FRAC * m)
    return {
        "band_seam": seam_ids[np.maximum(lab[bys, bxs], 1) - 1],
        "fix_w": (np.float32(1.0) - cushion(d[bys, bxs] / np.float32(fix_w))).astype(np.float32),
        "x": gy.astype(np.int64) * W + gx, "ny": ny, "ny2": ny2, "xy": np.column_stack([gy, gx]),
        "band": (bys + by0 + sy.start).astype(np.int64) * W + (bxs + bx0 + sx.start),
        "bxy": np.column_stack([bys + by0 + sy.start, bxs + bx0 + sx.start]),
        "band_w": (np.float32(1.0) - cushion(d[bys, bxs] / np.float32(band_w))).astype(np.float32),
    }


def _merge_seams(entries: list, basis: dict, sg_shape: tuple[int, int]) -> dict:
    """One layer's seam caches, with the support-grid binning used to spread the residual."""
    H, W = basis["_shape"]
    hs, ws = sg_shape
    cat = lambda k: np.concatenate([e[k] for e in entries])  # noqa: E731
    x, band, xy, bxy = cat("x"), cat("band"), cat("xy"), cat("bxy")
    offs = np.cumsum([0] + [len(e["x"]) for e in entries[:-1]])
    band_seam = np.concatenate([e["band_seam"] + o for e, o in zip(entries, offs)])

    def sg_coords(yx):  # working pixel centres in support-grid pixel coordinates, for cv2.remap
        sx_ = ((yx[:, 1] + 0.5) * ws / W - 0.5).astype(np.float32)
        sy_ = ((yx[:, 0] + 0.5) * hs / H - 0.5).astype(np.float32)
        return sx_, sy_

    def block(sx_, sy_):  # cv2.remap wants each map side < 32767: rows of _REMAP_W points
        n_ = len(sx_)
        rows_ = max(1, -(-n_ // _REMAP_W))
        bx = np.zeros(rows_ * _REMAP_W, np.float32)
        by = np.zeros(rows_ * _REMAP_W, np.float32)
        bx[:n_], by[:n_] = sx_, sy_
        return bx.reshape(rows_, _REMAP_W), by.reshape(rows_, _REMAP_W), n_

    sxx, syx = sg_coords(xy)
    cell = np.clip(np.rint(syx), 0, hs - 1).astype(np.int64) * ws + np.clip(np.rint(sxx), 0, ws - 1).astype(np.int64)
    cnt = np.bincount(cell, minlength=hs * ws).reshape(hs, ws).astype(np.float64)
    den = cv2.GaussianBlur(cnt, (0, 0), SEAM_SPREAD_SG)
    sup = basis["support"].ravel()
    return {
        "x": x, "ny": cat("ny"), "ny2": cat("ny2"), "cell": cell, "den": den, "map_x": block(sxx, syx), "map_b": block(*sg_coords(bxy)),
        "band": band, "band_w": cat("band_w"), "band_seam": band_seam, "fix_w": cat("fix_w"),
        "base_x": sup[x].copy(), "base_band": sup[band].copy(),
    }


def _restore_base(basis: dict) -> None:
    """Undo the last recompute's seam corrections (support back to its cached base)."""
    sup = basis["support"].ravel()
    for c in basis.get("_seams", {}).values():
        sup[c["band"]] = c["base_band"]
        sup[c["x"]] = c["base_x"]


def _remap(img: np.ndarray, maps: tuple) -> np.ndarray:
    """Bicubic samples of a support-grid field at cached scattered points."""
    mx, my, n = maps
    if not n:
        return np.zeros(0, np.float32)
    return cv2.remap(img, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE).ravel()[:n]


def _seam_correct(basis: dict, P: tuple) -> float:
    """Make every wall foot sit on the surface behind at the current settings.

    Layer by layer (lowest first, so the surfaces behind are already corrected): evaluate the
    highest lower neighbour's z at each seam pixel, extrapolated to the seam pixel along the
    neighbour's own part (`SEAM_EXTRAP_CLIP`), solve ``L S g + mix q = z_behind`` (the
    front pixel's actual height) for the support, and spread the residual against the
    cached base smoothly: binned on the support grid, normalised-Gaussian blurred, sampled
    bicubically at the seam and band pixels, faded out across the band. Returns a bound on the
    slope (part heights per px, times the gate) that the correction adds.
    """
    L = P[6]
    seams = basis.get("_seams", {})
    if not seams:
        return 0.0
    sup = basis["support"].ravel()
    if L < 1e-3:
        _restore_base(basis)
        return 0.0
    H, W = basis["_shape"]
    band_w = max(1.0, SEAM_BAND_FRAC * float(min(H, W)))
    hs, ws = next(iter(seams.values()))["den"].shape
    sig_px = SEAM_SPREAD_SG * max(H / hs, W / ws)
    grad = 0.0
    F = P[4]
    for lv in sorted(seams):
        c = seams[lv]
        ny = c["ny"]
        valid = ny >= 0
        zb = np.full(ny.shape, -np.inf, np.float32)
        zb[valid] = _z_at(basis, ny[valid], P)
        j = zb.argmax(axis=1)
        rows = np.arange(len(j))
        best = zb[rows, j]
        # extrapolate the highest surface behind to the seam pixel itself (see SEAM_EXTRAP_CLIP)
        n2 = c["ny2"][rows, j]
        e = np.flatnonzero(n2 >= 0)
        if len(e):
            ex = best[e] - _z_at(basis, n2[e], P)
            lim = np.float32(SEAM_EXTRAP_CLIP) * (np.float32(F) + np.float32(1.0 - F) * basis["dome"].ravel()[c["x"][e]])
            best[e] += np.clip(ex, -lim, lim)
        mix, g, q = _parts_at(basis, c["x"], P)
        lam_x = basis["_lam_part"][basis["labels"].ravel()[c["x"]]]  # the front part's step factor
        Lx = np.float32(L) * lam_x
        target = (best - mix * q) / (Lx * np.maximum(g, np.float32(_GATE_FLOOR)))
        # guards against the gate vanishing at the silhouette edge, scaled by the gate itself
        cap = np.float32(SEAM_TARGET_CAP) * np.maximum(best, np.float32(F)) / (
            np.maximum(Lx, np.float32(0.05)) * np.maximum(g, np.float32(_GATE_FLOOR))
        )
        np.clip(target, -cap, cap, out=target)  # negative: the part's own level term is already higher
        r = (target - c["base_x"]).astype(np.float64)
        acc = np.bincount(c["cell"], weights=r, minlength=c["den"].size).reshape(c["den"].shape)
        num = cv2.GaussianBlur(acc, (0, 0), SEAM_SPREAD_SG)
        v = np.where(c["den"] > 1e-9, num / np.maximum(c["den"], 1e-300), 0.0).astype(np.float32)
        vx = _remap(v, c["map_x"])
        fix = (r.astype(np.float32) - vx)  # exact on the seam: the smooth spread's error there ...
        sup[c["band"]] = (c["base_band"] + _remap(v, c["map_b"]) * c["band_w"]
                          + fix[c["band_seam"]] * c["fix_w"])  # ... fades out within SEAM_FIX_FRAC
        sup[c["x"]] = target
        if len(vx):  # weighted by the gate, as for the base support (see `_support`)
            gw = np.minimum(g + 0.1, 1.0) * lam_x
            grad = max(grad, float((np.abs(vx) * gw).max()) * (float(_HALF_PI) / band_w + 1.0 / sig_px)
                       + float((np.abs(fix) * gw).max()) * float(_HALF_PI) / max(2.0, SEAM_FIX_FRAC * float(min(H, W))))
    return grad


# --------------------------------------------------------------------------- reorder


# Entries `build_sculpt_basis` derives from the drawing alone. Neither a reorder (`rebuild_parts`)
# nor a recompute (`sculpt_profile`) writes them, so `copy_for_reorder` shares them.
_ORDER_INDEPENDENT = frozenset({
    "labels", "_shape", "_sil", "_sil_u8", "_fg_bbox", "_role", "_ys", "_xs", "_near", "_pa_labels",
    "_pa_r", "_sep_flat", "_sep_ana", "_pairs", "_n_parts", "_r_work", "_part_slices", "_part_px",
    "_row_ext", "_sub_corr", "d_open_px", "d_crease_all_px", "ink", "dome", "_g_dome",
})


def copy_for_reorder(basis: dict) -> dict:
    """A private copy of ``basis`` for `rebuild_parts` and recomputes to work on while the
    original stays in use (and unchanged): the order-dependent entries and the seam caches a
    recompute writes are copied, the order-independent arrays shared (about two thirds of the
    bytes; the copy costs ~13 bytes per working pixel)."""
    with basis.setdefault("_lock", threading.Lock()):
        out = {k: (v if k in _ORDER_INDEPENDENT else copy.deepcopy(v)) for k, v in basis.items() if k != "_lock"}
    out["_lock"] = threading.Lock()
    return out


def rebuild_parts(basis: dict, part_ids: list[int], layers_np: np.ndarray) -> None:
    """Re-derive walls, gate, creases, thickness and support after a reorder.

    Walls are rebuilt for the level groups (under the new order) that contain a listed part, a
    part whose layer changed, or a neighbour of either; support and seam caches for every layer
    from the lowest affected one up (parts in front rest on what moved, so they follow it).
    """
    n = basis["_n_parts"]
    new = _layer_vec(layers_np, n)
    old = basis["layer"]
    changed = {int(p) for p in part_ids if 1 <= int(p) <= n}
    changed |= set(np.flatnonzero(new != old).tolist())
    changed.discard(0)
    if not changed:
        basis["layer"] = new
        basis["_sculpt_order_key"] = sculpt_order_key(new)
        return
    affected = set(changed)
    for a, b in basis["_pairs"].tolist():
        if a in changed:
            affected.add(b)
        if b in changed:
            affected.add(a)
    groups = _groups(new, basis["_pairs"])
    redo = {int(groups[p]) for p in affected}
    lo = min(min(int(old[p]) for p in affected), min(int(new[p]) for p in affected))
    # a part whose step factor changes (`_layer_lambda`) needs its layer's support re-solved
    lam_new = _layer_lambda(basis, new)[0]
    moved = np.flatnonzero(lam_new != basis["_lam_part"])
    if len(moved):
        lo = min(lo, int(new[moved].min()))
    _restore_base(basis)
    basis["_tmp_dglob"] = cv2.distanceTransform(np.ascontiguousarray(basis["_role"] == 0).view(np.uint8), cv2.DIST_L2,
                                                cv2.DIST_MASK_PRECISE)
    try:
        _rebuild_order_fields(basis, new, groups_to_redo=redo, support_from=lo)
    finally:
        basis.pop("_tmp_dglob", None)


# --------------------------------------------------------------------------- profile


def _ceil_geo(x: float, step: float = 1.1) -> float:
    """``x`` rounded up to the grid ``step ** n`` (x > 0)."""
    return step ** math.ceil(math.log(x) / math.log(step) - 1e-9)


def _radii(
    basis: dict, params: Any, physical_width_mm: float, relief_z_mm: float, g_support: float = 0.0, span: float = 1.0
):
    """((R, R_lo, D_lo, Wc, F, cd, L), clamped) in working pixels.

    Draft clamp (decision 3): every interior slope must lean at least ``draft_angle_deg`` from
    vertical. In raw units (part heights per px) the allowed slope is ``k0 * span`` with
    ``k0 = cot(theta) * px_mm / (relief_z_mm * (1 - h_min) * _KNEE_GAIN)`` (the tone curve
    multiplies slopes by at most `_KNEE_GAIN`). The slope is bounded by
      q * (L (FMAX + FMAX_s) + (1 - L) FMAX) / Re  the wall mix: c_w = f f_s has
                                                   |grad c_w| <= (FMAX + FMAX_s) / Re, c_o FMAX / Re
      + L S FMAX / D                               the support gate (<= span * ...)
      + cd * pi/2 / Wc                             creases
      + (1 - F) * |grad dome| + L * |grad S|       dome and support
    (with a per-layer step factor lambda <= 1, L lambda replaces L per pixel, which only shrinks
    each term; ``g_support`` already carries lambda, `_support`)
    so the crease width gets a fixed share and the dome and support gradients what they need
    (these scale with ``span``, assumed >= 1 and re-run with the real span when it is smaller),
    and the walls and gate the rest: R_lo = (FMAX + L FMAX_s) / (share k0), D_lo >= FMAX / (share k0).
    A lower bound that cannot bind (below the thinnest part's radius, or the gate floor) is
    returned as its canonical value, so equal outputs come with equal parameters.
    """
    H, W = basis["_shape"]
    m = float(min(H, W))
    F, cd, L = _shape_params(params)
    pct = float(_get(params, "edge_roundness_pct", SCULPT_DEFAULTS["edge_roundness_pct"]))
    R_user = max(1e-3, pct / 100.0 * m)
    cw = float(_get(params, "crease_width_pct", SCULPT_DEFAULTS["crease_width_pct"]))
    Wc_user = max(1e-3, cw / 100.0 * m)
    Dlo = max(1e-3, GATE_MIN_FRAC * m)
    theta = min(float(_get(params, "draft_angle_deg", 0.0)), 89.0)
    R_lo, Wc = 1e-3, Wc_user
    clamped = False
    if theta > 1e-3:
        h_min = max(0.0, min(float(_get(params, "h_min", 0.08)), 0.95))
        px_mm = max(float(physical_width_mm), 1e-6) / float(max(W, 1))
        k0 = px_mm / (math.tan(math.radians(theta)) * max(float(relief_z_mm), 1e-6) * (1.0 - h_min) * _KNEE_GAIN)
        ks = k0 * min(1.0, span)
        Wc_min = cd * (math.pi / 2.0) / (_CREASE_SHARE * ks)
        Wc = max(Wc_user, Wc_min)
        used = cd * (math.pi / 2.0) / Wc + (1.0 - F) * float(basis.get("_g_dome", 0.0)) + L * g_support
        share = max(0.25, 1.0 - used / ks) * 0.95
        # rounded up to a 10 % geometric grid (only ever more conservative), so the seam
        # correction's small effect on the support share rarely changes the radii and forces
        # a second seam solve in `_prepare`
        R_lo = _ceil_geo((_FMAX + L * _FMAX_S) / (share * k0))
        Dlo = max(Dlo, _ceil_geo(_FMAX / (share * k0)))
        clamped = R_lo > min(R_user, float(basis.get("_thick_p05", R_user))) or Wc > Wc_user
        if R_lo <= _SMIN_K * min(float(basis.get("_thick_min", 0.0)), R_user):
            # max(smin(d_thick, R), R_lo) cannot bind: smin(T, R) >= _SMIN_K min(T, R) and
            # T >= thick_min, R >= R_user (task-4 review round 4: thick_min alone is not enough)
            R_lo = 1e-3
    R = max(R_user, R_lo)
    return (R, R_lo, Dlo, Wc, F, cd, L), bool(clamped)


def sculpt_radii(basis: dict, params: Any, physical_width_mm: float, relief_z_mm: float) -> tuple[float, float, bool]:
    """(R_px, Wc_px, clamped): edge roundness (outer shoulder) and crease width in working px.

    The steep foot of the wall is ``CUSHION_RHO * R``. Under the draft clamp the radius actually
    used per pixel, `edge_radius` ``max(smin(d_thick_px, R), R_lo)``, keeps every slope within
    the draft angle.
    """
    P, clamped = _radii(basis, params, physical_width_mm, relief_z_mm, max(basis.get("_g_support", {}).values(), default=0.0))
    return P[0], P[3], clamped


def _scratch(n: int) -> list[np.ndarray]:
    """Per-thread preallocated flat chunk buffers (reshaped per chunk, so views stay contiguous)."""
    cache = getattr(_TLS, "bufs", None)
    if cache is None or cache[0] < n:
        cache = _TLS.bufs = (n, [np.empty(n, np.float32) for _ in range(8)] + [np.empty(n, np.int32)])
    return cache[1]


def _chunk_cols(basis: dict, r: int, r2: int, c0: int, c1: int) -> tuple[int, int]:
    """Columns [a, b) holding every silhouette pixel of rows [r, r2) (inside the ROI)."""
    lo, hi = basis["_row_ext"]
    a, b = int(lo[r:r2].min()), int(hi[r:r2].max())
    return (max(a, c0), min(b, c1)) if b > a else (c0, c0)


def _raw_rows(basis, out, y0, y1, c0, c1, P) -> tuple[float, float]:
    """Raw z for rows [y0, y1) of the ROI into ``out``; returns the masked (min, max).

    Each chunk covers only the columns its rows have silhouette pixels in; the rest is 0.
    """
    cols = c1 - c0
    ch = max(1, _CHUNK_PX // max(cols, 1))
    full = _scratch(ch * cols)
    sil = basis["_sil_u8"]
    keys = (("dw", "d_wall_px"), ("ds", "d_seam_px"), ("do", "d_open_px"), ("dg", "d_gate_px"), ("dc", "d_crease_px"),
            ("dca", "d_crease_all_px"), ("dome", "dome"), ("T", "d_thick_px"), ("S", "support"), ("ink", "ink"))
    lam_rows = basis.get("_lam_rows")
    lo, hi = math.inf, -math.inf
    for r in range(y0, y1, ch):
        r2 = min(y1, r + ch)
        k = r2 - r
        a, b = _chunk_cols(basis, r, r2, c0, c1)
        out[r:r2, c0:a] = 0.0
        out[r:r2, b:c1] = 0.0
        if b <= a:
            continue
        z = out[r:r2, a:b]
        msk = sil[r:r2, a:b]
        f = {key: basis[name][r:r2, a:b] for key, name in keys}
        if lam_rows is not None and lam_rows[r:r2].any():  # rows with a per-layer step factor < 1
            f["lab"], f["lamp"] = basis["labels"][r:r2, a:b], basis["_lam_part"]
        n = k * (b - a)
        _kernel(f, P, [x[:n].reshape(k, b - a) for x in full], z)
        mn, mx, _, _ = cv2.minMaxLoc(z, mask=msk)
        lo, hi = min(lo, mn), max(hi, mx)
    return lo, hi


def _finish_rows(basis, out, y0, y1, c0, c1, tone: tuple) -> None:
    """Apply ``tone`` = (sa, sb, knee, h1, h0): x = sa z + sb, the soft knee above `KNEE_X`
    when ``knee``, then h0 + h1 y; zero outside the silhouette."""
    sa, sb, knee, h1, h0 = tone
    cols = c1 - c0
    ch = max(1, _CHUNK_PX // max(cols, 1))
    sil = basis["_sil_u8"]
    kx, kw = np.float32(KNEE_X), np.float32(1.0 - KNEE_X)
    for r in range(y0, y1, ch):
        r2 = min(y1, r + ch)
        a, b = _chunk_cols(basis, r, r2, c0, c1)
        if b <= a:
            continue
        z = out[r:r2, a:b]
        cv2.addWeighted(z, sa, z, 0.0, sb, dst=z)
        if knee:
            m = z > kx
            if m.any():
                v = z[m]
                v -= kx
                v /= kw
                np.tanh(v, out=v)
                v *= kw
                v += kx
                z[m] = v
        cv2.addWeighted(z, h1, z, 0.0, h0, dst=z)
        cv2.multiply(z, sil[r:r2, a:b], dst=z, dtype=cv2.CV_32F)


def _blocks(r0: int, r1: int, nw: int) -> list[tuple[int, int]]:
    k = max(1, min(2 * nw, r1 - r0))
    edges = np.linspace(r0, r1, k + 1).astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def _run(basis: dict, fn) -> list:
    """Run ``fn(y0, y1)`` over the foreground rows: inline for small images, else on the pool."""
    r0, r1, _c0, _c1 = basis["_fg_bbox"]
    H, W = basis["_shape"]
    if H * W <= _INLINE_PX or _N_WORKERS == 1:
        return [fn(r0, r1)]
    return list(_pool().map(lambda ab: fn(*ab), _blocks(r0, r1, _N_WORKERS)))


def _raw_profile(basis: dict, P: tuple, out: np.ndarray) -> tuple[float, float]:
    """Raw (un-normalised) z into ``out`` over the foreground bbox, zero elsewhere; (min, max)."""
    H, W = basis["_shape"]
    if out.shape != (H, W) or out.dtype != np.float32:
        raise ValueError(f"out must be float32 {(H, W)}, got {out.dtype} {out.shape}")
    bbox = basis["_fg_bbox"]
    if bbox is None:
        out.fill(0.0)
        return 0.0, 0.0
    r0, r1, c0, c1 = bbox
    out[:r0] = 0.0
    out[r1:] = 0.0
    out[r0:r1, :c0] = 0.0
    out[r0:r1, c1:] = 0.0
    res = _run(basis, lambda a, b: _raw_rows(basis, out, a, b, c0, c1, P))
    return min(x[0] for x in res), max(x[1] for x in res)


def _knee(x: float) -> float:
    return x if x <= KNEE_X else KNEE_X + (1.0 - KNEE_X) * math.tanh((x - KNEE_X) / (1.0 - KNEE_X))


def _tone(params: Any, zmin: float, zmax: float, zp: float) -> tuple:
    """(sa, sb, knee, h1, h0) for `_finish_rows`: raw z -> [h_min, 1] through the soft knee.

    ``zp`` is the raw height's `KNEE_PCT` percentile inside the silhouette. The scale
    ``s = max(zp - zmin, span / KNEE_XMAX)`` keeps the slope gain over a plain [zmin, zmax]
    normalisation at most `_KNEE_GAIN` (x / knee(x) grows with x and x_max <= KNEE_XMAX).
    """
    h_min = max(0.0, min(float(_get(params, "h_min", 0.08)), 0.95))
    span = zmax - zmin
    if not (span > 1e-9 and math.isfinite(span)):
        return 0.0, 0.0, False, 0.0, h_min + (1.0 - h_min) * 0.5
    sc = max(float(zp) - zmin, span / KNEE_XMAX)
    xmax = span / sc
    return 1.0 / sc, -zmin / sc, xmax > KNEE_X, (1.0 - h_min) / _knee(xmax), h_min


def _z_percentile(basis: dict, out: np.ndarray, zmax: float) -> float:
    """`KNEE_PCT` percentile of the raw z inside the silhouette, on a grid of about 1024 px
    (every pixel up to 1024 px, so preview and export agree)."""
    bbox = basis["_fg_bbox"]
    if bbox is None:
        return zmax
    H, W = basis["_shape"]
    st = max(1, int(math.ceil(min(H, W) / 1024.0)))
    r0, r1, c0, c1 = bbox
    v = out[r0:r1:st, c0:c1:st][basis["_sil"][r0:r1:st, c0:c1:st]]
    if v.size == 0:
        return zmax
    kth = int(round(KNEE_PCT / 100.0 * (v.size - 1)))
    return float(np.partition(v, kth)[kth])


def _g_support_now(basis: dict, P: tuple) -> float:
    """max(g |grad S_base|) with the gate at the settings ``P`` (draft budget, see `_support`)."""
    g = 0.0
    for gidx, gmag in basis.get("_gs_px", {}).values():
        gg = _parts_at(basis, gidx, P)[1]
        g = max(g, float((gmag * np.minimum(gg + 0.1, 1.0)).max()))
    return g


def _covers(P: tuple, Q: tuple) -> bool:
    """Radii ``P`` keep every slope bound that ``Q`` does (same sliders; bounds no smaller)."""
    return P == Q or (P[0] >= Q[0] and P[1] >= Q[1] and P[2] >= Q[2] and P[3] >= Q[3] and P[4:] == Q[4:])


def _prepare(basis: dict, params: Any, physical_width_mm: float, relief_z_mm: float, out: np.ndarray):
    """Radii, seam correction and raw z, with the draft budget re-checked against the result.

    Under the draft clamp the support's share of the slope budget is first bounded with the
    gate at the default radius (conservative); the clamp then enlarges the radii, which lowers
    the gate, so the bound is re-evaluated at the clamped settings and the radii relaxed.
    Everything here depends only on the basis and the call's own arguments (never on a
    previous call), so the same parameters always give the same relief.
    """
    g_base = max(basis.get("_g_support", {}).values(), default=0.0)
    theta = float(_get(params, "draft_angle_deg", 0.0))
    P, clamped = _radii(basis, params, physical_width_mm, relief_z_mm, g_base + basis.get("_g_corr_est", 0.0))
    g_corr = _seam_correct(basis, P)
    if theta > 1e-3:
        for _ in range(3):
            g_now = _g_support_now(basis, P) if clamped else g_base
            P_new, clamped_new = _radii(basis, params, physical_width_mm, relief_z_mm, g_now + g_corr)
            if _covers(P, P_new):  # the radii in use are at least as conservative as needed
                break
            P, clamped = P_new, clamped_new
            g_corr = _seam_correct(basis, P)
    zmin, zmax = _raw_profile(basis, P, out)
    if theta > 1e-3 and zmax - zmin < 1.0:
        # the crease / dome / support shares assumed a span of at least one part height
        g_now = _g_support_now(basis, P) if clamped else g_base
        P_new, clamped_new = _radii(basis, params, physical_width_mm, relief_z_mm, g_now + g_corr, span=max(zmax - zmin, 0.05))
        if not _covers(P, P_new):
            P, clamped = P_new, clamped_new
            _seam_correct(basis, P)
            zmin, zmax = _raw_profile(basis, P, out)
    return P, clamped, zmin, zmax, _z_percentile(basis, out, zmax)


def _limit_where_needed(out: np.ndarray, bbox, sil: np.ndarray, k: float) -> None:
    """Apply `_lipschitz_limit` over the foreground when any interior pixel is steeper than ``k``.

    Steepness is measured as the central-difference gradient magnitude; without a violation the
    profile is left untouched. The limiter itself always runs on the whole foreground box: its
    lowering travels along chamfer paths that can leave any window drawn around the violations,
    so a windowed run is not exact (and growing windows until stable cost several full passes).
    """
    r0, r1, c0, c1 = bbox
    roi = out[r0:r1, c0:c1]
    m = sil[r0:r1, c0:c1]
    if roi.shape[0] < 3 or roi.shape[1] < 3:
        return
    # twice the central difference, on the box's interior (its border rows are outline)
    gx = cv2.subtract(roi[1:-1, 2:], roi[1:-1, :-2])
    gy = cv2.subtract(roi[2:, 1:-1], roi[:-2, 1:-1])
    steep = cv2.compare(cv2.magnitude(gx, gy), 2.0 * k * 0.999, cv2.CMP_GT)
    inner = cv2.erode(np.ascontiguousarray(m).view(np.uint8), np.ones((5, 5), np.uint8))  # outline: the tail's
    if cv2.countNonZero(cv2.bitwise_and(steep, inner[1:-1, 1:-1])) == 0:
        return
    _lipschitz_limit(out, bbox, sil, k)


_CHAMFER_FIX = 1.0825  # chamfer step = k / this: 1/cos(22.5 deg) = 1.08239, plus float32 margin


def _sweep(a: np.ndarray, steps: tuple) -> None:
    """Row recurrences down then up: ``a[i, j] = min(a[i, j], a[i -+ 1, j + dj] + c)`` for each
    ``(dj, c)`` in ``steps`` (float32, in place)."""
    h, w = a.shape
    tmp = np.empty(w, np.float32)
    for d, rows in ((1, range(1, h)), (-1, range(h - 2, -1, -1))):
        for i in rows:
            p, r = a[i - d], a[i]
            for dj, c in steps:
                if dj == 0:
                    np.add(p, c, out=tmp)
                    np.minimum(r, tmp, out=r)
                elif dj == -1:  # from the upper-left / lower-left neighbour
                    np.add(p[:-1], c, out=tmp[1:])
                    np.minimum(r[1:], tmp[1:], out=r[1:])
                else:
                    np.add(p[1:], c, out=tmp[:-1])
                    np.minimum(r[:-1], tmp[:-1], out=r[:-1])


def _lipschitz_limit(out: np.ndarray, bbox, sil: np.ndarray, k: float) -> None:
    """Lower ``out`` in place to the largest surface with slope <= ``k`` (per px) everywhere.

    ``out(x) = min_y out(y) + k d(x, y)`` over the box, ``y`` in the silhouette, with ``d`` the
    8-neighbour chamfer distance (axis step ``ka``, diagonal step ``ka sqrt 2``). A shortest
    chamfer path is a run of vertical and diagonal steps, all the same way in y, then a run of
    horizontal ones, so two row sweeps (down, up) over the vertical and diagonal neighbours and
    then two sweeps over the rows' own neighbours (on the transpose) give the exact min-plus
    convolution; every sweep is one vectorised operation per row. ``ka = k / _CHAMFER_FIX``
    (1/cos 22.5 deg plus float32 margin), so the chamfer cost never exceeds the Euclidean one:
    the result's slope is <= k in every direction. Background pixels are no sources (the outer
    walls are the shared tail's exact envelope); they are restored to 0.
    """
    r0, r1, c0, c1 = bbox
    roi = out[r0:r1, c0:c1]
    m = sil[r0:r1, c0:c1]
    if roi.size == 0:
        return
    a = np.where(m, roi, np.float32(np.inf)).astype(np.float32)
    ka = np.float32(k / _CHAMFER_FIX)
    kd = np.float32(float(ka) * math.sqrt(2.0))
    _sweep(a, ((0, ka), (-1, kd), (1, kd)))
    at = np.ascontiguousarray(a.T)
    _sweep(at, ((0, ka),))
    np.copyto(roi, np.where(m, at.T, np.float32(0.0)))


def sculpt_profile(
    basis: dict,
    params: Mapping[str, Any] | Any,
    physical_width_mm: float,
    relief_z_mm: float,
    out: np.ndarray,
) -> None:
    """Write the normalised Sculpt relief into ``out`` (float32, the working shape).

    Inside the silhouette the result spans ``[h_min, 1]`` (then ``invert_depth`` flips it as in
    Classic); the background is exactly 0. Sets ``basis["_last_edge_clamped"]`` to whether the
    draft clamp raised the edge roundness or crease width. When it did, the profile is also
    passed through an exact interior slope limiter (`_lipschitz_limit`), so no interior slope
    exceeds the draft angle even where the construction bound is not tight (seam corrections).
    Thread-safe per basis (a lock); never call it from inside the engine's worker pool.
    """
    lock = basis.setdefault("_lock", threading.Lock())
    with lock:
        P, clamped, zmin, zmax, zp = _prepare(basis, params, physical_width_mm, relief_z_mm, out)
        basis["_last_edge_clamped"] = clamped
        if basis["_fg_bbox"] is None:
            return
        tone = _tone(params, zmin, zmax, zp)
        invert = bool(_get(params, "invert_depth", False))
        theta = min(float(_get(params, "draft_angle_deg", 0.0)), 89.0)
        limit = clamped and theta > 1e-3
        if invert and not limit:
            sa, sb, knee, h1, h0 = tone
            tone = (sa, sb, knee, -h1, 1.0 - h0)
        _r0, _r1, c0, c1 = basis["_fg_bbox"]
        _run(basis, lambda a, b: _finish_rows(basis, out, a, b, c0, c1, tone))
        if limit:
            H, W = basis["_shape"]
            k = (max(float(physical_width_mm), 1e-6) / W) / (math.tan(math.radians(theta)) * max(float(relief_z_mm), 1e-6))
            _limit_where_needed(out, basis["_fg_bbox"], basis["_sil"], k)
            if invert:
                _run(basis, lambda a, b: _finish_rows(basis, out, a, b, c0, c1, (-1.0, 1.0, False, 1.0, 0.0)))


# --------------------------------------------------------------------------- test support


def seam_mismatch(basis: dict, params: Mapping[str, Any] | Any, H: np.ndarray) -> float:
    """Largest height step where a front part meets the surface behind, in layer steps.

    Over every pair of 4-adjacent pixels across a seam where the parts' layers differ (except
    front pixels less than 1 px from the outline, where the seam ends), the front pixel's height
    in ``H`` (the `sculpt_profile` output for ``params``, not inverted) is compared with the
    highest lower neighbour's, and divided by the local layer step (one part height,
    ``fullness + (1 - fullness) * dome``, in ``H`` units). The wall foot sits exactly on the
    surface behind, so this is 0 up to float rounding.
    """
    Hs, Ws = basis["_shape"]
    labels = basis["labels"]
    p = {**dict(params), "draft_angle_deg": 0.0} if isinstance(params, Mapping) else params
    raw = np.zeros((Hs, Ws), np.float32)
    with basis.setdefault("_lock", threading.Lock()):
        P, _c, zmin, zmax, zp = _prepare(basis, p, 1.0, 0.0, raw)
    sa, sb, knee, h1, _h0 = _tone(params, zmin, zmax, zp)
    lay = basis["layer"][labels].astype(np.float32)
    Hf = np.asarray(H, np.float32)
    Hl = np.where(labels > 0, Hf, -np.inf).astype(np.float32)
    best = np.full((Hs, Ws), -np.inf, np.float32)
    Pp = np.pad(Hl, 1, constant_values=-np.inf)
    PL = np.pad(lay, 1, constant_values=np.inf)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nb = Pp[1 + dy : 1 + dy + Hs, 1 + dx : 1 + dx + Ws]
        nl = PL[1 + dy : 1 + dy + Hs, 1 + dx : 1 + dx + Ws]
        np.maximum(best, np.where((labels > 0) & (nl < lay), nb, -np.inf), out=best)
    sel = np.isfinite(best) & (basis["d_open_px"] >= 1)
    if not sel.any():
        return 0.0
    F = P[4]
    # dH/dz at the front pixel (the tone curve's local slope)
    x = raw[sel].astype(np.float64) * sa + sb
    dydx = np.where(knee & (x > KNEE_X), 1.0 - np.tanh((x - KNEE_X) / (1.0 - KNEE_X)) ** 2, 1.0)
    step = h1 * sa * dydx * (F + (1 - F) * basis["dome"][sel].astype(np.float32))
    return float(np.max(np.abs(Hf[sel] - best[sel]) / np.maximum(step, 1e-12)))
