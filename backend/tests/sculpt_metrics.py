"""Test-only helpers that measure how a relief compares to a reference depth map.

Helpers for comparing relief profiles.
Nothing here ships with the application. Masks may be bool or any numeric
dtype (non-zero = inside).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.stats import spearmanr

SILHOUETTE_THRESHOLD = 0.06
# Inside-figure values of a row profile are floored at this so they stay > 0;
# outside the figure a profile is exactly 0.
_INSIDE_FLOOR = 1e-3
# A notch must rise at least this much (normalised height) on both sides.
_NOTCH_RISE_EPS = 0.01

_BENCH_DIR = Path(__file__).resolve().parents[2] / "benchmarks"
THIGH_ROW_FRAC = 0.62
WALL_PX = (20, 40, 60)


def _bool(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask) > 0


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _to_unit_gray(depth: np.ndarray) -> np.ndarray:
    """Single-channel float32 in 0..1 from uint8 / float, gray / multi-channel input."""
    d = np.asarray(depth)
    if d.ndim == 3:
        d = d[..., :3].mean(axis=2)
    scale = 255.0 if np.issubdtype(np.asarray(depth).dtype, np.integer) else 1.0
    return np.ascontiguousarray(d, dtype=np.float32) / np.float32(scale)


def align_to(mask: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Place a competitor depth map (0..1) on the grid of ``mask``.

    The silhouette bounding box of ``depth`` (> 0.06) is resized onto the
    bounding box of ``mask``. Everything outside that box is 0. ``depth`` may
    be uint8 (scaled by 1/255) or float 0..1, gray or 3-channel (averaged).
    Raises ValueError if ``mask`` or the depth silhouette is empty.
    """
    m = _bool(mask)
    d = _to_unit_gray(depth)
    if not m.any():
        raise ValueError("align_to: mask is empty")
    sil = d > SILHOUETTE_THRESHOLD
    if not sil.any():
        raise ValueError("align_to: depth has no pixels above the silhouette threshold")
    cy0, cy1, cx0, cx1 = _bbox(sil)
    oy0, oy1, ox0, ox1 = _bbox(m)
    out = np.zeros(m.shape, np.float32)
    crop = np.ascontiguousarray(d[cy0:cy1, cx0:cx1])
    out[oy0:oy1, ox0:ox1] = cv2.resize(crop, (ox1 - ox0, oy1 - oy0), interpolation=cv2.INTER_AREA)
    return out


def row_profile(H: np.ndarray, mask: np.ndarray, frac: float) -> np.ndarray:
    """Row of ``H`` at ``frac`` of the figure's height, min-max normalised inside the figure.

    Inside pixels lie in [1e-3, 1]; outside pixels are exactly 0.
    """
    m = _bool(mask)
    y0, y1, _, _ = _bbox(m)
    y = min(max(int(y0 + frac * (y1 - y0)), y0), y1 - 1)
    row = np.asarray(H)[y].astype(np.float32)
    inside = m[y]
    out = np.zeros_like(row)
    if inside.any():
        v = row[inside]
        norm = (v - v.min()) / (v.max() - v.min() + 1e-9)
        out[inside] = _INSIDE_FLOOR + (1.0 - _INSIDE_FLOOR) * norm
    return out


def wall_rise(profile: np.ndarray, px: int) -> float:
    """How vertical the walls are: height ``px`` pixels inside each outline crossing / local top.

    Pixels > 0 are inside the figure. For every crossing (a run of inside pixels
    starts or ends) the local top is the maximum within 15% of the FULL row
    width (``len(profile)``, not the figure's width) from the crossing, going
    inward. A figure clipped at the canvas border counts as a crossing there
    too. Returns the minimum ratio over crossings; runs no longer than ``px``
    pixels are skipped, and NaN is returned if no crossing can be measured.
    A vertical wall gives 1.0, a gentle ramp gives a small value.
    """
    p = np.asarray(profile, np.float32)
    inside = p > 0
    reach = max(1, int(round(0.15 * len(p))))
    padded = np.concatenate([[False], inside, [False]]).astype(np.int8)
    d = np.diff(padded)
    starts = np.nonzero(d == 1)[0]
    ends = np.nonzero(d == -1)[0]  # exclusive
    ratios = []
    for s, e in zip(starts, ends):
        if e - s <= px:
            continue
        top_l = p[s : min(s + reach + 1, e)].max()
        ratios.append(p[s + px] / top_l)
        top_r = p[max(e - 1 - reach, s) : e].max()
        ratios.append(p[e - 1 - px] / top_r)
    return float(min(ratios)) if ratios else float("nan")


def notch_depth(profile: np.ndarray, x: int, half_window: int) -> float:
    """``1 - min/max`` of ``profile[x-hw:x+hw]`` (start clamped at 0): how deep a separating line cuts in.

    NaN if the window is empty or its maximum is not positive.
    """
    w = np.asarray(profile, np.float32)[max(0, x - half_window) : x + half_window]
    if w.size == 0 or w.max() <= 0:
        return float("nan")
    return float(1.0 - w.min() / w.max())


def find_centre_notch(profile: np.ndarray, search_frac: float = 0.10) -> int:
    """x of the separating line in a row: the true local minimum near the figure's centre.

    The centre is the midpoint of the first and last inside (> 0) pixel; the
    window is ``search_frac`` of the inside extent either side. Only inside
    pixels are considered. If the minimum is a flat bottom (ties), the centre
    of its first contiguous run is returned. Raises ValueError if the window
    has no inside pixels, or the minimum is not a true local minimum (the
    profile must rise by at least 0.01 on both sides of it within the window,
    so a monotone slope or a minimum at a window edge is rejected).
    """
    p = np.asarray(profile, np.float32)
    xs = np.nonzero(p > 0)[0]
    if xs.size == 0:
        raise ValueError("find_centre_notch: profile has no inside pixels")
    centre = (int(xs[0]) + int(xs[-1])) // 2
    half = max(1, int(search_frac * (xs[-1] - xs[0])))
    lo, hi = max(0, centre - half), min(len(p), centre + half + 1)
    seg = p[lo:hi]
    inside = seg > 0
    if not inside.any():
        raise ValueError("find_centre_notch: no inside pixels near the figure centre")
    vmin = seg[inside].min()
    ties = np.nonzero(inside & np.isclose(seg, vmin, rtol=0, atol=1e-6))[0]
    run_end = 0
    while run_end + 1 < len(ties) and ties[run_end + 1] == ties[run_end] + 1:
        run_end += 1
    a, b = int(ties[0]), int(ties[run_end])
    left, right = seg[:a][inside[:a]], seg[b + 1 :][inside[b + 1 :]]
    if (
        left.size == 0
        or right.size == 0
        or left.max() < vmin + _NOTCH_RISE_EPS
        or right.max() < vmin + _NOTCH_RISE_EPS
    ):
        raise ValueError("find_centre_notch: no true local minimum near the figure centre")
    return lo + (a + b) // 2


def part_order_agreement(
    H: np.ndarray, ref: np.ndarray, labels: np.ndarray, min_area: int = 300
) -> float:
    """Spearman correlation of per-part mean heights of ``H`` vs ``ref``.

    ``labels`` is an integer part image (0 = background); parts with an area
    of ``min_area`` pixels or less are ignored. Returns NaN with fewer than 3
    qualifying parts (a rank correlation of 2 points is meaningless).
    """
    ids = [k for k in np.unique(labels) if k > 0 and (labels == k).sum() > min_area]
    if len(ids) < 3:
        return float("nan")
    a = ndi.mean(ref, labels, ids)
    b = ndi.mean(H, labels, ids)
    return float(spearmanr(a, b).correlation)


def height_correlation(H: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float:
    """Pearson correlation of min-max-normalised heights inside ``mask & (ref > 0.06)``."""
    sel = _bool(mask) & (np.asarray(ref) > SILHOUETTE_THRESHOLD)

    def norm(a: np.ndarray) -> np.ndarray:
        v = np.asarray(a, np.float64)[sel]
        return (v - v.min()) / (v.max() - v.min() + 1e-12)

    return float(np.corrcoef(norm(H), norm(ref))[0, 1])
