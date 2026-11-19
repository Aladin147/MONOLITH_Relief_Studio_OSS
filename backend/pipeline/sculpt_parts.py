"""Parts, line roles and anchors for the Sculpt relief style.

The drawing is split into *parts* (4-connected white regions) and every dark pixel inside
the silhouette is classified as a line: outline (touches the background), separating
(between two different parts) or open stroke (a detail line that does not divide parts).
All of this runs on a fixed analysis grid (longest edge ``ANALYSIS_RES``). The production
entry point is `analyze_parts_from_source`, which preprocesses the original image directly at
the analysis resolution, so the result is bit-identical whatever working resolution the
project uses. `analyze_parts` analyses already-preprocessed arrays (tests, synthetic input).

`part_at` returns 0 on line pixels and on the background; use `part_near` (one click: a line
pixel gives its nearest part) or `labels_on_grid` (a whole grid, lines filled with the nearest
part) when a click should always land on a part.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from scipy import ndimage as ndi

from backend.pipeline.preprocessor import ImageSource, preprocess_lineart

ANALYSIS_RES = 2048
# Retain faint lines between adjacent regions.
WHITE_THRESHOLD = 0.65
SPECK_FRACTION = 0.002 / 100.0  # 0.002 % of the canvas area
SMALL_FRACTION = 0.05 / 100.0  # parts below 0.05 % of the canvas are ordered after the rest

ROLE_NONE, ROLE_OUTLINE, ROLE_SEPARATING, ROLE_OPEN = 0, 1, 2, 3


@dataclass
class PartsAnalysis:
    labels: np.ndarray  # int32 (H, W): 0 = background/lines, 1..N = parts
    line_role: np.ndarray  # uint8 (H, W): 0 none, 1 outline, 2 separating, 3 open stroke
    # line_owner: int32 (N_line_px, 2). Rows follow the C-order of np.nonzero(line_role > 0).
    #   separating: (lo, hi) with lo < hi (the lowest and highest part id within r; at T/X
    #               junctions the middle ids are not listed, their pairs come from nearby rows);
    #   outline:    (p, -1), or (-1, -1) when no part lies within r;
    #   open:       (p, p).
    line_owner: np.ndarray
    adjacency: dict[tuple[int, int], int]  # (a < b) -> shared line length in px
    anchors: np.ndarray  # float32 (N+1, 2): normalised (u, v), v down; row 0 unused (0, 0)
    shape: tuple[int, int]  # analysis grid (H, W)
    filled: np.ndarray | None = None  # lazy cache: labels with lines/silhouette filled
    order: Any = None  # lazy cache: the order solver's inputs (sculpt_overrides.order_inputs)


def _analysis_shape(h: int, w: int) -> tuple[int, int]:
    s = ANALYSIS_RES / float(max(h, w))
    return max(1, int(round(h * s))), max(1, int(round(w * s)))


def line_radius(shape: tuple[int, int]) -> int:
    """Line-to-part assignment radius ``r`` in px on a grid of ``shape`` (6 at 2048)."""
    return max(2, int(round(0.003 * max(shape))))


def _to_grid(a: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    a = np.ascontiguousarray(a, dtype=np.float32)
    if a.shape == shape:
        return a
    return cv2.resize(a, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)


def _label_parts(white: np.ndarray) -> np.ndarray:
    """4-connected components of `white`; specks merge into lines.

    Ids are 1..N: substantial parts first in raster order of their first pixel, then small
    ones (also raster order). A small part that appears or vanishes between two preprocess
    resolutions therefore only shifts the ids of other small parts, and two similar-sized
    substantial parts cannot swap ids on area noise.
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(white.astype(np.uint8), connectivity=4)
    min_area = max(1.0, SPECK_FRACTION * white.size)
    area = stats[:, cv2.CC_STAT_AREA]
    kept = np.nonzero(area >= min_area)[0]
    kept = kept[kept > 0]  # component ids from OpenCV are already in raster order
    small = area[kept] < SMALL_FRACTION * white.size
    order = np.concatenate([kept[~small], kept[small]])
    remap = np.zeros(n, np.int32)
    remap[order] = np.arange(1, len(order) + 1, dtype=np.int32)
    return remap[lab]


def _anchors(labels: np.ndarray, n: int) -> np.ndarray:
    h, w = labels.shape
    out = np.zeros((n + 1, 2), np.float32)
    objs = ndi.find_objects(labels, max_label=n)
    for p in range(1, n + 1):
        sl = objs[p - 1]
        if sl is None:
            continue
        y0, x0 = sl[0].start, sl[1].start
        m = np.pad((labels[sl] == p).astype(np.uint8), 1)  # pad so the bbox edge counts as outside
        dist = cv2.distanceTransform(m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        iy, ix = np.unravel_index(int(np.argmax(dist)), dist.shape)
        out[p, 0] = (x0 + ix - 1 + 0.5) / w
        out[p, 1] = (y0 + iy - 1 + 0.5) / h
    return out


def _skeleton_pair_lengths(sep_mask: np.ndarray, sep_pairs: np.ndarray) -> dict[tuple[int, int], int]:
    """Shared line length per part pair: skeleton pixels of the separating lines by owner pair.

    `sep_pairs` is (K, 2) with a < b for every separating pixel in raster order of `sep_mask`.
    """
    k = len(sep_pairs)
    idx = np.full(sep_mask.shape, -1, np.int64)
    idx[sep_mask] = np.arange(k)
    skel = cv2.ximgproc.thinning(sep_mask.astype(np.uint8) * 255, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN) > 0
    key_all = sep_pairs[:, 0].astype(np.int64) << 32 | sep_pairs[:, 1].astype(np.int64)
    keys, _ = np.unique(key_all, return_counts=True)
    key_sk = key_all[idx[skel & sep_mask]]
    ks, cs = np.unique(key_sk, return_counts=True)
    length = dict(zip(ks.tolist(), cs.tolist()))
    return {
        (int(key >> 32), int(key & 0xFFFFFFFF)): int(max(1, length.get(int(key), 0)))
        for key in keys.tolist()
    }


def analyze_parts(L_clean: np.ndarray, M_sil: np.ndarray, M_regions: np.ndarray) -> PartsAnalysis:
    """Analyse working-grid inputs on the fixed analysis grid.

    `L_clean` is accepted for interface symmetry; parts come from `M_regions` (white inside
    the silhouette) and lines are the remaining silhouette pixels.
    """
    del L_clean
    h0, w0 = M_sil.shape
    shape = _analysis_shape(h0, w0)
    sil = _to_grid(M_sil, shape) > 0.5
    white = (_to_grid(M_regions, shape) > WHITE_THRESHOLD) & sil

    labels = _label_parts(white)
    n = int(labels.max())
    line = sil & (labels == 0)

    r = line_radius(shape)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    lab_f = labels.astype(np.float32)
    max_l = cv2.dilate(lab_f, k)  # highest part id within r (0 = none)
    min_l = cv2.erode(np.where(labels > 0, lab_f, np.float32(1e9)), k)  # lowest part id within r
    near_bg = (
        cv2.dilate((~sil).astype(np.uint8), k, borderType=cv2.BORDER_CONSTANT, borderValue=1) > 0
    )

    has_part = max_l > 0
    multi = has_part & (max_l != min_l)
    role = np.zeros(labels.shape, np.uint8)
    role[line & ~multi] = ROLE_OPEN
    role[line & multi] = ROLE_SEPARATING
    role[line & near_bg & ~multi] = ROLE_OUTLINE  # a seam running to the edge stays a seam

    ly, lx = np.nonzero(line)
    owner = np.full((len(ly), 2), -1, np.int32)
    lo = min_l[ly, lx]
    hi = max_l[ly, lx]
    r_l = role[ly, lx]
    first = np.where(hi > 0, lo, -1).astype(np.int32)
    sep = r_l == ROLE_SEPARATING
    out_ = r_l == ROLE_OUTLINE
    opn = r_l == ROLE_OPEN
    owner[sep, 0] = lo[sep].astype(np.int32)
    owner[sep, 1] = hi[sep].astype(np.int32)
    owner[out_, 0] = first[out_]  # second side stays -1 (background)
    owner[opn, 0] = first[opn]
    owner[opn, 1] = first[opn]
    # open-stroke pixels with no part within r: borrow the nearest part
    orphan = opn & (first < 0)
    if orphan.any() and n > 0:
        _, (iy, ix) = ndi.distance_transform_edt(labels == 0, return_indices=True)
        near = labels[iy[ly[orphan], lx[orphan]], ix[ly[orphan], lx[orphan]]]
        owner[orphan, 0] = near
        owner[orphan, 1] = near

    adjacency: dict[tuple[int, int], int] = {}
    pair_ok = sep & (owner[:, 0] > 0) & (owner[:, 1] > 0) & (owner[:, 0] != owner[:, 1])
    if pair_ok.any():
        sep_mask = np.zeros(labels.shape, bool)
        sep_mask[ly[pair_ok], lx[pair_ok]] = True
        min_len = max(2, r)  # drop single-pixel contacts
        adjacency = {
            k_: v for k_, v in _skeleton_pair_lengths(sep_mask, owner[pair_ok]).items() if v >= min_len
        }

    return PartsAnalysis(
        labels=labels,
        line_role=role,
        line_owner=owner,
        adjacency=adjacency,
        anchors=_anchors(labels, n),
        shape=shape,
    )


def analyze_parts_from_source(
    image_source: ImageSource,
    *,
    bg_removal: bool = True,
    hd_restore: bool = True,
    polarity: str = "auto",
) -> PartsAnalysis:
    """Canonical analysis: preprocess the original image at the fixed analysis resolution."""
    d = preprocess_lineart(
        image_source,
        target_res=ANALYSIS_RES,
        bg_removal=bg_removal,
        hd_restore=hd_restore,
        polarity=polarity,
    )
    return analyze_parts(d["L_clean"], d["M_sil"], d["M_regions"])


def _filled_labels(pa: PartsAnalysis) -> np.ndarray:
    """Analysis-grid labels where EVERY pixel (lines, background) holds its nearest part."""
    if pa.filled is None:
        lab = pa.labels
        if lab.any():
            _, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
            lab = lab[iy, ix]
        pa.filled = lab.astype(np.int32, copy=False)
    return pa.filled


def labels_on_grid(
    pa: PartsAnalysis, shape: tuple[int, int], sil: np.ndarray | None = None
) -> np.ndarray:
    """Part labels on another grid; every silhouette pixel gets a part, at any resolution.

    The analysis grid is filled with nearest parts everywhere (one cheap EDT at 2048),
    nearest-neighbour upsampled, then masked with the working silhouette `sil` (shape must
    equal `shape`; default: the analysis silhouette). No full-resolution EDT.
    """
    H, W = shape
    h, w = pa.shape
    if sil is not None and np.shape(sil) != (H, W):
        raise ValueError(f"sil shape {np.shape(sil)} != {(H, W)}")
    filled = _filled_labels(pa)
    ys = np.minimum(((np.arange(H) + 0.5) * h / H).astype(np.int32), h - 1)
    xs = np.minimum(((np.arange(W) + 0.5) * w / W).astype(np.int32), w - 1)
    lab = filled[np.ix_(ys, xs)]
    if sil is None:
        inside = ((pa.labels > 0) | (pa.line_role > 0))[np.ix_(ys, xs)]
    else:
        inside = np.asarray(sil) > 0.5
    lab[~inside] = 0
    return lab


def part_at(pa: PartsAnalysis, u: float, v: float) -> int:
    """Part id at normalised (u, v), v downward; 0 = no part (background or a line pixel)."""
    h, w = pa.shape
    x = min(max(int(np.floor(u * w)), 0), w - 1)
    y = min(max(int(np.floor(v * h)), 0), h - 1)
    return int(pa.labels[y, x])


def part_near(pa: PartsAnalysis, u: float, v: float) -> int:
    """The part a click at normalised (u, v), v downward, selects: like `part_at`, but a line pixel
    inside the silhouette gives its nearest part. 0 on the background or outside [0, 1]."""
    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        return 0
    h, w = pa.shape
    x = min(int(np.floor(u * w)), w - 1)
    y = min(int(np.floor(v * h)), h - 1)
    p = int(pa.labels[y, x])
    if p == 0 and pa.line_role[y, x] > 0:
        p = int(_filled_labels(pa)[y, x])
    return p
