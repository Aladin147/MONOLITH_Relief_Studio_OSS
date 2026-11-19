"""Tests for the Sculpt parts / line-role / anchor analysis (backend.pipeline.sculpt_parts)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.pipeline.preprocessor import preprocess_lineart
from backend.pipeline.sculpt_parts import (
    ANALYSIS_RES,
    analyze_parts,
    analyze_parts_from_source,
    labels_on_grid,
    part_at,
)


def _inputs(white: np.ndarray, sil: np.ndarray):
    L = white.astype(np.float32)
    S = sil.astype(np.float32)
    return L, S, L * S


def _disc(shape, cx, cy, r):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def _two_discs():
    n = ANALYSIS_RES
    a = _disc((n, n), 800, 1024, 420)
    b = _disc((n, n), 1250, 1024, 420)
    union = a | b
    white = union.copy()
    white[:, 1022:1026] = False  # black dividing stroke through the whole union
    sil = cv2.dilate(union.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0  # dark outline rim
    return white, sil


def _two_discs_union_stroke():
    n = ANALYSIS_RES
    union = _disc((n, n), 800, 1024, 420) | _disc((n, n), 1250, 1024, 420)
    out = np.zeros((n, n), bool)
    out[:, 1022:1026] = union[:, 1022:1026]
    return out


def test_two_overlapping_discs():
    pa = analyze_parts(*_inputs(*_two_discs()))
    assert pa.shape == (ANALYSIS_RES, ANALYSIS_RES)
    assert pa.labels.dtype == np.int32 and pa.line_role.dtype == np.uint8
    assert int(pa.labels.max()) == 2
    assert set(np.unique(pa.labels)) == {0, 1, 2}
    # the dividing stroke is a separating line (role 2) in its middle
    assert (pa.line_role[1000:1050, 1022:1026] == 2).all()
    assert (pa.line_role == 1).any()  # outline exists too
    assert pa.adjacency.keys() == {(1, 2)}
    assert pa.adjacency[(1, 2)] > 400  # chord is ~700 px tall
    # owners of the separating pixels are the two parts
    sep = pa.line_owner[(pa.line_role[pa.line_role > 0]) == 2]
    assert {tuple(sorted(map(int, o))) for o in sep} == {(1, 2)}


def test_open_stroke_inside_part():
    n = ANALYSIS_RES
    sil = _disc((n, n), 1024, 1024, 600)
    white = sil.copy()
    white[1020:1024, 800:1100] = False  # short stroke, does not reach the edge
    pa = analyze_parts(*_inputs(white, sil))
    assert int(pa.labels.max()) == 1
    assert (pa.line_role[1020:1024, 820:1080] == 3).all()
    assert not pa.adjacency


def test_single_part_blob():
    n = ANALYSIS_RES
    sil = _disc((n, n), 1000, 1100, 700)
    pa = analyze_parts(*_inputs(sil, sil))
    assert int(pa.labels.max()) == 1
    assert pa.anchors.shape == (2, 2) and pa.anchors.dtype == np.float32
    assert not pa.adjacency
    # pole of inaccessibility of a disc is its centre
    u, v = pa.anchors[1]
    assert abs(u * n - 1000) < 3 and abs(v * n - 1100) < 3


def test_anchor_inside_part():
    pa = analyze_parts(*_inputs(*_two_discs()))
    for p in range(1, int(pa.labels.max()) + 1):
        u, v = pa.anchors[p]
        assert part_at(pa, float(u), float(v)) == p
    assert part_at(pa, 0.01, 0.01) == 0


def test_speck_merges_into_lines():
    white, sil = _two_discs()
    white[300:305, 300:305] = False
    sil[300:305, 300:305] = True  # 5x5 black hole
    s2 = sil.copy()
    w2 = white.copy()
    w2[1500:1504, 800:804] = True  # tiny isolated white speck far from parts
    s2[1500:1504, 800:804] = True
    pa = analyze_parts(*_inputs(w2, s2))
    assert int(pa.labels.max()) == 2


def test_labels_on_grid_fills_silhouette():
    pa = analyze_parts(*_inputs(*_two_discs()))
    lab = labels_on_grid(pa, (4096, 4096))
    assert lab.shape == (4096, 4096)
    sil_small = (pa.labels > 0) | (pa.line_role > 0)
    sil_big = np.kron(sil_small, np.ones((2, 2), bool))
    assert (lab[sil_big] > 0).all()
    assert (lab[~sil_big] == 0).all()
    assert set(np.unique(lab)) == {0, 1, 2}


def _synthetic_png(path: Path) -> None:
    img = np.zeros((1024, 1024), np.uint8)
    cv2.circle(img, (420, 520), 330, 255, -1, cv2.LINE_AA)
    cv2.circle(img, (450, 500), 170, 0, 5, cv2.LINE_AA)  # closed black ring -> second part
    cv2.imwrite(str(path), img)


def test_from_source_is_deterministic_synthetic(tmp_path):
    p = tmp_path / "d.png"
    _synthetic_png(p)
    a, b = analyze_parts_from_source(str(p)), analyze_parts_from_source(str(p))
    assert int(a.labels.max()) == 2
    assert np.array_equal(a.labels, b.labels) and np.array_equal(a.anchors, b.anchors)
    assert a.adjacency == b.adjacency


def test_labels_on_grid_rejects_wrong_sil_shape():
    pa = analyze_parts(*_inputs(*_two_discs()))
    with pytest.raises(ValueError):
        labels_on_grid(pa, (100, 100), np.ones((50, 50), np.float32))


def test_seam_runs_to_the_silhouette_edge():
    white, _ = _two_discs()
    sil = white | _two_discs_union_stroke()
    pa = analyze_parts(*_inputs(white, sil))
    # the dividing stroke at x=1022..1025 reaches the top/bottom of the union: still separating
    col = pa.line_role[:, 1023]
    ys = np.nonzero(col)[0]
    assert (col[ys[0] : ys[0] + 3] == 2).all() and (col[ys[-1] - 2 : ys[-1] + 1] == 2).all()


def test_adjacency_drops_single_pixel_contact():
    n = ANALYSIS_RES
    sil = np.zeros((n, n), bool)
    sil[500:1500, 500:1500] = True
    white = sil.copy()
    white[:, 1000:1004] = False  # real seam
    white[1200:1204, 500:1000] = False  # horizontal stroke splitting the left part
    white[1200:1204, 1000:1004] = False
    pa = analyze_parts(*_inputs(white, sil))
    assert pa.adjacency and all(v >= 2 for v in pa.adjacency.values())
    assert max(pa.adjacency.values()) > 100  # a real seam survives
    assert min(pa.adjacency.values()) >= 6  # contacts shorter than max(2, r) are dropped
