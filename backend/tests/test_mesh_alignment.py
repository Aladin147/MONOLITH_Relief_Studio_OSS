"""Mesh coordinates must sit on the same pixel-centre convention as the PNG / DXF exporters."""

from __future__ import annotations

import io

import numpy as np
import pytest
import trimesh

from backend.pipeline import outline_exporter as oe
from backend.pipeline import stl_exporter as ex


def _disc(size: int, cx: float, cy: float, r: float) -> np.ndarray:
    """Anti-aliased disc in [0, 1]; pixel (row, col) centre is at (col + 0.5, row + 0.5)."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    d = np.hypot(xx + 0.5 - cx, yy + 0.5 - cy)
    return np.clip(r - d + 0.5, 0.0, 1.0).astype(np.float32)


def _polygon_centroid(p: np.ndarray) -> np.ndarray:
    x, y = p[:, 0], p[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    a = cross.sum() / 2.0
    return np.array([((x + x1) * cross).sum() / (6 * a), ((y + y1) * cross).sum() / (6 * a)])


def test_slab_grid_axes_are_pixel_centres_plus_canvas_ring():
    X, Y, Zt, Zb, cm = ex._prepare_mesh_grid(np.zeros((30, 40), np.float32), 400.0, 10.0, 2.0, 0)
    assert cm is None
    assert X[0, 0] == 0.0 and X[0, -1] == pytest.approx(400.0)
    assert X[0, 1] == pytest.approx(5.0) and X[0, -2] == pytest.approx(395.0)  # p = 10 mm
    assert Y[0, 0] == pytest.approx(300.0) and Y[-1, 0] == 0.0  # 30 * 10 mm tall


def test_canvas_axis_mm():
    ax = ex.canvas_axis_mm(4, 8.0)
    np.testing.assert_allclose(ax, [0.0, 1.0, 3.0, 5.0, 7.0, 8.0])


def test_downsampled_height_comes_from_full_res_shape():
    X, Y, *_ = ex._prepare_mesh_grid(np.zeros((300, 400), np.float32), 400.0, 10.0, 2.0, 64)
    assert Y[0, 0] == pytest.approx(300.0) and X[0, -1] == pytest.approx(400.0)


@pytest.mark.parametrize("size,max_grid,tol_pitch,tol_mm", [(512, 0, None, 0.05), (512, 128, 0.25, None)])
def test_stl_feature_registers_with_dxf_outline(size, max_grid, tol_pitch, tol_mm):
    width = 512.0  # 1 mm / px
    L = _disc(size, 90.3, 430.7, 60.0)
    M = (L > 0.5).astype(np.float32)
    loops = oe.extract_outline(L, M, width, tolerance_mm=0.0)
    outer = [lp for lp in loops if not lp.is_hole]
    assert len(outer) == 1
    ox, oy_down = _polygon_centroid(outer[0].points)
    height = width * size / size
    oy = height - oy_down  # y-up

    X, Y, Zt, Zb, cm = ex._prepare_mesh_grid(L, width, 10.0, 2.0, max_grid)
    w = Zt - 2.0
    sx = float((w * X).sum() / w.sum())
    sy = float((w * Y).sum() / w.sum())
    tol = tol_mm if tol_mm is not None else tol_pitch * (width / (X.shape[1] - 2))
    assert abs(sx - ox) < tol and abs(sy - oy) < tol, (sx - ox, sy - oy)


def test_cutout_footprint_area_is_unbiased():
    L = _disc(256, 128.0, 128.0, 40.0)
    H = (L > 0.5).astype(np.float32)
    stl = ex.heightmap_to_watertight_stl(H, 256.0, 10.0, 0.0, max_grid=64)
    mesh = trimesh.load(io.BytesIO(stl), file_type="stl")
    bottom = mesh.area_faces[mesh.face_normals[:, 2] < -0.5].sum()
    assert bottom == pytest.approx(np.pi * 40.0**2, rel=0.03)
