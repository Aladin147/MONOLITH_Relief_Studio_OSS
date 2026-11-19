"""Adaptive, error-bounded heightfield mesh (RTIN) used by the STL / OBJ export."""

from __future__ import annotations

import os

import io
import time

import cv2
import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient

from backend import db
from backend import main as api_main
from backend.pipeline import adaptive_mesh as am
from backend.pipeline.adaptive_mesh import MeshTooLargeError, adaptive_heightfield_mesh
from backend.pipeline.stl_exporter import MeshData, encode_obj, encode_stl
from backend.tests.conftest import make_lineart_png


def smooth_random_field(shape: tuple[int, int], seed: int) -> np.ndarray:
    """Gaussian-blurred noise rescaled to [0, 1] (float32)."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(shape).astype(np.float32)
    sigma = max(1.0, 0.08 * max(shape))
    field = cv2.GaussianBlur(noise, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    lo, hi = float(field.min()), float(field.max())
    if hi - lo < 1e-12:
        return np.zeros(shape, np.float32)
    return ((field - lo) / (hi - lo)).astype(np.float32)


def _mesh(md: MeshData) -> trimesh.Trimesh:
    return trimesh.Trimesh(vertices=md.vertices, faces=md.faces)


def _brute_force_max_error(md: MeshData, H: np.ndarray, width_mm: float, relief: float, base: float) -> float:
    """Max |mesh - (H * relief + base)| over every pixel centre, rasterizing the top faces in mm."""
    rows, cols = H.shape
    p = width_mm / cols
    h_mm = p * rows
    target = H.astype(np.float64) * relief + base
    V = md.vertices
    top = md.faces[np.all(V[md.faces][:, :, 2] > 0.0, axis=1)]
    covered = np.zeros(H.shape, bool)
    worst = 0.0
    for f in top:
        (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = V[f]
        det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        c_lo = max(0, int(np.floor(min(x0, x1, x2) / p - 0.5)))
        c_hi = min(cols - 1, int(np.ceil(max(x0, x1, x2) / p - 0.5)))
        r_lo = max(0, int(np.floor((h_mm - max(y0, y1, y2)) / p - 0.5)))
        r_hi = min(rows - 1, int(np.ceil((h_mm - min(y0, y1, y2)) / p - 0.5)))
        for r in range(r_lo, r_hi + 1):
            y = h_mm - (r + 0.5) * p
            for c in range(c_lo, c_hi + 1):
                x = (c + 0.5) * p
                l0 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / det
                l1 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / det
                l2 = 1.0 - l0 - l1
                if min(l0, l1, l2) < -1e-9:
                    continue
                covered[r, c] = True
                worst = max(worst, abs(l0 * z0 + l1 * z1 + l2 * z2 - target[r, c]))
    assert covered.all(), "some heightmap samples are not under any top face"
    return worst


@pytest.mark.parametrize("shape", [(97, 131), (3, 200), (2, 2), (128, 128)])
def test_adaptive_mesh_odd_shapes(shape):
    H = smooth_random_field(shape, seed=1)
    md = adaptive_heightfield_mesh(H, 300.0, 10.0, 4.0, 0.02)
    m = _mesh(md)
    assert m.is_watertight and m.is_winding_consistent and m.volume > 0
    assert md.max_error_mm <= 0.02 + 1e-6
    assert m.bounds[0][:2] == pytest.approx([0, 0]) and m.bounds[1][0] == pytest.approx(300.0)
    assert m.bounds[1][1] == pytest.approx(300.0 * shape[0] / shape[1])
    assert md.vertices.dtype == np.float64 and md.faces.dtype == np.int64


@pytest.mark.parametrize("tol", [0.02, 0.2])
def test_adaptive_mesh_error_bound_is_real(tol):
    H = smooth_random_field((65, 65), seed=7)
    H[:, 40:43] *= 0.3  # a carved groove
    yy, xx = np.mgrid[0:65, 0:65]
    H = np.clip(H + 0.004 * xx + 0.003 * yy, 0.0, 1.0).astype(np.float32)  # ramps run into every edge
    md = adaptive_heightfield_mesh(H, 130.0, 10.0, 3.0, tol)
    worst = _brute_force_max_error(md, H, 130.0, 10.0, 3.0)
    assert worst <= tol + 1e-6
    assert md.max_error_mm == pytest.approx(worst, abs=1e-6)  # the reported error is the real one


def test_adaptive_mesh_flat_input():
    md = adaptive_heightfield_mesh(np.zeros((256, 320), np.float32), 100.0, 5.0, 2.0, 0.02)
    assert _mesh(md).is_watertight and len(md.faces) < 0.15 * 2 * 255 * 319
    assert md.max_error_mm == pytest.approx(0.0, abs=1e-9)


def test_adaptive_mesh_edge_subject():
    H = np.zeros((64, 64), np.float32)
    H[:, :20] = 1.0  # subject touches the left edge
    m = _mesh(adaptive_heightfield_mesh(H, 64.0, 10.0, 2.0, 0.02))
    left = m.vertices[np.isclose(m.vertices[:, 0], 0.0) & (m.vertices[:, 2] > 0)]
    assert left[:, 2].min() == pytest.approx(12.0, abs=1e-4)  # ring copies edge height, no dip


@pytest.mark.parametrize("raw", ["abc", "", "1.5e7", "0", "-5"])
def test_bad_triangle_cap_env_falls_back_to_default(monkeypatch, raw):
    from backend.pipeline.adaptive_mesh import DEFAULT_MAX_MESH_TRIANGLES, max_mesh_triangles

    monkeypatch.setenv("MONOLITH_MAX_MESH_TRIANGLES", raw)
    assert max_mesh_triangles() == DEFAULT_MAX_MESH_TRIANGLES
    monkeypatch.setenv("MONOLITH_MAX_MESH_TRIANGLES", "1234")
    assert max_mesh_triangles() == 1234


def test_adaptive_mesh_requires_a_base():
    with pytest.raises(ValueError):
        adaptive_heightfield_mesh(np.zeros((8, 8), np.float32), 10.0, 1.0, 0.0, 0.02)


def test_adaptive_mesh_triangle_cap(monkeypatch):
    H = smooth_random_field((97, 131), seed=1)
    monkeypatch.setenv("MONOLITH_MAX_MESH_TRIANGLES", "500")
    with pytest.raises(MeshTooLargeError) as info:
        adaptive_heightfield_mesh(H, 300.0, 10.0, 4.0, 0.02)
    assert isinstance(info.value, ValueError)
    assert info.value.cap == 500 and info.value.triangles > 500
    # The frontier is bounded too: extraction stops just past the cap, not at the full mesh.
    assert info.value.triangles <= 2 * 500
    monkeypatch.delenv("MONOLITH_MAX_MESH_TRIANGLES")
    assert len(adaptive_heightfield_mesh(H, 300.0, 10.0, 4.0, 0.02).faces) > 500


def test_adaptive_mesh_unconverged_rounds_fall_back_to_exact(monkeypatch):
    H = smooth_random_field((128, 128), seed=1)  # first extraction measures ~0.023 mm > 0.02
    thresholds = []
    real_extract = am._extract

    def recording_extract(E, n, threshold, max_triangles):
        thresholds.append(threshold)
        return real_extract(E, n, threshold, max_triangles)

    monkeypatch.setattr(am, "_extract", recording_extract)
    monkeypatch.setattr(am, "MAX_REFINE_ROUNDS", 0)
    md = adaptive_heightfield_mesh(H, 130.0, 10.0, 3.0, 0.02)
    assert thresholds == [pytest.approx(0.02 / 10.0), 0.0]  # no converged round -> exact pass
    assert md.max_error_mm <= 0.02
    m = _mesh(md)
    assert m.is_watertight and m.is_winding_consistent


def test_row_spans_enumerate_exactly_the_triangle_lattice_points():
    shapes = {(3, 1, -2, 5), (7, -3, 2, 4)}  # general triangles too, not just RTIN shapes
    for h in (1, 2, 4, 8):
        for sx in (-1, 1):
            for sy in (-1, 1):
                shapes.add((2 * h * sx, 0, h * sx, h * sy))  # axis hypotenuse, apex on either side
                shapes.add((0, 2 * h * sy, h * sx, h * sy))
                shapes.add((2 * h * sx, 2 * h * sy, 2 * h * sx, 0))  # diagonal hypotenuse
                shapes.add((2 * h * sx, 2 * h * sy, 0, 2 * h * sy))
    for bx, by, cx, cy in shapes:
        det = bx * cy - by * cx
        if det == 0:
            continue
        ys, lo, counts = am._row_spans(bx, by, cx, cy)
        got = {(int(lo[i]) + j, int(ys[i])) for i in range(ys.size) for j in range(int(counts[i]))}
        want = set()
        for y in range(min(0, by, cy), max(0, by, cy) + 1):
            for x in range(min(0, bx, cx), max(0, bx, cx) + 1):
                l1 = (x * cy - y * cx) / det
                l2 = (bx * y - by * x) / det
                if l1 >= 0 and l2 >= 0 and l1 + l2 <= 1:
                    want.add((x, y))
        assert got == want, (bx, by, cx, cy)


def test_encoders_round_trip_meshdata():
    md = adaptive_heightfield_mesh(smooth_random_field((40, 50), seed=3), 100.0, 8.0, 2.0, 0.05)
    stl = trimesh.load(io.BytesIO(encode_stl(md)), file_type="stl")
    obj = trimesh.load(io.BytesIO(encode_obj(md)), file_type="obj")
    for m in (stl, obj):
        assert m.is_watertight and m.is_winding_consistent
        assert len(m.faces) == len(md.faces)
        assert float(m.volume) == pytest.approx(float(_mesh(md).volume), rel=1e-5)


def test_benchmark_adaptive_stl_beats_grid():
    _, H = api_main._build_layers_from_image(
        api_main.BENCHMARK_IMAGE_PATH,
        target_res=1024,
        bg_removal=True,
        hd_restore=True,
        ai_mode="off",
        params=db.default_relief_params(),
    )
    rows, cols = H.shape
    t0 = time.perf_counter()
    md = adaptive_heightfield_mesh(H, 1000.0, 15.0, 5.0, 0.02)
    elapsed = time.perf_counter() - t0
    m = _mesh(md)
    assert m.is_watertight and m.is_winding_consistent
    assert md.max_error_mm <= 0.02 + 1e-6
    assert len(md.faces) < 2 * (rows - 1) * (cols - 1)
    if os.environ.get("MONOLITH_CHECK_TIMINGS") == "1":
        assert elapsed < 10.0, f"adaptive mesh at {cols}x{rows} took {elapsed:.2f} s"


def test_export_endpoint_adaptive_headers(isolated_env):
    with TestClient(api_main.app) as client:
        r = client.post(
            "/api/projects/create",
            files={"file": ("p.png", make_lineart_png(128), "image/png")},
            data={"target_res": "256"},
        )
        assert r.status_code == 200, r.text
        pid = r.json()["id"]

        exp = client.post(
            f"/api/projects/{pid}/export",
            json={"format": "stl"},
            headers={"Origin": "http://localhost:5173"},
        )
        assert exp.status_code == 200, exp.text
        assert exp.headers["x-mesh-mode"] == "adaptive"
        n_tri = int(exp.headers["x-triangles"])
        assert float(exp.headers["x-max-error-mm"]) <= 0.02
        mesh = trimesh.load(io.BytesIO(exp.content), file_type="stl")
        assert mesh.is_watertight and len(mesh.faces) == n_tri
        exposed = exp.headers["access-control-expose-headers"].lower()
        for name in ("x-mesh-mode", "x-triangles", "x-max-error-mm"):
            assert name in exposed

        obj = client.post(f"/api/projects/{pid}/export", json={"format": "obj", "mesh_tolerance_mm": 0.1})
        assert obj.status_code == 200 and obj.headers["x-mesh-mode"] == "adaptive"
        assert float(obj.headers["x-max-error-mm"]) <= 0.1

        grid = client.post(f"/api/projects/{pid}/export", json={"format": "stl", "mesh_mode": "grid",
                                                                   "mesh_resolution": 64})
        assert grid.status_code == 200 and grid.headers["x-mesh-mode"] == "grid"
        assert "x-max-error-mm" not in grid.headers

        bad = client.post(f"/api/projects/{pid}/export", json={"format": "stl", "mesh_tolerance_mm": 0.0})
        assert bad.status_code == 422

        assert client.post(f"/api/projects/{pid}/recompute", json={"base_z_mm": 0.0}).status_code == 200
        cut = client.post(f"/api/projects/{pid}/export", json={"format": "stl"})
        assert cut.status_code == 200
        assert cut.headers["x-mesh-mode"] == "grid"
        assert "x-max-error-mm" not in cut.headers
        assert int(cut.headers["x-triangles"]) == len(trimesh.load(io.BytesIO(cut.content), file_type="stl").faces)


def test_export_endpoint_refuses_oversized_adaptive_mesh(isolated_env, monkeypatch):
    with TestClient(api_main.app) as client:
        r = client.post(
            "/api/projects/create",
            files={"file": ("p.png", make_lineart_png(128), "image/png")},
            data={"target_res": "256"},
        )
        assert r.status_code == 200, r.text
        pid = r.json()["id"]
        monkeypatch.setenv("MONOLITH_MAX_MESH_TRIANGLES", "200")
        for fmt in ("stl", "obj"):
            big = client.post(f"/api/projects/{pid}/export", json={"format": fmt})
            assert big.status_code == 413, big.text
            detail = big.json()["detail"]
            assert "triangles" in detail and "mesh_tolerance_mm" in detail and "grid" in detail
        grid = client.post(f"/api/projects/{pid}/export", json={"format": "stl", "mesh_mode": "grid",
                                                                   "mesh_resolution": 64})
        assert grid.status_code == 200
