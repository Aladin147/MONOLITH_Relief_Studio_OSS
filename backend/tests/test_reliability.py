"""Regression tests for reliability / precision fixes (brush integrity, units, concurrency, validation)."""

from __future__ import annotations

import io

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient

from backend import main as api_main
from backend.tests.conftest import make_lineart_png

FULL_RES = 2048  # > PREVIEW_MAX_DIM so the preview and full-res grids differ


def _create(client: TestClient, res: int = FULL_RES, size: int = 512, style: str | None = None) -> str:
    data = {"target_res": str(res)}
    if style is not None:
        data["style"] = style
    r = client.post(
        "/api/projects/create",
        files={"file": ("p.png", make_lineart_png(size), "image/png")},
        data=data,
    )
    assert r.status_code == 200, r.text
    return r.json()["id"]


def _get_tex(client: TestClient, pid: str, max_dim: int = 1024) -> tuple[int, int, np.ndarray]:
    r = client.get(f"/api/projects/{pid}/float32-texture?max_dim={max_dim}")
    assert r.status_code == 200
    w, h = int(r.headers["x-width"]), int(r.headers["x-height"])
    return w, h, np.frombuffer(r.content, dtype=np.float32).reshape(h, w).copy()


def _put(client: TestClient, pid: str, w: int, h: int, arr: np.ndarray, mode: str | None = None):
    qs = f"width={w}&height={h}" + (f"&mode={mode}" if mode else "")
    return client.put(f"/api/projects/{pid}/float32-texture?{qs}", content=arr.astype("<f4").tobytes())


def _brush_delta(pid: str) -> np.ndarray:
    return api_main._RAM_CACHE[pid].full_layers["H_brush_delta"]


# ---------------------------------------------------------------------------
# 1. Brush uploads must only change what the user painted
# ---------------------------------------------------------------------------


def test_noop_absolute_stroke_after_preview_only_recompute_changes_nothing(isolated_env):
    with TestClient(api_main.app) as client:
        pid = _create(client)
        client.post(f"/api/projects/{pid}/recompute", json={"gamma": 0.9})
        client.post(f"/api/projects/{pid}/recompute?preview_only=true", json={"gamma": 0.7})
        w, h, tex = _get_tex(client, pid)
        assert _put(client, pid, w, h, tex).status_code == 200
        assert float(np.abs(_brush_delta(pid)).max()) < 1e-6


def test_noop_absolute_stroke_after_refetch_following_brush_changes_nothing(isolated_env):
    with TestClient(api_main.app) as client:
        pid = _create(client)
        w, h, tex = _get_tex(client, pid)
        edited = tex.copy()
        edited[h // 2 - 20 : h // 2 + 20, w // 2 - 20 : w // 2 + 20] += 0.05
        assert _put(client, pid, w, h, edited).status_code == 200
        before = _brush_delta(pid).copy()

        w, h, tex2 = _get_tex(client, pid)  # e.g. project re-opened
        assert _put(client, pid, w, h, tex2).status_code == 200
        np.testing.assert_allclose(_brush_delta(pid), before, atol=1e-6)


def test_delta_mode_only_touches_stroke_footprint(isolated_env):
    with TestClient(api_main.app) as client:
        pid = _create(client)
        w, h, _ = _get_tex(client, pid)
        delta = np.zeros((h, w), dtype=np.float32)
        y0, x0 = h // 2 - 30, w // 2 - 30
        delta[y0 : y0 + 60, x0 : x0 + 60] = 0.04
        assert _put(client, pid, w, h, delta, mode="delta").status_code == 200

        full = _brush_delta(pid)
        fh, fw = full.shape
        sy, sx = fh / h, fw / w
        # Outside the (slightly dilated) stroke footprint nothing changes at all.
        mask = np.ones_like(full, dtype=bool)
        mask[int((y0 - 2) * sy) : int((y0 + 62) * sy), int((x0 - 2) * sx) : int((x0 + 62) * sx)] = False
        assert float(np.abs(full[mask]).max()) == 0.0
        # Linear upsampling never overshoots the painted amplitude.
        assert float(full.max()) <= 0.04 + 1e-6
        assert float(full.max()) > 0.035


def test_delta_mode_rejects_unknown_mode(isolated_env):
    with TestClient(api_main.app) as client:
        pid = _create(client, res=512, size=256)
        w, h, tex = _get_tex(client, pid)
        assert _put(client, pid, w, h, tex, mode="bogus").status_code == 400


# ---------------------------------------------------------------------------
# 2. Frame values are true millimetres; legacy stored values migrate losslessly
# ---------------------------------------------------------------------------


def test_legacy_stored_frame_is_migrated_on_read(isolated_env):
    import json

    from backend import db

    db.init_db()
    legacy = db.default_relief_params()
    legacy["frame"] = {"enabled": True, "preset": "gallery_classic", "width_mm": 2.2, "thickness_mm": 1.1,
                       "back_protrusion_mm": 0.0, "bbox": [0.16, 0.48, 0.84, 0.96]}
    proj = db.create_project(name="legacy", original_filename="x.png", width=10, height=10, params=legacy)
    # Simulate a row written by the previous release (no "units" tag).
    with db.get_connection() as conn:
        conn.execute("UPDATE projects SET params_json = ? WHERE id = ?", (json.dumps(legacy), proj["id"]))
    frame = db.get_project(proj["id"])["params"]["frame"]
    assert frame["units"] == "mm"
    assert frame["width_mm"] == pytest.approx(22.0)
    assert frame["thickness_mm"] == pytest.approx(5.5)


def test_frame_width_is_continuous_through_api(isolated_env):
    with TestClient(api_main.app) as client:
        pid = _create(client, res=1024, size=256)
        covered = []
        for wmm in (24.9, 25.0):
            r = client.post(f"/api/projects/{pid}/recompute",
                            json={"h_min": 0.0, "frame": {"enabled": True, "width_mm": wmm,
                                                          "bbox": [0.05, 0.05, 0.95, 0.95]}})
            assert r.status_code == 200
            covered.append(float(np.mean(api_main._RAM_CACHE[pid].H_current > 0)))
        assert abs(covered[0] - covered[1]) < 0.01, covered


# ---------------------------------------------------------------------------
# 3. Out-of-order slider requests can never leave stale params behind
# ---------------------------------------------------------------------------


def test_stale_recompute_is_dropped_and_vault_matches_cache(isolated_env):
    with TestClient(api_main.app) as client:
        pid = _create(client, res=512, size=256)
        newer = client.post(f"/api/projects/{pid}/recompute?seq=2000", json={"gamma": 1.3})
        assert newer.status_code == 200 and newer.json()["stale"] is False
        h_after_newer = api_main._RAM_CACHE[pid].H_current.copy()

        # An older request that the threadpool happened to run last.
        older = client.post(f"/api/projects/{pid}/recompute?seq=1000", json={"gamma": 0.6})
        assert older.status_code == 200
        assert older.json()["stale"] is True
        assert older.json()["params"]["gamma"] == pytest.approx(1.3)

        assert client.get(f"/api/projects/{pid}").json()["params"]["gamma"] == pytest.approx(1.3)
        np.testing.assert_array_equal(api_main._RAM_CACHE[pid].H_current, h_after_newer)


def test_invalid_param_type_is_422_not_500(isolated_env):
    with TestClient(api_main.app, raise_server_exceptions=False) as client:
        pid = _create(client, res=512, size=256)
        r = client.post(f"/api/projects/{pid}/recompute", json={"physical_width_mm": "abc"})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# 4. Input validation & upload robustness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"relief_z_mm": -15},
        {"physical_width_mm": 0},
        {"gamma": 0},
        {"h_min": 2.0},
        {"frame": {"bbox": [0, 0, 2, 1]}},
        {"frame": {"width_mm": -1}},
    ],
)
def test_out_of_range_params_rejected(isolated_env, payload):
    with TestClient(api_main.app, raise_server_exceptions=False) as client:
        pid = _create(client, res=256, size=128)
        before = client.get(f"/api/projects/{pid}").json()["params"]
        r = client.post(f"/api/projects/{pid}/recompute", json=payload)
        assert r.status_code == 422, r.text
        assert client.get(f"/api/projects/{pid}").json()["params"] == before


@pytest.mark.parametrize("mesh_res", [0, -5, 100000])
def test_export_mesh_resolution_is_bounded(isolated_env, mesh_res):
    with TestClient(api_main.app, raise_server_exceptions=False) as client:
        pid = _create(client, res=256, size=128)
        r = client.post(f"/api/projects/{pid}/export", json={"format": "stl", "mesh_mode": "grid", "mesh_resolution": mesh_res})
        assert r.status_code == 422


def test_corrupt_upload_is_400_and_leaves_no_orphan(isolated_env):
    with TestClient(api_main.app, raise_server_exceptions=False) as client:
        projects_dir = isolated_env["data_dir"] / "projects"
        before = set(projects_dir.iterdir()) if projects_dir.exists() else set()
        r = client.post(
            "/api/projects/create",
            files={"file": ("broken.png", b"definitely not an image", "image/png")},
            data={"target_res": "256"},
        )
        assert r.status_code == 400
        after = set(projects_dir.iterdir()) if projects_dir.exists() else set()
        assert after == before


def test_oversized_image_rejected_before_decode(monkeypatch):
    from backend.pipeline import preprocessor

    monkeypatch.setattr(preprocessor, "MAX_SOURCE_PIXELS", 1000)
    with pytest.raises(ValueError, match="safety limit"):
        preprocessor.preprocess_lineart(make_lineart_png(64), target_res=64)


@pytest.mark.parametrize("orientation", range(1, 9))
def test_exif_orientation_matches_pil(orientation):
    import io

    from PIL import Image, ImageOps

    from backend.pipeline.preprocessor import _load_luminance_float32

    rng = np.random.default_rng(orientation)
    img = Image.fromarray(rng.integers(0, 256, size=(24, 40), dtype=np.uint8), mode="L")
    exif = Image.Exif()
    exif[0x0112] = orientation
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes(), quality=100, subsampling=0)
    data = buf.getvalue()

    expected = np.asarray(ImageOps.exif_transpose(Image.open(io.BytesIO(data)))).astype(np.float32) / 255.0
    got = _load_luminance_float32(data)
    assert got.shape == expected.shape
    assert float(np.abs(got - expected).max()) < 0.02


# ---------------------------------------------------------------------------
# 5. Resolution independence: the same drawing + params look the same at 1K and 2K
# ---------------------------------------------------------------------------


def _relief_at(client: TestClient, res: int) -> np.ndarray:
    pid = _create(client, res=res, size=512)
    r = client.post(f"/api/projects/{pid}/recompute", json={"groove_sigma": 12.0, "w_groove": 0.8,
                                                            "draft_angle_deg": 0.0})
    assert r.status_code == 200
    w, h, tex = _get_tex(client, pid, max_dim=512)
    return tex


def test_groove_width_is_resolution_independent(isolated_env):
    with TestClient(api_main.app) as client:
        import cv2

        a = _relief_at(client, 1024)
        b = _relief_at(client, 2048)
        # Compare inside the silhouette (the 1-px outline step itself is sampling noise
        # at 512 px). Raw-pixel groove sigma differed by ~0.17 here; now ~0.03.
        inside = ((a > 0) & (b > 0)).astype(np.uint8)
        interior = cv2.erode(inside, np.ones((9, 9), np.uint8)) > 0
        diff = float(np.abs(a - b)[interior].max())
        assert diff < 0.05, diff


def test_legacy_pixel_groove_sigma_is_migrated(isolated_env):
    import json

    from backend import db

    db.init_db()
    legacy = db.default_relief_params()
    legacy.pop("groove_sigma_units")
    legacy["groove_sigma"] = 2.5  # raw pixels on a 2048 px project
    proj = db.create_project(name="g", original_filename="x.png", width=2048, height=1500, params=legacy)
    with db.get_connection() as conn:
        conn.execute("UPDATE projects SET params_json = ? WHERE id = ?", (json.dumps(legacy), proj["id"]))
    params = db.get_project(proj["id"])["params"]
    assert params["groove_sigma_units"] == "px@4096"
    assert params["groove_sigma"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# 6. Preprocessing: smoother enlarged edges without losing 1-source-pixel lines
# ---------------------------------------------------------------------------


def _boundary_radius_std(mask: np.ndarray) -> float:
    import cv2

    m = (mask > 0.5).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    (cx, cy), _ = cv2.minEnclosingCircle(c.astype(np.float32))
    return float(np.hypot(c[:, 0] - cx, c[:, 1] - cy).std())


def test_enlarged_low_res_source_has_smoother_outline():
    import cv2

    from backend.pipeline import preprocess_lineart

    src = np.zeros((64, 64), np.uint8)
    cv2.circle(src, (32, 32), 24, 255, -1)
    prep = preprocess_lineart(src, target_res=1024)  # 16x enlargement
    # Fixed 1 px blur left 16 px source stair-steps (std ~3.7 px); now ~2.8 px.
    assert _boundary_radius_std(prep["M_sil"]) < 3.2


def test_one_source_pixel_line_survives_enlargement():
    import cv2

    from backend.pipeline import preprocess_lineart

    src = np.zeros((64, 64), np.uint8)
    cv2.circle(src, (32, 32), 26, 255, -1)
    src[32, 4:60] = 0  # 1 px contour line splitting the disc
    prep = preprocess_lineart(src, target_res=1024)
    reg = ((prep["M_regions"] > 0.5) & (prep["M_sil"] > 0.5)).astype(np.uint8)
    n, _ = cv2.connectedComponents(reg, connectivity=8)
    assert n - 1 == 2


# ---------------------------------------------------------------------------
# 7. Draft angle is a true geometric guarantee
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("theta", [5.0, 15.0, 30.0])
def test_draft_angle_is_exact(theta):
    import math

    import cv2

    from backend.pipeline import compute_basis_layers, preprocess_lineart, recompute_relief

    img = np.zeros((256, 256), np.uint8)
    cv2.circle(img, (128, 128), 100, 255, -1)
    layers = compute_basis_layers(preprocess_lineart(img, target_res=256))
    phys_w, rel_z = 100.0, 30.0
    H = recompute_relief(layers, {"draft_angle_deg": theta, "h_min": 0.08},
                         physical_width_mm=phys_w, relief_z_mm=rel_z)
    fg = layers["M_sil"] > 0.5
    d_mm = (layers["d_sil"].astype(np.float64) - 0.5) * (phys_w / 256)
    rise_per_mm = np.max(H[fg] * rel_z / np.maximum(d_mm[fg], 1e-9))
    wall_deg = math.degrees(math.atan(1.0 / rise_per_mm))
    # Previously a 30 deg request produced ~2 deg walls.
    assert wall_deg == pytest.approx(theta, abs=0.05)


# ---------------------------------------------------------------------------
# 8. Upscale / preprocess keep Local Edit sculpting
# ---------------------------------------------------------------------------


def test_upscale_preserves_brush_strokes_in_place(isolated_env):
    with TestClient(api_main.app) as client:
        pid = _create(client, res=512, size=256)
        w, h, _ = _get_tex(client, pid)
        delta = np.zeros((h, w), dtype=np.float32)
        # A carving stroke: a raise could be clipped at the top of the range (Sculpt crests reach 1).
        delta[h // 2 - 40 : h // 2 - 20, w // 2 - 10 : w // 2 + 10] = -0.05
        assert _put(client, pid, w, h, delta, mode="delta").status_code == 200

        r = client.post(f"/api/projects/{pid}/upscale", json={"target_res": 1024})
        assert r.status_code == 200
        full = api_main._RAM_CACHE[pid].full_layers["H_brush_delta"]
        fh, fw = full.shape
        assert max(fh, fw) == 1024
        # Stroke is still there, at the same relative position.
        assert float(full[int(fh * (0.5 - 30 / h)), fw // 2]) == pytest.approx(-0.05, abs=0.005)
        assert float(np.abs(full[: fh // 4]).max()) == 0.0


# ---------------------------------------------------------------------------
# 9. Classic black-ink-on-white-paper drawings are understood
# ---------------------------------------------------------------------------


def _ink_on_paper(size: int = 512, gap: int = 0) -> np.ndarray:
    import cv2

    img = np.full((size, size), 255, np.uint8)
    c, r = size // 2, size // 3
    cv2.circle(img, (c, c), r, 0, 6)  # outer contour
    cv2.line(img, (c - r, c), (c + r, c), 0, 6)  # interior contour line
    if gap:
        img[c - r - 8 : c - r + 8, c - gap // 2 : c + gap // 2] = 255  # break in the outline
    return img


@pytest.mark.parametrize("gap", [0, 4])
def test_black_on_white_drawing_gets_correct_silhouette(gap):
    import cv2

    from backend.pipeline import preprocess_lineart

    prep = preprocess_lineart(_ink_on_paper(gap=gap), target_res=512)
    assert prep["polarity"] == "black_on_white"
    sil = prep["M_sil"] > 0.5
    r = 512 // 3
    area_ratio = float(sil.sum()) / (np.pi * r * r)
    assert 0.9 < area_ratio < 1.1, area_ratio  # the disc, not the whole sheet of paper
    assert not sil[:5, :].any() and not sil[:, :5].any()
    reg = ((prep["M_regions"] > 0.5) & sil).astype(np.uint8)
    n, _ = cv2.connectedComponents(reg, connectivity=8)
    assert n - 1 == 2  # the two halves separated by the interior line


def test_artist_white_on_black_style_is_auto_detected():
    from backend.pipeline import preprocess_lineart

    prep = preprocess_lineart(make_lineart_png(256), target_res=256)
    assert prep["polarity"] == "white_on_black"


def test_api_black_on_white_upload_and_polarity_validation(isolated_env):
    import cv2

    ok, buf = cv2.imencode(".png", _ink_on_paper())
    with TestClient(api_main.app) as client:
        r = client.post("/api/projects/create",
                        files={"file": ("ink.png", buf.tobytes(), "image/png")},
                        data={"target_res": "512"})
        assert r.status_code == 200
        pid = r.json()["id"]
        assert r.json()["polarity"] == "auto"
        sil = api_main._RAM_CACHE[pid].full_layers["M_sil"] > 0.5
        assert sil.mean() < 0.5  # not the whole sheet of paper
        bad = client.post("/api/projects/create",
                          files={"file": ("ink.png", buf.tobytes(), "image/png")},
                          data={"target_res": "512", "polarity": "sideways"})
        assert bad.status_code == 400
        r2 = client.post(f"/api/projects/{pid}/preprocess",
                         json={"target_res": 512, "polarity": "white_on_black"})
        assert r2.status_code == 200 and r2.json()["polarity"] == "white_on_black"


# ---------------------------------------------------------------------------
# 10. FFT smoothing has no wrap-around at the canvas edges
# ---------------------------------------------------------------------------


def test_subject_cropped_by_canvas_edge_does_not_sag():
    from backend.pipeline import compute_basis_layers, preprocess_lineart

    n = 256
    img = np.zeros((n, n), np.uint8)
    img[:, : n // 2] = 255  # touches the left edge; the right half is empty
    H_local = compute_basis_layers(preprocess_lineart(img, target_res=n))["H_local"]
    # Periodic FFT pulled the left edge down to ~0.65 of the plateau.
    assert float(H_local[n // 2, 1]) > 0.97 * float(H_local[n // 2, n // 4])


# ---------------------------------------------------------------------------
# 11. Slab STL: fan-triangulated bottom is identical geometry at ~half the size
# ---------------------------------------------------------------------------


def test_slab_stl_bottom_fan_is_watertight_and_volume_exact():
    import io

    import trimesh

    from backend.pipeline import stl_exporter as ex

    rng = np.random.default_rng(0)
    H = np.clip(rng.random((97, 131)).astype(np.float32), 0, 1)
    fan = trimesh.load(io.BytesIO(ex.heightmap_to_watertight_stl(H, 300.0, 10.0, 4.0, max_grid=0)),
                       file_type="stl")
    X, Y, Zt, Zb, cm = ex._prepare_mesh_grid(H, 300.0, 10.0, 4.0, 0)
    old_tris = ex._assemble_mesh_triangles(np.stack([X, Y, Zt], -1), np.stack([X, Y, Zb], -1), cm)
    old = trimesh.Trimesh(**trimesh.triangles.to_kwargs(old_tris.astype(np.float64)))
    assert fan.is_watertight and fan.is_winding_consistent
    assert float(fan.volume) == pytest.approx(float(old.volume), rel=1e-5)
    assert len(fan.faces) < 0.55 * len(old.faces)

    obj = trimesh.load(io.BytesIO(ex.heightmap_to_obj(H, 300.0, 10.0, 4.0, max_grid=0)), file_type="obj")
    assert obj.is_watertight and float(obj.volume) == pytest.approx(float(old.volume), rel=1e-5)


def test_locked_cache_never_hands_out_a_swapped_out_entry(isolated_env, monkeypatch):
    import copy

    with TestClient(api_main.app) as client:
        pid = _create(client, res=256, size=128)
        real = api_main._load_or_rebuild_cache
        calls = []

        def racy_load(project_id):
            calls.append(project_id)
            cache = real(project_id)
            # First call returns an entry that a concurrent /preprocess has just replaced.
            return copy.copy(cache) if len(calls) == 1 else cache

        monkeypatch.setattr(api_main, "_load_or_rebuild_cache", racy_load)
        with api_main._locked_cache(pid) as cache:
            assert cache is api_main._RAM_CACHE[pid]
        assert len(calls) == 2


# ---------------------------------------------------------------------------
# 12. Export dimensions are the project's
# ---------------------------------------------------------------------------

_DIMS = {"physical_width_mm": 500.0, "relief_z_mm": 10.0, "base_z_mm": 3.0}


def _stl_bounds(client: TestClient, pid: str, **body):
    r = client.post(f"/api/projects/{pid}/export", json={"format": "stl", "mesh_mode": "grid", "mesh_resolution": 64, **body})
    assert r.status_code == 200, r.text
    return trimesh.load(io.BytesIO(r.content), file_type="stl").bounds


def test_export_uses_saved_dimensions_when_omitted(isolated_env):
    with TestClient(api_main.app) as client:
        pid = _create(client, res=256, size=128)
        client.post(f"/api/projects/{pid}/recompute", json=_DIMS)
        b = _stl_bounds(client, pid)
        assert b[1][0] - b[0][0] == pytest.approx(500.0, abs=1e-3)
        assert b[1][2] <= 13.0 + 1e-3


def test_export_dims_follow_preview_only_recompute(isolated_env):
    with TestClient(api_main.app) as client:
        pid = _create(client, res=256, size=128)
        client.post(f"/api/projects/{pid}/recompute?preview_only=true", json=_DIMS)
        b = _stl_bounds(client, pid)
        assert b[1][0] - b[0][0] == pytest.approx(500.0, abs=1e-3)


def test_export_rejects_mismatched_dimensions(isolated_env):
    with TestClient(api_main.app) as client:
        pid = _create(client, res=256, size=128)
        client.post(f"/api/projects/{pid}/recompute", json=_DIMS)
        r = client.post(f"/api/projects/{pid}/export",
                        json={"format": "stl", "mesh_mode": "grid", "mesh_resolution": 64, "width_mm": 1000.0})
        assert r.status_code == 409
        assert "apply the settings first" in r.json()["detail"]
        ok = client.post(f"/api/projects/{pid}/export",
                         json={"format": "stl", "mesh_mode": "grid", "mesh_resolution": 64, "width_mm": 500.0})
        assert ok.status_code == 200


def test_texture_reports_unique_levels(isolated_env):
    with TestClient(api_main.app) as client:
        pid = _create(client, res=256, size=128)
        r = client.get(f"/api/projects/{pid}/float32-texture?max_dim=1024")
        assert r.status_code == 200
        assert int(r.headers["x-unique-levels"]) > 1


# ---------------------------------------------------------------------------
# 13. Huge sources: the pixel limit holds, and big scans decode at reduced size
# ---------------------------------------------------------------------------
def _disc_image(mode, size, radius, fill):
    from PIL import Image, ImageDraw

    img = Image.new(mode, size, 0)
    cx, cy = size[0] // 2, size[1] // 2
    ImageDraw.Draw(img).ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=fill)
    return img


def _spy_imread(monkeypatch):
    from backend.pipeline import preprocessor as pre

    flags = []
    real = pre.cv2.imread

    def spy(path, flag=pre.cv2.IMREAD_COLOR):
        flags.append(flag)
        return real(path, flag)

    monkeypatch.setattr(pre.cv2, "imread", spy)
    return flags


def test_bomb_header_rejected_even_when_pil_refuses(tmp_path, monkeypatch):
    from PIL import Image

    from backend.pipeline import preprocessor as pre

    monkeypatch.setattr(pre, "MAX_SOURCE_PIXELS", 100_000_000)
    p = tmp_path / "big.png"
    Image.new("1", (21000, 20000), 0).save(p)  # ~51 KB on disk, 420 MP declared
    with pytest.raises(ValueError, match="safety limit"):
        pre.preprocess_lineart(p, target_res=1024)


def test_full_decode_modes_have_a_lower_pixel_cap(tmp_path, monkeypatch):
    from PIL import Image

    from backend.pipeline import preprocessor as pre

    monkeypatch.setattr(pre, "MAX_SOURCE_PIXELS", 10_000_000)
    monkeypatch.setattr(pre, "MAX_FULL_DECODE_PIXELS", 1_000_000)
    monkeypatch.setattr(pre, "LARGE_SOURCE_PIXELS", 2_000_000)
    rgba = tmp_path / "rgba.png"
    Image.new("RGBA", (2000, 1500), (0, 0, 0, 0)).save(rgba)  # 3 MP > full-decode cap
    flags = _spy_imread(monkeypatch)
    with pytest.raises(ValueError, match="safety limit") as exc:
        pre.preprocess_lineart(rgba, target_res=256)
    assert "RGBA" in str(exc.value) and "grayscale or RGB" in str(exc.value)
    assert flags == []  # refused before any decode

    gray = tmp_path / "gray.png"
    Image.new("L", (2000, 1500), 0).save(gray)  # same size, reducible mode: large path
    out = pre.preprocess_lineart(gray, target_res=256)
    assert max(out["original_shape"]) == 256
    assert flags  # decoded


def test_large_scan_uses_grayscale_preshrink(tmp_path, monkeypatch):
    import cv2

    from backend.pipeline import preprocessor as pre

    p = tmp_path / "scan.png"
    _disc_image("L", (12000, 10000), 3000, 255).save(p)
    flags = _spy_imread(monkeypatch)
    out = pre.preprocess_lineart(p, target_res=1024)
    assert flags == [cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION]
    assert not flags[0] & (
        cv2.IMREAD_REDUCED_GRAYSCALE_2 | cv2.IMREAD_REDUCED_GRAYSCALE_4 | cv2.IMREAD_REDUCED_GRAYSCALE_8
    )
    assert out["original_shape"] == (853, 1024)  # area pre-shrink to target_res
    area_ratio = float(out["M_sil"].sum()) / out["M_sil"].size
    expected = np.pi * 3000 ** 2 / (12000 * 10000)
    assert area_ratio == pytest.approx(expected, rel=0.01)


def test_large_path_keeps_thin_lines(tmp_path, monkeypatch):
    import cv2
    from PIL import Image

    from backend.pipeline import preprocessor as pre

    a = np.full((4096, 4096), 255, np.uint8)
    a[::500, :] = 0
    a[:, ::500] = 0
    p = tmp_path / "grid.png"
    Image.fromarray(a, "L").save(p)
    monkeypatch.setattr(pre, "LARGE_SOURCE_PIXELS", 1000)
    got = pre._load_luminance_float32(p, 1024)
    ref = cv2.resize(a, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    assert got.min() <= 0.8
    dark, dark_ref = int((got < 0.9).sum()), int((ref < 0.9).sum())
    assert dark_ref > 0
    assert abs(dark - dark_ref) <= 0.02 * dark_ref


def test_large_path_skips_alpha(tmp_path, monkeypatch):
    import cv2
    from PIL import Image

    from backend.pipeline import preprocessor as pre

    p = tmp_path / "alpha.png"
    Image.new("RGBA", (6000, 5000), (255, 255, 255, 0)).save(p)
    monkeypatch.setattr(pre, "LARGE_SOURCE_PIXELS", 1000)
    flags = _spy_imread(monkeypatch)
    pre.preprocess_lineart(p, target_res=512)
    assert flags == [cv2.IMREAD_UNCHANGED]


@pytest.mark.parametrize("large", [False, True])
def test_exif_orientation_applied_exactly_once(tmp_path, monkeypatch, large):
    from PIL import Image, ImageOps

    from backend.pipeline import preprocessor as pre

    img = Image.new("L", (3000, 1000), 255)
    img.paste(0, (0, 0, 300, 300))  # black square, top-left
    exif = Image.Exif()
    exif[0x0112] = 6
    p = tmp_path / "rot.jpg"
    img.save(p, exif=exif.tobytes(), quality=100, subsampling=0)
    if large:
        monkeypatch.setattr(pre, "LARGE_SOURCE_PIXELS", 1000)
    expected = np.asarray(ImageOps.exif_transpose(Image.open(p))).astype(np.float32) / 255.0
    got = pre._load_luminance_float32(p, 1000 if large else None)
    assert got.shape[0] > got.shape[1]
    h, w = got.shape
    eh, ew = expected.shape
    # Locate the mark's corner in PIL's result and require the same corner here.
    corner = lambda a: (float(a[: a.shape[0] // 8, : a.shape[1] // 3].mean()) < 0.3,
                        float(a[-(a.shape[0] // 8):, : a.shape[1] // 3].mean()) < 0.3,
                        float(a[: a.shape[0] // 8, -(a.shape[1] // 3):].mean()) < 0.3,
                        float(a[-(a.shape[0] // 8):, -(a.shape[1] // 3):].mean()) < 0.3)
    assert sum(corner(expected)) == 1
    assert corner(got) == corner(expected)


# ---------------------------------------------------------------------------
# 14. Memory: the RAM cache is bounded by bytes, and each project is lean
# ---------------------------------------------------------------------------


def test_cache_footprint_per_pixel(isolated_env):
    with TestClient(api_main.app) as client:
        pid = _create(client, res=2048, style="classic")
        c = api_main._RAM_CACHE[pid]
        h, w = c.H_current.shape
        ratio = api_main._cache_nbytes(c) / (h * w * 4)
        assert ratio <= 13.5  # was 17.55 before the lean cache
        # Sculpt: its basis (~30 B/px), the parts analysis (fixed 2048 grid, ~37 MB) and the
        # preview's own basis; 20.7 float32 pixels per pixel measured at 2048 (about 14 at 4K).
        pid = _create(client, res=2048)
        c = api_main._RAM_CACHE[pid]
        assert api_main._cache_nbytes(c) / (h * w * 4) <= 22.0


def test_ram_budget_evicts_lru_by_bytes(isolated_env, monkeypatch):
    with TestClient(api_main.app) as client:
        a = _create(client, res=512)
        one = api_main._cache_nbytes(api_main._RAM_CACHE[a])
        monkeypatch.setattr(api_main, "_RAM_BUDGET_BYTES", int(one * 1.5))
        before = api_main._RAM_CACHE[a].H_current.copy()
        b = _create(client, res=512)
        c = _create(client, res=512)
        assert list(api_main._RAM_CACHE) == [c]
        assert a not in api_main._RAM_CACHE and b not in api_main._RAM_CACHE
        # Reopening an evicted project restores exactly the same relief.
        assert client.get(f"/api/projects/{a}/preview-png").status_code == 200
        np.testing.assert_array_equal(api_main._RAM_CACHE[a].H_current, before)


def _evict_all_but_new(client, monkeypatch, a):
    one = api_main._cache_nbytes(api_main._RAM_CACHE[a])
    monkeypatch.setattr(api_main, "_RAM_BUDGET_BYTES", int(one * 1.2))
    b = _create(client, res=1024, size=256)
    assert a not in api_main._RAM_CACHE
    return b


def _brush_square_delta(client, pid):
    w, h, _ = _get_tex(client, pid)
    delta = np.zeros((h, w), np.float32)
    delta[h // 2 - 40 : h // 2 + 40, w // 2 - 40 : w // 2 + 40] = 0.05
    assert _put(client, pid, w, h, delta, mode="delta").status_code == 200


def test_brushed_then_byte_evicted_reload_keeps_brush_edits(isolated_env, monkeypatch):
    with TestClient(api_main.app) as client:
        a = _create(client, res=1024, size=256)
        _brush_square_delta(client, a)
        bd = api_main._RAM_CACHE[a].full_layers["H_brush_delta"].copy()
        Hc = api_main._RAM_CACHE[a].H_current.copy()
        assert np.abs(bd).max() > 0.04
        _evict_all_but_new(client, monkeypatch, a)
        monkeypatch.setattr(api_main, "_RAM_BUDGET_BYTES", 1 << 40)
        assert client.get(f"/api/projects/{a}/preview-png").status_code == 200
        c = api_main._RAM_CACHE[a]
        np.testing.assert_array_equal(c.full_layers["H_brush_delta"], bd)
        np.testing.assert_array_equal(c.H_current, Hc)


def test_brushed_preview_only_then_evicted_export_keeps_brush_edits(isolated_env, monkeypatch):
    with TestClient(api_main.app) as client:
        a = _create(client, res=1024, size=256)
        _brush_square_delta(client, a)
        bd = api_main._RAM_CACHE[a].full_layers["H_brush_delta"].copy()
        r = client.post(f"/api/projects/{a}/recompute?preview_only=true", json={"gamma": 0.7})
        assert r.status_code == 200
        _evict_all_but_new(client, monkeypatch, a)
        monkeypatch.setattr(api_main, "_RAM_BUDGET_BYTES", 1 << 40)
        r = client.post(f"/api/projects/{a}/export", json={"format": "png16"})
        assert r.status_code == 200
        c = api_main._RAM_CACHE[a]
        np.testing.assert_array_equal(c.full_layers["H_brush_delta"], bd)
        # Reference: a fresh full recompute with the same params and the same brush.
        ref = api_main.recompute_relief(
            c.full_layers, api_main.db.get_project(a)["params"], physical_width_mm=1000.0, relief_z_mm=15.0
        )
        np.testing.assert_allclose(c.H_current, ref, atol=1e-6)


def test_groove_change_does_not_pin_full_res_m_lines(isolated_env):
    with TestClient(api_main.app) as client:
        pid = _create(client, res=512, style="classic")
        c = api_main._RAM_CACHE[pid]
        assert "M_lines" not in c.full_layers
        g_before = c.full_layers["G_lines"].copy()
        r = client.post(f"/api/projects/{pid}/recompute", json={"groove_sigma": 5.0})
        assert r.status_code == 200
        assert "M_lines" not in c.full_layers
        # G_lines was rebuilt from the transient M_lines at the new (resolution-scaled) sigma.
        assert c.full_layers["groove_sigma"] == pytest.approx(5.0 * c.full_layers["_groove_scale"])
        assert not np.array_equal(c.full_layers["G_lines"], g_before)


def test_served_textures_keep_only_two_most_recent(isolated_env):
    with TestClient(api_main.app) as client:
        pid = _create(client, res=2048)
        for md in (256, 512, 1024, 2048):
            _get_tex(client, pid, max_dim=md)
        keys = list(api_main._RAM_CACHE[pid].served_textures)
        assert len(keys) == 2
        assert max(keys[-1]) == 2048
