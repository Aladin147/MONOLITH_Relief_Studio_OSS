"""Comprehensive API and integration tests for MONOLITH Relief Studio."""

from __future__ import annotations

import io
import os
import struct
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient

from backend import db
from backend import main as api_main

PERF_SLACK = float(os.environ.get("MONOLITH_PERF_SLACK", "1"))  # >1 on slow CI runners


def _make_synthetic_lineart_png(size: int = 256) -> bytes:
    """Create a clean black-background white-silhouette line-art PNG with interior contour."""
    img = np.zeros((size, size), dtype=np.uint8)
    center = (size // 2, size // 2)
    cv2.circle(img, center, size // 3, 255, -1)
    cv2.line(img, (size // 2 - size // 4, size // 2), (size // 2 + size // 4, size // 2), 0, max(2, size // 64))
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_db_crud_helpers(isolated_env):
    db_path = isolated_env["db_path"]
    db.init_db(db_path)

    proj = db.create_project(
        name="Test Piece",
        original_filename="test.png",
        width=1024,
        height=1024,
        ai_mode="off",
        db_path=db_path,
    )
    assert proj["name"] == "Test Piece"
    assert proj["params"]["gamma"] == pytest.approx(0.85)
    assert len(db.list_projects(db_path)) == 1

    updated = db.update_project(
        proj["id"],
        name="Updated Piece",
        params={"gamma": 1.15, "invert_depth": True},
        db_path=db_path,
    )
    assert updated is not None
    assert updated["name"] == "Updated Piece"
    assert updated["params"]["gamma"] == pytest.approx(1.15)
    assert updated["params"]["invert_depth"] is True

    preset = db.create_preset(name="Studio Preset", params={"w_local": 0.9}, db_path=db_path)
    assert preset["name"] == "Studio Preset"
    assert len(db.list_presets(db_path)) == 1
    assert db.delete_preset(preset["id"], db_path=db_path) is True
    assert db.delete_project(proj["id"], db_path=db_path) is True
    assert db.get_project(proj["id"], db_path=db_path) is None


def test_lifespan_autoseed_and_health(isolated_env):
    with TestClient(api_main.app) as client:
        res = client.get("/api/health")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "ok"
        assert body["projects_count"] >= 1

        projects = client.get("/api/projects").json()
        assert len(projects) == 1
        seeded = projects[0]
        pid = seeded["id"]
        assert seeded["ai_mode"] == "off"
        assert max(seeded["width"], seeded["height"]) == 1024

        orig_resp = client.get(f"/api/projects/{pid}/original-image")
        assert orig_resp.status_code == 200
        assert len(orig_resp.content) > 1000

        prev_resp = client.get(f"/api/projects/{pid}/preview-png")
        assert prev_resp.status_code == 200
        assert prev_resp.headers["content-type"] == "image/png"
        arr = cv2.imdecode(np.frombuffer(prev_resp.content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        assert arr is not None
        assert arr.dtype == np.uint16
        assert int(np.unique(arr).size) > 4000


def test_create_upload_and_benchmark_and_cache_files(isolated_env):
    data_dir = isolated_env["data_dir"]
    with TestClient(api_main.app) as client:
        # Missing file without use_benchmark -> 400
        bad = client.post("/api/projects/create", data={"name": "Invalid"})
        assert bad.status_code == 400

        # Upload custom image
        png_bytes = _make_synthetic_lineart_png(256)
        up = client.post(
            "/api/projects/create",
            files={"file": ("circle_line.png", png_bytes, "image/png")},
            data={
                "name": "Custom Circle",
                "target_res": "512",
                "bg_removal": "true",
                "hd_restore": "true",
                "ai_mode": "off",
                "bit_depth": "16bit",
            },
        )
        assert up.status_code == 200
        up_data = up.json()
        assert up_data["width"] == 512
        assert up_data["height"] == 512
        assert up_data["name"] == "Custom Circle"

        cache_dir = data_dir / "projects" / up_data["id"] / "cache"
        # A new project is Sculpt: the shared layers plus the Sculpt basis and parts analysis.
        for layer_name in (
            "M_sil.npy",
            "d_sil.npy",
            "L_clean.npy",
            "H_current.npy",
            "sculpt_basis.json",
            "parts.npz",
        ):
            assert (cache_dir / layer_name).is_file()
        classic = client.post(
            "/api/projects/create",
            files={"file": ("circle_line.png", png_bytes, "image/png")},
            data={"target_res": "512", "style": "classic"},
        )
        assert classic.status_code == 200
        cache_dir = data_dir / "projects" / classic.json()["id"] / "cache"
        for layer_name in (
            "M_sil.npy",
            "d_sil.npy",
            "H_global.npy",
            "H_local.npy",
            "G_lines.npy",
            "H_ai_guided.npy",
            "L_clean.npy",
            "H_current.npy",
        ):
            assert (cache_dir / layer_name).is_file()

        # Benchmark creation via endpoint
        bm = client.post(
            "/api/projects/create",
            data={
                "use_benchmark": "true",
                "name": "Benchmark Copy",
                "target_res": "1024",
                "ai_mode": "off",
            },
        )
        assert bm.status_code == 200
        assert max(bm.json()["width"], bm.json()["height"]) == 1024


def test_cached_recompute_speed_and_params_persistence(isolated_env):
    with TestClient(api_main.app) as client:
        bm = client.post(
            "/api/projects/create",
            data={
                "use_benchmark": "true",
                "name": "Speed Test 4K",
                "target_res": "4096",
                "ai_mode": "off",
                "style": "classic",  # the Classic 80 ms budget (Sculpt's: test_sculpt_engine)
            },
        )
        assert bm.status_code == 200
        pid = bm.json()["id"]

        # Interactive preview-only recompute using 1024 RAM cache must be < 25ms
        prev_res = client.post(
            f"/api/projects/{pid}/recompute?preview_only=true",
            json={
                "w_global": 0.50,
                "w_local": 0.70,
                "gamma": 0.90,
                "draft_angle_deg": 4.0,
                "frame": {
                    "enabled": True,
                    "preset": "gallery_classic",
                    "width_mm": 22.0,
                    "thickness_mm": 5.5,
                    "back_protrusion_mm": 0.0,
                    "bbox": [0.16, 0.48, 0.84, 0.96],
                },
            },
        )
        assert prev_res.status_code == 200
        if os.environ.get("MONOLITH_CHECK_TIMINGS") == "1":
            assert prev_res.json()["compute_ms"] < 25.0 * PERF_SLACK

        # Ensure preview-png materializes full-res cleanly even right after preview_only=true
        prev_png = client.get(f"/api/projects/{pid}/preview-png")
        assert prev_png.status_code == 200
        assert int(prev_png.headers["x-unique-levels"]) > 4000

        # Warm-up full-res draft cache at 4K
        warm = client.post(
            f"/api/projects/{pid}/recompute",
            json={
                "w_global": 0.50,
                "w_local": 0.70,
                "gamma": 0.90,
                "draft_angle_deg": 4.0,
                "frame": {
                    "enabled": True,
                    "preset": "gallery_classic",
                    "width_mm": 22.0,
                    "thickness_mm": 5.5,
                    "back_protrusion_mm": 0.0,
                    "bbox": [0.16, 0.48, 0.84, 0.96],
                },
            },
        )
        assert warm.status_code == 200

        # Measured cached 4K recompute must complete in < 80ms
        t0 = time.perf_counter()
        res = client.post(
            f"/api/projects/{pid}/recompute",
            json={
                "w_global": 0.55,
                "w_local": 0.75,
                "w_groove": 0.40,
                "gamma": 0.80,
                "brightness": 0.02,
                "contrast": 1.05,
                "invert_depth": True,
                "draft_angle_deg": 4.0,
                "frame": {
                    "enabled": True,
                    "preset": "gallery_classic",
                    "width_mm": 22.0,
                    "thickness_mm": 5.5,
                    "back_protrusion_mm": 0.0,
                    "bbox": [0.16, 0.48, 0.84, 0.96],
                },
            },
        )
        wall_ms = (time.perf_counter() - t0) * 1000.0
        assert res.status_code == 200
        body = res.json()
        if os.environ.get("MONOLITH_CHECK_TIMINGS") == "1":
            assert body["compute_ms"] < 80.0 * PERF_SLACK
        if os.environ.get("MONOLITH_CHECK_TIMINGS") == "1":
            assert wall_ms < 250.0  # HTTP roundtrip + FastAPI JSON overhead
        assert body["unique_levels"] > 4000
        assert body["params"]["invert_depth"] is True
        assert body["params"]["frame"]["enabled"] is True

        # Verify SQLite persistence
        fetched = client.get(f"/api/projects/{pid}").json()
        assert fetched["params"]["w_global"] == pytest.approx(0.55)
        assert fetched["params"]["frame"]["preset"] == "gallery_classic"


def test_float32_texture_stream_and_local_edit_put(isolated_env):
    with TestClient(api_main.app) as client:
        proj = client.get("/api/projects").json()[0]
        pid = proj["id"]

        # GET float32 texture
        tex_res = client.get(f"/api/projects/{pid}/float32-texture?max_dim=512")
        assert tex_res.status_code == 200
        assert tex_res.headers["content-type"] == "application/octet-stream"
        if os.environ.get("MONOLITH_CHECK_TIMINGS") == "1":
            assert float(tex_res.headers["x-compute-ms"]) < 25.0 * PERF_SLACK
        w = int(tex_res.headers["x-width"])
        h = int(tex_res.headers["x-height"])
        assert max(w, h) == 512
        assert len(tex_res.content) == w * h * 4

        arr = np.frombuffer(tex_res.content, dtype=np.float32).reshape((h, w)).copy()
        assert np.isfinite(arr).all()
        assert float(arr.min()) >= 0.0
        assert float(arr.max()) <= 1.0

        # Simulate Local Edit brush sculpting the center patch (deflate if near peak, else inflate)
        cy, cx = h // 2, w // 2
        before_center = float(arr[cy, cx])
        target_center = float(np.clip(before_center - 0.35 if before_center > 0.5 else before_center + 0.35, 0.0, 1.0))
        arr[cy - 10 : cy + 10, cx - 10 : cx + 10] = target_center

        put_res = client.put(
            f"/api/projects/{pid}/float32-texture",
            content=arr.tobytes(),
            headers={"X-Width": str(w), "X-Height": str(h)},
        )
        assert put_res.status_code == 200
        assert put_res.json()["status"] == "updated"

        # Read back and verify local edit persisted
        tex_after = client.get(f"/api/projects/{pid}/float32-texture?max_dim=512")
        arr_after = np.frombuffer(tex_after.content, dtype=np.float32).reshape((h, w))
        assert abs(float(arr_after[cy, cx]) - target_center) < 0.02

        # Also test query-param width/height form at max_dim=1024
        tex_1k = client.get(f"/api/projects/{pid}/float32-texture?max_dim=1024")
        w1 = int(tex_1k.headers["x-width"])
        h1 = int(tex_1k.headers["x-height"])
        arr_1k = np.frombuffer(tex_1k.content, dtype=np.float32).reshape((h1, w1)).copy()
        arr_1k[10:20, 10:20] = 0.42
        put_qs = client.put(
            f"/api/projects/{pid}/float32-texture?width={w1}&height={h1}",
            content=arr_1k.tobytes(),
        )
        assert put_qs.status_code == 200
        tex_1k_after = client.get(f"/api/projects/{pid}/float32-texture?max_dim=1024")
        arr_1k_after = np.frombuffer(tex_1k_after.content, dtype=np.float32).reshape((h1, w1))
        assert abs(float(arr_1k_after[15, 15]) - 0.42) < 1e-4


def test_exports_and_preprocess_and_upscale_and_cold_reload(isolated_env):
    with TestClient(api_main.app) as client:
        png_bytes = _make_synthetic_lineart_png(256)
        create_res = client.post(
            "/api/projects/create",
            files={"file": ("export_test.png", png_bytes, "image/png")},
            data={"name": "Export Piece", "target_res": "512"},
        )
        assert create_res.status_code == 200
        pid = create_res.json()["id"]

        # Preprocess with guided_overlap
        prep_res = client.post(
            f"/api/projects/{pid}/preprocess",
            json={"bg_removal": True, "hd_restore": True, "target_res": 512, "ai_mode": "guided_overlap"},
        )
        assert prep_res.status_code == 200
        assert prep_res.json()["ai_mode"] == "guided_overlap"

        # Upscale to 1024
        up_res = client.post(f"/api/projects/{pid}/upscale", json={"target_res": 1024})
        assert up_res.status_code == 200
        assert up_res.json()["width"] == 1024

        # Simulate server restart by clearing RAM cache
        api_main._RAM_CACHE.clear()

        # 16-bit PNG export
        exp_png = client.post(f"/api/projects/{pid}/export", json={"format": "png16"})
        assert exp_png.status_code == 200
        assert "attachment;" in exp_png.headers["content-disposition"]
        img16 = cv2.imdecode(np.frombuffer(exp_png.content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        assert img16 is not None and img16.dtype == np.uint16

        # 32-bit EXR export (verify OpenEXR magic bytes 0x76, 0x2f, 0x31, 0x01)
        exp_exr = client.post(f"/api/projects/{pid}/export", json={"format": "exr32"})
        assert exp_exr.status_code == 200
        assert exp_exr.content[:4] == struct.pack("<I", 20000630)

        # 16-bit TIFF export
        exp_tiff = client.post(f"/api/projects/{pid}/export", json={"format": "tiff16"})
        assert exp_tiff.status_code == 200
        tiff_arr = cv2.imdecode(np.frombuffer(exp_tiff.content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        assert tiff_arr is not None and tiff_arr.dtype == np.uint16

        assert client.post(f"/api/projects/{pid}/recompute", json={"physical_width_mm": 500.0, "relief_z_mm": 12.0, "base_z_mm": 4.0}).status_code == 200
        # Watertight STL export
        exp_stl = client.post(
            f"/api/projects/{pid}/export",
            json={
                "format": "stl",
                "width_mm": 500.0,
                "relief_z_mm": 12.0,
                "base_z_mm": 4.0,
                "mesh_mode": "grid", "mesh_resolution": 128,
            },
        )
        assert exp_stl.status_code == 200
        mesh = trimesh.load(io.BytesIO(exp_stl.content), file_type="stl")
        assert mesh.is_watertight is True

        # OBJ export
        exp_obj = client.post(
            f"/api/projects/{pid}/export",
            json={"format": "obj", "width_mm": 500.0, "relief_z_mm": 12.0, "base_z_mm": 4.0, "mesh_mode": "grid", "mesh_resolution": 64},
        )
        assert exp_obj.status_code == 200
        obj_text = exp_obj.content.decode("utf-8")
        assert "v " in obj_text and "f " in obj_text

        # Delete project
        del_res = client.delete(f"/api/projects/{pid}")
        assert del_res.status_code == 200
        assert client.get(f"/api/projects/{pid}").status_code == 404


def test_presets_api_crud(isolated_env):
    with TestClient(api_main.app) as client:
        initial = client.get("/api/presets").json()
        assert len(initial) >= 1

        created = client.post(
            "/api/presets",
            json={"name": "High Relief Test", "params": {"w_global": 0.7, "gamma": 0.75}},
        )
        assert created.status_code == 200
        preset_id = created.json()["id"]

        listed = client.get("/api/presets").json()
        assert any(p["id"] == preset_id for p in listed)

        deleted = client.delete(f"/api/presets/{preset_id}")
        assert deleted.status_code == 200
        assert client.delete(f"/api/presets/{preset_id}").status_code == 404


def test_brush_edits_persist_across_recompute_restart_and_reset(isolated_env):
    data_dir = isolated_env["data_dir"]
    with TestClient(api_main.app) as client:
        png_bytes = _make_synthetic_lineart_png(256)
        create_res = client.post(
            "/api/projects/create",
            files={"file": ("brush_persist.png", png_bytes, "image/png")},
            data={"name": "Brush Persist", "target_res": "512"},
        )
        assert create_res.status_code == 200
        pid = create_res.json()["id"]
        cache_dir = data_dir / "projects" / pid / "cache"
        assert not (cache_dir / "H_brush_delta.npy").exists()

        tex0 = client.get(f"/api/projects/{pid}/float32-texture?max_dim=512")
        assert tex0.status_code == 200
        w = int(tex0.headers["x-width"])
        h = int(tex0.headers["x-height"])
        arr0 = np.frombuffer(tex0.content, dtype=np.float32).reshape((h, w)).copy()

        # Sculpt a non-saturating center patch inside the silhouette.
        cy, cx = h // 2 - 20, w // 2
        base_val = float(arr0[cy, cx])
        delta_val = -0.35 if base_val > 0.6 else 0.35
        arr_edited = arr0.copy()
        arr_edited[cy - 10 : cy + 10, cx - 10 : cx + 10] = np.clip(
            arr0[cy - 10 : cy + 10, cx - 10 : cx + 10] + delta_val, 0.0, 1.0
        )

        put_res = client.put(
            f"/api/projects/{pid}/float32-texture?width={w}&height={h}",
            content=arr_edited.tobytes(),
        )
        assert put_res.status_code == 200
        assert (cache_dir / "H_brush_delta.npy").is_file()

        # 1. Interactive preview_only=true recompute must retain brush delta.
        prev_rec = client.post(
            f"/api/projects/{pid}/recompute?preview_only=true",
            json={"gamma": 1.25},
        )
        assert prev_rec.status_code == 200
        tex_prev = client.get(f"/api/projects/{pid}/float32-texture?max_dim=512")
        arr_prev = np.frombuffer(tex_prev.content, dtype=np.float32).reshape((h, w))

        # 2. Full recompute (preview_only=false) with changed gamma & frame enabled must retain brush delta.
        full_rec = client.post(
            f"/api/projects/{pid}/recompute",
            json={
                "gamma": 1.25,
                "frame": {
                    "enabled": True,
                    "preset": "gallery_classic",
                    "width_mm": 22.0,
                    "thickness_mm": 5.5,
                    "back_protrusion_mm": 0.0,
                    # Keep the frame band clear of the brushed patch.
                    "bbox": [0.1, 0.8, 0.9, 0.98],
                },
            },
        )
        assert full_rec.status_code == 200
        tex_full = client.get(f"/api/projects/{pid}/float32-texture?max_dim=512")
        arr_full = np.frombuffer(tex_full.content, dtype=np.float32).reshape((h, w)).copy()
        assert np.allclose(arr_prev[cy - 8 : cy + 8, cx - 8 : cx + 8], arr_full[cy - 8 : cy + 8, cx - 8 : cx + 8], atol=1e-4)

        # 3. Simulated cold restart: clear RAM cache and verify brush edits survive reload & subsequent recompute.
        api_main._RAM_CACHE.clear()
        tex_cold = client.get(f"/api/projects/{pid}/float32-texture?max_dim=512")
        assert tex_cold.status_code == 200
        arr_cold = np.frombuffer(tex_cold.content, dtype=np.float32).reshape((h, w))
        assert np.allclose(arr_cold, arr_full, atol=1e-5)

        rec_cold = client.post(f"/api/projects/{pid}/recompute", json={"gamma": 1.25})
        assert rec_cold.status_code == 200
        tex_cold_rec = client.get(f"/api/projects/{pid}/float32-texture?max_dim=512")
        arr_cold_rec = np.frombuffer(tex_cold_rec.content, dtype=np.float32).reshape((h, w)).copy()
        assert np.allclose(arr_cold_rec, arr_full, atol=1e-5)

        # 4. Reset brush edits cleanly via DELETE /brush-edits.
        reset_res = client.delete(f"/api/projects/{pid}/brush-edits")
        assert reset_res.status_code == 200
        assert reset_res.json()["status"] == "reset"
        assert not (cache_dir / "H_brush_delta.npy").exists()

        tex_reset = client.get(f"/api/projects/{pid}/float32-texture?max_dim=512")
        arr_reset = np.frombuffer(tex_reset.content, dtype=np.float32).reshape((h, w)).copy()
        assert abs(float(arr_cold_rec[cy, cx]) - float(arr_reset[cy, cx])) > 0.20

        # Confirm clean recompute matches reset output (delta truly gone, frame preserved).
        rec_clean = client.post(f"/api/projects/{pid}/recompute", json={"gamma": 1.25})
        assert rec_clean.status_code == 200
        tex_clean = client.get(f"/api/projects/{pid}/float32-texture?max_dim=512")
        arr_clean = np.frombuffer(tex_clean.content, dtype=np.float32).reshape((h, w))
        assert np.allclose(arr_reset, arr_clean, atol=1e-5)

        # 5. Re-preprocessing at a new resolution keeps the sculpting (resampled).
        put_again = client.put(
            f"/api/projects/{pid}/float32-texture?width={w}&height={h}",
            content=arr_edited.tobytes(),
        )
        assert put_again.status_code == 200
        assert (cache_dir / "H_brush_delta.npy").is_file()
        prep_res = client.post(
            f"/api/projects/{pid}/preprocess",
            json={"bg_removal": True, "hd_restore": True, "target_res": 512, "ai_mode": "off"},
        )
        assert prep_res.status_code == 200
        kept = np.load(cache_dir / "H_brush_delta.npy")
        assert kept.shape == (prep_res.json()["height"], prep_res.json()["width"])
        assert float(np.abs(kept).max()) > 0.01


def test_brush_delta_survives_slider_recompute_and_clear_brush_endpoint(isolated_env):
    data_dir = isolated_env["data_dir"]
    with TestClient(api_main.app) as client:
        png_bytes = _make_synthetic_lineart_png(256)
        create_res = client.post(
            "/api/projects/create",
            files={"file": ("brush_slider.png", png_bytes, "image/png")},
            data={"name": "Brush Slider", "target_res": "512"},
        )
        assert create_res.status_code == 200
        pid = create_res.json()["id"]
        cache_dir = data_dir / "projects" / pid / "cache"

        def _tex() -> np.ndarray:
            r = client.get(f"/api/projects/{pid}/float32-texture?max_dim=512")
            assert r.status_code == 200
            hh, ww = int(r.headers["x-height"]), int(r.headers["x-width"])
            return np.frombuffer(r.content, dtype=np.float32).reshape((hh, ww)).copy()

        arr0 = _tex()
        h, w = arr0.shape
        cy, cx = h // 2 - 20, w // 2
        delta_val = -0.2 if float(arr0[cy, cx]) > 0.5 else 0.2
        edited = arr0.copy()
        edited[cy - 10 : cy + 10, cx - 10 : cx + 10] += np.float32(delta_val)
        put_res = client.put(
            f"/api/projects/{pid}/float32-texture?width={w}&height={h}",
            content=np.clip(edited, 0.0, 1.0).astype(np.float32).tobytes(),
        )
        assert put_res.status_code == 200

        # Persisted delta == edited - unbrushed procedural heightmap.
        stored = np.load(cache_dir / "H_brush_delta.npy")
        assert stored.dtype == np.float32 and stored.shape == (h, w)
        assert np.allclose(stored[cy - 8 : cy + 8, cx - 8 : cx + 8], delta_val, atol=1e-4)
        assert np.allclose(stored[: cy - 30, :], 0.0, atol=1e-5)

        # Slider changes (gamma, draft angle, frame) must keep the stroke.
        slider_params = {
            "gamma": 1.3,
            "draft_angle_deg": 4.0,
            "frame": {"enabled": True, "width_mm": 22.0, "thickness_mm": 5.5, "bbox": [0.1, 0.8, 0.9, 0.98]},
        }
        for preview_only in ("true", "false"):
            rec = client.post(f"/api/projects/{pid}/recompute?preview_only={preview_only}", json=slider_params)
            assert rec.status_code == 200
        brushed = _tex()

        clear_res = client.post(f"/api/projects/{pid}/clear-brush")
        assert clear_res.status_code == 200
        assert clear_res.json()["status"] == "reset"
        assert not (cache_dir / "H_brush_delta.npy").exists()
        clean = _tex()

        diff = brushed - clean
        assert np.allclose(diff[cy - 8 : cy + 8, cx - 8 : cx + 8], delta_val, atol=1e-4)
        assert np.allclose(diff[: cy - 30, :], 0.0, atol=1e-5)

        # Clearing survives further slider changes (no stale delta resurrected).
        rec2 = client.post(f"/api/projects/{pid}/recompute", json={"gamma": 0.9})
        assert rec2.status_code == 200
        rec3 = client.post(f"/api/projects/{pid}/recompute", json=slider_params)
        assert rec3.status_code == 200
        assert np.allclose(_tex(), clean, atol=1e-5)


def test_lru_ram_cache_eviction_and_reload(isolated_env, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(api_main, "MAX_RAM_PROJECTS", 2)
    with TestClient(api_main.app) as client:
        png_bytes = _make_synthetic_lineart_png(256)
        pids: list[str] = []
        for idx in range(3):
            res = client.post(
                "/api/projects/create",
                files={"file": (f"lru_{idx}.png", png_bytes, "image/png")},
                data={"name": f"LRU Piece {idx}", "target_res": "256"},
            )
            assert res.status_code == 200
            pids.append(res.json()["id"])
            assert len(api_main._RAM_CACHE) <= 2

        p0, p1, p2 = pids
        assert p0 not in api_main._RAM_CACHE
        assert p1 in api_main._RAM_CACHE and p2 in api_main._RAM_CACHE

        # Promote p1 to MRU, then reload p0 from disk -> p2 must be evicted, p1 kept.
        t1 = client.get(f"/api/projects/{p1}/float32-texture?max_dim=256")
        assert t1.status_code == 200
        t0 = client.get(f"/api/projects/{p0}/preview-png")
        assert t0.status_code == 200
        assert len(api_main._RAM_CACHE) <= 2
        assert p0 in api_main._RAM_CACHE and p1 in api_main._RAM_CACHE
        assert p2 not in api_main._RAM_CACHE

        # Dirty preview_only recompute on p0 is flushed transparently when p0 is evicted.
        rec_p0 = client.post(f"/api/projects/{p0}/recompute?preview_only=true", json={"gamma": 1.65})
        assert rec_p0.status_code == 200
        # Load p2 and create p3 to evict p0.
        assert client.get(f"/api/projects/{p2}/float32-texture?max_dim=256").status_code == 200
        res3 = client.post(
            "/api/projects/create",
            files={"file": ("lru_3.png", png_bytes, "image/png")},
            data={"name": "LRU Piece 3", "target_res": "256"},
        )
        assert res3.status_code == 200
        assert len(api_main._RAM_CACHE) <= 2
        assert p0 not in api_main._RAM_CACHE

        # Reload p0 and verify gamma=1.65 is reflected in reloaded full texture & health count <= 2.
        p0_detail = client.get(f"/api/projects/{p0}").json()
        assert p0_detail["params"]["gamma"] == pytest.approx(1.65)
        t0_reloaded = client.get(f"/api/projects/{p0}/float32-texture?max_dim=256")
        assert t0_reloaded.status_code == 200
        arr_reloaded = np.frombuffer(t0_reloaded.content, dtype=np.float32).copy()
        assert client.post(f"/api/projects/{p0}/recompute", json={"gamma": 1.65}).status_code == 200
        t0_recomputed = client.get(f"/api/projects/{p0}/float32-texture?max_dim=256")
        arr_recomputed = np.frombuffer(t0_recomputed.content, dtype=np.float32)
        assert np.allclose(arr_reloaded, arr_recomputed, atol=1e-5)
        assert client.get("/api/health").json()["cached_projects"] <= 2


def test_png8_and_cutout_mesh_exports(isolated_env):
    with TestClient(api_main.app) as client:
        png_bytes = _make_synthetic_lineart_png(256)
        create_res = client.post(
            "/api/projects/create",
            files={"file": ("cutout_export.png", png_bytes, "image/png")},
            data={"name": "Cutout Piece", "target_res": "256"},
        )
        assert create_res.status_code == 200
        pid = create_res.json()["id"]

        # 8-bit PNG export
        exp_png8 = client.post(f"/api/projects/{pid}/export", json={"format": "png8"})
        assert exp_png8.status_code == 200
        assert exp_png8.headers["content-type"] == "image/png"
        assert "_8bit.png" in exp_png8.headers["content-disposition"]
        img8 = cv2.imdecode(np.frombuffer(exp_png8.content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        assert img8 is not None
        assert img8.dtype == np.uint8
        assert img8.ndim == 2
        assert int(img8.min()) == 0 and int(img8.max()) > 200

        # base_z_mm=0.0 STL export -> HTTP 200 & watertight cutout mesh with fewer faces than slab
        assert client.post(f"/api/projects/{pid}/recompute", json={"physical_width_mm": 400.0, "relief_z_mm": 12.0, "base_z_mm": 5.0}).status_code == 200
        exp_stl5 = client.post(
            f"/api/projects/{pid}/export",
            json={
                "format": "stl",
                "width_mm": 400.0,
                "relief_z_mm": 12.0,
                "base_z_mm": 5.0,
                "mesh_mode": "grid", "mesh_resolution": 96,
            },
        )
        assert exp_stl5.status_code == 200
        mesh_stl5 = trimesh.load(io.BytesIO(exp_stl5.content), file_type="stl")

        assert client.post(f"/api/projects/{pid}/recompute", json={"physical_width_mm": 400.0, "relief_z_mm": 12.0, "base_z_mm": 0.0}).status_code == 200
        exp_stl0 = client.post(
            f"/api/projects/{pid}/export",
            json={
                "format": "stl",
                "width_mm": 400.0,
                "relief_z_mm": 12.0,
                "base_z_mm": 0.0,
                "mesh_mode": "grid", "mesh_resolution": 96,
            },
        )
        assert exp_stl0.status_code == 200
        mesh_stl0 = trimesh.load(io.BytesIO(exp_stl0.content), file_type="stl")
        assert mesh_stl0.is_watertight is True
        assert mesh_stl0.is_winding_consistent is True
        assert float(mesh_stl0.volume) > 0.0
        # Cutout keeps only the silhouette footprint (no rectangular backing plaque).
        assert float(mesh_stl0.area_faces[mesh_stl0.face_normals[:, 2] < -0.99].sum()) < 0.95 * float(
            mesh_stl5.area_faces[mesh_stl5.face_normals[:, 2] < -0.99].sum()
        )

        assert client.post(f"/api/projects/{pid}/recompute", json={"physical_width_mm": 400.0, "relief_z_mm": 12.0, "base_z_mm": 0.0}).status_code == 200
        # base_z_mm=0.0 OBJ export -> HTTP 200 & watertight mesh
        exp_obj0 = client.post(
            f"/api/projects/{pid}/export",
            json={
                "format": "obj",
                "width_mm": 400.0,
                "relief_z_mm": 12.0,
                "base_z_mm": 0.0,
                "mesh_mode": "grid", "mesh_resolution": 64,
            },
        )
        assert exp_obj0.status_code == 200
        mesh_obj0 = trimesh.load(io.BytesIO(exp_obj0.content), file_type="obj")
        assert mesh_obj0.is_watertight is True
        assert float(mesh_obj0.volume) > 0.0
