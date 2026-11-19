"""Phase 1 Benchmark & Unit Verification Suite for MONOLITH float32 pipeline."""

from __future__ import annotations

import os
import io
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.pipeline import (
    compute_ai_guided_ordering,
    compute_basis_layers,
    encode_exr32,
    encode_png16,
    encode_tiff16,
    generate_frame,
    heightmap_to_obj,
    heightmap_to_watertight_stl,
    preprocess_lineart,
    recompute_relief,
)
from backend.pipeline.ai_ordering import _guided_filter_box

PERF_SLACK = float(os.environ.get("MONOLITH_PERF_SLACK", "1"))  # >1 on slow CI runners

BENCHMARK_IMAGE = REPO_ROOT / "benchmarks" / "geometric-study.png"


@pytest.fixture(scope="module")
def benchmark_4k() -> dict:
    """Run 4K preprocessing and basis decomposition once across benchmark assertions."""
    assert BENCHMARK_IMAGE.is_file(), f"Missing benchmark image at {BENCHMARK_IMAGE}"
    preprocessed = preprocess_lineart(
        BENCHMARK_IMAGE,
        target_res=4096,
        bg_removal=True,
        hd_restore=True,
    )
    layers = compute_basis_layers(preprocessed, ai_mode="off", groove_sigma=2.5)
    default_params = {
        "w_global": 0.45,
        "w_local": 0.65,
        "w_ai": 0.25,
        "w_groove": 0.35,
        "groove_sigma": 2.5,
        "gamma": 0.85,
        "brightness": 0.0,
        "contrast": 1.0,
        "h_min": 0.08,
        "invert_depth": False,
        "draft_angle_deg": 2.0,
        "frame": {"enabled": False},
    }
    heightmap = recompute_relief(
        layers,
        default_params,
        physical_width_mm=1000.0,
        relief_z_mm=15.0,
    )
    return {
        "preprocessed": preprocessed,
        "layers": layers,
        "params": default_params,
        "heightmap": heightmap,
    }


def test_benchmark_float32_types_and_4k_resolution(benchmark_4k: dict) -> None:
    prep = benchmark_4k["preprocessed"]
    layers = benchmark_4k["layers"]
    H_norm = benchmark_4k["heightmap"]

    assert max(prep["shape"]) == 4096
    for key in ("L_clean", "M_bg", "M_sil", "M_lines", "M_regions"):
        assert prep[key].dtype == np.float32, f"{key} must remain float32"

    for key in ("M_sil", "d_sil", "H_global", "H_local", "G_lines", "H_ai_guided"):
        assert layers[key].dtype == np.float32, f"layer {key} must remain float32"

    assert H_norm.dtype == np.float32
    assert np.all(layers["H_ai_guided"] == 0.0), "ai_mode='off' must produce zero AI tensor"


def test_benchmark_t_junction_and_background_zero(benchmark_4k: dict) -> None:
    prep = benchmark_4k["preprocessed"]
    H_norm = benchmark_4k["heightmap"]

    bg_mask = prep["M_bg"] > 0.5
    sil_mask = prep["M_sil"] > 0.5
    assert np.any(bg_mask) and np.any(sil_mask)

    # Outer background and enclosed calf gap must be strictly 0.0.
    assert np.all(H_norm[bg_mask] == 0.0)
    assert np.all(H_norm[prep["M_sil"] <= 0.01] == 0.0)

    # Interior lines must NOT leak into M_bg: verify most of the figure silhouette stays intact.
    sil_ratio = float(sil_mask.mean())
    assert 0.25 < sil_ratio < 0.45, f"Unexpected silhouette fraction {sil_ratio:.4f}"


def test_sample_hole_is_background(benchmark_4k):
    prep = benchmark_4k["preprocessed"]
    assert prep["M_bg"][656 * 4, 260 * 4] == 1
    assert benchmark_4k["heightmap"][656 * 4, 260 * 4] == 0


def test_preprocess_forces_wide_enclosed_dark_pocket_to_background() -> None:
    img = np.zeros((1024, 1024), dtype=np.uint8)
    cv2.circle(img, (512, 512), 400, 255, -1)
    # Small enclosed dark pocket (well below the 0.5% large-hole area) far wider than any stroke.
    cv2.circle(img, (512, 360), 28, 0, -1)
    # Thin interior contour line must stay part of the silhouette.
    cv2.line(img, (300, 640), (720, 640), 0, 3)

    prep = preprocess_lineart(img, target_res=1024, bg_removal=True, hd_restore=True)
    assert float(prep["M_bg"][360, 512]) == 1.0
    assert float(prep["M_sil"][360, 512]) == 0.0
    assert float(prep["M_sil"][640, 512]) == 1.0
    assert float(prep["M_bg"][20, 20]) == 1.0


def test_benchmark_16bit_png_precision_and_unique_levels(benchmark_4k: dict) -> None:
    prep = benchmark_4k["preprocessed"]
    H_norm = benchmark_4k["heightmap"]

    png_bytes = encode_png16(H_norm)
    decoded = cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded is not None
    assert decoded.dtype == np.uint16
    assert decoded.shape == H_norm.shape
    assert int(decoded.max()) == 65535

    bg_mask = prep["M_bg"] > 0.5
    sil_mask = prep["M_sil"] > 0.5
    assert int(decoded[bg_mask].max()) == 0

    unique_levels = len(np.unique(decoded[sil_mask]))
    assert unique_levels > 4000, f"Expected >4000 unique 16-bit levels inside silhouette, got {unique_levels}"


def test_benchmark_line_creases_and_glass_smooth_cores(benchmark_4k: dict) -> None:
    prep = benchmark_4k["preprocessed"]
    H_norm = benchmark_4k["heightmap"]

    sil_mask = prep["M_sil"] > 0.5
    line_mask = (prep["M_lines"] > 0.5) & sil_mask
    assert np.any(line_mask)

    # Dilate interior lines to inspect directly adjacent white limb pixels.
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    near_line = cv2.dilate(line_mask.astype(np.uint8), kern) > 0
    d_reg = cv2.distanceTransform((prep["M_regions"] > 0.5).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    adj_white = near_line & (prep["M_regions"] > 0.8) & (d_reg > 5.0)
    assert np.any(adj_white)

    mean_line_z = float(H_norm[line_mask].mean())
    mean_adj_z = float(H_norm[adj_white].mean())
    assert mean_line_z < mean_adj_z - 0.05, (
        f"Interior contour lines ({mean_line_z:.4f}) must sit strictly lower than adjacent limbs ({mean_adj_z:.4f})"
    )

    # Verify limb cores are glass-smooth with ultra-low Laplacian variance (no blotches/hallucinations).
    core_white = (prep["M_regions"] > 0.9) & (d_reg > 12.0)
    assert np.any(core_white)
    lap = cv2.Laplacian(H_norm, cv2.CV_32F, ksize=3)
    lap_var = float(np.var(lap[core_white]))
    assert lap_var < 1e-5, f"Limb core Laplacian variance too high (not smooth): {lap_var:.3e}"


def test_benchmark_watertight_stl_and_exports(benchmark_4k: dict) -> None:
    H_norm = benchmark_4k["heightmap"]

    stl_bytes = heightmap_to_watertight_stl(
        H_norm,
        width_mm=1000.0,
        relief_z_mm=15.0,
        base_z_mm=5.0,
        max_grid=512,
    )
    mesh = trimesh.load(io.BytesIO(stl_bytes), file_type="stl")
    assert mesh.is_watertight is True
    assert mesh.is_winding_consistent is True
    assert float(mesh.volume) > 0.0

    obj_bytes = heightmap_to_obj(
        H_norm,
        width_mm=1000.0,
        relief_z_mm=15.0,
        base_z_mm=5.0,
        max_grid=256,
    )
    mesh_obj = trimesh.load(io.BytesIO(obj_bytes), file_type="obj")
    assert mesh_obj.is_watertight is True

    tiff_bytes = encode_tiff16(H_norm)
    t16 = cv2.imdecode(np.frombuffer(tiff_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert t16 is not None and t16.dtype == np.uint16 and int(t16.max()) == 65535

    exr_bytes = encode_exr32(H_norm)
    assert exr_bytes[:4] == struct.pack("<I", 20000630)
    rows, cols = H_norm.shape
    # Verify last scanline in EXR matches float32 row exactly.
    last_row_bytes = exr_bytes[-4 * cols :]
    last_row = np.frombuffer(last_row_bytes, dtype="<f4")
    assert np.allclose(last_row, H_norm[-1], atol=0.0)


def test_cached_recompute_under_80ms_at_4k(benchmark_4k: dict) -> None:
    layers = benchmark_4k["layers"]
    tweaked_params = {
        "w_global": 0.50,
        "w_local": 0.70,
        "w_ai": 0.20,
        "w_groove": 0.40,
        "groove_sigma": 2.5,
        "gamma": 0.90,
        "brightness": 0.02,
        "contrast": 1.05,
        "h_min": 0.08,
        "invert_depth": False,
        "draft_angle_deg": 2.0,
        "frame": {"enabled": False},
    }

    # Measure multiple cached recomputes to avoid transient CPU noise.
    elapsed_ms = []
    for _ in range(3):
        t0 = time.perf_counter()
        out = recompute_relief(layers, tweaked_params, physical_width_mm=1000.0, relief_z_mm=15.0)
        elapsed_ms.append((time.perf_counter() - t0) * 1000.0)
        assert out.dtype == np.float32

    best_ms = min(elapsed_ms)
    if os.environ.get("MONOLITH_CHECK_TIMINGS") == "1":
        assert best_ms < 80.0 * PERF_SLACK, f"4K cached recompute took {best_ms:.2f}ms (must be <80ms x {PERF_SLACK:g})"


def test_frame_presets_and_recompute_variants(benchmark_4k: dict) -> None:
    prep = benchmark_4k["preprocessed"]
    layers = benchmark_4k["layers"]
    shape = prep["shape"]

    disabled = generate_frame(shape, {"enabled": False})
    assert disabled.dtype == np.float32 and np.all(disabled == 0.0)

    for preset in ("gallery_classic", "modern_flat", "flat", "stepped_bevel", "bevel", "circular_medallion"):
        fr = generate_frame(
            (512, 435),
            {
                "enabled": True,
                "preset": preset,
                "width_mm": 22.0,
                "thickness_mm": 8.0,
                "back_protrusion_mm": 0.0,
                "bbox": [0.16, 0.48, 0.84, 0.96],
            },
            physical_width_mm=1000.0,
            relief_z_mm=15.0,
        )
        assert fr.dtype == np.float32
        assert fr.min() >= 0.0 and fr.max() <= 1.0
        assert fr.max() > 0.2

    # Check invert_depth and dynamic groove_sigma on downsampled prep for speed.
    small_prep = preprocess_lineart(BENCHMARK_IMAGE, target_res=512, bg_removal=True, hd_restore=True)
    small_layers = compute_basis_layers(small_prep, ai_mode="guided_overlap", groove_sigma=2.0)
    assert small_layers["H_ai_guided"].dtype == np.float32
    assert small_layers["H_ai_guided"].max() > 0.5
    assert np.all(small_layers["H_ai_guided"][small_prep["M_bg"] > 0.5] == 0.0)

    inv = recompute_relief(
        small_layers,
        {"invert_depth": True, "groove_sigma": 4.0, "frame": {"enabled": False}},
    )
    assert inv.dtype == np.float32
    assert np.all(inv[small_prep["M_bg"] > 0.5] == 0.0)

    # Verify fallback box-filter guided filter works independently.
    fb = _guided_filter_box(small_prep["L_clean"], small_layers["H_ai_guided"], radius=8, eps=1e-3)
    assert fb.dtype == np.float32 and fb.shape == small_prep["shape"]


def test_preprocess_input_modes_and_toggles() -> None:
    raw_bytes = BENCHMARK_IMAGE.read_bytes()
    prep_bytes = preprocess_lineart(raw_bytes, target_res=256, bg_removal=False, hd_restore=False)
    assert prep_bytes["shape"] == (256, 218)
    assert np.all(prep_bytes["M_bg"] == 0.0)
    assert np.all(prep_bytes["M_sil"] == 1.0)
    assert prep_bytes["L_clean"].dtype == np.float32

    img_arr = cv2.imread(str(BENCHMARK_IMAGE), cv2.IMREAD_COLOR)
    prep_arr = preprocess_lineart(img_arr, target_res=256, bg_removal=True, hd_restore=True)
    assert prep_arr["L_clean"].dtype == np.float32
    assert prep_arr["M_bg"].max() == 1.0


def test_recompute_preserves_brush_delta_under_80ms_at_4k(benchmark_4k: dict) -> None:
    layers = dict(benchmark_4k["layers"])
    params = dict(benchmark_4k["params"])
    params["gamma"] = 1.05
    H_base = recompute_relief(layers, params, physical_width_mm=1000.0, relief_z_mm=15.0)

    H_px, W_px = H_base.shape
    delta = np.zeros((H_px, W_px), dtype=np.float32)
    r0, r1 = H_px // 2 - 120, H_px // 2 + 120
    c0, c1 = W_px // 2 - 120, W_px // 2 + 120
    delta[r0:r1, c0:c1] = np.float32(0.25)

    layers["H_brush_delta"] = delta
    layers["_has_brush_delta"] = False
    out_off = recompute_relief(layers, params, physical_width_mm=1000.0, relief_z_mm=15.0)
    assert np.allclose(out_off, H_base, atol=1e-6)

    layers["_has_brush_delta"] = True
    elapsed_ms = []
    out = None
    for _ in range(3):
        t0 = time.perf_counter()
        out = recompute_relief(layers, params, physical_width_mm=1000.0, relief_z_mm=15.0)
        elapsed_ms.append((time.perf_counter() - t0) * 1000.0)
        assert out.dtype == np.float32

    assert out is not None
    if os.environ.get("MONOLITH_CHECK_TIMINGS") == "1":
        assert min(elapsed_ms) < 80.0 * PERF_SLACK, f"4K cached recompute with brush delta took {min(elapsed_ms):.2f}ms"

    expected_patch = np.clip(H_base[r0:r1, c0:c1] + np.float32(0.25), np.float32(0.0), np.float32(1.0))
    assert np.allclose(out[r0:r1, c0:c1], expected_patch, atol=1e-5)
    assert np.allclose(out[:r0, :], H_base[:r0, :], atol=1e-6)


def test_preprocess_preserves_exterior_v_concavity() -> None:
    img = np.zeros((512, 512), dtype=np.uint8)
    cv2.circle(img, (256, 256), 160, 255, -1)
    # Exterior V-notch narrowing below morphological closing width (2*rad = 12 px) near its tip.
    notch = np.array([[246, 90], [266, 90], [256, 130]], dtype=np.int32)
    cv2.fillPoly(img, [notch], 0)
    # Interior thin black line must remain part of the silhouette, not background.
    cv2.line(img, (180, 260), (332, 260), 0, 2)

    prep = preprocess_lineart(img, target_res=512, bg_removal=True, hd_restore=True)
    assert float(prep["M_bg"][20, 20]) == 1.0
    assert float(prep["M_sil"][200, 256]) == 1.0
    assert float(prep["M_bg"][112, 256]) == 1.0
    assert float(prep["M_sil"][112, 256]) == 0.0
    assert float(prep["M_sil"][260, 256]) == 1.0
    assert float(prep["M_bg"][260, 256]) == 0.0


def test_frame_dimensions_are_true_millimetres() -> None:
    """Frame width/thickness are honoured literally and scale continuously (no hidden rescale)."""
    shape = (1024, 870)
    cfg = {
        "enabled": True,
        "preset": "modern_flat",
        "width_mm": 22.0,
        "thickness_mm": 5.5,
        "back_protrusion_mm": 0.0,
        "bbox": [0.16, 0.48, 0.84, 0.96],
    }
    mid_row = int(0.72 * shape[0])
    px_per_mm = shape[1] / 1000.0

    def band(width_mm: float) -> int:
        fr = generate_frame(shape, dict(cfg, width_mm=width_mm), physical_width_mm=1000.0, relief_z_mm=15.0)
        return int(np.count_nonzero(fr[mid_row, : shape[1] // 2] > 0.0))

    # 22 mm on a 1000 mm canvas rendered at 870 px wide -> ~19 px band.
    assert abs(band(22.0) - 22.0 * px_per_mm) <= 2
    # Monotonic and continuous across the old 25 mm switch-over (was a 10x jump).
    widths = [10.0, 20.0, 24.9, 25.0, 30.0, 60.0]
    bands = [band(w) for w in widths]
    assert bands == sorted(bands), bands
    assert abs(bands[2] - bands[3]) <= 1, bands

    fr = generate_frame(shape, cfg, physical_width_mm=1000.0, relief_z_mm=15.0)
    assert abs(float(fr.max()) - 5.5 / 15.0) < 0.02
    # Thin user-specified thickness is no longer silently raised.
    fr_thin = generate_frame(shape, dict(cfg, thickness_mm=0.3), physical_width_mm=1000.0, relief_z_mm=15.0)
    assert abs(float(fr_thin.max()) - 0.3 / 15.0) < 0.005


def test_legacy_frame_migration_preserves_geometry() -> None:
    from backend.pipeline.frame_generator import migrate_legacy_frame

    legacy = {"enabled": True, "preset": "gallery_classic", "width_mm": 2.2, "thickness_mm": 1.1}
    migrated = migrate_legacy_frame(legacy, physical_width_mm=1000.0, relief_z_mm=15.0, h_min=0.08)
    assert migrated["units"] == "mm"
    assert migrated["width_mm"] == pytest.approx(22.0)
    assert migrated["thickness_mm"] == pytest.approx(5.5)
    # Idempotent once tagged.
    assert migrate_legacy_frame(migrated, 1000.0, 15.0, 0.08) == migrated
    # Old crown floor (h_min + 0.06) is reproduced for tiny legacy thicknesses.
    floor = migrate_legacy_frame(dict(legacy, thickness_mm=0.2), 100.0, 15.0, 0.3)
    assert floor["thickness_mm"] == pytest.approx(0.36 * 15.0)


def test_brush_delta_applied_before_frame_compositing() -> None:
    prep = preprocess_lineart(BENCHMARK_IMAGE, target_res=512, bg_removal=True, hd_restore=True)
    layers = compute_basis_layers(prep, ai_mode="off", groove_sigma=2.0)
    frame_cfg = {
        "enabled": True,
        "preset": "gallery_classic",
        "width_mm": 22.0,
        "thickness_mm": 5.5,
        "bbox": [0.16, 0.48, 0.84, 0.96],
    }
    params = {"draft_angle_deg": 2.0, "frame": frame_cfg}
    base = recompute_relief(layers, params, include_frame=False)
    frame = generate_frame(base.shape, frame_cfg, physical_width_mm=1000.0, relief_z_mm=15.0)

    delta = np.full(base.shape, np.float32(-0.6), dtype=np.float32)
    layers["H_brush_delta"] = delta
    layers["_has_brush_delta"] = True
    out = recompute_relief(layers, params)
    expected = np.maximum(np.clip(base + delta, 0.0, 1.0), frame)
    assert np.allclose(out, expected, atol=1e-6)
    # Frame crown survives a deep carve stroke across it.
    assert float(out[frame > 0.3].min()) > 0.3

    unbrushed = recompute_relief(layers, params, include_brush=False)
    assert np.allclose(unbrushed, np.maximum(base, frame), atol=1e-6)
