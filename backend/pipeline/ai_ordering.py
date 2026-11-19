"""Engine B: Low-frequency line-locked AI/geometric region depth ordering."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import uniform_filter


def _guided_filter_box(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int,
    eps: float = 1e-3,
) -> np.ndarray:
    """Pure NumPy/SciPy box-filter Guided Filter fallback in float32."""
    I = guide.astype(np.float32, copy=False)
    p = src.astype(np.float32, copy=False)
    ksize = int(2 * max(1, radius) + 1)

    mean_I = uniform_filter(I, size=ksize, mode="nearest").astype(np.float32, copy=False)
    mean_p = uniform_filter(p, size=ksize, mode="nearest").astype(np.float32, copy=False)
    mean_Ip = uniform_filter(I * p, size=ksize, mode="nearest").astype(np.float32, copy=False)
    mean_II = uniform_filter(I * I, size=ksize, mode="nearest").astype(np.float32, copy=False)

    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + np.float32(eps))
    b = mean_p - a * mean_I

    mean_a = uniform_filter(a, size=ksize, mode="nearest").astype(np.float32, copy=False)
    mean_b = uniform_filter(b, size=ksize, mode="nearest").astype(np.float32, copy=False)

    return (mean_a * I + mean_b).astype(np.float32, copy=False)


def _apply_guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int,
    eps: float = 1e-3,
) -> np.ndarray:
    """Edge-preserving Guided Filter using OpenCV ximgproc with SciPy fallback."""
    guide_f32 = np.ascontiguousarray(guide, dtype=np.float32)
    src_f32 = np.ascontiguousarray(src, dtype=np.float32)
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
        out = cv2.ximgproc.guidedFilter(
            guide=guide_f32,
            src=src_f32,
            radius=int(radius),
            eps=float(eps),
        )
        return np.ascontiguousarray(out, dtype=np.float32)
    return _guided_filter_box(guide_f32, src_f32, radius=int(radius), eps=float(eps))


def _resolve_external_depth(
    external_depth: np.ndarray | None,
    target_shape: tuple[int, int],
) -> np.ndarray | None:
    """Resolve optional external depth map from argument or MONOLITH_AI_DEPTH_NPY."""
    depth_arr = external_depth
    if depth_arr is None:
        env_path = os.environ.get("MONOLITH_AI_DEPTH_NPY", "").strip()
        if env_path:
            depth_path = Path(env_path)
            if depth_path.is_file():
                loaded = np.load(str(depth_path), allow_pickle=False)
                if hasattr(loaded, "files"):
                    with loaded:
                        if not loaded.files:
                            raise ValueError(f"Empty archive at MONOLITH_AI_DEPTH_NPY={depth_path}")
                        depth_arr = loaded[loaded.files[0]]
                else:
                    depth_arr = loaded

    if depth_arr is None:
        return None

    arr = np.asarray(depth_arr)
    if arr.size == 0:
        raise ValueError("external_depth array is empty")

    if arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        elif arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[2] in (3, 4):
            arr = np.mean(arr[:, :, :3].astype(np.float32, copy=False), axis=2)
        else:
            raise ValueError(f"Unsupported 3D external_depth shape: {arr.shape}")
    elif arr.ndim != 2:
        raise ValueError(f"external_depth must be 2D (or squeezable to 2D), got shape {arr.shape}")

    if arr.dtype == np.uint8:
        d = arr.astype(np.float32) / np.float32(255.0)
    elif arr.dtype == np.uint16:
        d = arr.astype(np.float32) / np.float32(65535.0)
    elif np.issubdtype(arr.dtype, np.integer):
        max_val = float(np.iinfo(arr.dtype).max)
        d = arr.astype(np.float32) / np.float32(max_val if max_val > 0 else 1.0)
    else:
        d = np.array(arr, dtype=np.float32, copy=True)

    finite = np.isfinite(d)
    if not np.all(finite):
        fill_val = float(d[finite].min()) if np.any(finite) else 0.0
        d[~finite] = np.float32(fill_val)

    H, W = target_shape
    dh, dw = int(d.shape[0]), int(d.shape[1])
    if (dh, dw) != (H, W):
        interp = cv2.INTER_AREA if (H * W) < (dh * dw) else cv2.INTER_CUBIC
        d = cv2.resize(d, (W, H), interpolation=interp).astype(np.float32, copy=False)

    return np.ascontiguousarray(d, dtype=np.float32)


def _compute_topological_overlap_raw(
    M_sil: np.ndarray,
    M_regions: np.ndarray,
    sil_bin: np.ndarray,
    sil_area: float,
    H: int,
    W: int,
) -> np.ndarray | None:
    """Compute deterministic connected-component geometric overlap ordering before filtering."""
    reg_bin = (M_regions > np.float32(0.5)).astype(np.uint8)
    d_sil = cv2.distanceTransform(sil_bin, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    d_sil_max = max(float(d_sil.max()), 1.0)

    n_reg, labels, stats, centroids = cv2.connectedComponentsWithStats(reg_bin, connectivity=8)
    if n_reg <= 1:
        return None

    # Aggregate per-region mean silhouette depth cleanly in O(N) via bincount.
    flat_labels = labels.ravel()
    counts = np.bincount(flat_labels, minlength=n_reg).astype(np.float64)
    d_sil_sum = np.bincount(flat_labels, weights=d_sil.ravel(), minlength=n_reg)

    areas = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    valid_areas = areas[1:]
    max_area = max(float(valid_areas.max()) if valid_areas.size else 1.0, 1.0)
    log_max_area = float(np.log1p(max_area))

    region_depths = np.zeros(n_reg, dtype=np.float32)
    for k in range(1, n_reg):
        area_k = areas[k]
        if area_k <= 0.0:
            continue
        mean_dsil = (d_sil_sum[k] / max(counts[k], 1.0)) / d_sil_max
        # Smaller foreground segments (overlapping arms/hands) sit proud of large torso slabs,
        # while tiny noise/sliver specks are damped smoothly toward neutral depth.
        area_inv = 1.0 - (np.log1p(area_k) / max(log_max_area, 1e-6))
        confidence = min(1.0, area_k / max(0.002 * sil_area, 1.0))
        cy_norm = float(centroids[k, 1]) / max(float(H), 1.0)
        elev_prior = 1.0 - cy_norm

        score = (
            0.55 * confidence * area_inv
            + 0.30 * mean_dsil
            + 0.15 * elev_prior
        )
        region_depths[k] = np.float32(0.10 + 0.90 * np.clip(score, 0.0, 1.0))

    D_raw = region_depths[labels]

    # Bridge narrow interior contour gaps before guided filtering so dark line pixels
    # do not pull region plateaus toward zero inside the filter support window.
    bridge_rad = max(2, int(round(0.004 * float(min(H, W)))))
    bridge_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * bridge_rad + 1, 2 * bridge_rad + 1))
    D_filled = cv2.dilate(D_raw, bridge_kern)
    D_raw = np.where(labels > 0, D_raw, D_filled).astype(np.float32, copy=False)
    D_raw *= M_sil.astype(np.float32, copy=False)
    return D_raw


def compute_ai_guided_ordering(
    L_clean: np.ndarray,
    M_sil: np.ndarray,
    M_regions: np.ndarray,
    ai_mode: str = "off",
    external_depth: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute low-frequency, line-locked region depth ordering in float32 [0, 1].

    When `ai_mode == 'off'`, returns an array of zeros (zero hallucinations).
    When `ai_mode == 'guided_overlap'`:
      - If `external_depth` (or an existing `.npy` path in `MONOLITH_AI_DEPTH_NPY`) is
        provided, resizes it in float32, normalizes foreground values to [0, 1], masks
        by `M_sil`, and refines via an edge-preserving Guided Filter locked to `L_clean`.
      - Otherwise (default), runs the deterministic, zero-dependency connected-component
        topological overlap solver (`area_inv`, `mean_dsil`, `elev_prior`) + Guided Filter.
    """
    H, W = int(L_clean.shape[0]), int(L_clean.shape[1])
    mode = (ai_mode or "off").strip().lower()
    if mode in ("off", "none", "false", "0", ""):
        return np.zeros((H, W), dtype=np.float32)

    sil_bin = (M_sil > np.float32(0.5)).astype(np.uint8)
    sil_area = float(np.count_nonzero(sil_bin))
    if sil_area < 1.0:
        return np.zeros((H, W), dtype=np.float32)

    fg = M_sil > np.float32(0.5)
    ext_depth = _resolve_external_depth(external_depth, (H, W))
    if ext_depth is not None:
        fg_vals = ext_depth[fg]
        mn = float(fg_vals.min())
        mx = float(fg_vals.max())
        span = mx - mn
        if span > 1e-6:
            D_raw = np.clip(
                (ext_depth - np.float32(mn)) / np.float32(span),
                np.float32(0.0),
                np.float32(1.0),
            ).astype(np.float32, copy=False)
            D_raw *= M_sil.astype(np.float32, copy=False)
        else:
            # Uniform depth inside foreground -> zero raw input yields flat 0.5 after post-norm
            # without amplifying Guided Filter edge rolloff across the outer silhouette boundary.
            D_raw = np.zeros((H, W), dtype=np.float32)
    else:
        D_raw_opt = _compute_topological_overlap_raw(M_sil, M_regions, sil_bin, sil_area, H, W)
        if D_raw_opt is None:
            return np.zeros((H, W), dtype=np.float32)
        D_raw = D_raw_opt

    gf_radius = max(8, int(round(0.015 * float(min(H, W)))))
    D_guided = _apply_guided_filter(
        guide=L_clean,
        src=D_raw,
        radius=gf_radius,
        eps=1e-3,
    )

    D_guided = np.maximum(D_guided, np.float32(0.0), dtype=np.float32)
    D_guided *= M_sil.astype(np.float32, copy=False)
    D_guided[M_sil <= np.float32(0.01)] = np.float32(0.0)

    if np.any(fg):
        fg_vals = D_guided[fg]
        mn = float(fg_vals.min())
        mx = float(fg_vals.max())
        span = mx - mn
        if span > 1e-6:
            D_guided[fg] = ((fg_vals - np.float32(mn)) / np.float32(span)).astype(np.float32)
        else:
            D_guided[fg] = np.float32(0.5)

    np.clip(D_guided, 0.0, 1.0, out=D_guided)
    return np.ascontiguousarray(D_guided, dtype=np.float32)
