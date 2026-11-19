"""Stage 2: Deterministic float32 mathematical relief basis decomposition & <80ms recompute."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from scipy import fft as sfft

from .ai_ordering import compute_ai_guided_ordering
from .frame_generator import _cfg_get, generate_frame_patches


# `groove_sigma` in user params is expressed in pixels at this longest-edge resolution, so a
# relief looks the same at 2K, 4K or 8K. Engine layer dicts may carry `_groove_scale`
# (= longest edge / GROOVE_REFERENCE_RES) to convert; without it, params are raw pixels.
GROOVE_REFERENCE_RES = 4096
GROOVE_SIGMA_UNITS = "px@4096"


def groove_scale_for_shape(shape: tuple[int, ...]) -> float:
    """Pixels per groove_sigma reference unit for a grid of the given (H, W) shape."""
    return float(max(int(shape[0]), int(shape[1]))) / float(GROOVE_REFERENCE_RES)


def _param_get(params: Any, key: str, default: Any) -> Any:
    if params is None:
        return default
    if isinstance(params, dict):
        return params.get(key, default)
    return getattr(params, key, default)


def derive_m_lines(layers: dict[str, Any]) -> np.ndarray | None:
    """Interior contour lines: the stored `M_lines` if present, else (1 - L_clean) * M_sil."""
    stored = layers.get("M_lines")
    if stored is not None:
        return stored
    if "L_clean" in layers and "M_sil" in layers:
        return ((np.float32(1.0) - layers["L_clean"]) * layers["M_sil"]).astype(np.float32, copy=False)
    return None


def _compute_groove_field(M_lines: np.ndarray, groove_sigma: float) -> np.ndarray:
    """Compute normalized [0, 1] sculptural crease field from interior contour lines."""
    lines = np.ascontiguousarray(M_lines, dtype=np.float32)
    sigma = float(groove_sigma)
    if sigma > 1e-3:
        G = cv2.GaussianBlur(lines, (0, 0), sigmaX=sigma, sigmaY=sigma).astype(np.float32, copy=False)
    else:
        G = lines.copy()
    mx = float(G.max())
    if mx > 1e-6:
        G = (G / np.float32(mx)).astype(np.float32, copy=False)
    return np.ascontiguousarray(G, dtype=np.float32)


def compute_basis_layers(
    preprocessed: dict[str, Any],
    ai_mode: str = "off",
    groove_sigma: float = 2.5,
) -> dict[str, Any]:
    """
    Stage A (Heavy One-Time Compute): Extract cached float32 basis layers.

    Returns dict containing `M_sil`, `d_sil`, `H_global`, `H_local`, `G_lines`,
    `H_ai_guided` (all np.float32), plus `L_clean`, `M_lines`, and `groove_sigma`.
    """
    L_clean = np.ascontiguousarray(preprocessed["L_clean"], dtype=np.float32)
    M_sil = np.ascontiguousarray(preprocessed["M_sil"], dtype=np.float32)
    M_lines = np.ascontiguousarray(preprocessed["M_lines"], dtype=np.float32)
    M_regions = np.ascontiguousarray(preprocessed["M_regions"], dtype=np.float32)

    H, W = int(L_clean.shape[0]), int(L_clean.shape[1])
    min_dim = float(min(H, W))
    max_dim = float(max(H, W))

    # 1. Exact Euclidean Distance Transform of silhouette (float32).
    sil_bin = (M_sil > np.float32(0.5)).astype(np.uint8)
    d_sil = cv2.distanceTransform(sil_bin, cv2.DIST_L2, cv2.DIST_MASK_PRECISE).astype(np.float32, copy=False)

    # 2. Global silhouette anatomical cushion H_global in [0, 1].
    R_sil = max(16.0, 0.15 * min_dim)
    u_sil = np.clip(d_sil / np.float32(R_sil), np.float32(0.0), np.float32(1.0))
    H_global_raw = np.power(
        np.maximum(np.float32(0.0), np.float32(1.0) - (np.float32(1.0) - u_sil) ** 2),
        np.float32(0.75),
    ).astype(np.float32, copy=False)

    sig_global = max(8.0, 0.02 * min_dim)
    H_global = cv2.GaussianBlur(H_global_raw, (0, 0), sigmaX=sig_global, sigmaY=sig_global).astype(
        np.float32, copy=False
    )
    H_global *= M_sil
    g_max = float(H_global.max())
    if g_max > 1e-6:
        H_global /= np.float32(g_max)
    H_global[M_sil <= np.float32(0.01)] = np.float32(0.0)

    # 3. Local limb & enclosed region elliptical puffing + FFT screened Poisson smoothing.
    reg_bin = (M_regions > np.float32(0.5)).astype(np.uint8)
    d_reg = cv2.distanceTransform(reg_bin, cv2.DIST_L2, cv2.DIST_MASK_PRECISE).astype(np.float32, copy=False)
    R_reg = max(8.0, 0.06 * min_dim)
    u_reg = np.clip(d_reg / np.float32(R_reg), np.float32(0.0), np.float32(1.0))
    H_reg_raw = np.sqrt(
        np.maximum(np.float32(0.0), np.float32(1.0) - (np.float32(1.0) - u_reg) ** 2)
    ).astype(np.float32, copy=False)

    # Biharmonic / screened Poisson filter: 1 / (1 + l1*k^2 + l2*k^4) in float32/complex64.
    s_px = max(3.0, 0.0085 * max_dim)
    # The FFT treats the image as periodic: a subject cropped by one canvas edge would be
    # pulled toward whatever touches the opposite edge. Reflect-pad by ~6 filter lengths
    # (kernel tail < 0.3%) to a fast FFT size, filter, then crop back.
    pad = int(math.ceil(6.0 * s_px))
    pad_y = min(pad, H - 1)
    pad_x = min(pad, W - 1)
    Hp = sfft.next_fast_len(H + 2 * pad_y, real=True)
    Wp = sfft.next_fast_len(W + 2 * pad_x, real=True)
    top, left = pad_y, pad_x
    padded = np.pad(
        H_reg_raw,
        ((top, Hp - H - top), (left, Wp - W - left)),
        mode="reflect" if min(H, W) > 1 else "edge",
    )
    ky = (np.float32(2.0 * np.pi) * sfft.fftfreq(Hp).astype(np.float32))[:, None]
    kx = (np.float32(2.0 * np.pi) * sfft.rfftfreq(Wp).astype(np.float32))[None, :]
    k2 = kx * kx + ky * ky
    l1 = np.float32(0.5 * s_px * s_px)
    l2 = np.float32(0.15 * (s_px ** 4))
    filt = np.float32(1.0) / (np.float32(1.0) + l1 * k2 + l2 * (k2 * k2))

    spectrum = sfft.rfft2(padded, workers=-1)
    spectrum *= filt
    H_local = np.ascontiguousarray(
        sfft.irfft2(spectrum, s=(Hp, Wp), workers=-1)[top : top + H, left : left + W],
        dtype=np.float32,
    )
    np.maximum(H_local, np.float32(0.0), out=H_local)
    H_local *= M_sil
    l_max = float(H_local.max())
    if l_max > 1e-6:
        H_local /= np.float32(l_max)
    H_local[M_sil <= np.float32(0.01)] = np.float32(0.0)

    # 4. Sculptural contour line crease field G_lines.
    G_lines = _compute_groove_field(M_lines, groove_sigma)
    G_lines *= M_sil

    # 5. Optional line-locked AI/geometric region ordering.
    H_ai_guided = compute_ai_guided_ordering(L_clean, M_sil, M_regions, ai_mode=ai_mode)

    layers: dict[str, Any] = {
        "M_sil": np.ascontiguousarray(M_sil, dtype=np.float32),
        "d_sil": np.ascontiguousarray(d_sil, dtype=np.float32),
        "H_global": np.ascontiguousarray(H_global, dtype=np.float32),
        "H_local": np.ascontiguousarray(H_local, dtype=np.float32),
        "G_lines": np.ascontiguousarray(G_lines, dtype=np.float32),
        "H_ai_guided": np.ascontiguousarray(H_ai_guided, dtype=np.float32),
        "L_clean": L_clean,
        "M_lines": M_lines,
        "groove_sigma": float(groove_sigma),
    }
    _ensure_layer_cache(layers)
    return layers


def _ensure_layer_cache(layers: dict[str, Any]) -> None:
    """Populate transient ROI slices and workspace buffers on a layers dict for <50ms recompute."""
    if not isinstance(layers, dict):
        return
    M_sil = layers["M_sil"]
    fg_bool = layers.get("_fg_bool")
    if fg_bool is None or fg_bool.shape != M_sil.shape:
        fg_bool = M_sil > np.float32(0.01)
        layers["_fg_bool"] = fg_bool
        # Zero-copy uint8 view of the bool mask (cv2 masks want uint8).
        layers["_fg_u8"] = fg_bool.view(np.uint8)
        if np.array_equal(M_sil, fg_bool):
            # Exactly binary (full-res silhouette): already clean, no second copy needed.
            layers["_M_sil_clean"] = M_sil
        else:
            M_sil_clean = M_sil.copy()
            M_sil_clean[~fg_bool] = np.float32(0.0)
            layers["_M_sil_clean"] = M_sil_clean
        rows_any = np.any(fg_bool, axis=1)
        cols_any = np.any(fg_bool, axis=0)
        if np.any(rows_any) and np.any(cols_any):
            r_idx = np.flatnonzero(rows_any)
            c_idx = np.flatnonzero(cols_any)
            layers["_fg_bbox"] = (int(r_idx[0]), int(r_idx[-1]) + 1, int(c_idx[0]), int(c_idx[-1]) + 1)
        else:
            layers["_fg_bbox"] = None

    bbox = layers.get("_fg_bbox")
    if bbox is None or "H_global" not in layers:  # Sculpt-only layers carry no Classic arrays
        return
    r0, r1, c0, c1 = bbox
    W_px = int(M_sil.shape[1])
    if (c1 - c0) < int(0.85 * W_px):
        roi_shape = (r1 - r0, c1 - c0)
        roi_cache = layers.get("_roi_cache")
        if roi_cache is None or roi_cache.get("shape") != roi_shape:
            M_sil_clean = layers.get("_M_sil_clean", M_sil)
            fg_u8 = layers.get("_fg_u8")
            # Zero-copy row-strided views of the full layers (cv2 handles the stride at
            # the same speed); only the work buffer is allocated.
            layers["_roi_cache"] = {
                "shape": roi_shape,
                "M_sil": M_sil_clean[r0:r1, c0:c1],
                "fg_bool": fg_bool[r0:r1, c0:c1],
                "fg_u8": fg_u8[r0:r1, c0:c1] if fg_u8 is not None else fg_bool[r0:r1, c0:c1].view(np.uint8),
                "H_global": layers["H_global"][r0:r1, c0:c1],
                "H_local": layers["H_local"][r0:r1, c0:c1],
                "G_lines": layers["G_lines"][r0:r1, c0:c1],
                "H_ai_guided": layers["H_ai_guided"][r0:r1, c0:c1],
                "groove_sigma": float(layers.get("groove_sigma", 2.5)),
                "buf": np.empty(roi_shape, dtype=np.float32),
            }


def _apply_draft_limit(
    layers: dict[str, Any],
    out: np.ndarray,
    draft_angle_deg: float,
    physical_width_mm: float,
    relief_z_mm: float,
) -> None:
    """Clamp `out` in place to the draft-angle envelope (cached per geometry)."""
    r0, r1, _c0, _c1 = layers["_fg_bbox"]
    W_px = int(out.shape[1])
    px_mm = max(float(physical_width_mm), 1e-3) / float(max(W_px, 1))
    theta = math.radians(min(float(draft_angle_deg), 89.0))
    # Normalized height allowed per pixel of distance from the outline.
    slope = px_mm / (math.tan(theta) * max(float(relief_z_mm), 1e-6))
    key = (round(slope, 9), r0, r1, out.shape)
    cached = layers.get("_draft_limit")
    if cached is None or cached[0] != key:
        # d_sil is measured to the centre of the outermost foreground pixel; the outline
        # itself lies half a pixel further out.
        limit = (layers["d_sil"][r0:r1, :] - np.float32(0.5)) * np.float32(slope)
        np.clip(limit, np.float32(0.0), np.float32(1.0), out=limit)
        # Outside the silhouette only brush edits can exist; leave them untouched.
        limit[~layers["_fg_bool"][r0:r1, :]] = np.float32(1.0)
        limit = np.ascontiguousarray(limit, dtype=np.float32)
        layers["_draft_limit"] = (key, limit)
    else:
        limit = cached[1]
    rows = out[r0:r1, :]
    np.minimum(rows, limit, out=rows)


def recompute_relief(
    layers: dict[str, Any],
    params: dict[str, Any] | object | None = None,
    physical_width_mm: float = 1000.0,
    relief_z_mm: float = 15.0,
    *,
    include_brush: bool = True,
    include_frame: bool = True,
) -> np.ndarray:
    """
    Stage B (Fast Cached Composite, <80ms at 4K): Blend basis layers in pure float32.

    Supports `w_global`, `w_local`, `w_ai`, `w_groove`, `groove_sigma`, `gamma`,
    `brightness`, `contrast`, `h_min`, `invert_depth`, `draft_angle_deg`, and `frame`.

    When `layers["_has_brush_delta"]` is set, the float32 `layers["H_brush_delta"]`
    is added after the silhouette clip and before frame compositing.
    `include_brush=False` / `include_frame=False` return the unbrushed / unframed
    procedural relief (used to rebase incoming brush edits).

    `style` selects the profile: "classic" (the default when absent) or "sculpt", which reads
    the basis built by `sculpt_engine.build_sculpt_basis` from `layers["sculpt_basis"]` and
    ignores the Classic weights, gamma, brightness and contrast. Both share the tail: brush
    delta, exact draft limit, frame. Every Sculpt recompute sets the public key
    `layers["sculpt_edge_clamped"]` (bool): True when the draft-angle clamp raised the edge
    roundness or crease width above the user's setting (design decision 3), for the UI.
    """
    style = str(_param_get(params, "style", "classic") or "classic")
    if style == "sculpt":
        return _recompute_sculpt(
            layers, params, physical_width_mm, relief_z_mm, include_brush=include_brush, include_frame=include_frame
        )
    if style != "classic":
        raise ValueError(f"Unknown relief style {style!r}; expected 'classic' or 'sculpt'")
    M_sil = layers["M_sil"]
    d_sil = layers["d_sil"]
    H_global = layers["H_global"]
    H_local = layers["H_local"]
    G_lines = layers["G_lines"]
    H_ai_guided = layers["H_ai_guided"]

    H_px, W_px = int(M_sil.shape[0]), int(M_sil.shape[1])

    w_global = float(_param_get(params, "w_global", 0.45))
    w_local = float(_param_get(params, "w_local", 0.65))
    w_ai = float(_param_get(params, "w_ai", 0.25))
    w_groove = float(_param_get(params, "w_groove", 0.35))
    raw_sigma = _param_get(params, "groove_sigma", None)
    if raw_sigma is None:
        groove_sigma = float(layers.get("groove_sigma", 2.5))
    else:
        groove_sigma = float(raw_sigma) * float(layers.get("_groove_scale", 1.0))
    gamma = float(_param_get(params, "gamma", 0.85))
    brightness = float(_param_get(params, "brightness", 0.0))
    contrast = float(_param_get(params, "contrast", 1.0))
    h_min = float(_param_get(params, "h_min", 0.08))
    invert_depth = bool(_param_get(params, "invert_depth", False))
    draft_angle_deg = float(_param_get(params, "draft_angle_deg", 0.0))
    frame_cfg = _param_get(params, "frame", None)

    # Support interactive groove_sigma updates without re-running EDT/FFT.
    cached_sigma = float(layers.get("groove_sigma", 2.5))
    if abs(groove_sigma - cached_sigma) > 1e-3:
        M_lines = derive_m_lines(layers)  # transient: not stored, so it never pins RAM
        if M_lines is not None:
            G_lines = _compute_groove_field(M_lines, groove_sigma) * M_sil
            del M_lines
            layers["G_lines"] = G_lines
            layers["groove_sigma"] = groove_sigma

    _ensure_layer_cache(layers)
    fg_u8 = layers.get("_fg_u8")
    bbox = layers.get("_fg_bbox")
    M_sil_clean = layers.get("_M_sil_clean", M_sil)
    has_fg = bbox is not None
    out = np.zeros((H_px, W_px), dtype=np.float32)

    if has_fg:
        r0, r1, c0, c1 = bbox
        # Use cached contiguous ROI buffer when foreground occupies significantly less than full width,
        # or operate on contiguous row-slice directly.
        roi_cache = layers.get("_roi_cache")
        if roi_cache is not None and (c1 - c0) < int(0.85 * W_px):
            if abs(groove_sigma - float(roi_cache.get("groove_sigma", groove_sigma))) > 1e-3:
                roi_cache["G_lines"] = G_lines[r0:r1, c0:c1]
                roi_cache["groove_sigma"] = groove_sigma

            r_M_sil = roi_cache["M_sil"]
            r_fg_u8 = roi_cache["fg_u8"]
            r_Hg = roi_cache["H_global"]
            r_Hl = roi_cache["H_local"]
            r_Gl = roi_cache["G_lines"]
            r_Hai = roi_cache["H_ai_guided"]
            work = roi_cache["buf"]
        else:
            r_M_sil = M_sil_clean[r0:r1, :]
            r_fg_u8 = fg_u8[r0:r1, :]
            r_Hg = H_global[r0:r1, :]
            r_Hl = H_local[r0:r1, :]
            r_Gl = G_lines[r0:r1, :]
            r_Hai = H_ai_guided[r0:r1, :]
            work = out[r0:r1, :]

        # 1. SIMD linear layer blend.
        cv2.addWeighted(r_Hg, w_global, r_Hl, w_local, 0.0, dst=work)
        if abs(w_ai) > 1e-6:
            has_ai = layers.get("_has_ai")
            if has_ai is None:
                has_ai = bool(np.any(r_Hai > np.float32(1e-6)))
                if isinstance(layers, dict):
                    layers["_has_ai"] = has_ai
            if has_ai:
                cv2.addWeighted(work, 1.0, r_Hai, w_ai, 0.0, dst=work)
        if abs(w_groove) > 1e-6:
            cv2.addWeighted(work, 1.0, r_Gl, -w_groove, 0.0, dst=work)

        # 2. Foreground min-max normalization in [0, 1] using OpenCV SIMD minMaxLoc.
        mn, mx, _, _ = cv2.minMaxLoc(work, mask=r_fg_u8)
        span = mx - mn
        if span > 1e-6:
            inv_span = 1.0 / span
            cv2.addWeighted(work, inv_span, work, 0.0, -mn * inv_span, dst=work)
        else:
            work.fill(np.float32(0.5))

        # 3. Gamma curve via vectorized SIMD log+mul+exp.
        gamma_clamped = max(0.05, min(gamma, 5.0))
        if abs(gamma_clamped - 1.0) > 1e-4:
            np.clip(work, np.float32(1e-30), np.float32(1.0), out=work)
            cv2.log(work, dst=work)
            cv2.multiply(work, gamma_clamped, dst=work)
            cv2.exp(work, dst=work)
        else:
            np.clip(work, np.float32(0.0), np.float32(1.0), out=work)

        # 4. Brightness / contrast around 0.5 midpoint.
        if abs(contrast - 1.0) > 1e-4 or abs(brightness) > 1e-4:
            bias = 0.5 * (1.0 - contrast) + brightness
            cv2.addWeighted(work, contrast, work, 0.0, bias, dst=work)
            np.clip(work, np.float32(0.0), np.float32(1.0), out=work)

        # 5. Step silhouette above Z=0 floor via h_min and optional invert_depth.
        h_min_c = max(0.0, min(h_min, 0.95))
        if h_min_c > 1e-6:
            cv2.addWeighted(work, 1.0 - h_min_c, work, 0.0, h_min_c, dst=work)

        if invert_depth:
            cv2.addWeighted(work, -1.0, work, 0.0, 1.0, dst=work)

        # 6. Strictly zero background outside silhouette and ensure [0, 1].
        cv2.multiply(work, r_M_sil, dst=work)
        np.clip(work, np.float32(0.0), np.float32(1.0), out=work)

        if (c1 - c0) < int(0.85 * W_px):
            out[r0:r1, c0:c1] = work

    _finish_relief(
        layers,
        out,
        draft_angle_deg=draft_angle_deg,
        frame_cfg=frame_cfg,
        has_fg=has_fg,
        physical_width_mm=physical_width_mm,
        relief_z_mm=relief_z_mm,
        include_brush=include_brush,
        include_frame=include_frame,
    )
    return out


def _recompute_sculpt(
    layers: dict[str, Any],
    params: dict[str, Any] | object | None,
    physical_width_mm: float,
    relief_z_mm: float,
    *,
    include_brush: bool,
    include_frame: bool,
) -> np.ndarray:
    """Sculpt profile (sculpt_engine.sculpt_profile, invert included), then the shared tail."""
    from .sculpt_engine import sculpt_profile

    basis = layers.get("sculpt_basis") if isinstance(layers, dict) else None
    if basis is None:
        raise ValueError("style='sculpt' needs layers['sculpt_basis'] (sculpt_engine.build_sculpt_basis)")
    M_sil = layers["M_sil"]
    if "d_sil" not in layers:
        sil_bin = (M_sil > np.float32(0.5)).astype(np.uint8)
        layers["d_sil"] = cv2.distanceTransform(sil_bin, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    _ensure_layer_cache(layers)
    out = np.zeros(M_sil.shape, dtype=np.float32)
    sculpt_profile(basis, params, physical_width_mm, relief_z_mm, out)
    layers["sculpt_edge_clamped"] = bool(basis.get("_last_edge_clamped", False))
    _finish_relief(
        layers,
        out,
        draft_angle_deg=float(_param_get(params, "draft_angle_deg", 0.0)),
        frame_cfg=_param_get(params, "frame", None),
        has_fg=layers.get("_fg_bbox") is not None,
        physical_width_mm=physical_width_mm,
        relief_z_mm=relief_z_mm,
        include_brush=include_brush,
        include_frame=include_frame,
    )
    return out


def _finish_relief(
    layers: dict[str, Any],
    out: np.ndarray,
    *,
    draft_angle_deg: float,
    frame_cfg: Any,
    has_fg: bool,
    physical_width_mm: float,
    relief_z_mm: float,
    include_brush: bool,
    include_frame: bool,
) -> None:
    """The tail shared by every style: brush delta, exact draft limit, frame (in place)."""
    H_px, W_px = int(out.shape[0]), int(out.shape[1])
    # 7. Persistent local 3D brush edits (float32 delta) layered on the procedural
    # relief *before* frame compositing so slider recomputes never erase strokes.
    if include_brush and layers.get("_has_brush_delta", False):
        brush_delta = layers.get("H_brush_delta")
        if isinstance(brush_delta, np.ndarray) and brush_delta.shape == out.shape:
            if brush_delta.dtype != np.float32:
                brush_delta = brush_delta.astype(np.float32, copy=False)
            cv2.add(out, brush_delta, dst=out)
            np.clip(out, np.float32(0.0), np.float32(1.0), out=out)

    # Limit height by distance from the silhouette before adding the frame.
    if has_fg and draft_angle_deg > 1e-3:
        _apply_draft_limit(layers, out, draft_angle_deg, physical_width_mm, relief_z_mm)

    # 9. Procedural frame compositing over active frame patches only (cached in layers).
    if include_frame and frame_cfg is not None and bool(_cfg_get(frame_cfg, "enabled", False)):
        bbox_uv = _cfg_get(frame_cfg, "bbox", None) or [0.16, 0.48, 0.84, 0.96]
        frame_key = (
            H_px,
            W_px,
            str(_cfg_get(frame_cfg, "preset", "gallery_classic")),
            round(float(_cfg_get(frame_cfg, "width_mm", 2.2)), 5),
            round(float(_cfg_get(frame_cfg, "thickness_mm", 1.1)), 5),
            round(float(_cfg_get(frame_cfg, "back_protrusion_mm", 0.0)), 5),
            tuple(round(float(v), 5) for v in bbox_uv),
            round(float(physical_width_mm), 4),
            round(float(relief_z_mm), 4),
        )
        cached_frame = layers.get("_cached_frame") if isinstance(layers, dict) else None
        if cached_frame is not None and cached_frame[0] == frame_key:
            patches = cached_frame[1]
        else:
            patches = generate_frame_patches(
                (H_px, W_px),
                frame_cfg,
                physical_width_mm=physical_width_mm,
                relief_z_mm=relief_z_mm,
            )
            if isinstance(layers, dict):
                layers["_cached_frame"] = (frame_key, patches)

        for sr0, sr1, sc0, sc1, patch in patches:
            sub = out[sr0:sr1, sc0:sc1]
            np.maximum(sub, patch, out=sub)
