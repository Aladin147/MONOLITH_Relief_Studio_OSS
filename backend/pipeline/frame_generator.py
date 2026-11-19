"""Stage 3: Procedural frame heightfield generator in float32."""

from __future__ import annotations

from typing import Any

import numpy as np


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _normalize_preset(raw_preset: str | None) -> str:
    p = (raw_preset or "gallery_classic").strip().lower().replace("-", "_").replace(" ", "_")
    if p in ("gallery_classic", "classic", "ogee"):
        return "gallery_classic"
    if p in ("flat", "modern_flat", "flat_box"):
        return "modern_flat"
    if p in ("bevel", "stepped_bevel", "beveled", "beveled_molding"):
        return "stepped_bevel"
    if p in ("circular_medallion", "medallion", "circular"):
        return "circular_medallion"
    return "gallery_classic"


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    denom = max(edge1 - edge0, 1e-6)
    t = np.clip((x - np.float32(edge0)) / np.float32(denom), np.float32(0.0), np.float32(1.0))
    return (t * t * (np.float32(3.0) - np.float32(2.0) * t)).astype(np.float32, copy=False)


def _molding_profile(t: np.ndarray, preset: str) -> np.ndarray:
    """
    Evaluate molding cross-section height in [0, 1] for normalized position t in [0, 1],
    where t=0 is the outer border and t=1 is the inner border.
    """
    t = np.clip(t, np.float32(0.0), np.float32(1.0)).astype(np.float32, copy=False)

    if preset == "modern_flat":
        # Clean architectural flat bar with crisp micro-chamfers at outer/inner edges.
        rise = _smoothstep(0.0, 0.08, t)
        fall = np.float32(1.0) - _smoothstep(0.92, 1.0, t)
        return np.minimum(rise, fall).astype(np.float32, copy=False)

    if preset == "stepped_bevel":
        # Multi-tier stepped bevel profile: outer chamfer -> step plateau -> central bevel ridge -> inner step.
        prof = np.zeros_like(t, dtype=np.float32)
        m1 = t < 0.15
        m2 = (t >= 0.15) & (t < 0.30)
        m3 = (t >= 0.30) & (t < 0.50)
        m4 = (t >= 0.50) & (t < 0.70)
        m5 = (t >= 0.70) & (t < 0.85)
        m6 = t >= 0.85
        prof[m1] = (t[m1] / np.float32(0.15)) * np.float32(0.55)
        prof[m2] = np.float32(0.55)
        prof[m3] = np.float32(0.55) + ((t[m3] - np.float32(0.30)) / np.float32(0.20)) * np.float32(0.45)
        prof[m4] = np.float32(1.0) - ((t[m4] - np.float32(0.50)) / np.float32(0.20)) * np.float32(0.45)
        prof[m5] = np.float32(0.55)
        prof[m6] = (np.float32(1.0) - (t[m6] - np.float32(0.85)) / np.float32(0.15)) * np.float32(0.55)
        return np.clip(prof, 0.0, 1.0)

    # Classic gallery ogee molding (and circular_medallion uses the same sculpted ogee).
    prof = np.zeros_like(t, dtype=np.float32)
    m_outer_lip = t < 0.14
    m_cove = (t >= 0.14) & (t < 0.28)
    m_ogee = (t >= 0.28) & (t < 0.72)
    m_fillet = (t >= 0.72) & (t < 0.86)
    m_inner_lip = t >= 0.86

    # 1. Outer step rise 0 -> 0.52
    u1 = t[m_outer_lip] / np.float32(0.14)
    prof[m_outer_lip] = np.float32(0.52) * (u1 * u1 * (np.float32(3.0) - np.float32(2.0) * u1))

    # 2. Concave outer flute/cove: 0.52 -> dip to 0.38 -> 0.52
    u2 = (t[m_cove] - np.float32(0.14)) / np.float32(0.14)
    prof[m_cove] = np.float32(0.52) - np.float32(0.14) * np.sin(np.float32(np.pi) * u2)

    # 3. Main convex torus / ogee crown peaking smoothly at 1.0
    u3 = (t[m_ogee] - np.float32(0.28)) / np.float32(0.44)
    crown = np.sin(np.float32(np.pi) * u3)
    prof[m_ogee] = np.float32(0.52) + np.float32(0.48) * np.sqrt(np.maximum(crown, np.float32(0.0)))

    # 4. Inner stepped fillet bead: 0.52 -> 0.42 -> 0.50
    u4 = (t[m_fillet] - np.float32(0.72)) / np.float32(0.14)
    prof[m_fillet] = np.float32(0.52) - np.float32(0.10) * np.sin(np.float32(np.pi) * u4) - np.float32(0.04) * u4

    # 5. Inner rabbet drop: 0.48 -> 0.0
    u5 = (t[m_inner_lip] - np.float32(0.86)) / np.float32(0.14)
    prof[m_inner_lip] = np.float32(0.48) * (np.float32(1.0) - u5 * u5 * (np.float32(3.0) - np.float32(2.0) * u5))

    return np.clip(prof, 0.0, 1.0).astype(np.float32, copy=False)


FRAME_UNITS_MM = "mm"

# Legacy (pre-"units") frame values were reference-plaque units that the
# generator silently rescaled. Only used to migrate stored projects/presets once.
_LEGACY_PLAQUE_WIDTH_MM = 100.0
_LEGACY_RELIEF_Z_MM = 3.0
_LEGACY_WIDTH_LIMIT_MM = 25.0
_LEGACY_MIN_CANVAS_MM = 200.0
_LEGACY_CROWN_MARGIN = 0.06


def migrate_legacy_frame(
    frame: dict[str, Any],
    physical_width_mm: float,
    relief_z_mm: float,
    h_min: float,
) -> dict[str, Any]:
    """
    Convert a legacy frame dict (no `units` key) into true physical millimetres.

    Reproduces the geometry the old implicit rescaling generated, so existing projects
    and presets look identical after the upgrade, then tags the dict `units="mm"`.
    """
    out = dict(frame)
    if out.get("units") == FRAME_UNITS_MM:
        return out
    width_mm = float(out.get("width_mm", 2.2))
    thickness_mm = float(out.get("thickness_mm", 1.1))
    phys_w = max(float(physical_width_mm), 1e-3)
    rel_z = max(float(relief_z_mm), 1e-3)
    if width_mm < _LEGACY_WIDTH_LIMIT_MM and phys_w > _LEGACY_MIN_CANVAS_MM:
        width_mm *= phys_w / _LEGACY_PLAQUE_WIDTH_MM
        thickness_mm *= max(1.0, rel_z / _LEGACY_RELIEF_Z_MM)
    # The old generator also floored the crown at h_min + 0.06 of the relief depth.
    crown_floor_mm = min(1.0, max(0.0, float(h_min)) + _LEGACY_CROWN_MARGIN) * rel_z
    thickness_mm = min(max(thickness_mm, crown_floor_mm), rel_z)
    out["width_mm"] = round(width_mm, 4)
    out["thickness_mm"] = round(thickness_mm, 4)
    out["units"] = FRAME_UNITS_MM
    return out


def generate_frame_patches(
    shape: tuple[int, int],
    frame_cfg: dict[str, Any] | object | None,
    physical_width_mm: float = 1000.0,
    relief_z_mm: float = 15.0,
) -> list[tuple[int, int, int, int, np.ndarray]]:
    """
    Compute non-zero rectangular frame patches `[(r0, r1, c0, c1, float32_patch), ...]`.

    `width_mm`, `thickness_mm` and `back_protrusion_mm` are honoured as true physical
    millimetres on the `physical_width_mm` canvas (thickness is relative to the Z=0
    floor and capped at `relief_z_mm`, the top of the normalized heightmap).

    Evaluates only the active border strips/ROI so frame compositing onto a 4K
    heightmap avoids full-frame allocations and runs in ~1-2ms.
    """
    H, W = int(shape[0]), int(shape[1])

    enabled = bool(_cfg_get(frame_cfg, "enabled", False))
    if not enabled or frame_cfg is None:
        return []

    preset = _normalize_preset(_cfg_get(frame_cfg, "preset", "gallery_classic"))
    width_mm = float(_cfg_get(frame_cfg, "width_mm", 2.2))
    thickness_mm = float(_cfg_get(frame_cfg, "thickness_mm", 1.1))
    back_protrusion_mm = float(_cfg_get(frame_cfg, "back_protrusion_mm", 0.0))
    bbox = _cfg_get(frame_cfg, "bbox", None) or [0.16, 0.48, 0.84, 0.96]

    if width_mm <= 0.0 or thickness_mm <= 0.0:
        return []

    u0, v0, u1, v1 = [float(v) for v in bbox]
    u0, u1 = max(0.0, min(u0, u1)), min(1.0, max(u0, u1))
    v0, v1 = max(0.0, min(v0, v1)), min(1.0, max(v0, v1))
    if (u1 - u0) < 1e-4 or (v1 - v0) < 1e-4:
        return []

    phys_w = max(float(physical_width_mm), 1e-3)
    phys_h = phys_w * (float(H) / float(max(W, 1)))
    rel_z = max(float(relief_z_mm), 1e-3)

    x0_mm = u0 * phys_w
    x1_mm = u1 * phys_w
    y0_mm = v0 * phys_h
    y1_mm = v1 * phys_h

    min_frame_mm = 3.0 * (phys_w / float(max(W, 1)))
    eff_width_mm = max(width_mm, min_frame_mm)
    half_w_mm = 0.5 * eff_width_mm
    thickness_norm = float(np.clip(thickness_mm / rel_z, 0.0, 1.0))
    back_norm = float(np.clip(back_protrusion_mm / rel_z, 0.0, 1.0))

    px_per_mm_x = float(W) / phys_w
    px_per_mm_y = float(H) / phys_h
    pad_px_x = int(np.ceil(half_w_mm * px_per_mm_x)) + 2
    pad_px_y = int(np.ceil(half_w_mm * px_per_mm_y)) + 2

    c0 = max(0, int(np.floor(u0 * W)) - pad_px_x)
    c1 = min(W, int(np.ceil(u1 * W)) + pad_px_x)
    r0 = max(0, int(np.floor(v0 * H)) - pad_px_y)
    r1 = min(H, int(np.ceil(v1 * H)) + pad_px_y)
    if c1 <= c0 or r1 <= r0:
        return []

    patches: list[tuple[int, int, int, int, np.ndarray]] = []

    if back_norm > 0.0 or preset == "circular_medallion":
        xs_mm = ((np.arange(c0, c1, dtype=np.float32) + np.float32(0.5)) / np.float32(W)) * np.float32(phys_w)
        ys_mm = ((np.arange(r0, r1, dtype=np.float32) + np.float32(0.5)) / np.float32(H)) * np.float32(phys_h)

        if preset == "circular_medallion":
            cx = np.float32(0.5 * (x0_mm + x1_mm))
            cy = np.float32(0.5 * (y0_mm + y1_mm))
            rx = np.float32(max(0.5 * (x1_mm - x0_mm), 1e-3))
            ry = np.float32(max(0.5 * (y1_mm - y0_mm), 1e-3))
            X = xs_mm[None, :] - cx
            Y = ys_mm[:, None] - cy
            nx = X / rx
            ny = Y / ry
            rho = np.sqrt(nx * nx + ny * ny)
            grad_mag = np.sqrt((nx / rx) ** 2 + (ny / ry) ** 2)
            d_signed = (rho - np.float32(1.0)) / np.maximum(grad_mag, np.float32(1e-6))
        else:
            dx_1d = np.maximum(np.float32(x0_mm) - xs_mm, xs_mm - np.float32(x1_mm))[None, :]
            dy_1d = np.maximum(np.float32(y0_mm) - ys_mm, ys_mm - np.float32(y1_mm))[:, None]
            out_x = np.maximum(dx_1d, np.float32(0.0))
            out_y = np.maximum(dy_1d, np.float32(0.0))
            outside_dist = np.sqrt(out_x * out_x + out_y * out_y)
            inside_dist = np.minimum(np.maximum(dx_1d, dy_1d), np.float32(0.0))
            d_signed = (outside_dist + inside_dist).astype(np.float32, copy=False)

        roi = np.zeros((r1 - r0, c1 - c0), dtype=np.float32)
        if back_norm > 0.0:
            roi[d_signed <= np.float32(half_w_mm)] = np.float32(back_norm)

        border_mask = np.abs(d_signed) <= np.float32(half_w_mm)
        if np.any(border_mask):
            t = np.float32(0.5) * (np.float32(1.0) - d_signed[border_mask] / np.float32(half_w_mm))
            molding = _molding_profile(t, preset) * np.float32(thickness_norm)
            if back_norm > 0.0:
                roi[border_mask] = np.maximum(roi[border_mask], molding)
            else:
                roi[border_mask] = molding
        patches.append((r0, r1, c0, c1, roi))
    else:
        strip_y = min(r1 - r0, 2 * pad_px_y + 1)
        strip_x = min(c1 - c0, 2 * pad_px_x + 1)
        boxes = [
            (r0, min(r1, r0 + strip_y), c0, c1),
            (max(r0, r1 - strip_y), r1, c0, c1),
            (min(r1, r0 + strip_y), max(r0, r1 - strip_y), c0, min(c1, c0 + strip_x)),
            (min(r1, r0 + strip_y), max(r0, r1 - strip_y), max(c0, c1 - strip_x), c1),
        ]
        for sr0, sr1, sc0, sc1 in boxes:
            if sr1 <= sr0 or sc1 <= sc0:
                continue
            xs_mm = ((np.arange(sc0, sc1, dtype=np.float32) + np.float32(0.5)) / np.float32(W)) * np.float32(phys_w)
            ys_mm = ((np.arange(sr0, sr1, dtype=np.float32) + np.float32(0.5)) / np.float32(H)) * np.float32(phys_h)
            dx_1d = np.maximum(np.float32(x0_mm) - xs_mm, xs_mm - np.float32(x1_mm))[None, :]
            dy_1d = np.maximum(np.float32(y0_mm) - ys_mm, ys_mm - np.float32(y1_mm))[:, None]
            out_x = np.maximum(dx_1d, np.float32(0.0))
            out_y = np.maximum(dy_1d, np.float32(0.0))
            d_signed = np.sqrt(out_x * out_x + out_y * out_y) + np.minimum(np.maximum(dx_1d, dy_1d), np.float32(0.0))
            border_mask = np.abs(d_signed) <= np.float32(half_w_mm)
            if np.any(border_mask):
                t = np.float32(0.5) * (np.float32(1.0) - d_signed[border_mask] / np.float32(half_w_mm))
                sub = np.zeros((sr1 - sr0, sc1 - sc0), dtype=np.float32)
                sub[border_mask] = _molding_profile(t, preset) * np.float32(thickness_norm)
                patches.append((sr0, sr1, sc0, sc1, sub))

    return patches


def generate_frame(
    shape: tuple[int, int],
    frame_cfg: dict[str, Any] | object | None,
    physical_width_mm: float = 1000.0,
    relief_z_mm: float = 15.0,
) -> np.ndarray:
    """
    Generate a procedural molding frame heightfield in float32 [0.0, 1.0].

    Supports `enabled`, `preset` (`gallery_classic`, `modern_flat`/`flat`,
    `stepped_bevel`/`bevel`, `circular_medallion`), `width_mm`, `thickness_mm`,
    `back_protrusion_mm`, and `bbox` (`[u0, v0, u1, v1]`).
    """
    H, W = int(shape[0]), int(shape[1])
    out = np.zeros((H, W), dtype=np.float32)
    for sr0, sr1, sc0, sc1, patch in generate_frame_patches(
        shape,
        frame_cfg,
        physical_width_mm=physical_width_mm,
        relief_z_mm=relief_z_mm,
    ):
        out[sr0:sr1, sc0:sc1] = patch
    return out
