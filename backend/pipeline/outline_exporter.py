"""Stage 5: Sub-pixel silhouette outline extraction and SVG / DXF vector export (true mm).

The heightmap's silhouette is a binary pixel mask, so its edge is quantized to the pixel
grid. For a profile (cut-out) toolpath the outline should be a vector. We trace it with
marching squares on a field built from the smooth restored luminance `L_clean`, clamped so
that every silhouette pixel is above the iso-level and every background pixel below it:

  - topology matches `M_sil` exactly (interior contour lines, bridged T-junction gaps and
    reclaimed background pockets are never traced as extra loops);
  - the crossing position along each pixel edge uses the anti-aliased luminance, giving
    sub-pixel placement where the drawing's edge is a real white/black transition.

Coordinates: millimetres on the `physical_width_mm` canvas, pixel (r, c) centred at
((c + 0.5) * p, (r + 0.5) * p) with p = physical_width_mm / W, i.e. aligned with the 16-bit
PNG / EXR heightmap imported at the same physical width. SVG keeps the image's top-left
origin (y down); DXF uses the CAD convention (origin bottom-left, y up). Outer boundaries
are counter-clockwise and holes clockwise in the y-up frame, on separate layers / groups.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

ISO = np.float32(0.5)
_MARGIN = np.float32(1e-3)


@dataclass
class OutlineLoop:
    """A closed outline in canvas millimetres (y down, top-left origin)."""

    # (N, 2) float64 x/y in mm, first point not repeated. Visually counter-clockwise for
    # outer boundaries and clockwise for holes.
    points: np.ndarray
    is_hole: bool


def _outline_field(L_clean: np.ndarray, M_sil: np.ndarray, sigmoid_k: float | None = None) -> np.ndarray:
    """Luminance clamped to the silhouette mask's side of the iso-level, padded with background."""
    inside = M_sil > 0.5
    L = np.asarray(L_clean, dtype=np.float32)
    if sigmoid_k:
        # Undo the HD-restore sigmoid: the blurred luminance underneath is near-linear across
        # the edge, so linear interpolation along pixel edges places the crossing better
        # (worst-case error ~0.39 -> ~0.28 px in tests). Monotonic, so the 0.5 level is kept.
        Lc = np.clip(L, np.float32(1e-6), np.float32(1.0 - 1e-6))
        L = (ISO + np.log(Lc / (np.float32(1.0) - Lc)) / np.float32(sigmoid_k)).astype(np.float32)
    F = np.where(
        inside,
        np.maximum(L, ISO + _MARGIN),
        np.minimum(L, ISO - _MARGIN),
    ).astype(np.float32)
    # One pixel of background around the canvas closes contours that touch its border.
    return np.pad(F, 1, mode="constant", constant_values=0.0)


def _trace_loops(F: np.ndarray) -> list[np.ndarray]:
    """
    Marching squares on padded field F at ISO. Returns closed loops in padded (col, row)
    pixel-index coordinates, oriented with the inside on the right in image (y-down) axes.

    Only boundary cells are materialized, so cost scales with outline length rather than
    canvas area (a 4K canvas has ~14M cells but only tens of thousands on the outline).
    """
    Hp, Wp = F.shape
    ins = F > ISO
    case = (
        (ins[:-1, :-1].astype(np.uint8) << 3)
        | (ins[:-1, 1:].astype(np.uint8) << 2)
        | (ins[1:, 1:].astype(np.uint8) << 1)
        | ins[1:, :-1].astype(np.uint8)
    )
    n_h = Hp * (Wp - 1)  # horizontal edges (r, c)-(r, c+1) come first, then vertical ones

    def h_id(r, c):
        return r * (Wp - 1) + c

    def v_id(r, c):
        return n_h + r * Wp + c

    def edge_point(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Iso-crossing position (x=col, y=row) on each grid edge id."""
        ids = np.asarray(ids, dtype=np.int64)
        x = np.empty(ids.shape, dtype=np.float64)
        y = np.empty(ids.shape, dtype=np.float64)
        is_h = ids < n_h
        r, c = np.divmod(ids[is_h], Wp - 1)
        a, b = F[r, c].astype(np.float64), F[r, c + 1].astype(np.float64)
        x[is_h] = c + np.clip((0.5 - a) / (b - a), 0.0, 1.0)
        y[is_h] = r
        r, c = np.divmod(ids[~is_h] - n_h, Wp)
        a, b = F[r, c].astype(np.float64), F[r + 1, c].astype(np.float64)
        x[~is_h] = c
        y[~is_h] = r + np.clip((0.5 - a) / (b - a), 0.0, 1.0)
        return x, y

    def orient(e1, e2, qx, qy):
        """Order (e1, e2) so the inside reference point q lies to the right (y-down axes)."""
        x1, y1 = edge_point(e1)
        x2, y2 = edge_point(e2)
        cross = (x2 - x1) * (qy - y1) - (y2 - y1) * (qx - x1)
        keep = cross > 0
        return np.where(keep, e1, e2), np.where(keep, e2, e1)

    seg_from: list[np.ndarray] = []
    seg_to: list[np.ndarray] = []

    # Regular boundary cells: exactly two crossing edges.
    cr, cc = np.nonzero((case != 0) & (case != 15) & (case != 5) & (case != 10))
    if cr.size:
        tl, tr = ins[cr, cc], ins[cr, cc + 1]
        bl, br = ins[cr + 1, cc], ins[cr + 1, cc + 1]
        edges = np.stack([h_id(cr, cc), v_id(cr, cc + 1), h_id(cr + 1, cc), v_id(cr, cc)], axis=1)
        crosses = np.stack([tl != tr, tr != br, bl != br, tl != bl], axis=1)
        order = np.argsort(~crosses, axis=1, kind="stable")[:, :2]
        e1 = np.take_along_axis(edges, order[:, :1], axis=1)[:, 0]
        e2 = np.take_along_axis(edges, order[:, 1:2], axis=1)[:, 0]
        # Inside reference: centroid of the cell's inside corners.
        r0, c0 = cr.astype(np.float64), cc.astype(np.float64)
        cnt = tl.astype(np.float64) + tr + br + bl
        qx = (c0 * tl + (c0 + 1) * tr + (c0 + 1) * br + c0 * bl) / cnt
        qy = (r0 * tl + r0 * tr + (r0 + 1) * br + (r0 + 1) * bl) / cnt
        f, t = orient(e1, e2, qx, qy)
        seg_from.append(f)
        seg_to.append(t)

    # Saddle cells (rare): disambiguate with the cell-centre average.
    sr, sc = np.nonzero((case == 5) | (case == 10))
    if sr.size:
        f_list, t_list = [], []
        for r, c in zip(sr.tolist(), sc.tolist()):
            centre_in = float(F[r, c] + F[r, c + 1] + F[r + 1, c] + F[r + 1, c + 1]) * 0.25 > float(ISO)
            corners = (
                (ins[r, c], (c, r), h_id(r, c), v_id(r, c)),
                (ins[r, c + 1], (c + 1, r), h_id(r, c), v_id(r, c + 1)),
                (ins[r + 1, c + 1], (c + 1, r + 1), v_id(r, c + 1), h_id(r + 1, c)),
                (ins[r + 1, c], (c, r + 1), h_id(r + 1, c), v_id(r, c)),
            )
            for is_in, (qx, qy), ea, eb in corners:
                # Cut off each corner that differs from the centre's state.
                if bool(is_in) == centre_in:
                    continue
                ref = (qx, qy) if is_in else (c + 0.5, r + 0.5)
                f, t = orient(np.array([ea]), np.array([eb]), np.array([ref[0]]), np.array([ref[1]]))
                f_list.append(int(f[0]))
                t_list.append(int(t[0]))
        seg_from.append(np.array(f_list, dtype=np.int64))
        seg_to.append(np.array(t_list, dtype=np.int64))

    if not seg_from:
        return []
    src = np.concatenate(seg_from).astype(np.int64)
    dst = np.concatenate(seg_to).astype(np.int64)

    # Compact edge ids -> 0..K-1 so the successor table is outline-sized.
    uniq = np.unique(np.concatenate([src, dst]))
    s_idx = np.searchsorted(uniq, src)
    nxt = np.full(uniq.size, -1, dtype=np.int64)
    nxt[s_idx] = np.searchsorted(uniq, dst)
    ux, uy = edge_point(uniq)

    loops: list[np.ndarray] = []
    visited = np.zeros(uniq.size, dtype=bool)
    nxt_list = nxt.tolist()
    for start in s_idx.tolist():
        if visited[start]:
            continue
        ids = []
        cur = start
        while cur >= 0 and not visited[cur]:  # cur < 0 cannot happen on a padded field
            visited[cur] = True
            ids.append(cur)
            cur = nxt_list[cur]
        if len(ids) >= 3:
            idx = np.asarray(ids, dtype=np.int64)
            loops.append(np.stack([ux[idx], uy[idx]], axis=1))
    return loops


def _signed_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def extract_outline(
    L_clean: np.ndarray,
    M_sil: np.ndarray,
    physical_width_mm: float,
    tolerance_mm: float = 0.05,
    min_area_mm2: float | None = None,
    sigmoid_k: float | None = None,
) -> list[OutlineLoop]:
    """
    Trace the silhouette outline with sub-pixel accuracy, in canvas millimetres (y down).

    `tolerance_mm` is the Douglas-Peucker simplification tolerance (0 keeps every vertex).
    Loops smaller than `min_area_mm2` (default: 4 square pixels) are dropped as specks.
    Pass `sigmoid_k` (preprocessor.HD_SIGMOID_K) when L_clean came from HD restore.
    """
    if L_clean.shape != M_sil.shape:
        raise ValueError(f"L_clean {L_clean.shape} and M_sil {M_sil.shape} must match")
    H, W = int(M_sil.shape[0]), int(M_sil.shape[1])
    px_mm = max(float(physical_width_mm), 1e-6) / float(max(W, 1))
    min_area_px = (4.0 if min_area_mm2 is None else max(0.0, float(min_area_mm2)) / (px_mm * px_mm))
    tol_px = max(0.0, float(tolerance_mm)) / px_mm

    out: list[OutlineLoop] = []
    for loop in _trace_loops(_outline_field(L_clean, M_sil, sigmoid_k)):
        # Padded index -> original pixel index (pixel centres at integer coordinates).
        pts = loop - 1.0
        area = _signed_area(pts)  # y-down axes: inside-on-right outer loops have area > 0
        if abs(area) < min_area_px:
            continue
        if tol_px > 0.0 and pts.shape[0] > 3:
            simplified = cv2.approxPolyDP(pts.astype(np.float32).reshape(-1, 1, 2), tol_px, True)
            simplified = simplified.reshape(-1, 2).astype(np.float64)
            if simplified.shape[0] >= 3:
                pts = simplified
        # Traced loops keep the inside on their right (visually clockwise around material).
        # Reverse them so outer boundaries run counter-clockwise and holes clockwise, the
        # usual CAD/CAM convention (orientation is visual, so it survives the DXF y-flip).
        mm = (pts[::-1] + 0.5) * px_mm
        out.append(OutlineLoop(points=mm, is_hole=area < 0.0))
    return out


def _canvas_size_mm(shape: tuple[int, int], physical_width_mm: float) -> tuple[float, float]:
    H, W = int(shape[0]), int(shape[1])
    w_mm = float(physical_width_mm)
    return w_mm, w_mm * float(H) / float(max(W, 1))


def encode_svg(loops: list[OutlineLoop], shape: tuple[int, int], physical_width_mm: float) -> bytes:
    """SVG in true millimetres (width/height in mm, viewBox in mm, y down like the drawing)."""
    w_mm, h_mm = _canvas_size_mm(shape, physical_width_mm)

    def path(loop: OutlineLoop) -> str:
        p = loop.points
        coords = " L ".join(f"{x:.4f} {y:.4f}" for x, y in p)
        return f'    <path d="M {coords} Z"/>'

    outer = [path(lp) for lp in loops if not lp.is_hole]
    holes = [path(lp) for lp in loops if lp.is_hole]
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
        f'width="{w_mm:.4f}mm" height="{h_mm:.4f}mm" viewBox="0 0 {w_mm:.4f} {h_mm:.4f}">',
        "  <title>MONOLITH silhouette outline (mm)</title>",
        '  <g id="outline_outer" fill="none" stroke="#000000" stroke-width="0.1">',
        *outer,
        "  </g>",
        '  <g id="outline_holes" fill="none" stroke="#d00000" stroke-width="0.1">',
        *holes,
        "  </g>",
        "</svg>",
        "",
    ]
    return "\n".join(parts).encode("utf-8")


def encode_dxf(loops: list[OutlineLoop], shape: tuple[int, int], physical_width_mm: float) -> bytes:
    """
    ASCII DXF R12 (AC1009) with closed POLYLINE entities in millimetres, CAD axes (y up,
    origin at the canvas' bottom-left). Outer boundaries on layer OUTLINE_OUTER (CCW),
    holes on OUTLINE_HOLES (CW) for separate outside / inside profile toolpaths.
    """
    w_mm, h_mm = _canvas_size_mm(shape, physical_width_mm)
    lines: list[str] = []

    def g(code: int, value: object) -> None:
        lines.append(str(code))
        lines.append(str(value))

    g(0, "SECTION"); g(2, "HEADER")
    g(9, "$ACADVER"); g(1, "AC1009")
    g(9, "$INSUNITS"); g(70, 4)  # millimetres
    g(9, "$EXTMIN"); g(10, "0.0"); g(20, "0.0"); g(30, "0.0")
    g(9, "$EXTMAX"); g(10, f"{w_mm:.4f}"); g(20, f"{h_mm:.4f}"); g(30, "0.0")
    g(0, "ENDSEC")

    g(0, "SECTION"); g(2, "TABLES")
    g(0, "TABLE"); g(2, "LTYPE"); g(70, 1)
    g(0, "LTYPE"); g(2, "CONTINUOUS"); g(70, 0); g(3, "Solid line"); g(72, 65); g(73, 0); g(40, "0.0")
    g(0, "ENDTAB")
    g(0, "TABLE"); g(2, "LAYER"); g(70, 2)
    for name, colour in (("OUTLINE_OUTER", 7), ("OUTLINE_HOLES", 1)):
        g(0, "LAYER"); g(2, name); g(70, 0); g(62, colour); g(6, "CONTINUOUS")
    g(0, "ENDTAB")
    g(0, "ENDSEC")

    g(0, "SECTION"); g(2, "ENTITIES")
    for loop in loops:
        layer = "OUTLINE_HOLES" if loop.is_hole else "OUTLINE_OUTER"
        g(0, "POLYLINE"); g(8, layer); g(66, 1)
        g(10, "0.0"); g(20, "0.0"); g(30, "0.0")
        g(70, 1)  # closed
        # Flip to CAD axes (y up); visual orientation is unchanged (outer CCW, holes CW).
        for x, y in loop.points:
            g(0, "VERTEX"); g(8, layer)
            g(10, f"{x:.4f}"); g(20, f"{h_mm - y:.4f}"); g(30, "0.0")
        g(0, "SEQEND"); g(8, layer)
    g(0, "ENDSEC")
    g(0, "EOF")
    return ("\n".join(lines) + "\n").encode("ascii")
