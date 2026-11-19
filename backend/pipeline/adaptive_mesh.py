"""
Adaptive, error-bounded watertight heightfield mesh: a right-triangulated irregular network
(RTIN, after Evans et al. and Mapbox "Martini"), fully vectorized per level.

The canvas is the heightmap plus a 1-sample edge-replicated ring placed on the plaque edges
(`canvas_axis_mm`), so the plaque is exactly width x height. Lattice point (x = col, y = row)
of that canvas sits at (x_axis[x], y_axis[y]) mm. Inside the ring the map is affine with one
pitch p on both axes, so interpolation in lattice space equals interpolation in mm; the ring
itself is only p / 2 wide. Every midpoint within two samples of the canvas edge is therefore
forced to split: the triangles touching the ring are unit half-cells and no emitted triangle
crosses the non-uniform ring, which keeps the measured lattice-space error the true error in mm.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from .stl_exporter import MeshData, canvas_axis_mm

_log = logging.getLogger(__name__)

MAX_REFINE_ROUNDS = 4  # re-extractions with a tightened tolerance if the measured error exceeds it
DEFAULT_MAX_MESH_TRIANGLES = 15_000_000  # override with env MONOLITH_MAX_MESH_TRIANGLES
_MEASURE_CHUNK = 1 << 20  # samples gathered per vectorized measurement pass


class MeshTooLargeError(ValueError):
    """The adaptive mesh would exceed the triangle cap (raised before anything that large is built)."""

    def __init__(self, triangles: int, cap: int) -> None:
        self.triangles = int(triangles)
        self.cap = int(cap)
        super().__init__(
            f"Adaptive mesh needs at least {self.triangles:,} surface triangles (cap {self.cap:,}); "
            "use a larger mesh_tolerance_mm or mesh_mode 'grid'"
        )


def _parse_triangle_cap(raw: str | None) -> int:
    """Parse the env value; malformed or non-positive values fall back to the default (never raises)."""
    if raw is None:
        return DEFAULT_MAX_MESH_TRIANGLES
    try:
        value = int(raw)
    except (TypeError, ValueError):
        _log.warning("Ignoring invalid MONOLITH_MAX_MESH_TRIANGLES=%r; using %d", raw, DEFAULT_MAX_MESH_TRIANGLES)
        return DEFAULT_MAX_MESH_TRIANGLES
    if value <= 0:
        _log.warning("Ignoring non-positive MONOLITH_MAX_MESH_TRIANGLES=%r; using %d", raw, DEFAULT_MAX_MESH_TRIANGLES)
        return DEFAULT_MAX_MESH_TRIANGLES
    return value


def max_mesh_triangles() -> int:
    """Triangle cap for the adaptive top surface (env MONOLITH_MAX_MESH_TRIANGLES).

    Read per call (tests and operators may change the env), but parsed defensively: a bad value can
    never turn an export into a 500.
    """
    return _parse_triangle_cap(os.environ.get("MONOLITH_MAX_MESH_TRIANGLES"))


def _outer(A: np.ndarray, ys: np.ndarray, xs: np.ndarray, n: int) -> np.ndarray:
    """
    Outer-product gather A[ys][:, xs] (float32).

    Lattice points beyond A but inside the n x n RTIN grid read +inf (they lie off the canvas);
    points outside the RTIN grid do not exist and read 0 (neutral for a max of errors).
    """
    rows, cols = A.shape
    in_grid = ((ys >= 0) & (ys < n))[:, None] & ((xs >= 0) & (xs < n))[None, :]
    out = np.where(in_grid, np.float32(np.inf), np.float32(0.0)).astype(np.float32, copy=False)
    vy = (ys >= 0) & (ys < rows)
    vx = (xs >= 0) & (xs < cols)
    if vy.any() and vx.any():
        out[np.ix_(vy, vx)] = A[np.ix_(ys[vy], xs[vx])]
    return out


def _error_map(G: np.ndarray, n: int) -> np.ndarray:
    """
    RTIN error map E over the canvas G (heights), lattice-space, bottom-up by level.

    E at a hypotenuse midpoint is the interpolation error there, maxed with E at the midpoints
    of both sharing triangles' children, so a split always forces its neighbours to split too
    (crack-free). Midpoints within two samples of the canvas edge, and anything off the canvas,
    are +inf (always split).
    """
    rows, cols = G.shape
    E = np.zeros((rows, cols), dtype=np.float32)
    inf = np.float32(np.inf)

    def forced(ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
        fy = (ys <= 1) | (ys >= rows - 2)
        fx = (xs <= 1) | (xs >= cols - 2)
        return fy[:, None] | fx[None, :]

    h = 1
    while h <= (n - 1) // 2:
        s = 2 * h
        # a. Axis-aligned hypotenuses of length 2h: horizontal, then vertical.
        for horizontal in (True, False):
            xs = np.arange(h if horizontal else 0, cols, s)
            ys = np.arange(0 if horizontal else h, rows, s)
            if xs.size == 0 or ys.size == 0:
                continue
            if horizontal:
                ends = _outer(G, ys, xs - h, n) + _outer(G, ys, xs + h, n)
            else:
                ends = _outer(G, ys - h, xs, n) + _outer(G, ys + h, xs, n)
            e = np.abs(G[np.ix_(ys, xs)] - np.float32(0.5) * ends)
            if h >= 2:
                q = h // 2
                for dy in (-q, q):
                    for dx in (-q, q):
                        np.maximum(e, _outer(E, ys + dy, xs + dx, n), out=e)
            e[forced(ys, xs)] = inf
            E[np.ix_(ys, xs)] = e

        # b. Square centres (squares of side 2h); the diagonal runs through the corner whose
        #    coordinates are both odd multiples of 2h.
        xs = np.arange(h, cols, s)
        ys = np.arange(h, rows, s)
        if xs.size and ys.size:
            px = np.where(((xs - h) // s) % 2 == 1, xs - h, xs + h)
            py = np.where(((ys - h) // s) % 2 == 1, ys - h, ys + h)
            ends = _outer(G, py, px, n) + _outer(G, 2 * ys - py, 2 * xs - px, n)
            e = np.abs(G[np.ix_(ys, xs)] - np.float32(0.5) * ends)
            for dy, dx in ((0, -h), (0, h), (-h, 0), (h, 0)):
                np.maximum(e, _outer(E, ys + dy, xs + dx, n), out=e)
            e[forced(ys, xs)] = inf
            E[np.ix_(ys, xs)] = e
        h *= 2
    return E


def _extract(E: np.ndarray, n: int, threshold: float, max_triangles: int) -> np.ndarray:
    """
    Crack-free RTIN extraction: (K, 6) int32 triangles (ax, ay, bx, by, cx, cy) on the canvas.

    (a, b) is the hypotenuse and c the right-angle vertex. A triangle splits at m = (a + b) / 2
    iff its legs are longer than one cell and E[m] > threshold; children are (c, a, m), (b, c, m).
    Triangles wholly off the canvas are dropped as soon as they appear (no descendant can land on it).
    Every on-canvas frontier triangle yields at least one emitted triangle, so emitted + frontier
    is a lower bound on the result: MeshTooLargeError is raised as soon as it passes max_triangles.
    """
    rows, cols = E.shape
    t = n - 1
    tri = np.array([[0, 0, t, t, t, 0], [t, t, 0, 0, 0, t]], dtype=np.int32)
    emitted = []
    n_emitted = 0
    while tri.shape[0]:
        ax, ay, bx, by, cx, cy = tri.T
        on_canvas = (np.minimum(np.minimum(ax, bx), cx) < cols - 1) & (np.minimum(np.minimum(ay, by), cy) < rows - 1)
        if not on_canvas.all():
            tri = tri[on_canvas]
            ax, ay, bx, by, cx, cy = tri.T
        if n_emitted + tri.shape[0] > max_triangles:
            raise MeshTooLargeError(n_emitted + tri.shape[0], max_triangles)
        mx = (ax + bx) >> 1
        my = (ay + by) >> 1
        err = np.full(tri.shape[0], np.inf, dtype=np.float32)
        inb = (mx < cols) & (my < rows)
        err[inb] = E[my[inb], mx[inb]]
        split = ((np.abs(ax - cx) + np.abs(ay - cy)) > 1) & (err > threshold)
        emitted.append(tri[~split])
        n_emitted += emitted[-1].shape[0]
        ax, ay, bx, by, cx, cy, mx, my = (v[split] for v in (ax, ay, bx, by, cx, cy, mx, my))
        tri = np.concatenate(
            [np.stack([cx, cy, ax, ay, mx, my], axis=1), np.stack([bx, by, cx, cy, mx, my], axis=1)], axis=0
        )
    tri = np.concatenate(emitted, axis=0)
    sx = tri[:, 0] + tri[:, 2] + tri[:, 4]
    sy = tri[:, 1] + tri[:, 3] + tri[:, 5]
    return tri[(sx <= 3 * (cols - 1)) & (sy <= 3 * (rows - 1))]


def _row_spans(bx: int, by: int, cx: int, cy: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Lattice points of the canonical triangle (0, B, C) row by row: (ys, first x, count) per row.

    Each barycentric numerator (det * weight) is a * x + b with integer a and per-row b; requiring
    all three >= 0 bounds x exactly in integer arithmetic.
    """
    det = bx * cy - by * cx
    s = 1 if det > 0 else -1
    ys = np.arange(min(0, by, cy), max(0, by, cy) + 1, dtype=np.int64)
    lo = np.full(ys.size, min(0, bx, cx), dtype=np.int64)
    hi = np.full(ys.size, max(0, bx, cx), dtype=np.int64)
    ok = np.ones(ys.size, dtype=bool)
    a1, b1 = s * cy, -s * cx * ys  # det * weight of b
    a2, b2 = -s * by, s * bx * ys  # det * weight of c
    for a, b in ((a1, b1), (a2, b2), (-a1 - a2, abs(det) - b1 - b2)):
        if a > 0:
            lo = np.maximum(lo, -(b // a))  # x >= ceil(-b / a)
        elif a < 0:
            hi = np.minimum(hi, b // -a)  # x <= floor(b / -a)
        else:
            ok &= b >= 0
    return ys, lo, np.where(ok, np.maximum(hi - lo + 1, 0), 0)


def _measure(tri: np.ndarray, G: np.ndarray) -> float:
    """
    Max |plane - G| over every canvas lattice point, in G units.

    Triangles are grouped by shape (b - a, c - a); per group the lattice offsets inside the
    canonical triangle (vertices excluded: their error is exactly 0) and their barycentric
    weights are generated row-block by row-block (at most _MEASURE_CHUNK offsets at a time) and
    applied to all triangles of the group at once, so the transient memory stays bounded.
    """
    a = tri[:, 0:2].astype(np.int64)
    db = tri[:, 2:4].astype(np.int64) - a
    dc = tri[:, 4:6].astype(np.int64) - a
    span = int(max(np.abs(db).max(), np.abs(dc).max())) if tri.shape[0] else 0
    base = 2 * span + 1
    key = (((db[:, 0] + span) * base + (db[:, 1] + span)) * base + (dc[:, 0] + span)) * base + (dc[:, 1] + span)
    _, first, inverse = np.unique(key, return_index=True, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    bounds = np.concatenate([[0], np.cumsum(np.bincount(inverse, minlength=first.size))])

    worst = 0.0
    for g in range(first.size):
        bx, by, cx, cy = (int(v) for v in (*db[first[g]], *dc[first[g]]))
        ys, lo, counts = _row_spans(bx, by, cx, cy)
        if int(counts.sum()) <= 3:  # only the three vertices
            continue
        s = 1 if bx * cy - by * cx > 0 else -1
        ad = float(abs(bx * cy - by * cx))

        members = order[bounds[g] : bounds[g + 1]]
        tx, ty = tri[members, 0], tri[members, 1]
        z = np.stack(
            [G[ty, tx], G[tri[members, 3], tri[members, 2]], G[tri[members, 5], tri[members, 4]]], axis=1
        ).astype(np.float64)  # (K, 3)
        cum = np.cumsum(counts)
        r0 = 0
        while r0 < ys.size:
            done = int(cum[r0 - 1]) if r0 else 0
            r1 = max(r0 + 1, int(np.searchsorted(cum, done + _MEASURE_CHUNK, side="right")))
            c = counts[r0:r1]
            p = int(c.sum())
            starts = np.cumsum(c) - c
            oy = np.repeat(ys[r0:r1], c)
            ox = np.repeat(lo[r0:r1], c) + (np.arange(p, dtype=np.int64) - np.repeat(starts, c))
            r0 = r1
            keep = ~(((ox == 0) & (oy == 0)) | ((ox == bx) & (oy == by)) | ((ox == cx) & (oy == cy)))
            ox, oy = ox[keep], oy[keep]
            if ox.size == 0:
                continue
            n1 = s * (ox * cy - oy * cx)
            n2 = s * (bx * oy - by * ox)
            w = np.stack([ad - n1 - n2, n1, n2], axis=0) / ad  # (3, P) barycentric weights of a, b, c
            k_step = max(1, _MEASURE_CHUNK // ox.size)
            for k0 in range(0, members.size, k_step):
                k1 = min(members.size, k0 + k_step)
                sample = G[ty[k0:k1, None] + oy[None, :], tx[k0:k1, None] + ox[None, :]]
                plane = z[k0:k1] @ w
                worst = max(worst, float(np.abs(plane - sample).max()))
    return worst


def _solidify(
    tri: np.ndarray,
    G: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    relief_z_mm: float,
    base_z_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Top surface + 2-triangle walls on every boundary edge + a bottom fan, all outward CCW."""
    rows, cols = G.shape
    ids = tri[:, [1, 3, 5]].astype(np.int64) * cols + tri[:, [0, 2, 4]]
    used = np.zeros(rows * cols, dtype=bool)
    used[ids.ravel()] = True
    remap = np.cumsum(used, dtype=np.int32)  # canvas-sized: int32 halves it at 8K (~230 MB)
    remap -= 1
    vy, vx = np.divmod(np.flatnonzero(used), cols)
    top_v = np.stack(
        [x_axis[vx], y_axis[vy], G[vy, vx].astype(np.float64) * relief_z_mm + base_z_mm], axis=1
    )
    faces = remap[ids].astype(np.int64)
    del remap, used

    # Counter-clockwise seen from +Z.
    p = top_v[faces][:, :, :2]
    area2 = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 2, 0] - p[:, 0, 0]) * (
        p[:, 1, 1] - p[:, 0, 1]
    )
    faces[area2 < 0] = faces[area2 < 0][:, [0, 2, 1]]

    # Boundary = undirected edges used once. They lie on the canvas rectangle, so only edges
    # joining two perimeter vertices are candidates (an interior chord between two perimeter
    # vertices is still seen twice and dropped).
    nv = top_v.shape[0]
    on_perim = (vx == 0) | (vx == cols - 1) | (vy == 0) | (vy == rows - 1)
    edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    edges = edges[on_perim[edges[:, 0]] & on_perim[edges[:, 1]]]
    key = np.minimum(edges[:, 0], edges[:, 1]) * nv + np.maximum(edges[:, 0], edges[:, 1])
    _, first, counts = np.unique(key, return_index=True, return_counts=True)
    u, v = edges[first[counts == 1]].T  # directed along the CCW top faces

    nb = u.size
    bot = np.full(nv, -1, dtype=np.int64)
    bot[u] = nv + np.arange(nb, dtype=np.int64)
    if np.any(bot[v] < 0):
        raise RuntimeError("Adaptive mesh boundary is not a single closed loop")
    ub, vb = bot[u], bot[v]
    centre_idx = nv + nb
    bot_v = top_v[u].copy()
    bot_v[:, 2] = 0.0
    centre = np.array([[0.5 * float(x_axis[-1]), 0.5 * float(y_axis[0]), 0.0]])

    walls_1 = np.stack([u, ub, v], axis=1)
    walls_2 = np.stack([v, ub, vb], axis=1)
    fan = np.stack([np.full(nb, centre_idx, dtype=np.int64), vb, ub], axis=1)
    vertices = np.concatenate([top_v, bot_v, centre], axis=0)
    return vertices, np.concatenate([faces, walls_1, walls_2, fan], axis=0)


def adaptive_heightfield_mesh(
    H: np.ndarray,
    width_mm: float,
    relief_z_mm: float,
    base_z_mm: float,
    tolerance_mm: float,
) -> MeshData:
    """
    Watertight slab mesh of heightmap H in [0, 1] whose measured max vertical error is
    <= tolerance_mm at every heightmap sample (see module docstring).

    Z = H * relief_z_mm + base_z_mm on top, a flat bottom at Z = 0 and four perimeter walls.
    If the measured error exceeds the tolerance, the mesh is re-extracted with a tightened
    threshold (at most MAX_REFINE_ROUNDS times), then once more at threshold 0 (exact), so the
    returned `max_error_mm` (the final measured value) is always <= tolerance_mm. Raises
    MeshTooLargeError, before building anything that large, if the top surface would need more
    than `max_mesh_triangles()` triangles.
    """
    if not float(base_z_mm) > 0.0:
        raise ValueError("Adaptive mesh needs a solid base (base_z_mm > 0); use the grid mesh for cut-outs")
    if not float(tolerance_mm) > 0.0 or not float(relief_z_mm) >= 0.0:
        raise ValueError("tolerance_mm must be > 0 and relief_z_mm >= 0")
    h_arr = np.clip(np.asarray(H, dtype=np.float32), np.float32(0.0), np.float32(1.0))
    if h_arr.ndim != 2 or min(h_arr.shape) < 1:
        raise ValueError(f"Expected a non-empty 2D heightmap, got shape {h_arr.shape}")
    rows_full, cols_full = h_arr.shape

    w_mm = float(width_mm)
    h_mm = w_mm * float(rows_full) / float(cols_full)
    x_axis = canvas_axis_mm(cols_full, w_mm)
    y_axis = h_mm - canvas_axis_mm(rows_full, h_mm)  # y up: row 0 at height_mm

    G = np.pad(h_arr, 1, mode="edge")  # canvas heights in H units; mm = G * relief + base
    rows, cols = G.shape
    n = 2 ** max(1, (max(rows, cols) - 2).bit_length()) + 1
    E = _error_map(G, n)

    relief = float(relief_z_mm)
    unit = max(relief, 1e-12)  # E is in H units; compare in mm
    tol = float(tolerance_mm)
    cap = max_mesh_triangles()
    threshold_mm = tol
    for _ in range(MAX_REFINE_ROUNDS + 1):
        tri = None  # release the previous extraction before building the next one
        tri = _extract(E, n, threshold_mm / unit, cap)
        max_err = _measure(tri, G) * relief
        if max_err <= tol:
            break
        threshold_mm *= 0.9 * tol / max_err
    else:
        # Not converged: split wherever the error map is non-zero. A triangle with E = 0 at its
        # midpoint has every lattice point on its plane, so this mesh is exact (up to float32
        # rounding of E) and still subject to the triangle cap.
        tri = None
        tri = _extract(E, n, 0.0, cap)
        E = None  # free the error map before the final measure
        max_err = _measure(tri, G) * relief
        if max_err > tol:
            raise RuntimeError(f"Adaptive mesh error {max_err:.6f} mm exceeds tolerance {tol} mm")
    E = None

    vertices, faces = _solidify(tri, G, x_axis, y_axis, relief, float(base_z_mm))
    return MeshData(vertices=vertices, faces=faces, max_error_mm=float(max_err))
