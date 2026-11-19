"""Stage 4: Vectorized watertight binary STL, indexed OBJ, 8/16-bit PNG, 16-bit TIFF, and 32-bit OpenEXR exporters."""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass

import cv2
import numpy as np

_STL_RECORD_DTYPE = np.dtype(
    [
        ("normal", "<f4", 3),
        ("v0", "<f4", 3),
        ("v1", "<f4", 3),
        ("v2", "<f4", 3),
        ("attr", "<u2"),
    ]
)


def canvas_axis_mm(n_samples: int, length_mm: float) -> np.ndarray:
    """Sample centres `(j + 0.5) * length / n` plus the two canvas edges: `n + 2` values `[0, ..., length]`."""
    n = int(n_samples)
    axis = np.empty(n + 2, dtype=np.float64)
    axis[0] = 0.0
    axis[1:-1] = (np.arange(n, dtype=np.float64) + 0.5) * (float(length_mm) / n)
    axis[-1] = float(length_mm)
    return axis


def _prepare_mesh_grid(
    H: np.ndarray,
    width_mm: float,
    relief_z_mm: float,
    base_z_mm: float,
    max_grid: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Downsample heightmap cleanly and return X, Y, Z_top, Z_bot grids plus optional cutout cell mask.

    Coordinates follow the pixel-centre convention shared with the PNG / DXF exporters: sample j of
    a grid with `cols` columns sits at x = (j + 0.5) * width_mm / cols, rows run top to bottom
    from y = height_mm (y up), and height_mm comes from the FULL-RES shape.

    When base_z_mm > 1e-4, returns cell_mask=None for fast full-grid slab export. The grid gains a
    1-sample edge-replicated ring on the canvas edges, so the plaque is exactly width x height.
    When base_z_mm <= 1e-4, computes a manifold silhouette cell_mask (a cell is active iff the mean
    of its 4 corner foreground coverages is >= 0.5) with 2x2 diagonal checkerboard singularities
    repaired and enforces Z_top >= 0.25 mm on active vertices; X / Y are the sample centres only.
    """
    h_arr = np.clip(np.asarray(H, dtype=np.float32), np.float32(0.0), np.float32(1.0))
    if h_arr.ndim != 2:
        raise ValueError(f"Expected 2D heightmap for mesh export, got shape {h_arr.shape}")
    rows_full, cols_full = int(h_arr.shape[0]), int(h_arr.shape[1])

    cutout = not float(base_z_mm) > 1e-4
    fg_cov = (h_arr > np.float32(1e-4)).astype(np.float32) if cutout else None
    if max_grid > 0 and max(rows_full, cols_full) > max_grid:
        scale = float(max_grid) / float(max(rows_full, cols_full))
        new_cols = max(2, int(round(cols_full * scale)))
        new_rows = max(2, int(round(rows_full * scale)))
        h_grid = cv2.resize(h_arr, (new_cols, new_rows), interpolation=cv2.INTER_AREA).astype(np.float32, copy=False)
        np.clip(h_grid, np.float32(0.0), np.float32(1.0), out=h_grid)
        if cutout:
            fg_cov = cv2.resize(fg_cov, (new_cols, new_rows), interpolation=cv2.INTER_AREA)
    else:
        h_grid = h_arr

    rows, cols = int(h_grid.shape[0]), int(h_grid.shape[1])
    if rows < 2 or cols < 2:
        raise ValueError("Heightmap must be at least 2x2 to build a watertight mesh")

    w_mm = max(float(width_mm), 1e-3)
    h_mm = w_mm * (float(rows_full) / float(cols_full))
    x_axis = canvas_axis_mm(cols, w_mm)
    y_axis = h_mm - canvas_axis_mm(rows, h_mm)  # row 0 (top) is at the height_mm side

    if not cutout:
        h_pad = np.pad(h_grid, 1, mode="edge")
        X, Y = np.meshgrid(x_axis.astype(np.float32), y_axis.astype(np.float32))
        Z_bot = np.zeros(h_pad.shape, dtype=np.float32)
        Z_top = (h_pad * np.float32(relief_z_mm) + np.float32(base_z_mm)).astype(np.float32, copy=False)
        return X, Y, Z_top, Z_bot, None

    X, Y = np.meshgrid(x_axis[1:-1].astype(np.float32), y_axis[1:-1].astype(np.float32))
    Z_bot = np.zeros((rows, cols), dtype=np.float32)

    # Silhouette Cutout mode: a cell is foreground when the mean corner coverage is >= 0.5.
    cell_cov = 0.25 * (fg_cov[:-1, :-1] + fg_cov[1:, :-1] + fg_cov[:-1, 1:] + fg_cov[1:, 1:])
    cell_mask = cell_cov >= np.float32(0.5)
    if not np.any(cell_mask):
        cell_mask = np.ones((rows - 1, cols - 1), dtype=bool)
    else:
        # Repair 2x2 diagonal checkerboards so every active vertex has a single manifold umbrella.
        while True:
            c00 = cell_mask[:-1, :-1]
            c01 = cell_mask[:-1, 1:]
            c10 = cell_mask[1:, :-1]
            c11 = cell_mask[1:, 1:]
            diag = (c00 & c11 & ~c01 & ~c10) | (~c00 & ~c11 & c01 & c10)
            if not np.any(diag):
                break
            cell_mask[:-1, :-1] |= diag
            cell_mask[:-1, 1:] |= diag
            cell_mask[1:, :-1] |= diag
            cell_mask[1:, 1:] |= diag

    vert_active = np.zeros((rows, cols), dtype=bool)
    vert_active[:-1, :-1] |= cell_mask
    vert_active[1:, :-1] |= cell_mask
    vert_active[:-1, 1:] |= cell_mask
    vert_active[1:, 1:] |= cell_mask

    # Guarantee strictly positive wall height (>= 0.25 mm) on every active cutout vertex.
    Z_top = np.where(
        vert_active,
        np.maximum(h_grid * np.float32(relief_z_mm), np.float32(0.25)),
        np.float32(0.0),
    ).astype(np.float32, copy=False)
    return X, Y, Z_top, Z_bot, cell_mask


def _boundary_edge_masks(
    cell_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute North, South, West, and East boundary wall masks for active cutout cells."""
    north_mask = cell_mask & np.pad(~cell_mask[:-1, :], ((1, 0), (0, 0)), constant_values=True)
    south_mask = cell_mask & np.pad(~cell_mask[1:, :], ((0, 1), (0, 0)), constant_values=True)
    west_mask = cell_mask & np.pad(~cell_mask[:, :-1], ((0, 0), (1, 0)), constant_values=True)
    east_mask = cell_mask & np.pad(~cell_mask[:, 1:], ((0, 0), (0, 1)), constant_values=True)
    return north_mask, south_mask, west_mask, east_mask


def _perimeter_loop(grid: np.ndarray) -> np.ndarray:
    """Grid border entries as a closed loop, clockwise seen from +Z (row 0 is max Y)."""
    north = grid[0, :-1]  # +X along the top edge
    east = grid[:-1, -1]  # down the right edge
    south = grid[-1, :0:-1]  # -X along the bottom edge
    west = grid[:0:-1, 0]  # up the left edge
    return np.concatenate([north, east, south, west], axis=0)


def _assemble_mesh_triangles(
    top_grid: np.ndarray,
    bot_grid: np.ndarray,
    cell_mask: np.ndarray | None,
    bot_center: np.ndarray | int | None = None,
) -> np.ndarray:
    """
    Assemble outward-wound top, bottom, and boundary wall triangle batches.

    Works uniformly on (rows, cols, 3) coordinate arrays (STL) and (rows, cols)
    1-based index arrays (OBJ). For the full slab (`cell_mask is None`) with a
    `bot_center` (coordinate or index), the flat bottom is a fan from that centre over
    the shared perimeter (~2(rows+cols) triangles instead of 2(rows-1)(cols-1)),
    roughly halving file size while staying watertight.
    """
    tail_shape = top_grid.shape[2:]
    t00 = top_grid[:-1, :-1].reshape((-1, *tail_shape))
    t10 = top_grid[1:, :-1].reshape((-1, *tail_shape))
    t01 = top_grid[:-1, 1:].reshape((-1, *tail_shape))
    t11 = top_grid[1:, 1:].reshape((-1, *tail_shape))

    b00 = bot_grid[:-1, :-1].reshape((-1, *tail_shape))
    b10 = bot_grid[1:, :-1].reshape((-1, *tail_shape))
    b01 = bot_grid[:-1, 1:].reshape((-1, *tail_shape))
    b11 = bot_grid[1:, 1:].reshape((-1, *tail_shape))

    if cell_mask is None:
        tris_top_1 = np.stack([t00, t10, t01], axis=1)
        tris_top_2 = np.stack([t10, t11, t01], axis=1)
        if bot_center is not None:
            loop = _perimeter_loop(bot_grid)
            nxt = np.roll(loop, -1, axis=0)
            centre = np.broadcast_to(np.asarray(bot_center, dtype=loop.dtype), loop.shape)
            # Clockwise perimeter seen from +Z -> (centre, a, b) faces -Z (outward).
            tris_bot_1 = np.stack([centre, loop, nxt], axis=1)
            tris_bot_2 = tris_bot_1[:0]
        else:
            tris_bot_1 = np.stack([b00, b01, b10], axis=1)
            tris_bot_2 = np.stack([b10, b01, b11], axis=1)

        # Perimeter walls: North (+Y), South (-Y), West (-X), East (+X).
        wn_1 = np.stack([top_grid[0, :-1], top_grid[0, 1:], bot_grid[0, :-1]], axis=1)
        wn_2 = np.stack([top_grid[0, 1:], bot_grid[0, 1:], bot_grid[0, :-1]], axis=1)
        ws_1 = np.stack([top_grid[-1, 1:], top_grid[-1, :-1], bot_grid[-1, :-1]], axis=1)
        ws_2 = np.stack([top_grid[-1, 1:], bot_grid[-1, :-1], bot_grid[-1, 1:]], axis=1)
        ww_1 = np.stack([top_grid[1:, 0], top_grid[:-1, 0], bot_grid[:-1, 0]], axis=1)
        ww_2 = np.stack([top_grid[1:, 0], bot_grid[:-1, 0], bot_grid[1:, 0]], axis=1)
        we_1 = np.stack([top_grid[:-1, -1], top_grid[1:, -1], bot_grid[:-1, -1]], axis=1)
        we_2 = np.stack([top_grid[1:, -1], bot_grid[1:, -1], bot_grid[:-1, -1]], axis=1)
    else:
        cm = cell_mask.ravel()
        north_mask, south_mask, west_mask, east_mask = _boundary_edge_masks(cell_mask)
        nm = north_mask.ravel()
        sm = south_mask.ravel()
        wm = west_mask.ravel()
        em = east_mask.ravel()

        tris_top_1 = np.stack([t00[cm], t10[cm], t01[cm]], axis=1)
        tris_top_2 = np.stack([t10[cm], t11[cm], t01[cm]], axis=1)
        tris_bot_1 = np.stack([b00[cm], b01[cm], b10[cm]], axis=1)
        tris_bot_2 = np.stack([b10[cm], b01[cm], b11[cm]], axis=1)

        # Cutout boundary walls with identical outward winding:
        # North (+Y): (r,c)-(r,c+1); South (-Y): (r+1,c)-(r+1,c+1);
        # West (-X): (r,c)-(r+1,c); East (+X): (r,c+1)-(r+1,c+1).
        wn_1 = np.stack([t00[nm], t01[nm], b00[nm]], axis=1)
        wn_2 = np.stack([t01[nm], b01[nm], b00[nm]], axis=1)
        ws_1 = np.stack([t11[sm], t10[sm], b10[sm]], axis=1)
        ws_2 = np.stack([t11[sm], b10[sm], b11[sm]], axis=1)
        ww_1 = np.stack([t10[wm], t00[wm], b00[wm]], axis=1)
        ww_2 = np.stack([t10[wm], b00[wm], b10[wm]], axis=1)
        we_1 = np.stack([t01[em], t11[em], b01[em]], axis=1)
        we_2 = np.stack([t11[em], b11[em], b01[em]], axis=1)

    return np.concatenate(
        [
            tris_top_1,
            tris_top_2,
            tris_bot_1,
            tris_bot_2,
            wn_1,
            wn_2,
            ws_1,
            ws_2,
            ww_1,
            ww_2,
            we_1,
            we_2,
        ],
        axis=0,
    )


@dataclass
class MeshData:
    """Indexed triangle mesh in mm: 0-based faces wound counter-clockwise seen from outside."""

    vertices: np.ndarray  # (V, 3) float64, mm
    faces: np.ndarray  # (F, 3) int64, 0-based
    max_error_mm: float | None = None  # measured max vertical error (adaptive mesh only)


def encode_stl(mesh: MeshData) -> bytes:
    """Binary STL (float32 records) of an indexed mesh."""
    tris = np.asarray(mesh.vertices, dtype=np.float32)[np.asarray(mesh.faces, dtype=np.int64)]

    # Vectorized face normals.
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    normals = np.cross(e1, e2)
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, np.maximum(lens, np.float32(1e-12)), dtype=np.float32)

    n_tris = int(tris.shape[0])
    records = np.empty(n_tris, dtype=_STL_RECORD_DTYPE)
    records["normal"] = normals
    records["v0"] = tris[:, 0]
    records["v1"] = tris[:, 1]
    records["v2"] = tris[:, 2]
    records["attr"] = 0

    header = b"MONOLITH Relief Studio - Watertight Float32 Bas-Relief STL".ljust(80, b" ")[:80]
    count_bytes = struct.pack("<I", n_tris)
    return header + count_bytes + records.tobytes()


def encode_obj(mesh: MeshData) -> bytes:
    """Indexed Wavefront OBJ (1-based faces, 0.1 µm vertex precision) of an indexed mesh."""
    buf = io.BytesIO()
    buf.write(b"# MONOLITH Relief Studio - Watertight Wavefront OBJ\n")
    buf.write(b"o monolith_relief\n")
    np.savetxt(buf, np.asarray(mesh.vertices, dtype=np.float64), fmt="v %.4f %.4f %.4f")
    np.savetxt(buf, np.asarray(mesh.faces, dtype=np.int64) + 1, fmt="f %d %d %d")
    return buf.getvalue()


def grid_heightfield_mesh(
    H: np.ndarray,
    width_mm: float = 1000.0,
    relief_z_mm: float = 15.0,
    base_z_mm: float = 5.0,
    max_grid: int = 512,
) -> MeshData:
    """
    Regular-grid watertight mesh of a 2D float32 heightmap H in [0, 1] (area-downsampled to max_grid).

    When base_z_mm > 1e-4, emits a full-grid rectangular backing slab with Z_bot = 0.0.
    When base_z_mm <= 1e-4, emits a watertight silhouette-cutout mesh restricted to
    active foreground cells. Zero Python loops are used over pixels or edges; all
    triangles are built via vectorized NumPy.
    """
    X, Y, Z_top, Z_bot, cell_mask = _prepare_mesh_grid(H, width_mm, relief_z_mm, base_z_mm, max_grid)
    rows, cols = int(X.shape[0]), int(X.shape[1])

    V_top = np.stack([X, Y, Z_top], axis=-1).reshape(-1, 3)
    V_bot = np.stack([X, Y, Z_bot], axis=-1).reshape(-1, 3)
    centre = np.array([[0.5 * float(X[0, -1]), 0.5 * float(Y[0, 0]), 0.0]], dtype=np.float32)
    V_all = np.concatenate([V_top, V_bot, centre], axis=0)

    # Vertex index grids for top and bottom (+ the bottom fan centre).
    top_idx = np.arange(rows * cols, dtype=np.int64).reshape(rows, cols)
    bot_idx = top_idx + np.int64(rows * cols)
    centre_idx = 2 * rows * cols

    F_all = _assemble_mesh_triangles(top_idx, bot_idx, cell_mask, bot_center=centre_idx).astype(
        np.int64, copy=False
    )
    # Strip unreferenced vertices (background in cutout mode, interior bottom grid in slab mode).
    # O(N) mask + cumsum remap (np.unique would sort ~25M indices at a 2048 grid).
    used = np.zeros(V_all.shape[0], dtype=bool)
    used[F_all.ravel()] = True
    remap = np.cumsum(used, dtype=np.int64) - 1  # old index -> new index
    return MeshData(vertices=V_all[used].astype(np.float64), faces=remap[F_all], max_error_mm=None)


def heightmap_to_watertight_stl(
    H: np.ndarray,
    width_mm: float = 1000.0,
    relief_z_mm: float = 15.0,
    base_z_mm: float = 5.0,
    max_grid: int = 512,
) -> bytes:
    """Regular-grid watertight binary STL (see `grid_heightfield_mesh`)."""
    return encode_stl(grid_heightfield_mesh(H, width_mm, relief_z_mm, base_z_mm, max_grid))


def heightmap_to_obj(
    H: np.ndarray,
    width_mm: float = 1000.0,
    relief_z_mm: float = 15.0,
    base_z_mm: float = 5.0,
    max_grid: int = 512,
) -> bytes:
    """Regular-grid watertight indexed Wavefront OBJ (see `grid_heightfield_mesh`)."""
    return encode_obj(grid_heightfield_mesh(H, width_mm, relief_z_mm, base_z_mm, max_grid))


def encode_png8(H: np.ndarray) -> bytes:
    """Encode float32 [0, 1] heightmap to 8-bit grayscale PNG bytes (0..255)."""
    h_f32 = np.clip(np.asarray(H, dtype=np.float32), np.float32(0.0), np.float32(1.0))
    img_u8 = np.round(h_f32 * np.float32(255.0)).astype(np.uint8)
    ok, buf = cv2.imencode(".png", img_u8)
    if not ok:
        raise RuntimeError("Failed to encode 8-bit PNG")
    return buf.tobytes()


def encode_png16(H: np.ndarray) -> bytes:
    """Encode float32 [0, 1] heightmap to lossless 16-bit grayscale PNG bytes (0..65535)."""
    h_f32 = np.clip(np.asarray(H, dtype=np.float32), np.float32(0.0), np.float32(1.0))
    img_u16 = np.round(h_f32 * np.float32(65535.0)).astype(np.uint16)
    ok, buf = cv2.imencode(".png", img_u16)
    if not ok:
        raise RuntimeError("Failed to encode 16-bit PNG")
    return buf.tobytes()


def encode_tiff16(H: np.ndarray) -> bytes:
    """Encode float32 [0, 1] heightmap to lossless 16-bit grayscale TIFF bytes (0..65535)."""
    h_f32 = np.clip(np.asarray(H, dtype=np.float32), np.float32(0.0), np.float32(1.0))
    img_u16 = np.round(h_f32 * np.float32(65535.0)).astype(np.uint16)
    ok, buf = cv2.imencode(".tiff", img_u16)
    if not ok:
        raise RuntimeError("Failed to encode 16-bit TIFF")
    return buf.tobytes()


def _exr_attr(name: str, type_name: str, payload: bytes) -> bytes:
    return (
        name.encode("ascii")
        + b"\x00"
        + type_name.encode("ascii")
        + b"\x00"
        + struct.pack("<i", len(payload))
        + payload
    )


def encode_exr32(H: np.ndarray) -> bytes:
    """
    Encode float32 heightmap to uncompressed single-part scanline OpenEXR (32-bit float 'Y' channel).

    Implemented in pure Python + vectorized NumPy so EXR export works cleanly
    even when system libOpenEXR is not installed.
    """
    h_f32 = np.ascontiguousarray(np.asarray(H, dtype="<f4"))
    if h_f32.ndim != 2:
        raise ValueError(f"Expected 2D heightmap for EXR export, got shape {h_f32.shape}")

    rows, cols = int(h_f32.shape[0]), int(h_f32.shape[1])

    # OpenEXR 2.0 single-part scanline magic + version 2.
    magic_version = struct.pack("<II", 20000630, 2)

    # Channel list: single 'Y' channel, pixel_type=2 (FLOAT), pLinear=0, sampling=(1,1).
    chlist_payload = b"Y\x00" + struct.pack("<iB3xii", 2, 0, 1, 1) + b"\x00"
    box2i_payload = struct.pack("<iiii", 0, 0, cols - 1, rows - 1)

    header_parts = [
        _exr_attr("channels", "chlist", chlist_payload),
        _exr_attr("compression", "compression", struct.pack("<B", 0)),  # NO_COMPRESSION
        _exr_attr("dataWindow", "box2i", box2i_payload),
        _exr_attr("displayWindow", "box2i", box2i_payload),
        _exr_attr("lineOrder", "lineOrder", struct.pack("<B", 0)),  # INCREASING_Y
        _exr_attr("pixelAspectRatio", "float", struct.pack("<f", 1.0)),
        _exr_attr("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0)),
        _exr_attr("screenWindowWidth", "float", struct.pack("<f", 1.0)),
        b"\x00",
    ]
    header_bytes = magic_version + b"".join(header_parts)

    # Offset table (uint64 per scanline) + vectorized scanline block packing.
    offset_table_size = 8 * rows
    first_scanline_offset = len(header_bytes) + offset_table_size
    scanline_pixel_bytes = 4 * cols
    scanline_block_bytes = 8 + scanline_pixel_bytes

    offsets = (
        np.uint64(first_scanline_offset)
        + np.arange(rows, dtype=np.uint64) * np.uint64(scanline_block_bytes)
    ).astype("<u8")

    scanline_dtype = np.dtype([("y", "<i4"), ("size", "<i4"), ("pixels", "<f4", cols)])
    blocks = np.empty(rows, dtype=scanline_dtype)
    blocks["y"] = np.arange(rows, dtype=np.int32)
    blocks["size"] = scanline_pixel_bytes
    blocks["pixels"] = h_f32

    return header_bytes + offsets.tobytes() + blocks.tobytes()
