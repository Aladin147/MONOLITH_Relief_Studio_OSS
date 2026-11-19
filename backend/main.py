"""FastAPI backend for MONOLITH Relief Studio with two-stage caching and SQLite vault."""

from __future__ import annotations

import json
import math
import mimetypes
import os
import re
import shutil
import sys
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from backend import db
from backend.pipeline.outline_exporter import encode_dxf, encode_svg, extract_outline
from backend.pipeline.preprocessor import HD_SIGMOID_K, POLARITIES, preprocess_lineart
from backend.pipeline.relief_engine import (
    GROOVE_SIGMA_UNITS,
    compute_basis_layers,
    derive_m_lines,
    groove_scale_for_shape,
    recompute_relief,
)
from backend.pipeline import sculpt_cache, sculpt_overrides
from backend.pipeline.adaptive_mesh import MeshTooLargeError, adaptive_heightfield_mesh
from backend.pipeline.sculpt_engine import (
    FULLNESS_MIN,
    SCULPT_DEFAULTS,
    SCULPT_LATEST_VERSION,
    build_sculpt_basis,
    copy_for_reorder,
    rebuild_parts,
)
from backend.pipeline.sculpt_parts import PartsAnalysis, analyze_parts_from_source
from backend.pipeline.stl_exporter import (
    encode_exr32,
    encode_obj,
    encode_png8,
    encode_png16,
    encode_stl,
    encode_tiff16,
    grid_heightfield_mesh,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_IMAGE_PATH = PROJECT_ROOT / "benchmarks" / "geometric-study.png"
PREVIEW_MAX_DIM = 1024

LAYER_FILENAMES = (
    "M_sil",
    "d_sil",
    "H_global",
    "H_local",
    "G_lines",
    "H_ai_guided",
    "L_clean",
    "H_current",
)
# Layers both styles use, and the Classic-only ones. Only the active style's layers are held in
# RAM and on disk (design decision 5); the Sculpt basis lives in cache/sculpt_*.npy plus its
# manifest, and the parts analysis in cache/parts.npz (backend.pipeline.sculpt_cache).
COMMON_LAYER_FILENAMES = ("M_sil", "d_sil", "L_clean", "H_current")
CLASSIC_LAYER_KEYS = ("H_global", "H_local", "G_lines", "H_ai_guided")


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------


# Bounds are deliberately wider than the UI sliders: they reject nonsense (negative depth,
# zero width, NaN) and resource bombs, never a value a user could reasonably pick.
MAX_PHYSICAL_MM = 20000.0
MAX_MESH_RESOLUTION = int(os.environ.get("MONOLITH_MAX_MESH_RES", "2048"))


class FrameSettings(BaseModel):
    enabled: bool = False
    preset: str = "gallery_classic"
    width_mm: float = Field(default=22.0, ge=0.0, le=MAX_PHYSICAL_MM, allow_inf_nan=False)
    thickness_mm: float = Field(default=5.5, ge=0.0, le=MAX_PHYSICAL_MM, allow_inf_nan=False)
    back_protrusion_mm: float = Field(default=0.0, ge=0.0, le=MAX_PHYSICAL_MM, allow_inf_nan=False)
    bbox: list[float] = Field(default_factory=lambda: [0.16, 0.48, 0.84, 0.96], min_length=4, max_length=4)
    units: str = "mm"

    @field_validator("bbox")
    @classmethod
    def _bbox_in_unit_square(cls, v: list[float]) -> list[float]:
        if not all(math.isfinite(x) and -1e-6 <= x <= 1.0 + 1e-6 for x in v):
            raise ValueError("bbox values must be finite UV coordinates in [0, 1]")
        return [min(1.0, max(0.0, float(x))) for x in v]


class ReliefParams(BaseModel):
    w_global: float = Field(default=0.45, ge=0.0, le=5.0, allow_inf_nan=False)
    w_local: float = Field(default=0.65, ge=0.0, le=5.0, allow_inf_nan=False)
    w_ai: float = Field(default=0.25, ge=0.0, le=5.0, allow_inf_nan=False)
    w_groove: float = Field(default=0.35, ge=0.0, le=5.0, allow_inf_nan=False)
    # Pixels at a 4096 px longest edge (resolution independent; see relief_engine).
    groove_sigma: float = Field(default=2.5, ge=0.0, le=64.0, allow_inf_nan=False)
    groove_sigma_units: str = GROOVE_SIGMA_UNITS
    gamma: float = Field(default=0.85, ge=0.05, le=5.0, allow_inf_nan=False)
    brightness: float = Field(default=0.0, ge=-1.0, le=1.0, allow_inf_nan=False)
    contrast: float = Field(default=1.0, ge=0.0, le=5.0, allow_inf_nan=False)
    h_min: float = Field(default=0.08, ge=0.0, le=0.95, allow_inf_nan=False)
    invert_depth: bool = False
    draft_angle_deg: float = Field(default=2.0, ge=0.0, le=60.0, allow_inf_nan=False)
    physical_width_mm: float = Field(default=1000.0, ge=1.0, le=MAX_PHYSICAL_MM, allow_inf_nan=False)
    relief_z_mm: float = Field(default=15.0, gt=0.0, le=MAX_PHYSICAL_MM, allow_inf_nan=False)
    base_z_mm: float = Field(default=5.0, ge=0.0, le=MAX_PHYSICAL_MM, allow_inf_nan=False)
    frame: FrameSettings = Field(default_factory=FrameSettings)
    # Relief style. Stored projects and presets without it load as "classic" (db); new ones
    # default to "sculpt". The Sculpt sliders below are ignored by Classic and vice versa.
    style: Literal["sculpt", "classic"] = "sculpt"
    # Sculpt algorithm version; a recompute refuses one newer than SCULPT_LATEST_VERSION (409).
    sculpt_version: int = Field(default=SCULPT_LATEST_VERSION, ge=1)
    # Defaults come from the engine (sculpt_engine.SCULPT_DEFAULTS), so the two cannot drift.
    # Edge roundness and crease width are % of the shorter canvas side.
    edge_roundness_pct: float = Field(
        default=SCULPT_DEFAULTS["edge_roundness_pct"], ge=0.2, le=15.0, allow_inf_nan=False
    )
    fullness: float = Field(default=SCULPT_DEFAULTS["fullness"], ge=FULLNESS_MIN, le=1.0, allow_inf_nan=False)
    crease_depth: float = Field(default=SCULPT_DEFAULTS["crease_depth"], ge=0.0, le=0.6, allow_inf_nan=False)
    crease_width_pct: float = Field(
        default=SCULPT_DEFAULTS["crease_width_pct"], ge=0.1, le=5.0, allow_inf_nan=False
    )
    layering: float = Field(default=SCULPT_DEFAULTS["layering"], ge=0.0, le=1.0, allow_inf_nan=False)


class PreprocessRequest(BaseModel):
    bg_removal: bool = True
    hd_restore: bool = True
    # "auto" | "white_on_black" | "black_on_white"; None keeps the project's setting.
    polarity: str | None = None
    target_res: int = Field(default=4096, ge=128, le=8192)
    ai_mode: str = "off"


class UpscaleRequest(BaseModel):
    target_res: int = Field(default=4096, ge=128, le=8192)


class ExportRequest(BaseModel):
    format: str = "png16"
    # Physical dimensions come from the project's saved params. They are optional here only as a
    # consistency check: a value that differs from the saved one is rejected (409).
    width_mm: float | None = Field(default=None, ge=1.0, le=MAX_PHYSICAL_MM, allow_inf_nan=False)
    relief_z_mm: float | None = Field(default=None, gt=0.0, le=MAX_PHYSICAL_MM, allow_inf_nan=False)
    base_z_mm: float | None = Field(default=None, ge=0.0, le=MAX_PHYSICAL_MM, allow_inf_nan=False)
    # STL / OBJ: "adaptive" is an error-bounded RTIN mesh of the full-res heightmap (measured max
    # vertical error <= mesh_tolerance_mm at every sample); "grid" area-downsamples to
    # mesh_resolution. Cut-outs (base_z_mm == 0) always use the grid mesh. An adaptive mesh above
    # MONOLITH_MAX_MESH_TRIANGLES (default 15 M) is refused with 413 before it is built.
    mesh_mode: Literal["adaptive", "grid"] = "adaptive"
    mesh_tolerance_mm: float = Field(default=0.02, ge=0.001, le=5.0, allow_inf_nan=False)
    # Grid mode only. Capped: an uncapped grid at 8K is ~134M triangles (~6.7 GB) and takes the server down.
    mesh_resolution: int = Field(default=1024, ge=16, le=MAX_MESH_RESOLUTION)
    # SVG / DXF outline: Douglas-Peucker tolerance in mm (0 keeps every traced vertex).
    outline_tolerance_mm: float = Field(default=0.05, ge=0.0, le=10.0, allow_inf_nan=False)


class PresetCreateRequest(BaseModel):
    name: str
    params: dict[str, Any]


class PartOrderRequest(BaseModel):
    """A click-to-fix on the Sculpt part order. (u, v) are normalised image coordinates with v
    measured downward; ``reset_all`` ignores them (they may be omitted)."""

    action: Literal["forward", "back", "reset", "reset_all"]
    u: float | None = Field(default=None, allow_inf_nan=False)
    v: float | None = Field(default=None, allow_inf_nan=False)

    @model_validator(mode="after")
    def _point_for_part_actions(self) -> "PartOrderRequest":
        if self.action != "reset_all" and (self.u is None or self.v is None):
            raise ValueError(f"'{self.action}' needs the clicked point (u, v)")
        return self


# ---------------------------------------------------------------------------
# Two-Stage Cache State & Helpers
# ---------------------------------------------------------------------------


@dataclass
class ProjectMemoryCache:
    full_layers: dict[str, Any]
    preview_layers: dict[str, Any]
    H_current: np.ndarray
    H_preview: np.ndarray
    revision: int = 1
    full_dirty: bool = False
    cached_png_rev: int = -1
    cached_png_bytes: bytes | None = None
    pending_timer: threading.Timer | None = field(default=None)
    pending_flush: Future | None = field(default=None)
    # Exact float32 textures last handed to (or uploaded by) the client, keyed by (h, w).
    # Absolute-mode brush uploads diff against these so untouched pixels yield an exact
    # zero delta instead of baking preview-vs-full resampling error into the relief.
    served_textures: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    # Newest client request stamp applied by /recompute (stale requests are dropped).
    last_seq: float | None = None
    # After a Sculpt reorder: the basis whose on-disk copy was invalidated and is still to be
    # written (by `_sculpt_background_job`, or by an eviction / shutdown flush), the pending full
    # rebuild, the debounce timer that submits that job, and the job.
    sculpt_unsaved_basis: dict[str, Any] | None = None
    # A reorder whose full-basis rebuild is still to run: (new order, clicked part ids).
    sculpt_pending: tuple[np.ndarray, list[int]] | None = None
    sculpt_timer: threading.Timer | None = field(default=None)
    sculpt_job: Future | None = field(default=None)
    # Bumped by every change to the full basis or its pending order (reorder, in-place rebuild,
    # in-place save, style switch): the background job's off-lock result is used only if unchanged.
    sculpt_gen: int = 0


MAX_RAM_PROJECTS = max(1, int(os.environ.get("MONOLITH_MAX_RAM_PROJECTS", "3")))
def _physical_ram_bytes() -> int | None:
    """Total physical RAM in bytes, or None when it cannot be determined."""
    try:
        if sys.platform == "win32":
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            phys = int(status.ullTotalPhys)
        else:
            phys = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
        return phys if phys > 0 else None
    except (ValueError, OSError, AttributeError):
        return None


def _default_ram_budget_bytes() -> int:
    """40% of physical RAM (env MONOLITH_RAM_BUDGET_MB overrides); 4 GB if RAM is unknown."""
    raw = os.environ.get("MONOLITH_RAM_BUDGET_MB")
    if raw:
        return max(1, int(float(raw))) * 1024 * 1024
    phys = _physical_ram_bytes()
    if phys:
        return int(phys * 0.4)
    return 4096 * 1024 * 1024


_RAM_BUDGET_BYTES: int = _default_ram_budget_bytes()
MAX_UPLOAD_BYTES = int(os.environ.get("MONOLITH_MAX_UPLOAD_MB", "200")) * 1024 * 1024

_CACHE_MUTEX = threading.RLock()
_PROJECT_LOCKS: dict[str, threading.RLock] = {}
_RAM_CACHE: dict[str, ProjectMemoryCache] = {}
_EVICTING_EVENTS: dict[str, threading.Event] = {}
_FLUSH_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="monolith-flush")


def _get_project_lock(project_id: str) -> threading.RLock:
    with _CACHE_MUTEX:
        lock = _PROJECT_LOCKS.get(project_id)
        if lock is None:
            lock = threading.RLock()
            _PROJECT_LOCKS[project_id] = lock
        return lock


def _wait_if_evicting(project_id: str) -> None:
    """Wait outside all locks if another thread is flushing an evicted cache for this project."""
    while True:
        with _CACHE_MUTEX:
            ev = _EVICTING_EVENTS.get(project_id)
        if ev is None:
            return
        ev.wait(timeout=30.0)


def _touch_ram_cache(project_id: str) -> ProjectMemoryCache | None:
    """Return cached entry and promote it to MRU end under _CACHE_MUTEX."""
    with _CACHE_MUTEX:
        cache = _RAM_CACHE.get(project_id)
        if cache is not None:
            _RAM_CACHE.pop(project_id)
            _RAM_CACHE[project_id] = cache
        return cache


def _cache_nbytes(cache: ProjectMemoryCache) -> int:
    """Bytes held by every distinct ndarray reachable from a project cache (each buffer once)."""
    seen: set[int] = set()
    total = 0

    def walk(obj: Any) -> None:
        nonlocal total
        if isinstance(obj, np.ndarray):
            base = obj
            while isinstance(base.base, np.ndarray):
                base = base.base
            if id(base) not in seen:
                seen.add(id(base))
                total += int(base.nbytes)
        elif isinstance(obj, dict):
            for v in list(obj.values()):
                walk(v)
        elif isinstance(obj, (tuple, list)):
            for v in obj:
                walk(v)
        elif isinstance(obj, PartsAnalysis):  # the Sculpt parts analysis (2048-grid arrays)
            walk(list(vars(obj).values()))

    walk(cache.full_layers)
    walk(cache.preview_layers)
    walk(cache.served_textures)
    walk(cache.H_current)
    walk(cache.H_preview)
    return total


MAX_SERVED_TEXTURES = 2


def _remember_served_texture(cache: ProjectMemoryCache, key: tuple[int, int], tex: np.ndarray) -> None:
    """Record the texture last exchanged with the client; keep only the most recent few."""
    cache.served_textures.pop(key, None)
    cache.served_textures[key] = tex
    while len(cache.served_textures) > MAX_SERVED_TEXTURES:
        cache.served_textures.pop(next(iter(cache.served_textures)))


def _put_ram_cache_entry(
    project_id: str,
    mem_cache: ProjectMemoryCache,
) -> list[tuple[str, ProjectMemoryCache]]:
    """Insert or promote project_id in LRU _RAM_CACHE and return any evicted entries to flush."""
    evicted: list[tuple[str, ProjectMemoryCache]] = []
    limit = max(1, int(MAX_RAM_PROJECTS))
    with _CACHE_MUTEX:
        _RAM_CACHE.pop(project_id, None)
        _RAM_CACHE[project_id] = mem_cache
        while True:
            over_count = len(_RAM_CACHE) > limit
            over_bytes = (
                not over_count
                and len(_RAM_CACHE) > 1
                and sum(_cache_nbytes(c) for c in _RAM_CACHE.values()) > int(_RAM_BUDGET_BYTES)
            )
            if not (over_count or over_bytes):
                break
            oldest_id = next((k for k in _RAM_CACHE if k != project_id), None)
            if oldest_id is None:
                break
            oldest_cache = _RAM_CACHE.pop(oldest_id)
            _EVICTING_EVENTS[oldest_id] = threading.Event()
            evicted.append((oldest_id, oldest_cache))
    return evicted


def _flush_evicted_caches(evicted: list[tuple[str, ProjectMemoryCache]]) -> None:
    """Flush pending disk writes for evicted RAM cache entries outside _CACHE_MUTEX and caller locks.

    Every entry's evicting event is set, even when flushing an earlier one raises (the first
    error is re-raised after all of them): an unset event makes `_wait_if_evicting` block every
    later request for that project. Callers run this in a ``finally`` after the lock they held
    while calling `_put_ram_cache_entry`, so a failure in between cannot leak the events either.
    """
    first_error: BaseException | None = None
    for evicted_id, evicted_cache in evicted:
        try:
            ev_lock = _get_project_lock(evicted_id)
            fut: Future | None = None
            had_timer = False
            with ev_lock:
                if evicted_cache.pending_timer is not None:
                    evicted_cache.pending_timer.cancel()
                    evicted_cache.pending_timer = None
                    had_timer = True
                _cancel_sculpt_timer(evicted_cache)
                fut = evicted_cache.pending_flush
                sculpt_fut = evicted_cache.sculpt_job
            for f in (fut, sculpt_fut):
                if f is not None:
                    try:
                        f.result(timeout=30.0)
                    except Exception:
                        pass
            with ev_lock:
                with _CACHE_MUTEX:
                    still_evicted = _RAM_CACHE.get(evicted_id) is None
                if still_evicted and _project_dir(evicted_id).exists():
                    # a reorder's full rebuild and basis write, if still pending (its background
                    # job skips evicted entries); the full relief is then recomputed below
                    _persist_sculpt_basis(evicted_id, evicted_cache)
                    if evicted_cache.full_dirty:
                        proj = db.get_project(evicted_id)
                        params = proj["params"] if proj is not None else db.default_relief_params()
                        H_full = recompute_relief(
                            evicted_cache.full_layers,
                            params,
                            physical_width_mm=float(params.get("physical_width_mm", 1000.0)),
                            relief_z_mm=float(params.get("relief_z_mm", 15.0)),
                        )
                        H_full = np.ascontiguousarray(H_full, dtype=np.float32)
                        evicted_cache.H_current = H_full
                        evicted_cache.full_dirty = False
                        _save_current_to_disk(evicted_id, H_full)
                    elif had_timer:
                        _save_current_to_disk(evicted_id, evicted_cache.H_current)
        except BaseException as exc:  # flush the others first; the events are set below
            if first_error is None:
                first_error = exc
        finally:
            with _CACHE_MUTEX:
                ev = _EVICTING_EVENTS.pop(evicted_id, None)
                if ev is not None:
                    ev.set()
    if first_error is not None:
        raise first_error


def _projects_base_dir() -> Path:
    data_dir = Path(os.environ.get("MONOLITH_DATA_DIR", str(PROJECT_ROOT / "data")))
    return data_dir / "projects"


def _project_dir(project_id: str) -> Path:
    return _projects_base_dir() / project_id


def _project_cache_dir(project_id: str) -> Path:
    return _project_dir(project_id) / "cache"


def _meta_path(project_id: str) -> Path:
    return _project_dir(project_id) / "meta.json"


def _default_meta(
    *,
    bg_removal: bool = True,
    hd_restore: bool = True,
    target_res: int = 4096,
    bit_depth: str = "16bit",
    original_relpath: str = "original.png",
    groove_sigma: float = 2.5,
    polarity: str = "auto",
) -> dict[str, Any]:
    return {
        "polarity": polarity,
        "bg_removal": bool(bg_removal),
        "hd_restore": bool(hd_restore),
        "target_res": int(target_res),
        "bit_depth": bit_depth,
        "original_relpath": original_relpath,
        "groove_sigma": float(groove_sigma),
    }


def _read_meta(project_id: str) -> dict[str, Any]:
    path = _meta_path(project_id)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            defaults = _default_meta()
            defaults.update(data)
            return defaults
        except Exception:
            pass
    return _default_meta()


def _write_meta(project_id: str, meta: dict[str, Any]) -> None:
    pdir = _project_dir(project_id)
    pdir.mkdir(parents=True, exist_ok=True)
    tmp = pdir / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    os.replace(tmp, _meta_path(project_id))


def _normalize_ai_mode(ai_mode: str) -> str:
    mode = (ai_mode or "off").strip().lower()
    if mode in ("off", "none", "false", "0"):
        return "off"
    if mode in ("guided_overlap", "optimize", "subtle_overlap", "on", "guided"):
        return "guided_overlap"
    raise HTTPException(status_code=400, detail=f"Unsupported ai_mode: {ai_mode}")


def _normalize_polarity(polarity: str | None) -> str:
    value = (polarity or "auto").strip().lower()
    if value not in POLARITIES:
        raise HTTPException(status_code=400, detail=f"Unsupported polarity: {polarity}")
    return value


def _sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return cleaned.strip("._") or "monolith_relief"


def _detect_image_media_type(path: Path) -> str:
    try:
        head = path.read_bytes()[:16]
        if head.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if head.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
            return "image/webp"
        if head.startswith((b"II*\x00", b"MM\x00*")):
            return "image/tiff"
    except OSError:
        pass
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def _find_original_image_path(project_id: str) -> Path | None:
    pdir = _project_dir(project_id)
    meta = _read_meta(project_id)
    rel = meta.get("original_relpath")
    if rel:
        candidate = pdir / rel
        if candidate.is_file():
            return candidate
    for candidate in sorted(pdir.glob("original*")):
        if candidate.is_file():
            return candidate
    return None


def _count_unique_16bit_levels(H: np.ndarray) -> int:
    u16 = (np.clip(H, np.float32(0.0), np.float32(1.0)) * np.float32(65535.0)).astype(np.uint16)
    return int(np.count_nonzero(np.bincount(u16.ravel(), minlength=65536)))


def _downsample_float32(arr: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = arr.shape[:2]
    if max(h, w) <= max_dim:
        return np.ascontiguousarray(arr, dtype=np.float32)
    scale = float(max_dim) / float(max(h, w))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(arr.astype(np.float32, copy=False), (nw, nh), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(resized, dtype=np.float32)


BASIS_LAYER_KEYS = (
    "M_sil",
    "d_sil",
    "H_global",
    "H_local",
    "G_lines",
    "H_ai_guided",
    "L_clean",
    "M_lines",
)


# ---------------------------------------------------------------------------
# Relief style (Classic / Sculpt)
# ---------------------------------------------------------------------------


def _params_style(params: dict[str, Any] | None) -> str:
    """The relief style a params dict selects ("classic" when absent, as in the engine)."""
    return "sculpt" if str((params or {}).get("style") or "classic") == "sculpt" else "classic"


def _layers_style(full_layers: dict[str, Any]) -> str:
    """The style a layer dict holds the basis for."""
    return "sculpt" if "sculpt_basis" in full_layers else "classic"


def _check_sculpt_version(params: dict[str, Any]) -> None:
    """Refuse (409) a Sculpt version this build does not have, instead of a wrong look or a 500."""
    if _params_style(params) != "sculpt":
        return
    try:
        version = int(params.get("sculpt_version", SCULPT_LATEST_VERSION))
    except (TypeError, ValueError):
        return  # left to the params validation (422)
    if version > SCULPT_LATEST_VERSION:
        raise HTTPException(
            status_code=409,
            detail=(
                f"This project uses Sculpt version {version}, but this MONOLITH build supports "
                f"Sculpt up to version {SCULPT_LATEST_VERSION}. Update MONOLITH, or set "
                f"sculpt_version to {SCULPT_LATEST_VERSION} or style to 'classic'."
            ),
        )


def _require_known_sculpt_version(project_id: str) -> None:
    """409 before rendering a stored project whose Sculpt version this build does not have: a
    texture, PNG or export of it would silently use another algorithm (design decision 6). Only
    /recompute, which can change the version, renders nothing before its own check."""
    proj = db.get_project(project_id)
    if proj is not None:
        _check_sculpt_version(proj["params"])


def _sculpt_order(pa: PartsAnalysis, meta: dict[str, Any] | None) -> np.ndarray:
    """The project's part order (layer per part): the automatic order from the drawing with the
    click-to-fix overrides stored in ``meta`` (None for a new project) applied.

    Every basis build takes its order from here, so a preprocess or upscale re-applies the
    overrides to the new analysis (`sculpt_overrides`).
    """
    return sculpt_overrides.solve(pa, sculpt_overrides.stored(meta))


def _parts_source(meta: dict[str, Any]) -> dict[str, Any]:
    """The preprocess settings the parts analysis depends on (its cache key)."""
    return {
        "bg_removal": bool(meta.get("bg_removal", True)),
        "hd_restore": bool(meta.get("hd_restore", True)),
        "polarity": str(meta.get("polarity", "auto")),
    }


def _classic_layers(pre: dict[str, Any], *, ai_mode: str, params: dict[str, Any]) -> dict[str, Any]:
    """Classic basis layers from preprocessed arrays (L_clean, M_sil, M_lines, M_regions, shape)."""
    groove_scale = groove_scale_for_shape(pre["shape"])
    basis = compute_basis_layers(
        pre,
        ai_mode=ai_mode,
        groove_sigma=float(params.get("groove_sigma", 2.5)) * groove_scale,
    )
    # M_lines == (1 - L_clean) * M_sil is derived on demand, never held at full resolution.
    return {
        "M_sil": np.ascontiguousarray(basis["M_sil"], dtype=np.float32),
        "d_sil": np.ascontiguousarray(basis["d_sil"], dtype=np.float32),
        "H_global": np.ascontiguousarray(basis["H_global"], dtype=np.float32),
        "H_local": np.ascontiguousarray(basis["H_local"], dtype=np.float32),
        "G_lines": np.ascontiguousarray(basis["G_lines"], dtype=np.float32),
        "H_ai_guided": np.ascontiguousarray(basis["H_ai_guided"], dtype=np.float32),
        "L_clean": np.ascontiguousarray(pre["L_clean"], dtype=np.float32),
        # Pixel sigma the cached G_lines was built with (engine-internal, full-res px).
        "groove_sigma": float(basis["groove_sigma"]),
        "_groove_scale": groove_scale,
    }


def _preprocessed_from_layers(full_layers: dict[str, Any]) -> dict[str, Any]:
    """The preprocess outputs Classic needs, rebuilt exactly from the cached L_clean and M_sil."""
    L = full_layers["L_clean"]
    M = full_layers["M_sil"]
    return {
        "L_clean": L,
        "M_sil": M,
        "M_lines": ((np.float32(1.0) - L) * M).astype(np.float32),
        "M_regions": (L * M).astype(np.float32),
        "shape": tuple(M.shape[:2]),
    }


def _sculpt_layers(
    L_clean: np.ndarray,
    M_sil: np.ndarray,
    pa: PartsAnalysis,
    order: np.ndarray,
    *,
    source: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Sculpt layers: the shared working arrays plus the Sculpt basis (no Classic arrays)."""
    M = np.ascontiguousarray(M_sil, dtype=np.float32)
    L = np.ascontiguousarray(L_clean, dtype=np.float32)
    groove_scale = groove_scale_for_shape(M.shape)
    d_sil = cv2.distanceTransform(
        (M > np.float32(0.5)).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    ).astype(np.float32, copy=False)
    return {
        "M_sil": M,
        "d_sil": d_sil,
        "L_clean": L,
        # Working-resolution L_clean: lines and ink weight come from it, not the analysis grid.
        "sculpt_basis": build_sculpt_basis(pa, order, M, M.shape, L_clean=L),
        "sculpt_parts": pa,
        "_parts_source": dict(source),
        "groove_sigma": float(params.get("groove_sigma", 2.5)) * groove_scale,
        "_groove_scale": groove_scale,
    }


def _project_parts(project_id: str, meta: dict[str, Any]) -> tuple[PartsAnalysis, bool]:
    """(parts analysis, freshly analysed): from cache/parts.npz, else from the original image."""
    source = _parts_source(meta)
    pa = sculpt_cache.load_parts(_project_cache_dir(project_id), source)
    if pa is not None:
        return pa, False
    orig_path = _find_original_image_path(project_id)
    if orig_path is None:
        raise HTTPException(status_code=404, detail=f"Original image missing for '{project_id}'")
    return analyze_parts_from_source(str(orig_path), **source), True


def _attach_sculpt_layers(
    project_id: str, full_layers: dict[str, Any], meta: dict[str, Any], *, save: bool
) -> None:
    """Add the Sculpt basis to layers holding the shared arrays: read from the disk cache when it
    is current, else built from the cached L_clean / M_sil (and written back when ``save``)."""
    pa, fresh = _project_parts(project_id, meta)
    M = full_layers["M_sil"]
    cache_dir = _project_cache_dir(project_id)
    basis = None if fresh else sculpt_cache.load_basis(cache_dir, pa, M.shape[:2])
    source = _parts_source(meta)
    if basis is None:
        basis = build_sculpt_basis(pa, _sculpt_order(pa, meta), M, M.shape[:2], L_clean=full_layers["L_clean"])
        if save:
            if fresh:
                sculpt_cache.save_parts(cache_dir, pa, source)
            sculpt_cache.save_basis(cache_dir, basis, pa)
    full_layers["sculpt_basis"] = basis
    full_layers["sculpt_parts"] = pa
    full_layers["_parts_source"] = source


def _set_brush_delta_layers(
    full_layers: dict[str, Any],
    preview_layers: dict[str, Any] | None,
    brush_delta: np.ndarray | None = None,
) -> None:
    """Attach float32 H_brush_delta and _has_brush_delta flags to full and preview layers."""
    ref_shape = full_layers["M_sil"].shape[:2]
    if isinstance(brush_delta, np.ndarray):
        arr = np.ascontiguousarray(brush_delta, dtype=np.float32)
        if arr.shape != ref_shape:
            arr = np.ascontiguousarray(
                cv2.resize(arr, (ref_shape[1], ref_shape[0]), interpolation=cv2.INTER_CUBIC),
                dtype=np.float32,
            )
        has_delta = bool(np.any(np.abs(arr) > np.float32(1e-6)))
    else:
        arr = np.zeros(ref_shape, dtype=np.float32)
        has_delta = False

    full_layers["H_brush_delta"] = arr
    full_layers["_has_brush_delta"] = has_delta

    if preview_layers is not None:
        prev_shape = preview_layers["M_sil"].shape[:2]
        if has_delta:
            prev_arr = _downsample_float32(arr, PREVIEW_MAX_DIM)
            if prev_arr.shape != prev_shape:
                prev_arr = np.ascontiguousarray(
                    cv2.resize(arr, (prev_shape[1], prev_shape[0]), interpolation=cv2.INTER_AREA),
                    dtype=np.float32,
                )
        else:
            prev_arr = np.zeros(prev_shape, dtype=np.float32)
        preview_layers["H_brush_delta"] = prev_arr
        preview_layers["_has_brush_delta"] = has_delta


def _current_brush_delta(project_id: str) -> np.ndarray | None:
    """Return the project's persisted brush delta (RAM cache first, then disk), if any."""
    with _CACHE_MUTEX:
        cache = _RAM_CACHE.get(project_id)
    if cache is not None:
        if cache.full_layers.get("_has_brush_delta", False):
            arr = cache.full_layers.get("H_brush_delta")
            if isinstance(arr, np.ndarray):
                return arr
        return None
    path = _project_cache_dir(project_id) / "H_brush_delta.npy"
    if path.is_file():
        return np.load(path)
    return None


def _resample_brush_delta(delta: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    """Resample a brush delta onto a new (H, W) grid; linear when enlarging, area when shrinking."""
    if delta is None or not np.any(np.abs(delta) > np.float32(1e-6)):
        return None
    arr = np.ascontiguousarray(delta, dtype=np.float32)
    if arr.shape == tuple(shape):
        return arr
    h, w = int(shape[0]), int(shape[1])
    interp = cv2.INTER_AREA if (h * w) < arr.size else cv2.INTER_LINEAR
    return np.ascontiguousarray(cv2.resize(arr, (w, h), interpolation=interp), dtype=np.float32)


def _preview_shape(h: int, w: int, max_dim: int = PREVIEW_MAX_DIM) -> tuple[int, int]:
    """The preview grid for an (h, w) working grid: the same grid up to ``max_dim``."""
    if max(h, w) <= max_dim:
        return int(h), int(w)
    scale = float(max_dim) / float(max(h, w))
    return max(1, int(round(h * scale))), max(1, int(round(w * scale)))


def _build_sculpt_preview_layers(full_layers: dict[str, Any], max_dim: int) -> dict[str, Any]:
    """Sculpt preview layers: a separate Sculpt basis built on the preview grid.

    The basis is rebuilt from the downsampled silhouette and luminance with the same parts
    analysis and order, not resampled from the full basis: its sub-pixel distances and
    thickness field do not survive resampling linearly. Up to ``max_dim`` the full layers
    (and basis) are shared.
    """
    ref = full_layers["M_sil"]
    h, w = ref.shape[:2]
    full_basis = full_layers["sculpt_basis"]
    if max(h, w) <= max_dim:
        preview: dict[str, Any] = {
            k: np.ascontiguousarray(full_layers[k], dtype=np.float32) for k in ("M_sil", "d_sil", "L_clean")
        }
        preview["sculpt_basis"] = full_basis
        ph, pw, pixel_scale = h, w, 1.0
    else:
        scale = float(max_dim) / float(max(h, w))
        pw = max(1, int(round(w * scale)))
        ph = max(1, int(round(h * scale)))
        pixel_scale = float(pw) / float(w)
        preview = {}
        for key in ("M_sil", "d_sil", "L_clean"):
            resized = cv2.resize(full_layers[key], (pw, ph), interpolation=cv2.INTER_AREA)
            if key == "d_sil":
                resized = resized * pixel_scale
            preview[key] = np.ascontiguousarray(resized, dtype=np.float32)
        preview["sculpt_basis"] = build_sculpt_basis(
            full_layers["sculpt_parts"], full_basis["layer"], preview["M_sil"], (ph, pw), L_clean=preview["L_clean"]
        )
    preview["groove_sigma"] = float(full_layers.get("groove_sigma", 2.5)) * pixel_scale
    preview["_pixel_scale"] = pixel_scale
    preview["_groove_scale"] = groove_scale_for_shape((ph, pw))
    brush_delta = full_layers.get("H_brush_delta")
    if bool(full_layers.get("_has_brush_delta", False)) and isinstance(brush_delta, np.ndarray) and brush_delta.shape == (h, w):
        if (ph, pw) == (h, w):
            prev_delta = np.ascontiguousarray(brush_delta, dtype=np.float32)
        else:
            prev_delta = np.ascontiguousarray(
                cv2.resize(brush_delta, (pw, ph), interpolation=cv2.INTER_AREA), dtype=np.float32
            )
        preview["H_brush_delta"] = prev_delta
        preview["_has_brush_delta"] = True
    else:
        preview["H_brush_delta"] = np.zeros((ph, pw), dtype=np.float32)
        preview["_has_brush_delta"] = False
    return preview


def _build_preview_layers(full_layers: dict[str, Any], max_dim: int = PREVIEW_MAX_DIM) -> dict[str, Any]:
    if _layers_style(full_layers) == "sculpt":
        return _build_sculpt_preview_layers(full_layers, max_dim)
    ref = full_layers["M_sil"]
    h, w = ref.shape[:2]
    base_sigma = float(full_layers.get("groove_sigma", 2.5))
    has_brush = bool(full_layers.get("_has_brush_delta", False))
    brush_delta = full_layers.get("H_brush_delta")
    if max(h, w) <= max_dim:
        preview: dict[str, Any] = {
            k: np.ascontiguousarray(full_layers[k], dtype=np.float32)
            for k in BASIS_LAYER_KEYS
            if k in full_layers and isinstance(full_layers[k], np.ndarray)
        }
        if "M_lines" not in preview:
            m_lines = derive_m_lines(full_layers)
            if m_lines is not None:
                preview["M_lines"] = np.ascontiguousarray(m_lines, dtype=np.float32)
        preview["groove_sigma"] = base_sigma
        preview["_pixel_scale"] = 1.0
        preview["_groove_scale"] = groove_scale_for_shape((h, w))
        if has_brush and isinstance(brush_delta, np.ndarray) and brush_delta.shape == (h, w):
            preview["H_brush_delta"] = np.ascontiguousarray(brush_delta, dtype=np.float32)
            preview["_has_brush_delta"] = True
        else:
            preview["H_brush_delta"] = np.zeros((h, w), dtype=np.float32)
            preview["_has_brush_delta"] = False
        return preview

    scale = float(max_dim) / float(max(h, w))
    pw = max(1, int(round(w * scale)))
    ph = max(1, int(round(h * scale)))
    pixel_scale = float(pw) / float(w)

    preview = {}
    for key in BASIS_LAYER_KEYS:
        arr = full_layers.get(key)
        if key == "M_lines" and arr is None:
            arr = derive_m_lines(full_layers)  # transient full-res product, freed right after
        if not isinstance(arr, np.ndarray):
            continue
        resized = cv2.resize(arr.astype(np.float32, copy=False), (pw, ph), interpolation=cv2.INTER_AREA)
        if key == "d_sil":
            resized = resized * pixel_scale
        preview[key] = np.ascontiguousarray(resized, dtype=np.float32)
    preview["groove_sigma"] = base_sigma * pixel_scale
    preview["_pixel_scale"] = pixel_scale
    preview["_groove_scale"] = groove_scale_for_shape((ph, pw))
    if has_brush and isinstance(brush_delta, np.ndarray) and brush_delta.shape == (h, w):
        prev_delta = _downsample_float32(brush_delta, max_dim)
        if prev_delta.shape != (ph, pw):
            prev_delta = np.ascontiguousarray(
                cv2.resize(brush_delta, (pw, ph), interpolation=cv2.INTER_AREA),
                dtype=np.float32,
            )
        preview["H_brush_delta"] = prev_delta
        preview["_has_brush_delta"] = True
    else:
        preview["H_brush_delta"] = np.zeros((ph, pw), dtype=np.float32)
        preview["_has_brush_delta"] = False
    return preview


def _atomic_write_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_suffix(".npy.tmp")
    with tmp.open("wb") as fh:
        np.save(fh, np.ascontiguousarray(arr, dtype=np.float32))
    os.replace(tmp, path)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _save_layers_to_disk(project_id: str, full_layers: dict[str, Any], *, common: bool = True) -> None:
    """Write the active style's layers (and the shared ones when ``common``); remove the other
    style's, so only the active style is cached (design decision 5)."""
    cache_dir = _project_cache_dir(project_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Drop the Sculpt manifest first: a Sculpt cache is never read against other working arrays.
    sculpt_cache.clear_basis(cache_dir)
    keys = ("M_sil", "d_sil", "H_global", "H_local", "G_lines", "H_ai_guided", "L_clean")
    for key in keys:
        if key in full_layers and (common or key in CLASSIC_LAYER_KEYS):
            _atomic_write_npy(cache_dir / f"{key}.npy", full_layers[key])
    if _layers_style(full_layers) == "sculpt":
        for key in CLASSIC_LAYER_KEYS:
            (cache_dir / f"{key}.npy").unlink(missing_ok=True)
        pa = full_layers["sculpt_parts"]
        sculpt_cache.save_parts(cache_dir, pa, full_layers["_parts_source"])
        sculpt_cache.save_basis(cache_dir, full_layers["sculpt_basis"], pa)


def _save_all_layers_to_disk(
    project_id: str,
    full_layers: dict[str, np.ndarray],
    H_current: np.ndarray,
    png_bytes: bytes | None = None,
) -> bytes:
    cache_dir = _project_cache_dir(project_id)
    _save_layers_to_disk(project_id, full_layers)
    _atomic_write_npy(cache_dir / "H_current.npy", H_current)
    encoded_png = png_bytes if png_bytes is not None else encode_png16(H_current)
    _atomic_write_bytes(_project_dir(project_id) / "depth_16bit.png", encoded_png)
    return encoded_png


def _save_current_to_disk(project_id: str, H_current: np.ndarray, png_bytes: bytes | None = None) -> bytes:
    cache_dir = _project_cache_dir(project_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_npy(cache_dir / "H_current.npy", H_current)
    encoded_png = png_bytes if png_bytes is not None else encode_png16(H_current)
    _atomic_write_bytes(_project_dir(project_id) / "depth_16bit.png", encoded_png)
    return encoded_png


def _schedule_current_flush(project_id: str, cache: ProjectMemoryCache, revision: int, H_snapshot: np.ndarray) -> None:
    if cache.pending_timer is not None:
        cache.pending_timer.cancel()

    def _run_flush() -> None:
        lock = _get_project_lock(project_id)
        with lock:
            with _CACHE_MUTEX:
                current_cache = _RAM_CACHE.get(project_id)
            if current_cache is not cache or current_cache.revision != revision:
                return
            current_cache.pending_timer = None
            target_h = current_cache.H_current

        def _flush_job() -> None:
            png_bytes = encode_png16(target_h)
            with lock:
                # Re-check under the lock: a newer revision, a preprocess rebuild or a
                # delete may have landed while encoding; never overwrite fresher files.
                with _CACHE_MUTEX:
                    c2 = _RAM_CACHE.get(project_id)
                if c2 is not cache or c2.revision != revision or not _project_dir(project_id).exists():
                    return
                _save_current_to_disk(project_id, target_h, png_bytes=png_bytes)
                c2.cached_png_rev = revision
                c2.cached_png_bytes = png_bytes

        with lock:
            with _CACHE_MUTEX:
                c3 = _RAM_CACHE.get(project_id)
            if c3 is cache and c3.revision == revision:
                c3.pending_flush = _FLUSH_EXECUTOR.submit(_flush_job)

    timer = threading.Timer(0.15, _run_flush)
    timer.daemon = True
    cache.pending_timer = timer
    timer.start()


def _await_pending_flush(project_id: str, *, flush_now: bool = True) -> None:
    with _CACHE_MUTEX:
        first = _RAM_CACHE.get(project_id)
    if first is not None and first.sculpt_job is not None:
        # a Sculpt job's slow phase runs off the lock: let it finish before saving in place
        try:
            first.sculpt_job.result(timeout=60.0)
        except Exception:
            pass
    lock = _get_project_lock(project_id)
    with lock:
        with _CACHE_MUTEX:
            cache = _RAM_CACHE.get(project_id)
        if cache is None:
            return
        if cache.pending_timer is not None:
            cache.pending_timer.cancel()
            cache.pending_timer = None
            if flush_now and _project_dir(project_id).exists():
                png_bytes = encode_png16(cache.H_current)
                _save_current_to_disk(project_id, cache.H_current, png_bytes=png_bytes)
                cache.cached_png_rev = cache.revision
                cache.cached_png_bytes = png_bytes
        if flush_now:  # else the entry is about to be replaced or deleted, or is kept (brush)
            _cancel_sculpt_timer(cache)
            _persist_sculpt_basis(project_id, cache)
        futs = (cache.pending_flush, cache.sculpt_job)
    for fut in futs:
        if fut is None:
            continue
        try:
            fut.result(timeout=30.0)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Sculpt reorder (click-to-fix): the preview at once, the full basis and its disk copy after
# ---------------------------------------------------------------------------

# A reorder answers from the preview basis (rebuilt at once, ~0.15-0.4 s at a 1024 preview).
# This long after the last click a background job rebuilds a private copy of the full basis
# (0.5-1 s at 4K), recomputes the full relief and writes the basis (~470 MB, ~0.35 s), all off
# the project lock, then swaps them in within milliseconds; a request that needs the full relief
# sooner rebuilds the live basis in place first. Quick successive clicks rebuild and save once.
SCULPT_SAVE_DELAY_S = 1.0


def _cancel_sculpt_timer(cache: ProjectMemoryCache) -> None:
    if cache.sculpt_timer is not None:
        cache.sculpt_timer.cancel()
        cache.sculpt_timer = None


def _drop_ram_entry(project_id: str, cache: ProjectMemoryCache) -> None:
    """Forget a RAM entry left inconsistent by a failure: the next access reloads from disk."""
    with _CACHE_MUTEX:
        if _RAM_CACHE.get(project_id) is cache:
            _RAM_CACHE.pop(project_id, None)


def _sculpt_layer(cache: ProjectMemoryCache) -> np.ndarray:
    """The project's current part order: a pending reorder's, else the full basis's."""
    if cache.sculpt_pending is not None:
        return cache.sculpt_pending[0]
    return cache.full_layers["sculpt_basis"]["layer"]


def _sync_sculpt_full_order(project_id: str, cache: ProjectMemoryCache) -> None:
    """Rebuild the full basis for a pending reorder (caller holds the lock).

    Every path that computes the full relief or writes the basis calls this first. The full
    relief stays dirty (`_ensure_full_current` recomputes it). On failure the RAM entry is
    dropped, so the next access reloads from disk and the saved overrides.
    """
    pending = cache.sculpt_pending
    if pending is None:
        return
    cache.sculpt_pending = None
    basis = cache.full_layers.get("sculpt_basis")
    if basis is None:
        return
    layer, parts = pending
    cache.sculpt_gen += 1
    try:
        rebuild_parts(basis, parts, layer)
    except BaseException:
        _drop_ram_entry(project_id, cache)
        raise
    cache.sculpt_unsaved_basis = basis
    cache.full_dirty = True


def _persist_sculpt_basis(project_id: str, cache: ProjectMemoryCache) -> None:
    """Write a reorder's basis to the disk cache if it is still unsaved (caller holds the lock).

    A pending full rebuild runs first. Skipped when the layers no longer hold that basis (a style
    switch or rebuild replaced it and saved its own) or the project was deleted.
    """
    _sync_sculpt_full_order(project_id, cache)
    basis = cache.sculpt_unsaved_basis
    if basis is None:
        return
    full = cache.full_layers
    cache.sculpt_gen += 1
    if full.get("sculpt_basis") is basis and _project_dir(project_id).exists():
        sculpt_cache.save_basis(_project_cache_dir(project_id), basis, full["sculpt_parts"])
    cache.sculpt_unsaved_basis = None


def _sync_sculpt_preview_order(cache: ProjectMemoryCache) -> None:
    """Make sure the separate preview basis holds the current order (caller holds the lock).
    A reorder rebuilds it at once; this is the guard before a preview recompute."""
    full_b = cache.full_layers.get("sculpt_basis")
    prev_b = cache.preview_layers.get("sculpt_basis")
    if full_b is None or prev_b is None or prev_b is full_b:
        return
    layer = _sculpt_layer(cache)
    if not np.array_equal(prev_b["layer"], layer):
        rebuild_parts(prev_b, [], layer)


def _sculpt_background_job(project_id: str, cache: ProjectMemoryCache) -> None:
    """After a reorder, with the project lock held only for milliseconds at a time.

    1. Under the lock: a private copy of the full basis (`copy_for_reorder`, ~13 bytes per pixel;
       the order-independent arrays are shared), the pending order and the params.
    2. Off the lock: rebuild the copy for the pending order, recompute the full relief from it
       and write it to a staging folder (`sculpt_cache.stage_basis`).
    3. Under the lock, only if nothing changed meanwhile (same ``sculpt_gen``): commit the staged
       files (renames) and swap the copy in; the full relief and the preview following it are
       taken too when no recompute ran meanwhile (same revision). Otherwise the work is
       discarded and the job re-armed for the newer state.

    Clicks and slider recomputes therefore never wait behind a full rebuild (a full /recompute
    while the order is pending answers from the preview basis), and a request that reads the
    full relief before the swap rebuilds the live basis in place
    (`_sync_sculpt_full_order`), which also supersedes this job. With no pending order (a reader
    already rebuilt in place, or a preview grid that is the full grid) only steps 1-3 of the
    write run. The entry being no longer live (rebuilt, evicted, deleted) ends the job.
    """

    def live() -> bool:
        with _CACHE_MUTEX:
            return _RAM_CACHE.get(project_id) is cache

    lock = _get_project_lock(project_id)
    cache_dir = _project_cache_dir(project_id)
    with lock:
        full = cache.full_layers
        basis = full.get("sculpt_basis")
        pending = cache.sculpt_pending
        if not live() or basis is None or (pending is None and cache.sculpt_unsaved_basis is not basis):
            return
        gen, rev = cache.sculpt_gen, cache.revision
        work = copy_for_reorder(basis)
        layers = dict(full)
        layers["sculpt_basis"] = work
        pa = full["sculpt_parts"]
        proj = db.get_project(project_id)
        params = proj["params"] if proj is not None else db.default_relief_params()

    staging: Path | None = None
    H_full: np.ndarray | None = None
    try:
        if pending is not None:
            rebuild_parts(work, pending[1], pending[0])
            H_full = np.ascontiguousarray(
                recompute_relief(
                    layers,
                    params,
                    physical_width_mm=float(params.get("physical_width_mm", 1000.0)),
                    relief_z_mm=float(params.get("relief_z_mm", 15.0)),
                ),
                dtype=np.float32,
            )
        staging = sculpt_cache.stage_basis(cache_dir, work, pa)
    except Exception:
        # e.g. the project was deleted meanwhile; the in-place paths (readers, eviction,
        # shutdown) still rebuild and save the live basis
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        return

    trash: Path | None = None
    with lock:
        if live() and cache.sculpt_gen == gen and cache.full_layers.get("sculpt_basis") is basis:
            trash = sculpt_cache.commit_staged_basis(staging, cache_dir)
            staging = None
            if pending is not None:
                cache.full_layers["sculpt_basis"] = work
                cache.full_layers["sculpt_edge_clamped"] = bool(layers.get("sculpt_edge_clamped", False))
                cache.sculpt_pending = None
                if cache.revision == rev:  # no recompute, brush or click since the snapshot
                    cache.H_current = H_full
                    cache.full_dirty = False
                    ph, pw = cache.H_preview.shape
                    cache.H_preview = np.ascontiguousarray(
                        cv2.resize(H_full, (pw, ph), interpolation=cv2.INTER_AREA), dtype=np.float32
                    )
                    cache.revision += 1
                    _schedule_current_flush(project_id, cache, cache.revision, H_full)
            cache.sculpt_unsaved_basis = None
            cache.sculpt_gen += 1
        elif (
            live()
            and cache.sculpt_timer is None
            and (cache.sculpt_pending is not None or cache.sculpt_unsaved_basis is not None)
        ):
            _schedule_sculpt_job(project_id, cache)  # superseded: start over on the newer state
    for d in (staging, trash):
        if d is not None:
            shutil.rmtree(d, ignore_errors=True)


def _schedule_sculpt_job(project_id: str, cache: ProjectMemoryCache) -> None:
    """(Re)arm the debounce timer that submits `_sculpt_background_job` (caller holds the lock)."""
    _cancel_sculpt_timer(cache)

    def _fire() -> None:
        with _get_project_lock(project_id):
            if cache.sculpt_timer is not timer:
                return  # cancelled or re-armed meanwhile
            cache.sculpt_timer = None
            cache.sculpt_job = _FLUSH_EXECUTOR.submit(_sculpt_background_job, project_id, cache)

    timer = threading.Timer(SCULPT_SAVE_DELAY_S, _fire)
    timer.daemon = True
    cache.sculpt_timer = timer
    timer.start()


def _apply_sculpt_order(
    project_id: str,
    cache: ProjectMemoryCache,
    params: dict[str, Any],
    meta: dict[str, Any],
    overrides: list[dict[str, Any]],
    layer: np.ndarray,
    part: int,
) -> None:
    """Apply a new part order to a live Sculpt cache (caller holds the project lock).

    1. The on-disk basis (built for the old order) is made unreadable before the new overrides
       are saved, so a crash at any later point reloads a basis built from the saved overrides.
    2. The preview basis is rebuilt for the moved parts and their neighbours and the preview
       relief recomputed, as for a preview /recompute; the revision is bumped.
    3. The full basis follows (`_sync_sculpt_full_order`): from the debounced background job, or
       first thing in any request that needs the full relief, which stays dirty until then. When
       the preview grid is the full grid (<= 1024 px) the full relief is rebuilt here at once.
    If a rebuild fails the RAM entry is dropped, so the next access reloads from disk.
    """
    full_b = cache.full_layers["sculpt_basis"]
    prev_b = cache.preview_layers["sculpt_basis"]
    shared = prev_b is full_b
    parts = [part] if part > 0 else []
    _cancel_sculpt_timer(cache)
    cache.sculpt_gen += 1
    sculpt_cache.invalidate_basis(_project_cache_dir(project_id))
    meta[sculpt_overrides.META_KEY] = overrides
    _write_meta(project_id, meta)
    layers = cache.full_layers if shared else cache.preview_layers
    try:
        rebuild_parts(prev_b, parts, layer)
        H = np.ascontiguousarray(
            recompute_relief(
                layers,
                params,
                physical_width_mm=float(params.get("physical_width_mm", 1000.0)),
                relief_z_mm=float(params.get("relief_z_mm", 15.0)),
            ),
            dtype=np.float32,
        )
    except BaseException:
        _drop_ram_entry(project_id, cache)
        raise
    cache.H_preview = H
    cache.revision += 1
    if shared:
        cache.sculpt_unsaved_basis = full_b
        cache.H_current = H
        cache.full_dirty = False
        cache.cached_png_bytes = _save_current_to_disk(project_id, H)
        cache.cached_png_rev = cache.revision
    else:
        before = cache.sculpt_pending[1] if cache.sculpt_pending is not None else []
        cache.sculpt_pending = (layer, sorted(set(before) | set(parts)))
        cache.full_dirty = True
    _schedule_sculpt_job(project_id, cache)


def _build_layers_from_image(
    image_path: Path,
    *,
    target_res: int,
    bg_removal: bool,
    hd_restore: bool,
    ai_mode: str,
    params: dict[str, Any],
    polarity: str = "auto",
    meta: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Preprocess the image and build the layers of the style ``params`` selects, plus the relief.

    Sculpt also analyses the parts on the fixed analysis grid (from the original image, with the
    same preprocess settings) and orders them; ``meta`` is the project's metadata (the order may
    depend on it), None for a new project.
    """
    preprocessed = preprocess_lineart(
        str(image_path),
        target_res=int(target_res),
        bg_removal=bool(bg_removal),
        hd_restore=bool(hd_restore),
        polarity=polarity,
    )
    if _params_style(params) == "sculpt":
        source = {"bg_removal": bool(bg_removal), "hd_restore": bool(hd_restore), "polarity": str(polarity)}
        pa = analyze_parts_from_source(str(image_path), **source)
        full_layers = _sculpt_layers(
            preprocessed["L_clean"], preprocessed["M_sil"], pa, _sculpt_order(pa, meta), source=source, params=params
        )
    else:
        full_layers = _classic_layers(preprocessed, ai_mode=ai_mode, params=params)
    del preprocessed

    H_current = recompute_relief(
        full_layers,
        params,
        physical_width_mm=float(params.get("physical_width_mm", 1000.0)),
        relief_z_mm=float(params.get("relief_z_mm", 15.0)),
    )
    H_current = np.ascontiguousarray(H_current, dtype=np.float32)
    return full_layers, H_current


def _ensure_full_current(project_id: str, cache: ProjectMemoryCache, params: dict[str, Any] | None = None) -> np.ndarray:
    """Materialize full-res H_current if a previous preview_only recompute (or a Sculpt reorder)
    marked it dirty."""
    _sync_sculpt_full_order(project_id, cache)
    if not cache.full_dirty:
        return cache.H_current
    resolved_params = params
    if resolved_params is None:
        proj = db.get_project(project_id)
        resolved_params = proj["params"] if proj is not None else db.default_relief_params()
    H_full = recompute_relief(
        cache.full_layers,
        resolved_params,
        physical_width_mm=float(resolved_params.get("physical_width_mm", 1000.0)),
        relief_z_mm=float(resolved_params.get("relief_z_mm", 15.0)),
    )
    H_full = np.ascontiguousarray(H_full, dtype=np.float32)
    cache.H_current = H_full
    cache.full_dirty = False
    _schedule_current_flush(project_id, cache, cache.revision, H_full)
    return H_full


def _load_or_rebuild_cache(project_id: str) -> ProjectMemoryCache:
    _wait_if_evicting(project_id)
    lock = _get_project_lock(project_id)
    evicted: list[tuple[str, ProjectMemoryCache]] = []
    with lock:
        cached = _touch_ram_cache(project_id)
        if cached is not None:
            return cached

        project = db.get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")

        cache_dir = _project_cache_dir(project_id)
        style = _params_style(project["params"])
        common_files = [cache_dir / f"{name}.npy" for name in COMMON_LAYER_FILENAMES]
        # basis writes a crash interrupted (no job runs for a project that is not in RAM)
        sculpt_cache.remove_staging(cache_dir)

        if all(p.is_file() for p in common_files):
            meta = _read_meta(project_id)
            full_layers: dict[str, Any] = {}
            classic_cached = all((cache_dir / f"{name}.npy").is_file() for name in LAYER_FILENAMES)
            if style == "classic" and classic_cached:
                for name in ("M_sil", "d_sil", "H_global", "H_local", "G_lines", "H_ai_guided", "L_clean"):
                    full_layers[name] = np.ascontiguousarray(np.load(cache_dir / f"{name}.npy"), dtype=np.float32)
                # meta records the pixel sigma the on-disk G_lines.npy was built with.
                full_layers["groove_sigma"] = float(meta.get("groove_sigma", 2.5))
                full_layers["_groove_scale"] = groove_scale_for_shape(full_layers["M_sil"].shape)
            else:
                for name in ("M_sil", "d_sil", "L_clean"):
                    full_layers[name] = np.ascontiguousarray(np.load(cache_dir / f"{name}.npy"), dtype=np.float32)
                if style == "classic":
                    # The other style was active: build Classic from the cached working arrays.
                    full_layers.update(
                        _classic_layers(
                            _preprocessed_from_layers(full_layers), ai_mode=project["ai_mode"], params=project["params"]
                        )
                    )
                    _save_layers_to_disk(project_id, full_layers, common=False)
                    meta["groove_sigma"] = float(full_layers["groove_sigma"])
                    _write_meta(project_id, meta)
                else:
                    full_layers["groove_sigma"] = float(project["params"].get("groove_sigma", 2.5)) * groove_scale_for_shape(
                        full_layers["M_sil"].shape
                    )
                    full_layers["_groove_scale"] = groove_scale_for_shape(full_layers["M_sil"].shape)
                    # Read the basis from cache/sculpt_*.npy when current, else rebuild and save it.
                    _attach_sculpt_layers(project_id, full_layers, meta, save=True)

            brush_path = cache_dir / "H_brush_delta.npy"
            loaded_brush = np.load(brush_path) if brush_path.is_file() else None
            _set_brush_delta_layers(full_layers, None, loaded_brush)
            # H_current.npy is flushed on a debounce, so after a hard kill it can lag the saved
            # params/brush; the layers and params are authoritative, so recompute from them.
            params = project["params"]
            H_current = np.ascontiguousarray(
                recompute_relief(
                    full_layers,
                    params,
                    physical_width_mm=float(params.get("physical_width_mm", 1000.0)),
                    relief_z_mm=float(params.get("relief_z_mm", 15.0)),
                ),
                dtype=np.float32,
            )
            preview_layers = _build_preview_layers(full_layers, PREVIEW_MAX_DIM)
            H_preview = _downsample_float32(H_current, PREVIEW_MAX_DIM)

            # The on-disk PNG may be equally stale; re-encode and persist it from the recompute.
            png_bytes = _save_current_to_disk(project_id, H_current)
            mem_cache = ProjectMemoryCache(
                full_layers=full_layers,
                preview_layers=preview_layers,
                H_current=H_current,
                H_preview=H_preview,
                revision=1,
                cached_png_rev=1,
                cached_png_bytes=png_bytes,
            )
            evicted = _put_ram_cache_entry(project_id, mem_cache)
        else:
            orig_path = _find_original_image_path(project_id)
            if orig_path is None:
                raise HTTPException(status_code=404, detail=f"Cache and original image missing for '{project_id}'")

            meta = _read_meta(project_id)
            full_layers, H_current = _build_layers_from_image(
                orig_path,
                target_res=int(meta.get("target_res", max(project["width"], project["height"]))),
                bg_removal=bool(meta.get("bg_removal", True)),
                hd_restore=bool(meta.get("hd_restore", True)),
                ai_mode=project["ai_mode"],
                params=project["params"],
                polarity=str(meta.get("polarity", "auto")),
                meta=meta,
            )
            brush_path = cache_dir / "H_brush_delta.npy"
            loaded_brush = np.load(brush_path) if brush_path.is_file() else None
            _set_brush_delta_layers(full_layers, None, loaded_brush)
            if full_layers.get("_has_brush_delta", False):
                # Re-run the composite so the brush delta lands before frame compositing.
                params = project["params"]
                H_current = np.ascontiguousarray(
                    recompute_relief(
                        full_layers,
                        params,
                        physical_width_mm=float(params.get("physical_width_mm", 1000.0)),
                        relief_z_mm=float(params.get("relief_z_mm", 15.0)),
                    ),
                    dtype=np.float32,
                )

            png_bytes = _save_all_layers_to_disk(project_id, full_layers, H_current)
            # Record the pixel sigma the freshly written G_lines.npy was built with.
            meta["groove_sigma"] = float(full_layers["groove_sigma"])
            _write_meta(project_id, meta)
            preview_layers = _build_preview_layers(full_layers, PREVIEW_MAX_DIM)
            H_preview = _downsample_float32(H_current, PREVIEW_MAX_DIM)
            mem_cache = ProjectMemoryCache(
                full_layers=full_layers,
                preview_layers=preview_layers,
                H_current=H_current,
                H_preview=H_preview,
                revision=1,
                cached_png_rev=1,
                cached_png_bytes=png_bytes,
            )
            evicted = _put_ram_cache_entry(project_id, mem_cache)

    if evicted:
        _flush_evicted_caches(evicted)
    return mem_cache


@contextmanager
def _locked_cache(project_id: str):
    """
    Yield the project's *live* RAM cache entry while holding the project lock.

    Loading the cache and acquiring the lock are separate steps, so a concurrent
    /preprocess (or LRU eviction) could swap the entry in between; writing to the stale
    object would silently lose the update. Re-check identity under the lock and retry.
    """
    lock = _get_project_lock(project_id)
    for _ in range(8):
        cache = _load_or_rebuild_cache(project_id)
        lock.acquire()
        with _CACHE_MUTEX:
            live = _RAM_CACHE.get(project_id) is cache
        if live:
            try:
                yield cache
            finally:
                lock.release()
            return
        lock.release()
    raise HTTPException(status_code=503, detail="Project cache is being rebuilt; please retry")


def _build_project_payload(project: dict[str, Any], meta: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved_meta = meta if meta is not None else _read_meta(project["id"])
    payload = dict(project)
    payload["bit_depth"] = resolved_meta.get("bit_depth", "16bit")
    payload["bg_removal"] = bool(resolved_meta.get("bg_removal", True))
    payload["hd_restore"] = bool(resolved_meta.get("hd_restore", True))
    payload["target_res"] = int(resolved_meta.get("target_res", max(project["width"], project["height"])))
    payload["polarity"] = str(resolved_meta.get("polarity", "auto"))
    # Newest Sculpt algorithm this build offers ("Update to latest Sculpt" in the UI).
    payload["sculpt_latest_version"] = SCULPT_LATEST_VERSION
    return payload


def _create_project_from_source(
    *,
    raw_bytes: bytes,
    original_filename: str,
    name: str,
    target_res: int,
    bg_removal: bool,
    hd_restore: bool,
    ai_mode: str,
    bit_depth: str,
    polarity: str = "auto",
    style: str | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    normalized_ai = _normalize_ai_mode(ai_mode)
    polarity = _normalize_polarity(polarity)
    project_id = str(uuid.uuid4())
    pdir = _project_dir(project_id)
    pdir.mkdir(parents=True, exist_ok=True)

    ext = Path(original_filename).suffix.lower()
    if ext not in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"):
        ext = ".png"
    orig_rel = f"original{ext}"
    orig_path = pdir / orig_rel
    orig_path.write_bytes(raw_bytes)

    params = db.default_relief_params()  # new projects: Sculpt at the latest version
    if style is not None:
        params["style"] = style
    try:
        full_layers, H_current = _build_layers_from_image(
            orig_path,
            target_res=target_res,
            bg_removal=bg_removal,
            hd_restore=hd_restore,
            ai_mode=normalized_ai,
            params=params,
            polarity=polarity,
        )
    except (ValueError, cv2.error) as exc:
        # Unreadable / oversized image: don't leave an orphan project folder behind.
        shutil.rmtree(pdir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Could not process image: {exc}") from exc
    _set_brush_delta_layers(full_layers, None, None)
    height, width = H_current.shape
    png_bytes = _save_all_layers_to_disk(project_id, full_layers, H_current)

    meta = _default_meta(
        bg_removal=bg_removal,
        hd_restore=hd_restore,
        target_res=target_res,
        bit_depth=bit_depth,
        original_relpath=orig_rel,
        groove_sigma=float(full_layers["groove_sigma"]),
        polarity=polarity,
    )
    _write_meta(project_id, meta)

    project = db.create_project(
        project_id=project_id,
        name=name,
        original_filename=original_filename,
        width=width,
        height=height,
        ai_mode=normalized_ai,
        params=params,
    )

    preview_layers = _build_preview_layers(full_layers, PREVIEW_MAX_DIM)
    _set_brush_delta_layers(full_layers, preview_layers, None)
    # Preview mirrors the authoritative full-res relief (what you see is what you export).
    H_preview = _downsample_float32(H_current, PREVIEW_MAX_DIM)

    lock = _get_project_lock(project_id)
    with lock:
        evicted = _put_ram_cache_entry(
            project_id,
            ProjectMemoryCache(
                full_layers=full_layers,
                preview_layers=preview_layers,
                H_current=H_current,
                H_preview=H_preview,
                revision=1,
                cached_png_rev=1,
                cached_png_bytes=png_bytes,
            ),
        )
    if evicted:
        _flush_evicted_caches(evicted)

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    unique_levels = _count_unique_16bit_levels(H_preview)
    result = _build_project_payload(project, meta)
    result["compute_ms"] = elapsed_ms
    result["unique_levels"] = unique_levels
    return result


def _seed_defaults_if_empty() -> None:
    if not db.list_presets():
        base_params = db.default_relief_params()  # the defaults: the Sculpt style, as new projects
        db.create_preset(name="Minimalist Line-Lock (Default)", params=base_params)
        deep_params = db.merge_relief_params(  # Classic weights: a Classic preset
            {"style": "classic", "w_global": 0.60, "w_local": 0.80, "w_groove": 0.45, "gamma": 0.78, "relief_z_mm": 22.0}
        )
        db.create_preset(name="Deep Architectural Cushion", params=deep_params)
        framed_params = db.merge_relief_params(  # its name is the Classic style's: a Classic preset
            {
                "style": "classic",
                "frame": {
                    "enabled": True,
                    "preset": "gallery_classic",
                    "width_mm": 22.0,
                    "thickness_mm": 5.5,
                    "back_protrusion_mm": 0.0,
                    "bbox": [0.16, 0.48, 0.84, 0.96],
                }
            }
        )
        db.create_preset(name="Framed Gallery Classic", params=framed_params)

    if not db.list_projects() and BENCHMARK_IMAGE_PATH.is_file():
        seed_res = int(os.environ.get("MONOLITH_SEED_RES", "2048"))
        _create_project_from_source(
            raw_bytes=BENCHMARK_IMAGE_PATH.read_bytes(),
            original_filename=BENCHMARK_IMAGE_PATH.name,
            name="Geometric Study",
            target_res=seed_res,
            bg_removal=True,
            hd_restore=True,
            ai_mode="off",
            bit_depth="16bit",
        )


# ---------------------------------------------------------------------------
# FastAPI Lifespan & App
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.init_db()
    _seed_defaults_if_empty()
    try:
        yield
    finally:
        with _CACHE_MUTEX:
            pids = list(_RAM_CACHE.keys())
        for pid in pids:
            _await_pending_flush(pid)


app = FastAPI(
    title="MONOLITH Relief Studio API",
    version="1.0.0",
    lifespan=lifespan,
)

# The UI talks to the API through the Vite proxy (same origin), so cross-origin access is
# only needed for local tooling. Never allow arbitrary websites to drive the vault.
_CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "MONOLITH_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173",
    ).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Width",
        "X-Height",
        "X-Compute-Ms",
        "X-Unique-Levels",
        "X-Mesh-Mode",
        "X-Triangles",
        "X-Max-Error-Mm",
        "Content-Disposition",
    ],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health_check() -> dict[str, Any]:
    db.init_db()
    with _CACHE_MUTEX:
        cached_projects = len(_RAM_CACHE)
    return {
        "status": "ok",
        "service": "monolith-relief-studio",
        "projects_count": len(db.list_projects()),
        "cached_projects": cached_projects,
    }


@app.get("/api/projects")
def get_projects() -> list[dict[str, Any]]:
    return [_build_project_payload(proj) for proj in db.list_projects()]


@app.get("/api/projects/{project_id}")
def get_project_detail(project_id: str) -> dict[str, Any]:
    proj = db.get_project(project_id)
    if proj is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")
    return _build_project_payload(proj)


@app.post("/api/projects/create")
async def create_project_endpoint(
    file: UploadFile | None = File(default=None),
    use_benchmark: bool = Form(default=False),
    name: str | None = Form(default=None),
    target_res: int = Form(default=4096),
    bg_removal: bool = Form(default=True),
    hd_restore: bool = Form(default=True),
    ai_mode: str = Form(default="off"),
    bit_depth: str = Form(default="16bit"),
    polarity: str = Form(default="auto"),
    style: str = Form(default="sculpt"),
) -> dict[str, Any]:
    if target_res < 128 or target_res > 8192:
        raise HTTPException(status_code=400, detail="target_res must be between 128 and 8192")
    if style not in ("sculpt", "classic"):
        raise HTTPException(status_code=400, detail="style must be 'sculpt' or 'classic'")

    if use_benchmark or file is None:
        if not use_benchmark:
            raise HTTPException(status_code=400, detail="Upload an image file or set use_benchmark=true")
        if not BENCHMARK_IMAGE_PATH.is_file():
            raise HTTPException(status_code=404, detail="Benchmark image not found")
        raw_bytes = BENCHMARK_IMAGE_PATH.read_bytes()
        original_filename = BENCHMARK_IMAGE_PATH.name
        project_name = name or "Geometric Study"
    else:
        raw_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        if len(raw_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
            )
        original_filename = file.filename or "upload.png"
        project_name = name or Path(original_filename).stem or "Untitled Relief"

    # Multi-second CPU work: run in the threadpool so the event loop keeps serving
    # other requests (health checks, texture fetches) meanwhile.
    return await run_in_threadpool(
        _create_project_from_source,
        raw_bytes=raw_bytes,
        original_filename=original_filename,
        name=project_name,
        target_res=target_res,
        bg_removal=bg_removal,
        hd_restore=hd_restore,
        ai_mode=ai_mode,
        bit_depth=bit_depth,
        polarity=polarity,
        style=style,
    )


@app.post("/api/projects/{project_id}/preprocess")
def preprocess_project_endpoint(project_id: str, body: PreprocessRequest) -> dict[str, Any]:
    proj = db.get_project(project_id)
    if proj is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")
    if body.target_res < 128 or body.target_res > 8192:
        raise HTTPException(status_code=400, detail="target_res must be between 128 and 8192")

    orig_path = _find_original_image_path(project_id)
    if orig_path is None:
        raise HTTPException(status_code=404, detail="Original image missing on disk")

    _wait_if_evicting(project_id)
    _await_pending_flush(project_id, flush_now=False)
    # Hold the project lock for the whole rebuild: slider recomputes that arrive meanwhile
    # queue up and then apply to the new layers instead of racing the old cache.
    lock = _get_project_lock(project_id)
    with lock:
        t0 = time.perf_counter()
        normalized_ai = _normalize_ai_mode(body.ai_mode)
        meta = _read_meta(project_id)
        polarity = _normalize_polarity(body.polarity or str(meta.get("polarity", "auto")))
        latest_proj = db.get_project(project_id) or proj
        params = latest_proj["params"]

        prior_brush = _current_brush_delta(project_id)
        full_layers, H_current = _build_layers_from_image(
            orig_path,
            target_res=body.target_res,
            bg_removal=body.bg_removal,
            hd_restore=body.hd_restore,
            ai_mode=normalized_ai,
            params=params,
            polarity=polarity,
            meta=meta,
        )
        brush_path = _project_cache_dir(project_id) / "H_brush_delta.npy"
        # Keep the artist's Local Edit sculpting: resample it onto the new grid (same
        # drawing, so same aspect) instead of silently discarding it.
        _set_brush_delta_layers(full_layers, None, _resample_brush_delta(prior_brush, H_current.shape))
        if full_layers.get("_has_brush_delta", False):
            H_current = np.ascontiguousarray(
                recompute_relief(
                    full_layers,
                    params,
                    physical_width_mm=float(params.get("physical_width_mm", 1000.0)),
                    relief_z_mm=float(params.get("relief_z_mm", 15.0)),
                ),
                dtype=np.float32,
            )
            _atomic_write_npy(brush_path, full_layers["H_brush_delta"])
        else:
            brush_path.unlink(missing_ok=True)
        height, width = H_current.shape
        png_bytes = _save_all_layers_to_disk(project_id, full_layers, H_current)

        meta["polarity"] = polarity
        meta["bg_removal"] = bool(body.bg_removal)
        meta["hd_restore"] = bool(body.hd_restore)
        meta["target_res"] = int(body.target_res)
        meta["groove_sigma"] = float(full_layers["groove_sigma"])
        _write_meta(project_id, meta)

        updated = db.update_project(
            project_id,
            width=width,
            height=height,
            ai_mode=normalized_ai,
        )
        assert updated is not None

        preview_layers = _build_preview_layers(full_layers, PREVIEW_MAX_DIM)
        H_preview = _downsample_float32(H_current, PREVIEW_MAX_DIM)

        with _CACHE_MUTEX:
            prev_entry = _RAM_CACHE.get(project_id)
            prev_rev = prev_entry.revision if prev_entry is not None else 0
        rev = prev_rev + 1
        evicted = _put_ram_cache_entry(
            project_id,
            ProjectMemoryCache(
                full_layers=full_layers,
                preview_layers=preview_layers,
                H_current=H_current,
                H_preview=H_preview,
                revision=rev,
                cached_png_rev=rev,
                cached_png_bytes=png_bytes,
            ),
        )
    if evicted:
        _flush_evicted_caches(evicted)

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    payload = _build_project_payload(updated, meta)
    payload["compute_ms"] = elapsed_ms
    payload["unique_levels"] = _count_unique_16bit_levels(H_preview)
    return payload


@app.post("/api/projects/{project_id}/upscale")
def upscale_project_endpoint(project_id: str, body: UpscaleRequest) -> dict[str, Any]:
    proj = db.get_project(project_id)
    if proj is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")
    if body.target_res < 128 or body.target_res > 8192:
        raise HTTPException(status_code=400, detail="target_res must be between 128 and 8192")

    meta = _read_meta(project_id)
    return preprocess_project_endpoint(
        project_id,
        PreprocessRequest(
            bg_removal=bool(meta.get("bg_removal", True)),
            hd_restore=bool(meta.get("hd_restore", True)),
            target_res=int(body.target_res),
            ai_mode=proj["ai_mode"],
        ),
    )


def _merge_params(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, val in updates.items():
        if key == "frame" and isinstance(val, dict):
            frame_merged = dict(merged.get("frame", {}))
            frame_merged.update(val)
            merged["frame"] = frame_merged
        else:
            merged[key] = val
    return merged


def _validate_params(raw: dict[str, Any]) -> ReliefParams:
    try:
        return ReliefParams(**raw)
    except ValidationError as exc:
        # Surface as a regular 422 instead of an unhandled 500.
        raise RequestValidationError(exc.errors(include_url=False)) from exc


def _switch_style(project_id: str, cache: ProjectMemoryCache, params: dict[str, Any], ai_mode: str) -> None:
    """Swap the cache to the layers of ``params``' style (caller holds the project lock).

    The other style's basis is built once from the cached working arrays (seconds at 4K), then
    replaces the old style's arrays in RAM and on disk. The brush delta, the working arrays and
    the served textures carry over unchanged. On failure the RAM entry is dropped, so the next
    access reloads the still-saved style from disk.
    """
    old = cache.full_layers
    full: dict[str, Any] = {k: old[k] for k in ("M_sil", "d_sil", "L_clean", "groove_sigma", "_groove_scale") if k in old}
    brush = old.get("H_brush_delta") if old.get("_has_brush_delta", False) else None
    meta = _read_meta(project_id)
    try:
        if _params_style(params) == "sculpt":
            _attach_sculpt_layers(project_id, full, meta, save=False)
        else:
            full.update(_classic_layers(_preprocessed_from_layers(full), ai_mode=ai_mode, params=params))
        _set_brush_delta_layers(full, None, brush)
        preview = _build_preview_layers(full, PREVIEW_MAX_DIM)
        _set_brush_delta_layers(full, preview, brush)
        _save_layers_to_disk(project_id, full, common=False)
    except BaseException:
        with _CACHE_MUTEX:
            if _RAM_CACHE.get(project_id) is cache:
                _RAM_CACHE.pop(project_id, None)
        raise
    if _layers_style(full) == "classic":
        meta["groove_sigma"] = float(full["groove_sigma"])  # the pixel sigma of the new G_lines.npy
        _write_meta(project_id, meta)
    cache.full_layers = full
    cache.preview_layers = preview
    # a pending reorder's rebuild and save belonged to the replaced basis (its disk copy is gone)
    cache.sculpt_pending = None
    cache.sculpt_unsaved_basis = None
    cache.sculpt_gen += 1


@app.post("/api/projects/{project_id}/recompute")
def recompute_project_endpoint(
    project_id: str,
    body: dict[str, Any],
    preview_only: bool = Query(default=False),
    seq: float | None = Query(
        default=None,
        description="Client-side monotonic request stamp (e.g. epoch ms). Requests older than "
        "the newest one already applied are ignored and answered with stale=true.",
    ),
) -> dict[str, Any]:
    first = db.get_project(project_id)
    if first is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")
    # Refuse an unknown Sculpt version before loading (or building) any layers.
    _check_sculpt_version(_merge_params(first["params"], body))

    evicted: list[tuple[str, ProjectMemoryCache]] = []
    try:
        return _recompute_locked(project_id, body, preview_only, seq, evicted)
    finally:
        # Outside the project lock, and also when the recompute after a style switch failed:
        # the evicting events registered by `_put_ram_cache_entry` must always be set.
        if evicted:
            _flush_evicted_caches(evicted)


def _recompute_locked(
    project_id: str,
    body: dict[str, Any],
    preview_only: bool,
    seq: float | None,
    evicted: list[tuple[str, ProjectMemoryCache]],
) -> dict[str, Any]:
    """The body of /recompute under the project lock; projects evicted by a style switch's RAM
    budget check are appended to ``evicted`` for the caller to flush."""
    with _locked_cache(project_id) as cache:
        # Read, merge, compute and persist under one lock so concurrent slider requests
        # (which the threadpool may run out of order) can never leave the vault holding
        # older params than the cached relief.
        proj = db.get_project(project_id)
        if proj is None:
            raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")

        if seq is not None and cache.last_seq is not None and seq < cache.last_seq:
            fh, fw = cache.H_current.shape
            return {
                "id": project_id,
                "stale": True,
                "compute_ms": 0.0,
                "unique_levels": _count_unique_16bit_levels(cache.H_preview),
                "width": int(fw),
                "height": int(fh),
                "params": proj["params"],
                "sculpt_edge_clamped": bool(cache.full_layers.get("sculpt_edge_clamped", False)),
            }

        validated = _validate_params(_merge_params(proj["params"], body))
        params_dict = validated.model_dump()
        _check_sculpt_version(params_dict)
        if _layers_style(cache.full_layers) != validated.style:
            # Style switch: build the other style's basis once (seconds), drop the old one.
            _switch_style(project_id, cache, params_dict, proj["ai_mode"])
            # Saved at once: every other path recomputes from the saved params, whose style
            # must match the cached layers even if this recompute fails below.
            db.update_project(project_id, params=params_dict)
            evicted.extend(_put_ram_cache_entry(project_id, cache))  # re-check the RAM budget

        t0 = time.perf_counter()  # compute time only (excludes vault I/O)
        # A reorder's full rebuild is still pending (its background job is debounced or running):
        # rebuilding the live full basis here would hold the lock for seconds at 4K-8K and make
        # the next click wait behind it. Answer from the preview basis instead, as a preview
        # recompute does, and leave the full relief dirty: the job (re-armed below if idle)
        # re-reads these params, and its revision guard refuses a full relief computed before
        # this request; every full-relief reader still rebuilds in place first.
        pending_reorder = not preview_only and cache.sculpt_pending is not None
        if not preview_only and not pending_reorder:
            H_full = recompute_relief(
                cache.full_layers,
                params_dict,
                physical_width_mm=validated.physical_width_mm,
                relief_z_mm=validated.relief_z_mm,
            )
            H_full = np.ascontiguousarray(H_full, dtype=np.float32)
            cache.H_current = H_full
            cache.full_dirty = False
            ph, pw = cache.H_preview.shape
            fh, fw = H_full.shape
            if (ph, pw) == (fh, fw):
                H_preview = H_full
            else:
                H_preview = np.ascontiguousarray(
                    cv2.resize(H_full, (pw, ph), interpolation=cv2.INTER_AREA),
                    dtype=np.float32,
                )
        else:
            _sync_sculpt_preview_order(cache)  # after a reorder whose background job has not run yet
            H_preview = recompute_relief(
                cache.preview_layers,
                params_dict,
                physical_width_mm=validated.physical_width_mm,
                relief_z_mm=validated.relief_z_mm,
            )
            H_preview = np.ascontiguousarray(H_preview, dtype=np.float32)
            H_full = cache.H_current
            cache.full_dirty = True
            if pending_reorder and cache.sculpt_timer is None and (cache.sculpt_job is None or cache.sculpt_job.done()):
                _schedule_sculpt_job(project_id, cache)
        used = cache.full_layers if not preview_only and not pending_reorder else cache.preview_layers
        # Sculpt: the draft clamp raised Edge roundness / crease width (design decision 3).
        edge_clamped = validated.style == "sculpt" and bool(used.get("sculpt_edge_clamped", False))

        cache.H_preview = H_preview
        cache.revision += 1
        rev = cache.revision
        height, width = H_full.shape
        unique_levels = _count_unique_16bit_levels(H_preview)

        if not preview_only and not pending_reorder:
            if max(height, width) <= PREVIEW_MAX_DIM:
                png_bytes = _save_current_to_disk(project_id, H_full)
                cache.cached_png_rev = rev
                cache.cached_png_bytes = png_bytes
            else:
                _schedule_current_flush(project_id, cache, rev, H_full)

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        db.update_project(project_id, params=params_dict)
        if seq is not None:
            cache.last_seq = seq

    return {
        "id": project_id,
        "stale": False,
        "compute_ms": elapsed_ms,
        "unique_levels": unique_levels,
        "width": int(width),
        "height": int(height),
        "params": params_dict,
        "sculpt_edge_clamped": edge_clamped,
    }


def _sculpt_project(project_id: str) -> dict[str, Any]:
    """The project, or 404 when missing and 409 when it is not in the Sculpt style."""
    proj = db.get_project(project_id)
    if proj is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")
    if _params_style(proj["params"]) != "sculpt":
        raise HTTPException(status_code=409, detail="Parts and their order exist only in the Sculpt style")
    return proj


def _parts_payload(project_id: str, cache: ProjectMemoryCache, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    full = cache.full_layers
    if _layers_style(full) != "sculpt":
        raise HTTPException(status_code=409, detail="Parts and their order exist only in the Sculpt style")
    overrides = sculpt_overrides.stored(meta if meta is not None else _read_meta(project_id))
    return sculpt_overrides.parts_payload(full["sculpt_parts"], _sculpt_layer(cache), overrides, cache.H_preview.shape)


def _cold_parts_payload(project_id: str, project: dict[str, Any]) -> dict[str, Any]:
    """The parts payload of a project that is not in RAM, without loading it: no render and no
    disk write. The analysis comes from cache/parts.npz (or is analysed from the original image
    and not saved), the order is solved from the overrides in meta.json, which is the order the
    project's basis is built with, and the label map has the preview size it would load with."""
    meta = _read_meta(project_id)
    pa, _fresh = _project_parts(project_id, meta)
    overrides = sculpt_overrides.stored(meta)
    shape = _preview_shape(int(project["height"]), int(project["width"]))
    return sculpt_overrides.parts_payload(pa, sculpt_overrides.solve(pa, overrides), overrides, shape)


@app.get("/api/projects/{project_id}/parts")
def get_parts_endpoint(project_id: str) -> dict[str, Any]:
    """The Sculpt part-label map at the preview size and each part's layer and override.

    Read-only: a project not in RAM is answered from its stored analysis and overrides instead of
    being loaded (loading renders and rewrites H_current / depth_16bit.png). 409 for Classic and,
    like every other endpoint, for a Sculpt version this build does not have.
    """
    project = _sculpt_project(project_id)
    _check_sculpt_version(project["params"])
    _wait_if_evicting(project_id)
    with _get_project_lock(project_id):
        cache = _touch_ram_cache(project_id)
        if cache is not None:
            return _parts_payload(project_id, cache)
        return _cold_parts_payload(project_id, project)


@app.post("/api/projects/{project_id}/parts/order")
def set_part_order_endpoint(project_id: str, body: PartOrderRequest) -> dict[str, Any]:
    """Click-to-fix: bring the part under (u, v) forward, send it back, or reset overrides."""
    _check_sculpt_version(_sculpt_project(project_id)["params"])
    with _locked_cache(project_id) as cache:
        t0 = time.perf_counter()
        # Re-read under the lock: a concurrent /recompute may have changed style or params.
        proj = _sculpt_project(project_id)
        params = proj["params"]
        _check_sculpt_version(params)
        if _layers_style(cache.full_layers) != "sculpt":
            raise HTTPException(status_code=409, detail="Parts and their order exist only in the Sculpt style")
        pa = cache.full_layers["sculpt_parts"]
        part = 0
        if body.action != "reset_all":
            part = sculpt_overrides.click_part(pa, body.u, body.v)
            if part <= 0:
                raise HTTPException(status_code=404, detail="No part at the clicked point")
        meta = _read_meta(project_id)
        old = sculpt_overrides.stored(meta)
        new = sculpt_overrides.apply_action(pa, old, part, body.action)
        if new != old:  # else a repeat: nothing changes
            layer = sculpt_overrides.solve(pa, new)
            if np.array_equal(layer, _sculpt_layer(cache)):
                meta[sculpt_overrides.META_KEY] = new  # same order, same relief: no rebuild
                _write_meta(project_id, meta)
            else:
                _apply_sculpt_order(project_id, cache, params, meta, new, layer, part)
        payload = _parts_payload(project_id, cache, meta)
        payload["compute_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    return payload


@app.get("/api/projects/{project_id}/float32-texture")
def get_float32_texture(
    project_id: str,
    max_dim: int = Query(default=1024, ge=32, le=8192),
) -> Response:
    t0 = time.perf_counter()
    _require_known_sculpt_version(project_id)
    with _locked_cache(project_id) as cache:
        ph, pw = cache.H_preview.shape
        fh, fw = cache.H_current.shape
        if max_dim == PREVIEW_MAX_DIM or (max_dim >= max(fh, fw) and (ph, pw) == (fh, fw)):
            tex = cache.H_preview if max_dim == PREVIEW_MAX_DIM else _ensure_full_current(project_id, cache)
        elif max_dim >= max(fh, fw):
            tex = _ensure_full_current(project_id, cache)
        elif max_dim < max(ph, pw):
            tex = _downsample_float32(cache.H_preview, max_dim)
        else:
            tex = _downsample_float32(_ensure_full_current(project_id, cache), max_dim)

        tex = np.ascontiguousarray(tex, dtype=np.float32)
        h, w = tex.shape
        payload = tex.tobytes()
        _remember_served_texture(cache, (h, w), tex)
        unique_levels = _count_unique_16bit_levels(tex)

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return Response(
        content=payload,
        media_type="application/octet-stream",
        headers={
            "X-Width": str(w),
            "X-Height": str(h),
            "X-Unique-Levels": str(unique_levels),
            "X-Compute-Ms": str(elapsed_ms),
            "Cache-Control": "no-store",
        },
    )


@app.put("/api/projects/{project_id}/float32-texture")
async def put_float32_texture(
    project_id: str,
    request: Request,
    width: int | None = Query(default=None, ge=1, le=8192),
    height: int | None = Query(default=None, ge=1, le=8192),
    x_width: int | None = Header(default=None, alias="X-Width"),
    x_height: int | None = Header(default=None, alias="X-Height"),
    mode: str = Query(default="absolute"),
) -> dict[str, Any]:
    """
    Persist Local Edit brush strokes.

    `mode=delta` (preferred): body is the float32 stroke delta (edited - pre-stroke) at
    width x height. `mode=absolute` (legacy): body is the full edited texture; the delta
    is taken against the exact texture last served at that size.
    """
    target_w = width if width is not None else x_width
    target_h = height if height is not None else x_height
    if target_w is None or target_h is None or target_w <= 0 or target_h <= 0:
        raise HTTPException(
            status_code=400,
            detail="Missing or invalid texture dimensions (provide width/height query params or X-Width/X-Height headers)",
        )

    mode = (mode or "absolute").strip().lower()
    if mode not in ("absolute", "delta"):
        raise HTTPException(status_code=400, detail=f"Unsupported brush upload mode: {mode}")

    raw_bytes = await request.body()
    expected_bytes = target_w * target_h * 4
    if len(raw_bytes) != expected_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"Float32 buffer byte length mismatch: expected {expected_bytes}, got {len(raw_bytes)}",
        )

    uploaded = np.frombuffer(raw_bytes, dtype="<f4").reshape((target_h, target_w)).astype(np.float32)
    if mode == "delta":
        uploaded = np.nan_to_num(uploaded, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(uploaded, np.float32(-1.0), np.float32(1.0), out=uploaded)
    else:
        uploaded = np.nan_to_num(uploaded, nan=0.0, posinf=1.0, neginf=0.0)
        np.clip(uploaded, np.float32(0.0), np.float32(1.0), out=uploaded)

    # Heavy full-resolution recompute + PNG encode must not block the asyncio event loop.
    return await run_in_threadpool(_apply_brush_upload, project_id, uploaded, mode)


def _apply_brush_upload(project_id: str, uploaded: np.ndarray, mode: str) -> dict[str, Any]:
    target_h, target_w = uploaded.shape
    _require_known_sculpt_version(project_id)
    _await_pending_flush(project_id, flush_now=False)
    with _locked_cache(project_id) as cache:
        H_base = _ensure_full_current(project_id, cache)
        full_h, full_w = H_base.shape
        if mode == "delta":
            delta_small = uploaded
        else:
            # Diff against the exact texture the client was given (or last uploaded) so
            # untouched pixels produce an exact zero delta. Falling back to a resample of
            # the full-res relief would bake preview-vs-full differences into the brush layer.
            base_small = cache.served_textures.get((target_h, target_w))
            if base_small is None:
                if (target_h, target_w) == (full_h, full_w):
                    base_small = H_base
                else:
                    base_small = cv2.resize(H_base, (target_w, target_h), interpolation=cv2.INTER_AREA)
            delta_small = (uploaded - base_small).astype(np.float32, copy=False)
            _remember_served_texture(cache, (target_h, target_w), uploaded)

        if (target_h, target_w) == (full_h, full_w):
            delta_full = np.ascontiguousarray(delta_small, dtype=np.float32)
        else:
            # INTER_LINEAR never overshoots, so stroke edges cannot ring around the
            # silhouette step the way bicubic upsampling does.
            delta_full = np.ascontiguousarray(
                cv2.resize(delta_small, (full_w, full_h), interpolation=cv2.INTER_LINEAR),
                dtype=np.float32,
            )

        # Rebase against the unbrushed, unframed procedural relief so H_brush_delta is
        # exactly (edited - procedural) and survives any later slider recompute.
        proj = db.get_project(project_id)
        params = proj["params"] if proj is not None else db.default_relief_params()
        phys_w = float(params.get("physical_width_mm", 1000.0))
        rel_z = float(params.get("relief_z_mm", 15.0))
        H_proc = recompute_relief(
            cache.full_layers,
            params,
            physical_width_mm=phys_w,
            relief_z_mm=rel_z,
            include_brush=False,
            include_frame=False,
        )
        existing_delta = cache.full_layers.get("H_brush_delta")
        if not isinstance(existing_delta, np.ndarray) or existing_delta.shape != (full_h, full_w):
            existing_delta = np.zeros((full_h, full_w), dtype=np.float32)
        brushed = H_proc + existing_delta
        brushed += delta_full
        np.clip(brushed, np.float32(0.0), np.float32(1.0), out=brushed)
        accum_delta = np.ascontiguousarray(brushed - H_proc, dtype=np.float32)
        del brushed, H_proc

        _set_brush_delta_layers(cache.full_layers, cache.preview_layers, accum_delta)
        cache_dir = _project_cache_dir(project_id)
        cache_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_npy(cache_dir / "H_brush_delta.npy", accum_delta)

        if cache.pending_timer is not None:
            cache.pending_timer.cancel()
            cache.pending_timer = None

        H_new = np.ascontiguousarray(
            recompute_relief(cache.full_layers, params, physical_width_mm=phys_w, relief_z_mm=rel_z),
            dtype=np.float32,
        )
        cache.H_current = H_new
        cache.full_dirty = False
        # Preview mirrors the authoritative full-res relief so later fetches show exactly
        # what will be exported.
        ph, pw = cache.H_preview.shape
        if (ph, pw) == (full_h, full_w):
            cache.H_preview = H_new
        else:
            cache.H_preview = np.ascontiguousarray(
                cv2.resize(H_new, (pw, ph), interpolation=cv2.INTER_AREA), dtype=np.float32
            )

        cache.revision += 1
        rev = cache.revision
        png_bytes = _save_current_to_disk(project_id, H_new)
        cache.cached_png_rev = rev
        cache.cached_png_bytes = png_bytes
        unique_levels = _count_unique_16bit_levels(cache.H_preview)

    db.update_project(project_id)
    return {
        "id": project_id,
        "width": int(full_w),
        "height": int(full_h),
        "unique_levels": unique_levels,
        "status": "updated",
    }


@app.post("/api/projects/{project_id}/clear-brush")
def clear_brush_endpoint(project_id: str) -> dict[str, Any]:
    """Discard all manual Local Edit brush strokes (H_brush_delta) and restore the procedural relief."""
    return _clear_brush_edits(project_id)


@app.delete("/api/projects/{project_id}/brush-edits")
def reset_brush_edits_endpoint(project_id: str) -> dict[str, Any]:
    """Backward-compatible alias of POST /clear-brush."""
    return _clear_brush_edits(project_id)


def _clear_brush_edits(project_id: str) -> dict[str, Any]:
    proj = db.get_project(project_id)
    if proj is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")
    _check_sculpt_version(proj["params"])

    _await_pending_flush(project_id, flush_now=False)

    with _locked_cache(project_id) as cache:
        if cache.pending_timer is not None:
            cache.pending_timer.cancel()
            cache.pending_timer = None

        _set_brush_delta_layers(cache.full_layers, cache.preview_layers, None)
        (_project_cache_dir(project_id) / "H_brush_delta.npy").unlink(missing_ok=True)
        _sync_sculpt_full_order(project_id, cache)  # a reorder's full rebuild, if pending

        latest_proj = db.get_project(project_id)
        params = latest_proj["params"] if latest_proj is not None else proj["params"]
        H_full = recompute_relief(
            cache.full_layers,
            params,
            physical_width_mm=float(params.get("physical_width_mm", 1000.0)),
            relief_z_mm=float(params.get("relief_z_mm", 15.0)),
        )
        H_full = np.ascontiguousarray(H_full, dtype=np.float32)

        ph, pw = cache.H_preview.shape
        fh, fw = H_full.shape
        if (ph, pw) == (fh, fw):
            H_preview = H_full
        else:
            H_preview = np.ascontiguousarray(
                cv2.resize(H_full, (pw, ph), interpolation=cv2.INTER_AREA),
                dtype=np.float32,
            )

        cache.H_current = H_full
        cache.H_preview = H_preview
        cache.full_dirty = False
        cache.revision += 1
        rev = cache.revision
        png_bytes = _save_current_to_disk(project_id, H_full)
        cache.cached_png_rev = rev
        cache.cached_png_bytes = png_bytes
        unique_levels = _count_unique_16bit_levels(cache.H_preview)

    db.update_project(project_id)
    return {
        "id": project_id,
        "status": "reset",
        "unique_levels": unique_levels,
    }


@app.get("/api/projects/{project_id}/preview-png")
def get_preview_png(project_id: str) -> Response:
    _require_known_sculpt_version(project_id)
    with _locked_cache(project_id) as cache:
        H_full = _ensure_full_current(project_id, cache)
        if cache.cached_png_bytes is not None and cache.cached_png_rev == cache.revision:
            png_bytes = cache.cached_png_bytes
        else:
            png_bytes = encode_png16(H_full)
            cache.cached_png_rev = cache.revision
            cache.cached_png_bytes = png_bytes
        h, w = H_full.shape
        unique_levels = _count_unique_16bit_levels(cache.H_preview)

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "X-Width": str(w),
            "X-Height": str(h),
            "X-Unique-Levels": str(unique_levels),
            "Cache-Control": "no-store",
        },
    )


@app.get("/api/projects/{project_id}/original-image")
def get_original_image(project_id: str) -> FileResponse:
    proj = db.get_project(project_id)
    if proj is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")
    orig_path = _find_original_image_path(project_id)
    if orig_path is None:
        raise HTTPException(status_code=404, detail="Original image not found")
    return FileResponse(
        path=orig_path,
        media_type=_detect_image_media_type(orig_path),
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/projects/{project_id}/export")
def export_project_endpoint(project_id: str, body: ExportRequest) -> Response:
    proj = db.get_project(project_id)
    if proj is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")
    _check_sculpt_version(proj["params"])

    fmt = body.format.strip().lower()
    with _locked_cache(project_id) as cache:
        H_full = _ensure_full_current(project_id, cache)
        H_export = np.ascontiguousarray(H_full.copy(), dtype=np.float32)
        # Read the saved params together with the heights they produced.
        saved = (db.get_project(project_id) or proj)["params"]
        saved_dims = {
            "width_mm": float(saved["physical_width_mm"]),
            "relief_z_mm": float(saved["relief_z_mm"]),
            "base_z_mm": float(saved["base_z_mm"]),
        }
        for key, saved_val in saved_dims.items():
            given = getattr(body, key)
            if given is not None and abs(float(given) - saved_val) > 1e-6:
                raise HTTPException(
                    status_code=409,
                    detail="Export dimensions differ from the project's saved settings; apply the settings first",
                )
        width_mm = saved_dims["width_mm"]
        relief_z_mm = saved_dims["relief_z_mm"]
        base_z_mm = saved_dims["base_z_mm"]
        if fmt in ("svg", "dxf"):
            L_clean = cache.full_layers["L_clean"]
            M_sil = cache.full_layers["M_sil"]

    safe_stem = _sanitize_filename(Path(proj["name"]).stem or proj["name"])
    t0 = time.perf_counter()
    mesh_headers: dict[str, str] = {}

    if fmt == "png16":
        payload = encode_png16(H_export)
        media_type = "image/png"
        filename = f"{safe_stem}_16bit.png"
    elif fmt == "png8":
        payload = encode_png8(H_export)
        media_type = "image/png"
        filename = f"{safe_stem}_8bit.png"
    elif fmt == "exr32":
        payload = encode_exr32(H_export)
        media_type = "application/octet-stream"
        filename = f"{safe_stem}_32bit.exr"
    elif fmt == "tiff16":
        payload = encode_tiff16(H_export)
        media_type = "image/tiff"
        filename = f"{safe_stem}_16bit.tiff"
    elif fmt in ("stl", "obj"):
        # The adaptive mesh needs a solid slab; cut-outs (no base) fall back to the grid mesh.
        if body.mesh_mode == "adaptive" and base_z_mm > 1e-4:
            try:
                mesh = adaptive_heightfield_mesh(
                    H_export, width_mm, relief_z_mm, base_z_mm, float(body.mesh_tolerance_mm)
                )
            except MeshTooLargeError as exc:
                raise HTTPException(status_code=413, detail=str(exc)) from exc
            mesh_headers = {"X-Mesh-Mode": "adaptive", "X-Max-Error-Mm": f"{mesh.max_error_mm:.4f}"}
        else:
            mesh = grid_heightfield_mesh(
                H_export,
                width_mm=width_mm,
                relief_z_mm=relief_z_mm,
                base_z_mm=base_z_mm,
                max_grid=int(body.mesh_resolution),
            )
            mesh_headers = {"X-Mesh-Mode": "grid"}
        mesh_headers["X-Triangles"] = str(int(mesh.faces.shape[0]))
        if fmt == "stl":
            payload = encode_stl(mesh)
            media_type = "application/octet-stream"
            filename = f"{safe_stem}.stl"
        else:
            payload = encode_obj(mesh)
            media_type = "text/plain"
            filename = f"{safe_stem}.obj"
    elif fmt in ("svg", "dxf"):
        # Sub-pixel silhouette footprint (Z = 0 outline) for profile / cut-out toolpaths,
        # aligned with the PNG / EXR heightmap at the same physical width.
        loops = extract_outline(
            L_clean,
            M_sil,
            physical_width_mm=width_mm,
            tolerance_mm=float(body.outline_tolerance_mm),
            sigmoid_k=HD_SIGMOID_K if bool(_read_meta(project_id).get("hd_restore", True)) else None,
        )
        if not loops:
            raise HTTPException(status_code=400, detail="No silhouette outline to export")
        if fmt == "svg":
            payload = encode_svg(loops, M_sil.shape, width_mm)
            media_type = "image/svg+xml"
            filename = f"{safe_stem}_outline.svg"
        else:
            payload = encode_dxf(loops, M_sil.shape, width_mm)
            media_type = "application/dxf"
            filename = f"{safe_stem}_outline.dxf"
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported export format: {body.format}")

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    h, w = H_export.shape
    return Response(
        content=payload,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Width": str(w),
            "X-Height": str(h),
            "X-Compute-Ms": str(elapsed_ms),
            **mesh_headers,
        },
    )


@app.delete("/api/projects/{project_id}")
def delete_project_endpoint(project_id: str) -> dict[str, str]:
    proj = db.get_project(project_id)
    if proj is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")

    _wait_if_evicting(project_id)
    _await_pending_flush(project_id, flush_now=False)
    lock = _get_project_lock(project_id)
    with lock:
        with _CACHE_MUTEX:
            _RAM_CACHE.pop(project_id, None)
        db.delete_project(project_id)
        shutil.rmtree(_project_dir(project_id), ignore_errors=True)

    return {"status": "deleted", "id": project_id}


# ---------------------------------------------------------------------------
# Presets Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/presets")
def get_presets_endpoint() -> list[dict[str, Any]]:
    db.init_db()
    return db.list_presets()


@app.post("/api/presets")
def create_preset_endpoint(body: PresetCreateRequest) -> dict[str, Any]:
    db.init_db()
    clean_name = body.name.strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="Preset name cannot be empty")
    validated = _validate_params(db.merge_relief_params(body.params))
    return db.create_preset(name=clean_name, params=validated.model_dump())


@app.delete("/api/presets/{preset_id}")
def delete_preset_endpoint(preset_id: str) -> dict[str, str]:
    db.init_db()
    if not db.delete_preset(preset_id):
        raise HTTPException(status_code=404, detail=f"Preset '{preset_id}' not found")
    return {"status": "deleted", "id": preset_id}


# ---------------------------------------------------------------------------
# Built UI (single-port packaging). Must stay LAST so API routes win.
# ---------------------------------------------------------------------------


def resolve_ui_dir() -> Path | None:
    """Locate the built UI: MONOLITH_UI_DIR, else frontend/dist, else None."""
    env = os.environ.get("MONOLITH_UI_DIR")
    if env:
        candidate = Path(env)
        if (candidate / "index.html").is_file():
            return candidate
    dist = PROJECT_ROOT / "frontend" / "dist"
    if (dist / "index.html").is_file():
        return dist
    return None


def mount_ui(target: FastAPI, ui_dir: Path) -> None:
    """Serve ``ui_dir`` from ``target``: /assets statically, SPA fallback elsewhere."""
    # Windows reads MIME types from the registry, where .js is sometimes text/plain;
    # browsers refuse to run module scripts served that way (blank page).
    mimetypes.add_type("text/javascript", ".js")
    mimetypes.add_type("text/css", ".css")
    mimetypes.add_type("image/svg+xml", ".svg")
    ui_root = ui_dir.resolve()
    index_path = ui_root / "index.html"
    assets_dir = ui_root / "assets"
    if assets_dir.is_dir():
        target.mount("/assets", StaticFiles(directory=assets_dir), name="ui-assets")

    @target.get("/{full_path:path}", include_in_schema=False)
    def serve_ui(full_path: str) -> FileResponse:
        if full_path == "api" or full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not Found")
        if full_path:
            try:
                candidate = (ui_root / full_path).resolve()
                if candidate.is_file() and candidate.is_relative_to(ui_root) and candidate != index_path:
                    return FileResponse(candidate)
            except (ValueError, OSError):
                pass  # e.g. an encoded NUL byte: fall through to the SPA index
        return FileResponse(index_path, headers={"Cache-Control": "no-cache"})


if (_ui_dir := resolve_ui_dir()) is not None:
    mount_ui(app, _ui_dir)
