"""On-disk cache of the Sculpt basis and the parts analysis (a project's ``cache/`` folder).

Files:

- ``sculpt_basis.json``: the manifest, written last. It records `SCULPT_CACHE_FORMAT`, the
  working shape, the parts key (a hash of the analysis labels the basis was built from) and the
  basis dict's structure: every ndarray is a separate ``sculpt_<path>.npy`` file (dtype and
  shape recorded), other values are JSON (ints, floats, strings, tuples, slices, nested dicts).
- ``sculpt_*.npy``: the arrays.
- ``.sculpt-staging-*/``: a basis being written off the project lock (`stage_basis`), made the
  cached one by renames (`commit_staged_basis`); leftovers of a crash are removed on load.
- ``parts.npz``: the `PartsAnalysis` (labels, line roles and owners, adjacency, anchors) with its
  own format and the preprocess settings it was analysed with.

A cache is read only when its format, shape and parts key match and every array has the recorded
dtype and shape; anything else (a missing manifest, an older format, a short write) returns None
and the caller rebuilds. Bump `SCULPT_CACHE_FORMAT` whenever `build_sculpt_basis` changes what an
array holds or its dtype: e.g. an earlier build's uint8 ink read as float16 would make every
crease 255x too deep, so such a cache must be rebuilt, never read.

The engine-internal lock, the uint8 view of the silhouette and the analysis labels (shared with
the `PartsAnalysis`) are not stored; `load_basis` recreates them.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import uuid
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from backend.pipeline.sculpt_engine import _restore_base
from backend.pipeline.sculpt_parts import ANALYSIS_RES, PartsAnalysis

SCULPT_CACHE_FORMAT = 1
PARTS_FORMAT = 1
MANIFEST = "sculpt_basis.json"
PARTS_FILE = "parts.npz"
ARRAY_PREFIX = "sculpt_"
STAGING_PREFIX = ".sculpt-staging-"  # directories inside the cache folder

# rebuilt on load, never stored
_SKIP = ("_lock", "_sil_u8", "_pa_labels", "_last_edge_clamped")
# dtypes the current engine builds (a cheap second guard next to the format number)
_EXPECTED_DTYPES = {
    "ink": np.float16,
    "d_wall_px": np.float16,
    "d_seam_px": np.float16,
    "d_open_px": np.float16,
    "d_gate_px": np.float16,
    "d_crease_px": np.float16,
    "d_crease_all_px": np.float16,
    "d_thick_px": np.float16,
    "dome": np.float32,
    "support": np.float32,
}


def parts_key(pa: PartsAnalysis) -> str:
    """Identifies the analysis a basis was built from."""
    return hashlib.sha1(np.ascontiguousarray(pa.labels, dtype=np.int32).tobytes()).hexdigest()


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        np.save(fh, np.ascontiguousarray(arr), allow_pickle=False)
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _file_name(path: str) -> str:
    return ARRAY_PREFIX + re.sub(r"[^A-Za-z0-9_.-]+", "_", path) + ".npy"


def _encode(v: Any, path: str, arrays: dict[str, np.ndarray]) -> Any:
    if isinstance(v, np.ndarray):
        name = _file_name(path)
        if name in arrays:
            raise ValueError(f"duplicate cache file name for {path!r}")
        arrays[name] = v
        return {"__npy__": name, "dtype": v.dtype.str, "shape": list(v.shape)}
    if v is None or isinstance(v, (bool, str, int, float)) and not isinstance(v, np.generic):
        return v
    if isinstance(v, np.generic):
        return {"__np__": v.dtype.str, "v": v.item()}
    if isinstance(v, slice):
        return {"__slice__": [v.start, v.stop, v.step]}
    if isinstance(v, tuple):
        return {"__tuple__": [_encode(x, f"{path}.{i}", arrays) for i, x in enumerate(v)]}
    if isinstance(v, list):
        return {"__list__": [_encode(x, f"{path}.{i}", arrays) for i, x in enumerate(v)]}
    if isinstance(v, dict):
        return {"__dict__": [[_encode(k, f"{path}.key", {}), _encode(x, f"{path}.{k}", arrays)] for k, x in v.items()]}
    raise TypeError(f"cannot cache {type(v).__name__} at {path!r}")


def _decode(v: Any, cache_dir: Path) -> Any:
    if not isinstance(v, dict):
        return v
    if "__npy__" in v:
        arr = np.load(cache_dir / v["__npy__"], allow_pickle=False)
        if arr.dtype.str != v["dtype"] or list(arr.shape) != v["shape"]:
            raise ValueError(f"{v['__npy__']}: {arr.dtype.str} {arr.shape} != {v['dtype']} {v['shape']}")
        return arr
    if "__np__" in v:
        return np.dtype(v["__np__"]).type(v["v"])
    if "__slice__" in v:
        return slice(*v["__slice__"])
    if "__tuple__" in v:
        return tuple(_decode(x, cache_dir) for x in v["__tuple__"])
    if "__list__" in v:
        return [_decode(x, cache_dir) for x in v["__list__"]]
    if "__dict__" in v:
        return {_decode(k, cache_dir): _decode(x, cache_dir) for k, x in v["__dict__"]}
    raise ValueError(f"unknown cache entry {sorted(v)}")


def clear_basis(cache_dir: Path) -> None:
    """Remove the Sculpt basis files (the manifest first, so a half-removed cache is never read)."""
    (cache_dir / MANIFEST).unlink(missing_ok=True)
    if cache_dir.is_dir():
        for f in cache_dir.glob(ARRAY_PREFIX + "*"):
            f.unlink(missing_ok=True)


def invalidate_basis(cache_dir: Path) -> None:
    """Make the cached basis unreadable at once by removing only its manifest (milliseconds; the
    arrays, ~470 MB at 4K, stay until the next `save_basis` or `clear_basis` replaces them).
    `load_basis` never reads arrays without a manifest, and `save_basis` writes it last."""
    (cache_dir / MANIFEST).unlink(missing_ok=True)


def clear_all(cache_dir: Path) -> None:
    """Remove the Sculpt basis and the parts analysis."""
    clear_basis(cache_dir)
    (cache_dir / PARTS_FILE).unlink(missing_ok=True)


def save_basis(cache_dir: Path, basis: dict[str, Any], pa: PartsAnalysis) -> None:
    """Write ``basis`` (one .npy per array, then the manifest). Its support is stored without the
    last recompute's seam corrections (each recompute rewrites them)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    _write_basis(cache_dir, basis, pa)


def stage_basis(cache_dir: Path, basis: dict[str, Any], pa: PartsAnalysis) -> Path:
    """Write ``basis`` into a new staging folder inside ``cache_dir`` (which must exist: a deleted
    project is never recreated) and return it. The cached basis is untouched until
    `commit_staged_basis`, so the slow write needs no lock."""
    staging = cache_dir / f"{STAGING_PREFIX}{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        _write_basis(staging, basis, pa)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return staging


def commit_staged_basis(staging: Path, cache_dir: Path) -> Path:
    """Make a staged basis the cached one with renames only (milliseconds). The manifest goes
    first and comes back last, so a crash in between leaves no readable cache. Returns a folder
    holding the replaced arrays, for the caller to delete outside its locks."""
    if not (staging / MANIFEST).is_file():
        raise FileNotFoundError(staging / MANIFEST)
    (cache_dir / MANIFEST).unlink(missing_ok=True)
    trash = cache_dir / f"{STAGING_PREFIX}old-{uuid.uuid4().hex}"
    trash.mkdir()
    for f in cache_dir.glob(ARRAY_PREFIX + "*"):
        if f.name != MANIFEST and f.is_file():
            os.replace(f, trash / f.name)
    for f in staging.glob(ARRAY_PREFIX + "*.npy"):
        os.replace(f, cache_dir / f.name)
    os.replace(staging / MANIFEST, cache_dir / MANIFEST)
    staging.rmdir()
    return trash


def remove_staging(cache_dir: Path) -> None:
    """Delete staging folders left by a crash (call only when no save can be running)."""
    if cache_dir.is_dir():
        for d in cache_dir.glob(STAGING_PREFIX + "*"):
            shutil.rmtree(d, ignore_errors=True)


def _write_basis(cache_dir: Path, basis: dict[str, Any], pa: PartsAnalysis) -> None:
    with basis.setdefault("_lock", threading.Lock()):
        _restore_base(basis)
        arrays: dict[str, np.ndarray] = {}
        enc = {k: _encode(v, k, arrays) for k, v in basis.items() if k not in _SKIP}
        clear_basis(cache_dir)
        for name, arr in arrays.items():
            _atomic_save_npy(cache_dir / name, arr)
    manifest = {
        "format": SCULPT_CACHE_FORMAT,
        "shape": list(basis["_shape"]),
        "parts_key": parts_key(pa),
        "basis": enc,
    }
    _atomic_write_text(cache_dir / MANIFEST, json.dumps(manifest))


def load_basis(cache_dir: Path, pa: PartsAnalysis, shape: tuple[int, int]) -> dict[str, Any] | None:
    """The cached basis for ``pa`` on the working grid ``shape``, or None when absent or stale."""
    path = cache_dir / MANIFEST
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("format") != SCULPT_CACHE_FORMAT:
            return None
        if tuple(manifest.get("shape", ())) != tuple(int(s) for s in shape):
            return None
        if manifest.get("parts_key") != parts_key(pa):
            return None
        basis = {k: _decode(v, cache_dir) for k, v in manifest["basis"].items()}
    except (OSError, ValueError, KeyError, TypeError, EOFError):  # EOFError: a zero-length .npy
        return None
    for key, dt in _EXPECTED_DTYPES.items():
        arr = basis.get(key)
        if not isinstance(arr, np.ndarray) or arr.dtype != dt or arr.shape != tuple(shape):
            return None
    if not isinstance(basis.get("_sil"), np.ndarray) or basis["_sil"].dtype != np.bool_:
        return None
    basis["_shape"] = tuple(basis["_shape"])
    basis["_sil_u8"] = basis["_sil"].view(np.uint8)
    basis["_pa_labels"] = pa.labels
    basis["_lock"] = threading.Lock()
    return basis


def _source_stamp(source: dict[str, Any]) -> str:
    return json.dumps({"analysis_res": ANALYSIS_RES, **source}, sort_keys=True)


def save_parts(cache_dir: Path, pa: PartsAnalysis, source: dict[str, Any]) -> None:
    """Write ``parts.npz``; ``source`` is the preprocess settings the analysis used."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    adj = sorted(pa.adjacency.items())
    tmp = cache_dir / (PARTS_FILE + ".tmp")
    with tmp.open("wb") as fh:
        np.savez(
            fh,
            format=np.array(PARTS_FORMAT, np.int64),
            source=np.array(_source_stamp(source)),
            labels=np.ascontiguousarray(pa.labels),
            line_role=np.ascontiguousarray(pa.line_role),
            line_owner=np.ascontiguousarray(pa.line_owner),
            adjacency_pairs=np.array([k for k, _ in adj], np.int64).reshape(-1, 2),
            adjacency_len=np.array([v for _, v in adj], np.int64),
            anchors=np.ascontiguousarray(pa.anchors),
            shape=np.array(pa.shape, np.int64),
        )
    os.replace(tmp, cache_dir / PARTS_FILE)


def load_parts(cache_dir: Path, source: dict[str, Any]) -> PartsAnalysis | None:
    """The cached analysis if it was made with the same ``source`` settings, else None."""
    path = cache_dir / PARTS_FILE
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            if int(z["format"]) != PARTS_FORMAT or str(z["source"]) != _source_stamp(source):
                return None
            pairs, lens = z["adjacency_pairs"], z["adjacency_len"]
            return PartsAnalysis(
                labels=z["labels"],
                line_role=z["line_role"],
                line_owner=z["line_owner"],
                adjacency={(int(a), int(b)): int(n) for (a, b), n in zip(pairs.tolist(), lens.tolist())},
                anchors=z["anchors"],
                shape=(int(z["shape"][0]), int(z["shape"][1])),
            )
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        # a damaged file (truncated, zero-length, not a zip) is a cache miss: analysed again
        return None
