"""SQLite persistence layer for MONOLITH Local Vault (projects & presets)."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from backend.pipeline.frame_generator import FRAME_UNITS_MM, migrate_legacy_frame
from backend.pipeline.relief_engine import GROOVE_REFERENCE_RES, GROOVE_SIGMA_UNITS
from backend.pipeline.sculpt_engine import FULLNESS_MIN, SCULPT_DEFAULTS, SCULPT_LATEST_VERSION

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("MONOLITH_DATA_DIR", str(PROJECT_ROOT / "data")))
PROJECTS_DIR = DATA_DIR / "projects"
DEFAULT_DB_PATH = Path(os.environ.get("MONOLITH_DB_PATH", str(DATA_DIR / "monolith_vault.db")))


def default_relief_params() -> dict[str, Any]:
    """Return the default ReliefParams dictionary matching the MONOLITH schema."""
    return {
        "w_global": 0.45,
        "w_local": 0.65,
        "w_ai": 0.25,
        "w_groove": 0.35,
        "groove_sigma": 2.5,
        "groove_sigma_units": GROOVE_SIGMA_UNITS,
        "gamma": 0.85,
        "brightness": 0.0,
        "contrast": 1.0,
        "h_min": 0.08,
        "invert_depth": False,
        "draft_angle_deg": 2.0,
        "physical_width_mm": 1000.0,
        "relief_z_mm": 15.0,
        "base_z_mm": 5.0,
        "frame": {
            "enabled": False,
            "preset": "gallery_classic",
            # True physical millimetres (see frame_generator.migrate_legacy_frame).
            "width_mm": 22.0,
            "thickness_mm": 5.5,
            "back_protrusion_mm": 0.0,
            "bbox": [0.16, 0.48, 0.84, 0.96],
            "units": FRAME_UNITS_MM,
        },
        # New projects and presets are Sculpt at the latest version (stored ones without a
        # style load as Classic: `_load_stored_params`).
        "style": "sculpt",
        "sculpt_version": SCULPT_LATEST_VERSION,
        **SCULPT_DEFAULTS,
    }


def merge_relief_params(raw_params: dict[str, Any] | None) -> dict[str, Any]:
    """Deep-merge partial relief parameters onto defaults."""
    base = default_relief_params()
    if not raw_params:
        return base
    for key, val in raw_params.items():
        if key == "frame" and isinstance(val, dict):
            frame_merged = dict(base["frame"])
            frame_merged.update(val)
            base["frame"] = frame_merged
        else:
            base[key] = val
    return base


def _load_stored_params(params_json: str, longest_px: int | None = None) -> dict[str, Any]:
    """
    Decode stored params and migrate legacy units:
      - frames without `units` (reference-plaque values) -> true millimetres;
      - groove_sigma in raw full-res pixels -> pixels at GROOVE_REFERENCE_RES, using the
        project's stored resolution (`longest_px`), so the relief is unchanged;
      - no `style` (saved before the Sculpt style existed) -> "classic", so the look of an
        existing project or preset never changes under the user;
      - `fullness` below the engine's floor -> the floor (the engine already clamps it there,
        so the relief is unchanged and the params still validate).
    Used for projects and presets alike.
    """
    raw = json.loads(params_json) or {}
    if "style" not in raw:
        raw["style"] = "classic"
    try:
        if "fullness" in raw and float(raw["fullness"]) < FULLNESS_MIN:
            raw["fullness"] = FULLNESS_MIN
    except (TypeError, ValueError):
        pass
    if raw.get("groove_sigma_units") != GROOVE_SIGMA_UNITS:
        if longest_px and "groove_sigma" in raw:
            raw["groove_sigma"] = round(
                float(raw["groove_sigma"]) * GROOVE_REFERENCE_RES / float(longest_px), 4
            )
        raw["groove_sigma_units"] = GROOVE_SIGMA_UNITS
    frame = raw.get("frame")
    if isinstance(frame, dict) and frame.get("units") != FRAME_UNITS_MM:
        defaults = default_relief_params()
        raw["frame"] = migrate_legacy_frame(
            frame,
            physical_width_mm=float(raw.get("physical_width_mm", defaults["physical_width_mm"])),
            relief_z_mm=float(raw.get("relief_z_mm", defaults["relief_z_mm"])),
            h_min=float(raw.get("h_min", defaults["h_min"])),
        )
    return merge_relief_params(raw)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_db_path(db_path: Path | str | None = None) -> Path:
    if db_path is not None:
        return Path(db_path)
    env_path = os.environ.get("MONOLITH_DB_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_DB_PATH


@contextmanager
def get_connection(db_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    """Open a configured SQLite connection with WAL mode and Row factory."""
    path = resolve_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path | str | None = None) -> None:
    """Create projects and presets tables if they do not exist."""
    with get_connection(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                ai_mode TEXT DEFAULT 'off',
                params_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS presets (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                params_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )


def _row_to_project(row: sqlite3.Row) -> dict[str, Any]:
    params = _load_stored_params(
        row["params_json"], longest_px=max(int(row["width"]), int(row["height"]))
    )
    return {
        "id": row["id"],
        "name": row["name"],
        "original_filename": row["original_filename"],
        "width": int(row["width"]),
        "height": int(row["height"]),
        "ai_mode": row["ai_mode"],
        "params": params,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _row_to_preset(row: sqlite3.Row) -> dict[str, Any]:
    params = _load_stored_params(row["params_json"])
    return {
        "id": row["id"],
        "name": row["name"],
        "params": params,
        "created_at": row["created_at"],
    }


def create_project(
    *,
    project_id: str | None = None,
    name: str,
    original_filename: str,
    width: int,
    height: int,
    ai_mode: str = "off",
    params: dict[str, Any] | None = None,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Insert a new project into the vault and return its dictionary record."""
    pid = project_id or str(uuid.uuid4())
    merged_params = merge_relief_params(params)
    now = _utc_now_iso()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO projects (
                id, name, original_filename, width, height, ai_mode, params_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pid,
                name,
                original_filename,
                int(width),
                int(height),
                ai_mode,
                json.dumps(merged_params),
                now,
                now,
            ),
        )
        row = conn.execute("SELECT * FROM projects WHERE id = ?", (pid,)).fetchone()
    assert row is not None
    return _row_to_project(row)


def list_projects(db_path: Path | str | None = None) -> list[dict[str, Any]]:
    """Return all projects ordered from most recently updated to oldest."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM projects ORDER BY updated_at DESC, created_at DESC, rowid DESC"
        ).fetchall()
    return [_row_to_project(r) for r in rows]


def get_project(project_id: str, db_path: Path | str | None = None) -> dict[str, Any] | None:
    """Fetch a single project by id, or None if not found."""
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    return _row_to_project(row) if row is not None else None


def update_project(
    project_id: str,
    *,
    name: str | None = None,
    original_filename: str | None = None,
    width: int | None = None,
    height: int | None = None,
    ai_mode: str | None = None,
    params: dict[str, Any] | None = None,
    db_path: Path | str | None = None,
) -> dict[str, Any] | None:
    """Update mutable fields of a project and refresh updated_at."""
    existing = get_project(project_id, db_path=db_path)
    if existing is None:
        return None

    new_name = name if name is not None else existing["name"]
    new_filename = original_filename if original_filename is not None else existing["original_filename"]
    new_width = int(width) if width is not None else existing["width"]
    new_height = int(height) if height is not None else existing["height"]
    new_ai_mode = ai_mode if ai_mode is not None else existing["ai_mode"]
    new_params = merge_relief_params(params) if params is not None else existing["params"]
    now = _utc_now_iso()

    with get_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE projects
            SET name = ?, original_filename = ?, width = ?, height = ?, ai_mode = ?, params_json = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                new_name,
                new_filename,
                new_width,
                new_height,
                new_ai_mode,
                json.dumps(new_params),
                now,
                project_id,
            ),
        )
        row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    return _row_to_project(row) if row is not None else None


def delete_project(project_id: str, db_path: Path | str | None = None) -> bool:
    """Delete a project record by id. Returns True if a row was removed."""
    with get_connection(db_path) as conn:
        cur = conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        return cur.rowcount > 0


def create_preset(
    *,
    name: str,
    params: dict[str, Any],
    preset_id: str | None = None,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Insert a new relief parameter preset and return its record."""
    pid = preset_id or str(uuid.uuid4())
    merged_params = merge_relief_params(params)
    now = _utc_now_iso()
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO presets (id, name, params_json, created_at) VALUES (?, ?, ?, ?)",
            (pid, name, json.dumps(merged_params), now),
        )
        row = conn.execute("SELECT * FROM presets WHERE id = ?", (pid,)).fetchone()
    assert row is not None
    return _row_to_preset(row)


def list_presets(db_path: Path | str | None = None) -> list[dict[str, Any]]:
    """Return all saved presets ordered newest first."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM presets ORDER BY created_at DESC, rowid DESC"
        ).fetchall()
    return [_row_to_preset(r) for r in rows]


def get_preset(preset_id: str, db_path: Path | str | None = None) -> dict[str, Any] | None:
    """Fetch a preset by id, or None if missing."""
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM presets WHERE id = ?", (preset_id,)).fetchone()
    return _row_to_preset(row) if row is not None else None


def delete_preset(preset_id: str, db_path: Path | str | None = None) -> bool:
    """Delete a preset by id. Returns True if removed."""
    with get_connection(db_path) as conn:
        cur = conn.execute("DELETE FROM presets WHERE id = ?", (preset_id,))
        return cur.rowcount > 0
