"""Shared pytest fixtures for MONOLITH backend tests."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend import main as api_main  # noqa: E402


@pytest.fixture()
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolate SQLite DB, filesystem vault, and RAM cache per test."""
    data_dir = tmp_path / "data"
    db_path = data_dir / "monolith_vault.db"
    monkeypatch.setenv("MONOLITH_DATA_DIR", str(data_dir))
    monkeypatch.setenv("MONOLITH_DB_PATH", str(db_path))
    monkeypatch.setenv("MONOLITH_SEED_RES", "1024")
    api_main._RAM_CACHE.clear()
    api_main._PROJECT_LOCKS.clear()
    api_main._EVICTING_EVENTS.clear()
    yield {"data_dir": data_dir, "db_path": db_path}
    api_main._RAM_CACHE.clear()
    api_main._PROJECT_LOCKS.clear()
    api_main._EVICTING_EVENTS.clear()


def make_lineart_png(size: int = 256, height: int | None = None) -> bytes:
    """White silhouette on black with an interior contour line (the artist's drawing style)."""
    h = height or size
    img = np.zeros((h, size), dtype=np.uint8)
    cv2.ellipse(img, (size // 2, h // 2), (size // 3, h // 3), 0, 0, 360, 255, -1)
    cv2.line(img, (size // 2 - size // 4, h // 2), (size // 2 + size // 4, h // 2), 0, max(2, size // 64))
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()
