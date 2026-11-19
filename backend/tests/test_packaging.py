"""Tests for single-port packaging: UI serving, reload recompute, and the Windows bundle layout."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import main
from backend import main as api_main
from backend.tests.test_reliability import _create, _get_tex, _put


def _make_ui(tmp_path):
    ui = tmp_path / "ui"
    (ui / "assets").mkdir(parents=True)
    (ui / "index.html").write_text("<html>ui</html>", encoding="utf-8")
    (ui / "assets" / "app.js").write_text("js", encoding="utf-8")
    return ui


def test_spa_fallback_and_api_404(tmp_path):
    ui = _make_ui(tmp_path)
    test_app = FastAPI()

    @test_app.get("/api/ping")
    def ping():
        return {"ok": True}

    main.mount_ui(test_app, ui)
    client = TestClient(test_app)

    r = client.get("/")
    assert r.status_code == 200 and r.text == "<html>ui</html>"
    assert r.headers["cache-control"] == "no-cache"

    r = client.get("/assets/app.js")
    assert r.status_code == 200 and r.text == "js"

    r = client.get("/editor/deep")
    assert r.status_code == 200 and r.text == "<html>ui</html>"

    r = client.get("/api/ping")
    assert r.status_code == 200 and r.json() == {"ok": True}

    r = client.get("/api/nope")
    assert r.status_code == 404
    assert "detail" in r.json()


def test_path_traversal_is_not_served(tmp_path):
    ui = _make_ui(tmp_path)
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    test_app = FastAPI()
    main.mount_ui(test_app, ui)
    client = TestClient(test_app)
    for path in ("/..%2fsecret.txt", "/%00"):
        r = client.get(path)
        assert r.status_code == 200
        assert "secret" not in r.text
        assert r.text == "<html>ui</html>"


def test_resolve_ui_dir_env_and_missing(tmp_path, monkeypatch):
    ui = _make_ui(tmp_path)
    monkeypatch.setenv("MONOLITH_UI_DIR", str(ui))
    assert main.resolve_ui_dir() == ui

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("MONOLITH_UI_DIR", str(empty))
    dist = main.PROJECT_ROOT / "frontend" / "dist"
    expected = dist if (dist / "index.html").is_file() else None
    assert main.resolve_ui_dir() == expected

    monkeypatch.delenv("MONOLITH_UI_DIR")
    assert main.resolve_ui_dir() == expected


def test_physical_ram_bytes_positive():
    assert (api_main._physical_ram_bytes() or 0) > 0


def test_reload_recomputes_stale_current(isolated_env):
    with TestClient(api_main.app) as client:
        pid = _create(client, res=256)
        r = client.post(f"/api/projects/{pid}/recompute", json={"gamma": 0.7})
        assert r.status_code == 200, r.text
        w, h, _ = _get_tex(client, pid, max_dim=8192)
        delta = np.zeros((h, w), np.float32)
        delta[h // 2 - 20 : h // 2 + 20, w // 2 - 20 : w // 2 + 20] = 0.05
        assert _put(client, pid, w, h, delta, mode="delta").status_code == 200
        _, _, ref = _get_tex(client, pid, max_dim=8192)

        # Simulate a hard kill: RAM gone, H_current.npy stale (params/brush already on disk).
        api_main._RAM_CACHE.clear()
        cur = api_main._project_cache_dir(pid) / "H_current.npy"
        np.save(cur, np.zeros_like(ref))

        _, _, got = _get_tex(client, pid, max_dim=8192)
        np.testing.assert_allclose(got, ref, atol=1e-6)
        # The preview PNG must match the recomputed relief, not the stale one.
        png = api_main._RAM_CACHE[pid].cached_png_bytes
        assert png == api_main.encode_png16(api_main._RAM_CACHE[pid].H_current)


def test_bundle_layout_skip_python(tmp_path):
    root = Path(__file__).resolve().parents[2]
    if not (root / "frontend" / "dist" / "index.html").is_file():
        pytest.skip("frontend/dist not built")
    script = root / "packaging" / "windows" / "build_bundle.py"
    subprocess.run(
        [sys.executable, str(script), "--out", str(tmp_path), "--version", "abc1234 2026-11-14", "--skip-python"],
        check=True,
        cwd=root,
    )
    bundle = tmp_path / "MONOLITH"
    for rel in (
        "Start MONOLITH.bat",
        "Open MONOLITH data folder.bat",
        "READ ME FIRST.txt",
        "app/backend/main.py",
        "app/backend/launcher.py",
        "app/benchmarks/geometric-study.png",
        "app/ui/index.html",
        "app/VERSION.txt",
    ):
        assert (bundle / rel).is_file(), rel
    assert (bundle / "LICENSE").read_bytes() == (root / "LICENSE").read_bytes()
    samples = bundle / "app" / "benchmarks"
    assert {p.name for p in samples.iterdir()} == {"geometric-study.png"}
    assert not (bundle / "app" / "backend" / "tests").exists()
    assert not list((bundle / "app").rglob("__pycache__"))
    assert (bundle / "app" / "VERSION.txt").read_text(encoding="utf-8").strip() == "abc1234 2026-11-14"
    assert b"\r\n" in (bundle / "Start MONOLITH.bat").read_bytes()
    zpath = tmp_path / "MONOLITH-20261114-abc1234-windows.zip"
    assert zpath.is_file()
    with zipfile.ZipFile(zpath) as zf:
        assert "MONOLITH/Start MONOLITH.bat" in zf.namelist()


def test_mount_ui_forces_web_mime_types(tmp_path):
    import mimetypes

    mimetypes.add_type("text/plain", ".js")  # what a bad Windows registry can do
    mimetypes.add_type("text/plain", ".css")
    ui = _make_ui(tmp_path)
    (ui / "assets" / "app.css").write_text("a{}", encoding="utf-8")
    (ui / "assets" / "logo.svg").write_text("<svg/>", encoding="utf-8")
    test_app = FastAPI()
    main.mount_ui(test_app, ui)
    client = TestClient(test_app)
    assert "javascript" in client.get("/assets/app.js").headers["content-type"]
    assert "text/css" in client.get("/assets/app.css").headers["content-type"]
    assert "image/svg+xml" in client.get("/assets/logo.svg").headers["content-type"]
