"""Sculpt style through the API: params, migration, the cached basis and the style switch."""

from __future__ import annotations

import io
import json
import math
import sqlite3
import threading

import cv2
import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend import db
from backend import main as api_main
from backend.pipeline import sculpt_cache
from backend.pipeline.sculpt_engine import FULLNESS_MIN, SCULPT_DEFAULTS, SCULPT_LATEST_VERSION
from backend.tests.test_sculpt_engine import EXAMPLE, _figure, _interior_slope_excess

SCULPT_KEYS = ("style", "sculpt_version", *SCULPT_DEFAULTS)


def _figure_png(n: int = 512) -> bytes:
    """White parts on black: a torso, a head and an arm in front of it, two open strokes."""
    white, _sil = _figure(n)
    ok, buf = cv2.imencode(".png", (white * 255).astype(np.uint8))
    assert ok
    return buf.tobytes()


def _create(client: TestClient, res: int = 1024, style: str | None = None, data: bytes | None = None, name="f.png") -> dict:
    form = {"target_res": str(res)}
    if style is not None:
        form["style"] = style
    r = client.post("/api/projects/create", files={"file": (name, data or _figure_png(), "image/png")}, data=form)
    assert r.status_code == 200, r.text
    return r.json()


def _cache(pid: str) -> api_main.ProjectMemoryCache:
    return api_main._RAM_CACHE[pid]


def _full(client: TestClient, pid: str) -> np.ndarray:
    """The authoritative full-resolution relief (materialised if a preview recompute left it dirty)."""
    r = client.get(f"/api/projects/{pid}/float32-texture?max_dim=8192")
    assert r.status_code == 200
    w, h = int(r.headers["x-width"]), int(r.headers["x-height"])
    return np.frombuffer(r.content, dtype=np.float32).reshape(h, w).copy()


def _cache_dir(env, pid: str):
    return env["data_dir"] / "projects" / pid / "cache"


# --------------------------------------------------------------------------- params and migration


def test_relief_params_defaults_match_engine():
    """The Sculpt slider defaults come from the engine (SCULPT_DEFAULTS), in the model and the vault."""
    p = api_main.ReliefParams()
    assert p.style == "sculpt" and p.sculpt_version == SCULPT_LATEST_VERSION
    assert {k: getattr(p, k) for k in SCULPT_DEFAULTS} == SCULPT_DEFAULTS
    d = db.default_relief_params()
    assert d["style"] == "sculpt" and d["sculpt_version"] == SCULPT_LATEST_VERSION
    assert {k: d[k] for k in SCULPT_DEFAULTS} == SCULPT_DEFAULTS
    assert api_main.ReliefParams(**d).model_dump() == {**d, "frame": api_main.FrameSettings(**d["frame"]).model_dump()}
    for bad in (
        {"fullness": FULLNESS_MIN - 0.01},
        {"fullness": 1.01},
        {"edge_roundness_pct": 0.1},
        {"edge_roundness_pct": 15.5},
        {"crease_depth": -0.1},
        {"crease_depth": 0.61},
        {"crease_width_pct": 0.05},
        {"crease_width_pct": 5.5},
        {"layering": -0.1},
        {"layering": 1.1},
        {"style": "photo"},
        {"sculpt_version": 0},
    ):
        with pytest.raises(ValidationError):
            api_main.ReliefParams(**bad)


def test_new_project_defaults_to_sculpt(isolated_env):
    with TestClient(api_main.app) as client:
        body = _create(client)
        pid = body["id"]
        assert body["params"]["style"] == "sculpt"
        assert body["params"]["sculpt_version"] == SCULPT_LATEST_VERSION
        assert {k: body["params"][k] for k in SCULPT_DEFAULTS} == SCULPT_DEFAULTS
        assert body["sculpt_latest_version"] == SCULPT_LATEST_VERSION
        assert client.get(f"/api/projects/{pid}").json()["sculpt_latest_version"] == SCULPT_LATEST_VERSION
        assert all(p["sculpt_latest_version"] == SCULPT_LATEST_VERSION for p in client.get("/api/projects").json())
        # only the Sculpt layers, in RAM and on disk (design decision 5)
        full = _cache(pid).full_layers
        assert "sculpt_basis" in full and "H_global" not in full
        cdir = _cache_dir(isolated_env, pid)
        for name in ("M_sil.npy", "d_sil.npy", "L_clean.npy", "H_current.npy", sculpt_cache.MANIFEST, sculpt_cache.PARTS_FILE):
            assert (cdir / name).is_file(), name
        assert not any((cdir / f"{k}.npy").exists() for k in api_main.CLASSIC_LAYER_KEYS)
        # a project created as Classic holds only the Classic layers
        cl = _create(client, style="classic")
        assert cl["params"]["style"] == "classic"
        assert "H_global" in _cache(cl["id"]).full_layers and "sculpt_basis" not in _cache(cl["id"]).full_layers
        assert not (_cache_dir(isolated_env, cl["id"]) / sculpt_cache.MANIFEST).exists()
        assert client.post("/api/projects/create", files={"file": ("f.png", _figure_png(), "image/png")},
                           data={"style": "photo"}).status_code == 400


def test_stored_project_without_style_loads_classic(isolated_env):
    """A project saved before the Sculpt style existed (no `style` in its params) opens as Classic
    and keeps its exact relief; a stored Fullness below the floor loads clamped instead of 422."""
    with TestClient(api_main.app) as client:
        pid = _create(client, style="classic")["id"]
        H_classic = _full(client, pid)
        legacy = {k: v for k, v in client.get(f"/api/projects/{pid}").json()["params"].items() if k not in SCULPT_KEYS}
        with sqlite3.connect(isolated_env["db_path"]) as conn:
            conn.execute("UPDATE projects SET params_json = ? WHERE id = ?", (json.dumps(legacy), pid))
        api_main._RAM_CACHE.clear()
        params = client.get(f"/api/projects/{pid}").json()["params"]
        assert params["style"] == "classic"
        assert np.array_equal(_full(client, pid), H_classic)
        r = client.post(f"/api/projects/{pid}/recompute", json={})
        assert r.status_code == 200 and r.json()["params"]["style"] == "classic"
        assert np.array_equal(_full(client, pid), H_classic)

        # an under-floor Fullness (saved before the floor) is clamped on load
        with sqlite3.connect(isolated_env["db_path"]) as conn:
            conn.execute("UPDATE projects SET params_json = ? WHERE id = ?",
                         (json.dumps({**legacy, "style": "sculpt", "fullness": 0.05}), pid))
        assert client.get(f"/api/projects/{pid}").json()["params"]["fullness"] == FULLNESS_MIN
        r = client.post(f"/api/projects/{pid}/recompute", json={"layering": 0.6})
        assert r.status_code == 200, r.text
        assert r.json()["params"]["fullness"] == FULLNESS_MIN and r.json()["params"]["style"] == "sculpt"


def test_preset_without_style_is_classic(isolated_env):
    with TestClient(api_main.app) as client:  # seeds the default presets
        old = {k: v for k, v in db.default_relief_params().items() if k not in SCULPT_KEYS}
        old["w_global"] = 0.6
        with sqlite3.connect(isolated_env["db_path"]) as conn:
            conn.execute("INSERT INTO presets (id, name, params_json, created_at) VALUES (?, ?, ?, ?)",
                         ("old-preset", "Old", json.dumps(old), "2026-01-01T00:00:00Z"))
        stored = db.get_preset("old-preset")
        assert stored["params"]["style"] == "classic" and stored["params"]["w_global"] == 0.6
        listed = {p["id"]: p for p in client.get("/api/presets").json()}
        assert listed["old-preset"]["params"]["style"] == "classic"
        # presets carry their style; one saved now keeps what it was saved with
        made = client.post("/api/presets", json={"name": "S", "params": {"style": "sculpt", "layering": 0.8}}).json()
        assert made["params"]["style"] == "sculpt" and made["params"]["layering"] == 0.8
        # each seeded preset's style matches its name: the Classic-weights and the "Classic" framed
        # one are Classic (final review M3), the default one Sculpt (the default style)
        seeded = {p["name"]: p["params"] for p in client.get("/api/presets").json()}
        assert seeded["Deep Architectural Cushion"]["style"] == "classic"
        assert seeded["Framed Gallery Classic"]["style"] == "classic"
        assert seeded["Framed Gallery Classic"]["frame"]["enabled"] is True
        assert seeded["Minimalist Line-Lock (Default)"]["style"] == "sculpt"
        assert seeded["Minimalist Line-Lock (Default)"] == db.default_relief_params()


def test_unknown_sculpt_version_is_refused(isolated_env):
    with TestClient(api_main.app) as client:
        pid = _create(client)["id"]
        r = client.post(f"/api/projects/{pid}/recompute", json={"sculpt_version": SCULPT_LATEST_VERSION + 1})
        assert r.status_code == 409 and "version" in r.json()["detail"].lower()
        r = client.post(f"/api/projects/{pid}/recompute?preview_only=true", json={"sculpt_version": SCULPT_LATEST_VERSION + 1})
        assert r.status_code == 409
        assert client.get(f"/api/projects/{pid}").json()["params"]["sculpt_version"] == SCULPT_LATEST_VERSION
        # a stored project from a newer build: refused until the version or style is changed
        p = client.get(f"/api/projects/{pid}").json()["params"]
        with sqlite3.connect(isolated_env["db_path"]) as conn:
            conn.execute("UPDATE projects SET params_json = ? WHERE id = ?",
                         (json.dumps({**p, "sculpt_version": SCULPT_LATEST_VERSION + 1}), pid))
        assert client.post(f"/api/projects/{pid}/recompute", json={"layering": 0.7}).status_code == 409
        assert client.post(f"/api/projects/{pid}/recompute", json={"style": "classic"}).status_code == 200
        r = client.post(f"/api/projects/{pid}/recompute", json={"style": "sculpt", "sculpt_version": SCULPT_LATEST_VERSION})
        assert r.status_code == 200 and r.json()["params"]["sculpt_version"] == SCULPT_LATEST_VERSION


def test_unknown_sculpt_version_blocks_every_render(isolated_env):
    """A stored project with a Sculpt version this build lacks is not rendered with another
    algorithm anywhere (texture, PNG, export, brush): 409, until /recompute changes the version."""
    with TestClient(api_main.app) as client:
        pid = _create(client, res=512)["id"]
        h, w = _cache(pid).H_preview.shape
        p = client.get(f"/api/projects/{pid}").json()["params"]
        with sqlite3.connect(isolated_env["db_path"]) as conn:
            conn.execute("UPDATE projects SET params_json = ? WHERE id = ?",
                         (json.dumps({**p, "sculpt_version": SCULPT_LATEST_VERSION + 1}), pid))
        delta = np.zeros((h, w), np.float32).tobytes()

        def calls():
            return {
                "texture": client.get(f"/api/projects/{pid}/float32-texture?max_dim=512"),
                "preview-png": client.get(f"/api/projects/{pid}/preview-png"),
                "export png16": client.post(f"/api/projects/{pid}/export", json={"format": "png16"}),
                "export stl": client.post(f"/api/projects/{pid}/export", json={"format": "stl", "mesh_mode": "grid", "mesh_resolution": 64}),
                "brush": client.put(f"/api/projects/{pid}/float32-texture?width={w}&height={h}&mode=delta", content=delta),
                "clear-brush": client.post(f"/api/projects/{pid}/clear-brush"),
            }

        for evict in (False, True):  # with the project in RAM, and cold
            if evict:
                api_main._RAM_CACHE.clear()
            for name, r in calls().items():
                assert r.status_code == 409 and "version" in r.json()["detail"].lower(), (evict, name, r.status_code)
        # the recompute that sets a known version is not blocked; then everything renders again
        r = client.post(f"/api/projects/{pid}/recompute", json={"sculpt_version": SCULPT_LATEST_VERSION})
        assert r.status_code == 200, r.text
        assert all(r.status_code == 200 for r in calls().values())
        # a Classic project ignores sculpt_version
        p = client.get(f"/api/projects/{pid}").json()["params"]
        with sqlite3.connect(isolated_env["db_path"]) as conn:
            conn.execute("UPDATE projects SET params_json = ? WHERE id = ?",
                         (json.dumps({**p, "style": "classic", "sculpt_version": SCULPT_LATEST_VERSION + 1}), pid))
        api_main._RAM_CACHE.clear()
        assert client.get(f"/api/projects/{pid}/float32-texture?max_dim=512").status_code == 200


def test_failed_recompute_after_style_switch_sets_evicting_events(isolated_env, monkeypatch):
    """A style switch re-checks the RAM budget and may evict other projects. If the recompute then
    fails, the evicted projects are still flushed and their evicting events set; otherwise every
    later request for them would wait in `_wait_if_evicting` forever (task-5 review)."""

    with TestClient(api_main.app, raise_server_exceptions=False) as client:
        a = _create(client, res=512)["id"]
        b = _create(client, res=512, style="classic")["id"]
        # A: a preview-only recompute leaves its full relief dirty (only the flush writes it)
        assert client.post(f"/api/projects/{a}/recompute?preview_only=true", json={"layering": 0.9}).status_code == 200
        ca = _cache(a)
        assert ca.full_dirty
        expected_a = api_main.recompute_relief(ca.full_layers, db.get_project(a)["params"], physical_width_mm=1000.0, relief_z_mm=15.0)
        monkeypatch.setattr(api_main, "_RAM_BUDGET_BYTES", 1)  # any put evicts every other project
        real = api_main.recompute_relief

        def failing(layers, *args, **kw):
            live = api_main._RAM_CACHE.get(b)
            if live is not None and layers is live.full_layers:
                raise RuntimeError("recompute failed after the switch")
            return real(layers, *args, **kw)

        monkeypatch.setattr(api_main, "recompute_relief", failing)
        r = client.post(f"/api/projects/{b}/recompute", json={"style": "sculpt"})
        assert r.status_code == 500
        assert a not in api_main._RAM_CACHE
        assert api_main._EVICTING_EVENTS == {}
        # A's dirty relief was flushed to disk
        disk = np.load(_cache_dir(isolated_env, a) / "H_current.npy")
        np.testing.assert_array_equal(disk, expected_a)
        monkeypatch.setattr(api_main, "recompute_relief", real)
        monkeypatch.setattr(api_main, "_RAM_BUDGET_BYTES", 1 << 40)
        # A still loads (in a thread: a leaked event would block it forever)
        out = {}
        th = threading.Thread(target=lambda: out.setdefault("r", client.get(f"/api/projects/{a}/preview-png")), daemon=True)
        th.start()
        th.join(timeout=60)
        assert not th.is_alive() and out["r"].status_code == 200
        # B: the switch was saved, so it reopens as Sculpt
        assert client.get(f"/api/projects/{b}").json()["params"]["style"] == "sculpt"
        assert client.post(f"/api/projects/{b}/recompute", json={}).status_code == 200


def test_flush_evicted_sets_every_event_when_a_flush_fails(isolated_env):
    """One evicted project whose flush raises does not leave the others' evicting events unset."""
    db.init_db()
    bad = []
    for pid in ("p1", "p2"):
        api_main._project_dir(pid).mkdir(parents=True)
        z = np.zeros((4, 4), np.float32)
        bad.append((pid, api_main.ProjectMemoryCache(full_layers={}, preview_layers={}, H_current=z, H_preview=z, full_dirty=True)))
        api_main._EVICTING_EVENTS[pid] = threading.Event()
    events = dict(api_main._EVICTING_EVENTS)
    with pytest.raises((KeyError, ValueError)):  # recompute on empty layers fails for both
        api_main._flush_evicted_caches(bad)
    assert api_main._EVICTING_EVENTS == {} and all(e.is_set() for e in events.values())


def test_recompute_reports_edge_clamped(isolated_env):
    with TestClient(api_main.app) as client:
        pid = _create(client)["id"]
        r = client.post(f"/api/projects/{pid}/recompute", json={})
        assert r.status_code == 200 and r.json()["sculpt_edge_clamped"] is False
        steep = {"physical_width_mm": 100.0, "draft_angle_deg": 30.0}
        for q in ("", "?preview_only=true"):
            r = client.post(f"/api/projects/{pid}/recompute{q}", json=steep)
            assert r.status_code == 200 and r.json()["sculpt_edge_clamped"] is True, q
        r = client.post(f"/api/projects/{pid}/recompute", json={**steep, "style": "classic"})
        assert r.status_code == 200 and r.json()["sculpt_edge_clamped"] is False


# --------------------------------------------------------------------------- style switch and caches


def test_style_switch_roundtrip_keeps_brush(isolated_env):
    with TestClient(api_main.app) as client:
        pid = _create(client, res=1536)["id"]
        h, w = _cache(pid).H_preview.shape
        delta = np.zeros((h, w), np.float32)
        delta[h // 2 - 30 : h // 2 + 30, w // 2 - 30 : w // 2 + 30] = 0.04
        r = client.put(f"/api/projects/{pid}/float32-texture?width={w}&height={h}&mode=delta", content=delta.tobytes())
        assert r.status_code == 200
        brush = _cache(pid).full_layers["H_brush_delta"].copy()
        assert float(np.abs(brush).max()) > 0.03
        H_before = _full(client, pid)

        r = client.post(f"/api/projects/{pid}/recompute", json={"style": "classic"})
        assert r.status_code == 200 and r.json()["params"]["style"] == "classic"
        full = _cache(pid).full_layers
        assert "H_global" in full and "sculpt_basis" not in full and "sculpt_basis" not in _cache(pid).preview_layers
        assert np.array_equal(full["H_brush_delta"], brush)
        H_classic = _full(client, pid)
        assert float(np.abs(H_classic - H_before).max()) > 0.05
        cdir = _cache_dir(isolated_env, pid)
        assert (cdir / "H_global.npy").is_file() and not (cdir / sculpt_cache.MANIFEST).exists()

        r = client.post(f"/api/projects/{pid}/recompute", json={"style": "sculpt"})
        assert r.status_code == 200 and r.json()["params"]["style"] == "sculpt"
        full = _cache(pid).full_layers
        assert "sculpt_basis" in full and "H_global" not in full
        assert np.array_equal(full["H_brush_delta"], brush)
        assert np.array_equal(_full(client, pid), H_before)
        assert (cdir / sculpt_cache.MANIFEST).is_file() and not (cdir / "H_global.npy").exists()
        assert np.array_equal(np.load(cdir / "H_brush_delta.npy"), brush)


def test_style_switch_in_preview_recompute(isolated_env):
    """A slider preview that carries a new style switches too; the full relief follows on fetch."""
    with TestClient(api_main.app) as client:
        pid = _create(client, res=1536)["id"]
        H_sculpt = _full(client, pid)
        r = client.post(f"/api/projects/{pid}/recompute?preview_only=true", json={"style": "classic"})
        assert r.status_code == 200
        assert _cache(pid).full_dirty and "H_global" in _cache(pid).full_layers
        H_classic = _full(client, pid)
        assert float(np.abs(H_classic - H_sculpt).max()) > 0.05
        api_main._RAM_CACHE.clear()
        assert np.array_equal(_full(client, pid), H_classic)


def test_sculpt_export_stl_watertight_and_draft(isolated_env):
    """An adaptive STL of a Sculpt relief is watertight and every wall respects the draft angle:
    interior slopes (layer steps, creases) through `_interior_slope_excess`, the outer walls
    against the exact envelope, on the heightmap and on the mesh's vertices."""
    pw, rz, base, theta, tol = 200.0, 15.0, 5.0, 20.0, 0.02
    with TestClient(api_main.app) as client:
        pid = _create(client, res=512)["id"]
        r = client.post(f"/api/projects/{pid}/recompute",
                        json={"physical_width_mm": pw, "relief_z_mm": rz, "base_z_mm": base, "draft_angle_deg": theta})
        assert r.status_code == 200 and r.json()["sculpt_edge_clamped"] is True
        exp = client.post(f"/api/projects/{pid}/export", json={"format": "stl", "mesh_mode": "adaptive", "mesh_tolerance_mm": tol})
        assert exp.status_code == 200, exp.text
        assert exp.headers["x-mesh-mode"] == "adaptive" and float(exp.headers["x-max-error-mm"]) <= tol
        mesh = trimesh.load(io.BytesIO(exp.content), file_type="stl")
        assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0

        H = _cache(pid).H_current
        S = _cache(pid).full_layers["M_sil"]
        d_sil = _cache(pid).full_layers["d_sil"]
        excess, n = _interior_slope_excess(H, S, d_sil, pw, rz, theta)
        assert n > 1000 and excess <= 1.0 + 1e-6, excess
        rows, cols = H.shape
        px = pw / cols
        cot = 1.0 / math.tan(math.radians(theta))
        env = np.maximum(d_sil.astype(np.float64) - 0.5, 0.0) * px * cot  # mm of height allowed
        fg = S > 0.5
        assert float((H[fg] * rz - env[fg]).max()) <= 1e-4
        # the mesh: every vertex above the base stays under the envelope (+ the mesh tolerance)
        V = np.asarray(mesh.vertices)
        top = V[:, 2] > base + 1e-6
        c = np.clip(np.round(V[top, 0] / px - 0.5).astype(int), 0, cols - 1)
        r_ = np.clip(np.round((rows * px - V[top, 1]) / px - 0.5).astype(int), 0, rows - 1)
        assert top.sum() > 100
        assert float((V[top, 2] - base - env[r_, c]).max()) <= tol + 1e-4


def test_reload_from_disk_sculpt(isolated_env, monkeypatch):
    """Evicted and reloaded, a Sculpt project reads its basis from cache/sculpt_*.npy (no full
    rebuild) and gives the identical relief."""
    with TestClient(api_main.app) as client:
        pid = _create(client, res=1536)["id"]
        r = client.post(f"/api/projects/{pid}/recompute", json={"layering": 0.8, "crease_depth": 0.3, "draft_angle_deg": 3.0})
        assert r.status_code == 200
        H1 = _full(client, pid)
        builds = []
        real = api_main.build_sculpt_basis
        monkeypatch.setattr(api_main, "build_sculpt_basis", lambda pa, lay, M, shape, **kw: builds.append(tuple(shape)) or real(pa, lay, M, shape, **kw))
        api_main._RAM_CACHE.clear()
        H2 = _full(client, pid)
        assert np.array_equal(H1, H2)
        assert builds == [_cache(pid).H_preview.shape]  # only the preview basis is built
        # the reloaded basis carries the same arrays as a fresh build
        b = _cache(pid).full_layers["sculpt_basis"]
        assert b["ink"].dtype == np.float16 and b["labels"].dtype == np.uint16


def test_old_format_sculpt_cache_is_rebuilt(isolated_env):
    """A cache in an older format (here: uint8 ink, from before the float16 ink) is rebuilt, never
    read: read as the current format it would make every crease 255x too deep."""
    with TestClient(api_main.app) as client:
        pid = _create(client, res=1024)["id"]
        H1 = _full(client, pid)
        cdir = _cache_dir(isolated_env, pid)
        man_path = cdir / sculpt_cache.MANIFEST

        def tamper(fmt: int, record_dtype: bool):
            man = json.loads(man_path.read_text())
            ink = man["basis"]["ink"]
            arr = np.load(cdir / ink["__npy__"])
            np.save(cdir / ink["__npy__"], np.round(arr.astype(np.float32) * 255).astype(np.uint8))
            if record_dtype:
                ink["dtype"] = "|u1"
            man["format"] = fmt
            man_path.write_text(json.dumps(man))

        for fmt, record in ((sculpt_cache.SCULPT_CACHE_FORMAT - 1, False), (sculpt_cache.SCULPT_CACHE_FORMAT, True)):
            api_main._RAM_CACHE.clear()
            tamper(fmt, record)
            H2 = _full(client, pid)
            assert _cache(pid).full_layers["sculpt_basis"]["ink"].dtype == np.float16
            assert np.array_equal(H1, H2), fmt
            man = json.loads(man_path.read_text())
            assert man["format"] == sculpt_cache.SCULPT_CACHE_FORMAT and man["basis"]["ink"]["dtype"] == "<f2"
            assert np.load(cdir / man["basis"]["ink"]["__npy__"]).dtype == np.float16
        # a stale parts analysis (other preprocess settings) is not used either
        with np.load(cdir / sculpt_cache.PARTS_FILE) as z:
            assert json.loads(str(z["source"]))["bg_removal"] is True
        assert sculpt_cache.load_parts(cdir, {"bg_removal": False, "hd_restore": True, "polarity": "auto"}) is None


@pytest.mark.parametrize("damage", ["truncated", "garbage", "empty"])
def test_damaged_parts_file_is_a_cache_miss(isolated_env, damage):
    """A damaged cache/parts.npz (disk damage: the write itself is atomic) is a cache miss, not
    a permanent 500: zipfile.BadZipFile (not an OSError) and EOFError are caught, the parts are
    analysed again from the original image and the project reloads with the same relief."""
    with TestClient(api_main.app) as client:
        pid = _create(client, res=512)["id"]
        H1 = _full(client, pid)
        api_main._await_pending_flush(pid)
        cdir = _cache_dir(isolated_env, pid)
        path = cdir / sculpt_cache.PARTS_FILE
        raw = path.read_bytes()
        path.write_bytes({"truncated": raw[: len(raw) // 2], "garbage": b"\x00not a zip" * 100, "empty": b""}[damage])
        assert sculpt_cache.load_parts(cdir, {"bg_removal": True, "hd_restore": True, "polarity": "auto"}) is None
        api_main._RAM_CACHE.clear()
        assert client.get(f"/api/projects/{pid}/parts").status_code == 200
        assert np.array_equal(_full(client, pid), H1)


def test_damaged_basis_npy_is_a_cache_miss(isolated_env):
    """A zero-length basis .npy (np.load raises EOFError) makes the basis a cache miss too."""
    with TestClient(api_main.app) as client:
        pid = _create(client, res=512)["id"]
        H1 = _full(client, pid)
        api_main._await_pending_flush(pid)
        cdir = _cache_dir(isolated_env, pid)
        man = json.loads((cdir / sculpt_cache.MANIFEST).read_text())
        (cdir / man["basis"]["support"]["__npy__"]).write_bytes(b"")
        api_main._RAM_CACHE.clear()
        assert np.array_equal(_full(client, pid), H1)


def test_preprocess_rebuilds_sculpt_cache(isolated_env):
    with TestClient(api_main.app) as client:
        pid = _create(client, res=512)["id"]
        r = client.post(f"/api/projects/{pid}/upscale", json={"target_res": 1024})
        assert r.status_code == 200 and r.json()["width"] == 1024
        b = _cache(pid).full_layers["sculpt_basis"]
        assert b["labels"].shape == (1024, 1024)
        man = json.loads((_cache_dir(isolated_env, pid) / sculpt_cache.MANIFEST).read_text())
        assert man["shape"] == [1024, 1024]
        H1 = _full(client, pid)
        api_main._RAM_CACHE.clear()
        assert np.array_equal(_full(client, pid), H1)


def test_ram_budget_counts_sculpt_basis(isolated_env):
    with TestClient(api_main.app) as client:
        pid = _create(client, res=1536)["id"]
        cache = _cache(pid)
        seen: dict[int, int] = {}

        def arrays(o):
            if isinstance(o, np.ndarray):
                base = o
                while isinstance(base.base, np.ndarray):
                    base = base.base
                seen[id(base)] = base.nbytes
            elif isinstance(o, dict):
                for v in o.values():
                    arrays(v)
            elif isinstance(o, (list, tuple)):
                for v in o:
                    arrays(v)

        arrays(cache.full_layers["sculpt_basis"])
        arrays(cache.preview_layers["sculpt_basis"])
        pa = cache.full_layers["sculpt_parts"]
        arrays([pa.labels, pa.line_role, pa.line_owner, pa.anchors])
        basis_bytes = sum(seen.values())
        assert basis_bytes > 20 * 1536 * 1536  # the full basis is ~30 bytes per working pixel
        assert api_main._cache_nbytes(cache) >= basis_bytes + 3 * 4 * 1536 * 1536


# --------------------------------------------------------------------------- preview vs full


@pytest.fixture(scope="module")
def example_bytes():
    if not EXAMPLE.exists():
        pytest.fail("geometric example missing")
    return EXAMPLE.read_bytes()


def test_sculpt_preview_matches_full(isolated_env, example_bytes):
    """The 1024 preview (its own Sculpt basis on the preview grid) is close to the full 4K relief
    downsampled to the preview grid, inside the silhouette (>= 2 preview px from the outline)."""
    with TestClient(api_main.app) as client:
        pid = _create(client, res=4096, data=example_bytes, name="example.png")["id"]
        for params in ({}, {"layering": 1.0, "fullness": 0.8, "edge_roundness_pct": 4.0}):
            r = client.post(f"/api/projects/{pid}/recompute?preview_only=true", json=params)
            assert r.status_code == 200
            prev = _cache(pid).H_preview.copy()
            full = _full(client, pid)
            down = cv2.resize(full, (prev.shape[1], prev.shape[0]), interpolation=cv2.INTER_AREA)
            sil = _cache(pid).preview_layers["M_sil"] > 0.5
            inner = cv2.erode(sil.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            err = np.abs(prev - down)[inner]
            mean, p99 = float(err.mean()), float(np.percentile(err, 99))
            assert mean <= PREVIEW_MEAN_TOL and p99 <= PREVIEW_P99_TOL, (params, mean, p99)


PREVIEW_MEAN_TOL = 0.011
PREVIEW_P99_TOL = 0.055
