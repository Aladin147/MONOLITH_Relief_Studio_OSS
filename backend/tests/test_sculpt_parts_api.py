"""Sculpt parts API: the part-label payload and click-to-fix front/back overrides (task 6)."""

from __future__ import annotations

import os

import base64
import json
import random
import sqlite3
import threading
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import db
from backend import main as api_main
from backend.pipeline import sculpt_cache, sculpt_overrides
from backend.pipeline.sculpt_engine import SCULPT_LATEST_VERSION, build_sculpt_basis
from backend.pipeline.sculpt_order import auto_layers
from backend.pipeline.sculpt_parts import analyze_parts, analyze_parts_from_source, part_at
from backend.tests.test_sculpt_api import _cache, _cache_dir, _create, _figure_png, _full
from backend.tests.test_sculpt_engine import EXAMPLE, PERF_SLACK, _figure

# The figure (white parts on black, see test_sculpt_engine._figure): part ids are raster order of
# the first pixel, so the head is 1, the torso 2 and the arm 3. Points well inside each part:
HEAD_UV, TORSO_UV, ARM_UV = (0.5, 0.12), (0.40, 0.75), (0.20, 0.43)


def _parts(client: TestClient, pid: str) -> dict:
    r = client.get(f"/api/projects/{pid}/parts")
    assert r.status_code == 200, r.text
    return r.json()


def _order(client: TestClient, pid: str, action: str, uv=None, status: int = 200) -> dict:
    body = {"action": action}
    if uv is not None:
        body.update(u=uv[0], v=uv[1])
    r = client.post(f"/api/projects/{pid}/parts/order", json=body)
    assert r.status_code == status, r.text
    return r.json()


def _layers(payload: dict) -> dict[int, int]:
    return {p["id"]: p["layer"] for p in payload["parts"]}


def _pa(pid: str):
    return _cache(pid).full_layers["sculpt_parts"]


def _settle(pid: str) -> None:
    """Wait for the background work a reorder schedules (relief PNG flush, basis save)."""
    api_main._await_pending_flush(pid)


def _neighbours(pa, p: int) -> list[int]:
    return sculpt_overrides.order_inputs(pa).neighbours.get(p, [])


def _meta(env, pid: str) -> dict:
    return json.loads((env["data_dir"] / "projects" / pid / "meta.json").read_text())


# --------------------------------------------------------------------------- payload


def test_parts_payload(isolated_env):
    """Labels at the preview grid (max dim 1024), nearest from the analysis labels, lines 0; one
    entry per part with its layer, automatic layer, override and anchor."""
    with TestClient(api_main.app) as client:
        pid = _create(client, res=1536)["id"]
        body = _parts(client, pid)
        pa = _pa(pid)
        w, h = body["width"], body["height"]
        assert (h, w) == _cache(pid).H_preview.shape and max(w, h) == 1024
        raw = base64.b64decode(body["labels"])
        assert len(raw) == w * h * 2
        lab = np.frombuffer(raw, dtype="<u2").reshape(h, w)
        ys = np.minimum(((np.arange(h) + 0.5) * pa.shape[0] / h).astype(int), pa.shape[0] - 1)
        xs = np.minimum(((np.arange(w) + 0.5) * pa.shape[1] / w).astype(int), pa.shape[1] - 1)
        assert np.array_equal(lab, pa.labels[np.ix_(ys, xs)])
        line = pa.line_role[np.ix_(ys, xs)] > 0
        assert line.any() and not lab[line].any()  # line pixels are 0
        n = int(pa.labels.max())
        assert [p["id"] for p in body["parts"]] == list(range(1, n + 1))
        auto = auto_layers(pa)
        basis_layer = _cache(pid).full_layers["sculpt_basis"]["layer"]
        for p in body["parts"]:
            assert p["layer"] == int(basis_layer[p["id"]]) == int(auto[p["id"]]) == p["auto_layer"]
            assert p["override"] is None
            u, v = p["anchor"]
            assert part_at(pa, u, v) == p["id"]  # v measured downward
            assert lab[min(h - 1, int(v * h)), min(w - 1, int(u * w))] == p["id"]
        assert body["dropped_overrides"] == 0
        assert part_at(pa, *TORSO_UV) == 2 and part_at(pa, *HEAD_UV) == 1 and part_at(pa, *ARM_UV) == 3


def test_parts_409_for_classic(isolated_env):
    with TestClient(api_main.app) as client:
        pid = _create(client, res=512, style="classic")["id"]
        assert client.get(f"/api/projects/{pid}/parts").status_code == 409
        r = client.post(f"/api/projects/{pid}/parts/order", json={"u": 0.4, "v": 0.75, "action": "forward"})
        assert r.status_code == 409
        assert client.get("/api/projects/nope/parts").status_code == 404
        assert client.post("/api/projects/nope/parts/order", json={"action": "reset_all"}).status_code == 404


def _file_stamps(env, pid: str) -> dict[str, tuple[int, int]]:
    pdir = env["data_dir"] / "projects" / pid
    return {f.name: (f.stat().st_mtime_ns, f.stat().st_size)
            for f in [pdir / "depth_16bit.png", pdir / "meta.json", *sorted((pdir / "cache").iterdir())] if f.is_file()}


def test_unknown_sculpt_version_refused_warm_and_cold(isolated_env):
    """A Sculpt version this build lacks: GET /parts and POST /parts/order are refused (409), with
    the project in RAM and cold, and nothing on disk changes (review I2)."""
    with TestClient(api_main.app) as client:
        pid = _create(client, res=512)["id"]
        p = client.get(f"/api/projects/{pid}").json()["params"]
        with sqlite3.connect(isolated_env["db_path"]) as conn:
            conn.execute("UPDATE projects SET params_json = ? WHERE id = ?",
                         (json.dumps({**p, "sculpt_version": SCULPT_LATEST_VERSION + 1}), pid))
        for cold in (False, True):
            if cold:
                _settle(pid)
                api_main._RAM_CACHE.clear()
            before = _file_stamps(isolated_env, pid)
            r = client.get(f"/api/projects/{pid}/parts")
            assert r.status_code == 409 and "version" in r.json()["detail"].lower(), cold
            r = client.post(f"/api/projects/{pid}/parts/order", json={"u": TORSO_UV[0], "v": TORSO_UV[1], "action": "forward"})
            assert r.status_code == 409 and "version" in r.json()["detail"].lower()
            assert client.post(f"/api/projects/{pid}/parts/order", json={"action": "reset_all"}).status_code == 409
            assert _file_stamps(isolated_env, pid) == before, cold
            assert (pid in api_main._RAM_CACHE) is not cold  # a cold project stays unloaded
        assert sculpt_overrides.META_KEY not in _meta(isolated_env, pid)


def test_cold_get_parts_is_read_only(isolated_env):
    """GET /parts on a project that is not in RAM answers from parts.npz and meta.json: the same
    payload as when it is loaded, no render, no file written, the project left unloaded."""
    with TestClient(api_main.app) as client:
        pid = _create(client, res=1536)["id"]
        _order(client, pid, "forward", TORSO_UV)
        _settle(pid)
        warm = _parts(client, pid)
        api_main._RAM_CACHE.clear()
        before = _file_stamps(isolated_env, pid)
        builds = []
        real = api_main.recompute_relief
        try:
            api_main.recompute_relief = lambda *a, **k: builds.append(1) or real(*a, **k)
            cold = _parts(client, pid)
        finally:
            api_main.recompute_relief = real
        assert builds == [] and pid not in api_main._RAM_CACHE
        assert _file_stamps(isolated_env, pid) == before
        assert cold == warm
        # no parts.npz (another build's cache): analysed from the original image, still nothing written
        (_cache_dir(isolated_env, pid) / sculpt_cache.PARTS_FILE).unlink()
        before = _file_stamps(isolated_env, pid)
        assert _parts(client, pid) == warm
        assert _file_stamps(isolated_env, pid) == before and pid not in api_main._RAM_CACHE


# --------------------------------------------------------------------------- clicks


def test_forward_puts_part_in_front_of_neighbours(isolated_env):
    with TestClient(api_main.app) as client:
        pid = _create(client, res=1536)["id"]
        pa = _pa(pid)
        torso = part_at(pa, *TORSO_UV)
        before = _parts(client, pid)
        rev0 = _cache(pid).revision
        H0 = _full(client, pid)
        body = _order(client, pid, "forward", TORSO_UV)
        lay = _layers(body)
        nbrs = _neighbours(pa, torso)
        assert len(nbrs) == 2
        assert all(lay[torso] > lay[q] for q in nbrs), lay
        assert {p["id"]: p["override"] for p in body["parts"]} == {1: None, 2: "front", 3: None}
        assert [p["auto_layer"] for p in body["parts"]] == [p["auto_layer"] for p in before["parts"]]
        assert isinstance(body["compute_ms"], float) and body["dropped_overrides"] == 0
        assert len(base64.b64decode(body["labels"])) == body["width"] * body["height"] * 2
        cache = _cache(pid)
        assert cache.revision > rev0
        # the click answers from the preview basis, rebuilt at once
        params = db.get_project(pid)["params"]
        prev_basis = cache.preview_layers["sculpt_basis"]
        assert prev_basis is not cache.full_layers["sculpt_basis"]
        assert [int(prev_basis["layer"][p]) for p in sorted(lay)] == [lay[p] for p in sorted(lay)]
        assert np.array_equal(cache.H_preview, api_main.recompute_relief(cache.preview_layers, params, physical_width_mm=1000.0, relief_z_mm=15.0))
        # the full basis follows before anything reads the full relief, which is recomputed from it
        # (and close to a fresh build's)
        H1 = _full(client, pid)
        full_basis = cache.full_layers["sculpt_basis"]
        assert [int(full_basis["layer"][p]) for p in sorted(lay)] == [lay[p] for p in sorted(lay)]
        assert not np.array_equal(H0, H1)
        assert np.array_equal(H1, api_main.recompute_relief(cache.full_layers, params, physical_width_mm=1000.0, relief_z_mm=15.0))
        fresh = dict(cache.full_layers)
        fresh["sculpt_basis"] = build_sculpt_basis(pa, full_basis["layer"], fresh["M_sil"], fresh["M_sil"].shape, L_clean=fresh["L_clean"])
        H_fresh = api_main.recompute_relief(fresh, params, physical_width_mm=1000.0, relief_z_mm=15.0)
        assert float(np.abs(H_fresh - H1).max()) < 1e-4
        # stored as the part's own anchor and relation
        assert _meta(isolated_env, pid)[sculpt_overrides.META_KEY] == [
            {"anchor": [round(float(x), 6) for x in pa.anchors[torso]], "relation": "front"}
        ]
        # send back: behind every touching part
        body = _order(client, pid, "back", HEAD_UV)
        lay = _layers(body)
        head = part_at(pa, *HEAD_UV)
        assert all(lay[head] < lay[q] for q in _neighbours(pa, head)), lay
        assert all(lay[torso] > lay[q] for q in _neighbours(pa, torso) if q != head), lay


def test_click_on_a_line_picks_the_nearest_part(isolated_env):
    with TestClient(api_main.app) as client:
        pid = _create(client, res=1024)["id"]
        pa = _pa(pid)
        ly, lx = np.nonzero(pa.line_role > 0)
        i = len(ly) // 2
        u, v = (lx[i] + 0.5) / pa.shape[1], (ly[i] + 0.5) / pa.shape[0]
        assert part_at(pa, u, v) == 0
        body = _order(client, pid, "forward", (u, v))
        assert sum(p["override"] == "front" for p in body["parts"]) == 1


def test_repeat_forward_is_noop(isolated_env, monkeypatch):
    with TestClient(api_main.app) as client:
        pid = _create(client, res=1536)["id"]
        first = _order(client, pid, "forward", TORSO_UV)
        _settle(pid)
        H = _full(client, pid)
        cache = _cache(pid)
        rev, H_prev = cache.revision, cache.H_preview.copy()
        meta = _meta(isolated_env, pid)
        calls = []
        monkeypatch.setattr(api_main, "rebuild_parts", lambda *a, **k: calls.append(a))
        second = _order(client, pid, "forward", (TORSO_UV[0] + 0.02, TORSO_UV[1]))  # same part, another point
        assert calls == []
        assert _cache(pid) is cache and cache.revision == rev and not cache.full_dirty
        assert cache.H_current.tobytes() == H.tobytes() and cache.H_preview.tobytes() == H_prev.tobytes()
        assert _meta(isolated_env, pid) == meta
        assert _layers(second) == _layers(first)
        assert [p["override"] for p in second["parts"]] == [p["override"] for p in first["parts"]]
        assert _full(client, pid).tobytes() == H.tobytes()
        # reset on a part without an override changes nothing either
        _order(client, pid, "reset", HEAD_UV)
        assert cache.revision == rev and calls == []


def test_reset_and_reset_all(isolated_env):
    with TestClient(api_main.app) as client:
        pid = _create(client, res=1024)["id"]
        auto = _layers(_parts(client, pid))
        H_auto = _full(client, pid)
        _order(client, pid, "forward", TORSO_UV)
        body = _order(client, pid, "back", ARM_UV)
        assert {p["id"]: p["override"] for p in body["parts"]} == {1: None, 2: "front", 3: "back"}
        body = _order(client, pid, "reset", TORSO_UV)
        assert {p["id"]: p["override"] for p in body["parts"]} == {1: None, 2: None, 3: "back"}
        assert len(_meta(isolated_env, pid)[sculpt_overrides.META_KEY]) == 1
        # reset_all ignores (u, v): no point, or a point on the background, never 404
        body = _order(client, pid, "reset_all")
        assert all(p["override"] is None for p in body["parts"]) and _layers(body) == auto
        assert _meta(isolated_env, pid)[sculpt_overrides.META_KEY] == []
        assert np.array_equal(_full(client, pid), H_auto)
        _order(client, pid, "forward", TORSO_UV)
        body = _order(client, pid, "reset_all", (0.01, 0.01))
        assert _layers(body) == auto
        # other actions need a point on a part
        for action in ("forward", "back", "reset"):
            _order(client, pid, action, (0.01, 0.01), status=404)
            _order(client, pid, action, (1.5, 0.5), status=404)
            _order(client, pid, action, None, status=422)
        _order(client, pid, "sideways", TORSO_UV, status=422)


def test_contradicting_clicks_last_one_wins(isolated_env):
    """Overrides are hard constraints; when a click contradicts an earlier one, the later wins."""
    with TestClient(api_main.app) as client:
        pid = _create(client, res=1024)["id"]
        pa = _pa(pid)
        torso, arm = part_at(pa, *TORSO_UV), part_at(pa, *ARM_UV)
        lay = _layers(_order(client, pid, "forward", TORSO_UV))
        assert lay[torso] > lay[arm]
        lay = _layers(_order(client, pid, "forward", ARM_UV))  # arm over torso, after torso over arm
        assert lay[arm] > lay[torso]
        lay = _layers(_order(client, pid, "forward", TORSO_UV))  # and back again
        assert lay[torso] > lay[arm]
        lay = _layers(_order(client, pid, "back", TORSO_UV))
        assert all(lay[torso] < lay[q] for q in _neighbours(pa, torso))
        assert _cache(pid).full_layers["sculpt_basis"]["layer"].tolist()[1:] == [lay[p] for p in sorted(lay)]


def _check_last_click(pa, overrides, layer, part, action):
    inp = sculpt_overrides.order_inputs(pa)
    for q in inp.neighbours.get(part, []):
        if (q, part) in inp.resting:
            continue
        if action == "forward":
            assert layer[part] > layer[q], (overrides, part, q)
        else:
            assert layer[part] < layer[q], (overrides, part, q)


@pytest.mark.parametrize("which", ["figure", "artist"])
def test_contradicting_click_storm_never_fails(which):
    """Random click sequences: the solve never fails or hangs, and the last click always holds."""
    if which == "figure":
        white, sil = _figure(1024)
        L = (white & sil).astype(np.float32)
        pa = analyze_parts(L, sil.astype(np.float32), L)
    else:
        if not EXAMPLE.exists():
            pytest.fail("geometric example missing")
        pa = analyze_parts_from_source(str(EXAMPLE))
    inp = sculpt_overrides.order_inputs(pa)
    movable = sorted(inp.neighbours)
    rng = random.Random(7)
    overrides: list[dict] = []
    t0 = time.perf_counter()
    for _ in range(150):
        part = rng.choice(movable)
        action = rng.choice(("forward", "forward", "back", "back", "reset"))
        overrides = sculpt_overrides.apply_action(pa, overrides, part, action)
        layer = sculpt_overrides.solve(pa, overrides)
        assert layer.dtype == np.int32 and layer.shape == (inp.n_parts + 1,)
        if action != "reset":
            _check_last_click(pa, overrides, layer, part, action)
        # every stored override (latest per part) holds unless a later click overruled it
        assert sculpt_overrides.dropped(pa, overrides) == 0
    if os.environ.get("MONOLITH_CHECK_TIMINGS") == "1":
        assert time.perf_counter() - t0 < 30.0


def test_resting_part_moves_with_the_part_it_rests_on():
    """Design decision 1: parts resting on the moved part keep their relation to it. A small part
    inside the torso (a palm resting on it, the containment cue) stays in front when the torso is
    brought forward or sent back."""
    n = 2048  # drawn at the analysis grid (at 1024 the ring's resampled pixels miss the 70 % share)
    white, sil = _figure(n)
    lines = np.zeros((n, n), np.uint8)
    cv2.circle(lines, (int(0.62 * n), int(0.78 * n)), int(0.05 * n), 1, max(2, n // 512))
    white = white & (lines == 0)
    L = (white & sil).astype(np.float32)
    pa = analyze_parts(L, sil.astype(np.float32), L)
    torso, palm = part_at(pa, *TORSO_UV), part_at(pa, 0.62, 0.78)
    inp = sculpt_overrides.order_inputs(pa)
    assert palm > 0 and palm != torso and (palm, torso) in inp.resting
    assert inp.auto[palm] > inp.auto[torso]
    anchor = [round(float(x), 6) for x in pa.anchors[torso]]
    resting = [q for q in inp.neighbours[torso] if (q, torso) in inp.resting]  # the palm (and the arm)
    others = [q for q in inp.neighbours[torso] if q not in resting]  # the head
    assert palm in resting and others
    for rel in ("front", "back"):
        lay = sculpt_overrides.solve(pa, [{"anchor": anchor, "relation": rel}])
        if rel == "front":
            assert all(lay[torso] > lay[q] for q in others)
        else:
            assert all(lay[torso] < lay[q] for q in others)
        assert all(lay[q] > lay[torso] for q in resting), (rel, lay)
    # the palm itself can still be sent behind the torso
    lay = sculpt_overrides.solve(pa, [{"anchor": [float(x) for x in pa.anchors[palm]], "relation": "back"}])
    assert lay[palm] < lay[torso]


# --------------------------------------------------------------------------- re-apply, dropped


def test_override_survives_upscale_2k_to_4k(isolated_env):
    with TestClient(api_main.app) as client:
        pid = _create(client, res=2048)["id"]
        body = _order(client, pid, "forward", TORSO_UV)
        torso = part_at(_pa(pid), *TORSO_UV)
        r = client.post(f"/api/projects/{pid}/upscale", json={"target_res": 4096})
        assert r.status_code == 200 and r.json()["width"] == 4096
        after = _parts(client, pid)
        assert part_at(_pa(pid), *TORSO_UV) == torso
        assert {p["id"]: p["override"] for p in after["parts"]}[torso] == "front"
        assert _layers(after) == _layers(body)
        assert after["dropped_overrides"] == 0
        assert _cache(pid).full_layers["sculpt_basis"]["layer"].tolist()[1:] == [_layers(body)[p] for p in sorted(_layers(body))]
        # a plain reprocess re-applies them too
        r = client.post(f"/api/projects/{pid}/preprocess", json={"target_res": 1024})
        assert r.status_code == 200
        assert _layers(_parts(client, pid)) == _layers(body)


def test_override_dropped_when_anchor_misses(isolated_env):
    with TestClient(api_main.app) as client:
        pid = _create(client, res=1024)["id"]
        auto = _layers(_parts(client, pid))
        assert _layers(_order(client, pid, "forward", TORSO_UV)) != auto
        _settle(pid)
        meta = _meta(isolated_env, pid)
        meta[sculpt_overrides.META_KEY][0]["anchor"] = [0.01, 0.01]  # the background
        (isolated_env["data_dir"] / "projects" / pid / "meta.json").write_text(json.dumps(meta))
        r = client.post(f"/api/projects/{pid}/preprocess", json={"target_res": 1024})
        assert r.status_code == 200
        body = _parts(client, pid)
        assert body["dropped_overrides"] == 1
        assert _layers(body) == auto and all(p["override"] is None for p in body["parts"])
        # kept in meta.json (a reprocess that brings the part back re-applies it); reset_all clears it
        assert len(_meta(isolated_env, pid)[sculpt_overrides.META_KEY]) == 1
        body = _order(client, pid, "reset_all")
        assert body["dropped_overrides"] == 0 and _meta(isolated_env, pid)[sculpt_overrides.META_KEY] == []


def _split_block(n: int, split: bool):
    """Two blocks side by side; with ``split`` a drawn line separates them (two parts), without it
    they are one part, as when a reprocess loses the separating line."""
    white = np.zeros((n, n), bool)
    white[n // 4 : 3 * n // 4, n // 8 : 7 * n // 8] = True
    sil = white.copy()
    if split:
        white[:, n // 2 - max(2, n // 512) : n // 2 + max(2, n // 512)] = False
    L = (white & sil).astype(np.float32)
    return analyze_parts(L, sil.astype(np.float32), L)


def test_override_shadowed_by_a_merge_is_counted_dropped():
    """Final review I6: overrides on two parts that a reprocess merges resolve to one part; the
    older is shadowed (the later click wins) and is counted in dropped_overrides, never silently
    lost. Both stay stored, and a click on the merged part replaces both."""
    pa2, pa1 = _split_block(1024, True), _split_block(1024, False)
    assert int(pa2.labels.max()) == 2 and int(pa1.labels.max()) == 1
    left, right = part_at(pa2, 0.3, 0.5), part_at(pa2, 0.7, 0.5)
    assert {left, right} == {1, 2}
    ov = sculpt_overrides.apply_action(pa2, [], left, "forward")
    ov = sculpt_overrides.apply_action(pa2, ov, right, "back")
    assert len(ov) == 2 and sculpt_overrides.dropped(pa2, ov) == 0
    # the merged analysis: both anchors hit part 1
    assert {part_at(pa1, *o["anchor"]) for o in ov} == {1}
    assert sculpt_overrides.dropped(pa1, ov) == 1
    body = sculpt_overrides.parts_payload(pa1, sculpt_overrides.solve(pa1, ov), ov, (256, 256))
    assert body["dropped_overrides"] == 1
    assert [p["override"] for p in body["parts"]] == ["back"]  # the later click
    # one anchor on the background as well: both kinds are counted
    assert sculpt_overrides.dropped(pa1, [{"anchor": [0.01, 0.01], "relation": "front"}, *ov]) == 2
    # a click on the merged part replaces both stored overrides: nothing dropped any more
    ov1 = sculpt_overrides.apply_action(pa1, ov, 1, "forward")
    assert len(ov1) == 1 and sculpt_overrides.dropped(pa1, ov1) == 0


# --------------------------------------------------------------------------- persistence


def test_reorder_reloads_identically(isolated_env, monkeypatch):
    """After a reorder the on-disk Sculpt basis is the reordered one: evicted and reloaded, the
    project reads it (no full build) and gives the same relief byte for byte."""
    with TestClient(api_main.app) as client:
        pid = _create(client, res=1536)["id"]
        _order(client, pid, "forward", TORSO_UV)
        H1 = _full(client, pid)
        lay1 = _cache(pid).full_layers["sculpt_basis"]["layer"].copy()
        _settle(pid)
        builds = []
        real = api_main.build_sculpt_basis
        monkeypatch.setattr(api_main, "build_sculpt_basis", lambda pa, lay, M, shape, **kw: builds.append(tuple(shape)) or real(pa, lay, M, shape, **kw))
        api_main._RAM_CACHE.clear()
        assert np.array_equal(_full(client, pid), H1)
        assert np.array_equal(_cache(pid).full_layers["sculpt_basis"]["layer"], lay1)
        assert builds == [_cache(pid).H_preview.shape]  # only the preview basis is built


def test_reorder_then_eviction_reloads_identically(isolated_env, monkeypatch):
    """A project evicted right after a reorder (its basis save still pending) is flushed by the
    eviction and reloads with the reordered relief, byte for byte."""
    with TestClient(api_main.app) as client:
        monkeypatch.setattr(api_main, "_sculpt_background_job", lambda *args: None)  # never runs
        real = api_main.build_sculpt_basis
        for materialised in (True, False):  # the full relief read before the eviction, or not
            monkeypatch.setattr(api_main, "MAX_RAM_PROJECTS", 3)
            a = _create(client, res=1536)["id"]
            _order(client, a, "back", ARM_UV)
            cache_a = _cache(a)
            assert cache_a.sculpt_pending is not None  # the full basis has not followed yet
            assert not (_cache_dir(isolated_env, a) / sculpt_cache.MANIFEST).exists()  # stale basis unreadable at once
            if materialised:
                _full(client, a)
                assert cache_a.sculpt_pending is None and cache_a.sculpt_unsaved_basis is not None
            monkeypatch.setattr(api_main, "MAX_RAM_PROJECTS", 1)
            b = _create(client, res=512)["id"]  # evicts a: its flush rebuilds and saves the basis
            assert a not in api_main._RAM_CACHE and b in api_main._RAM_CACHE
            assert cache_a.sculpt_pending is None and cache_a.sculpt_unsaved_basis is None
            assert (_cache_dir(isolated_env, a) / sculpt_cache.MANIFEST).exists()
            H1 = cache_a.H_current.copy()  # the reordered relief the eviction flushed
            assert not cache_a.full_dirty
            builds = []
            monkeypatch.setattr(api_main, "build_sculpt_basis", lambda pa, lay, M, shape, **kw: builds.append(tuple(shape)) or real(pa, lay, M, shape, **kw))
            assert np.array_equal(_full(client, a), H1)
            assert builds == [_cache(a).H_preview.shape], materialised
            monkeypatch.setattr(api_main, "build_sculpt_basis", real)


def test_background_job_completes_the_reorder(isolated_env, monkeypatch):
    """Shortly after the last click, without further requests: the full basis takes the new order,
    the full relief is recomputed and the preview follows it, and the basis is on disk."""
    monkeypatch.setattr(api_main, "SCULPT_SAVE_DELAY_S", 0.05)
    with TestClient(api_main.app) as client:
        pid = _create(client, res=1536)["id"]
        _order(client, pid, "forward", TORSO_UV)
        lay = _layers(_order(client, pid, "back", ARM_UV))  # two quick clicks: one rebuild and save
        cache = _cache(pid)
        man = _cache_dir(isolated_env, pid) / sculpt_cache.MANIFEST
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline and not (man.exists() and cache.sculpt_unsaved_basis is None):
            time.sleep(0.05)
        _settle(pid)  # the job and its relief flush have finished
        assert man.exists() and cache.sculpt_pending is None and not cache.full_dirty
        assert cache.full_layers["sculpt_basis"]["layer"].tolist()[1:] == [lay[p] for p in sorted(lay)]
        params = db.get_project(pid)["params"]
        H = api_main.recompute_relief(cache.full_layers, params, physical_width_mm=1000.0, relief_z_mm=15.0)
        assert np.array_equal(cache.H_current, H)
        ph, pw = cache.H_preview.shape
        assert np.array_equal(cache.H_preview, cv2.resize(H, (pw, ph), interpolation=cv2.INTER_AREA))
        assert np.array_equal(np.load(_cache_dir(isolated_env, pid) / "H_current.npy"), H)
        api_main._RAM_CACHE.clear()
        assert np.array_equal(_full(client, pid), H)


def test_crash_before_basis_save_rebuilds_with_overrides(isolated_env, monkeypatch):
    """If the process dies before the background save, the cleared basis is rebuilt on load from
    the stored overrides: the same order, the same relief within the rebuild tolerance."""
    with TestClient(api_main.app) as client:
        pid = _create(client, res=1024)["id"]
        monkeypatch.setattr(api_main, "_sculpt_background_job", lambda *args: None)
        lay = _layers(_order(client, pid, "forward", TORSO_UV))
        H1 = _full(client, pid)
        api_main._RAM_CACHE.clear()  # "crash": nothing flushed
        assert not (_cache_dir(isolated_env, pid) / sculpt_cache.MANIFEST).exists()
        H2 = _full(client, pid)
        assert _layers(_parts(client, pid)) == lay
        assert float(np.abs(H2 - H1).max()) < 1e-4
        assert (_cache_dir(isolated_env, pid) / sculpt_cache.MANIFEST).exists()


def test_preview_basis_follows_reorder(isolated_env):
    """The preview basis (its own grid) takes the new order: a preview recompute after a reorder
    matches a preview basis built fresh with that order."""
    with TestClient(api_main.app) as client:
        pid = _create(client, res=1536)["id"]
        lay = _layers(_order(client, pid, "forward", TORSO_UV))
        pb = _cache(pid).preview_layers["sculpt_basis"]
        assert pb["layer"].tolist()[1:] == [lay[p] for p in sorted(lay)]  # at once, before any recompute
        r = client.post(f"/api/projects/{pid}/recompute?preview_only=true", json={"layering": 0.7})
        assert r.status_code == 200
        cache = _cache(pid)
        pb = cache.preview_layers["sculpt_basis"]
        order = api_main._sculpt_layer(cache)
        assert pb is not cache.full_layers["sculpt_basis"] and np.array_equal(pb["layer"], order)
        fresh = dict(cache.preview_layers)
        fresh["sculpt_basis"] = build_sculpt_basis(_pa(pid), order, fresh["M_sil"], fresh["M_sil"].shape, L_clean=fresh["L_clean"])
        params = db.get_project(pid)["params"]
        H_fresh = api_main.recompute_relief(fresh, params, physical_width_mm=1000.0, relief_z_mm=15.0)
        assert float(np.abs(H_fresh - cache.H_preview).max()) < 1e-4


def test_concurrent_reorders_and_recomputes(isolated_env):
    """Reorders and slider recomputes racing on one project leave a consistent state: the basis
    holds the order the stored overrides give, and the relief is the one its layers compute."""
    with TestClient(api_main.app) as client:
        pid = _create(client, res=1024)["id"]
        errors: list[BaseException] = []

        def reorders():
            try:
                for action, uv in [("forward", TORSO_UV), ("forward", ARM_UV), ("back", HEAD_UV), ("reset", ARM_UV)] * 2:
                    r = client.post(f"/api/projects/{pid}/parts/order", json={"u": uv[0], "v": uv[1], "action": action})
                    assert r.status_code == 200, r.text
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def sliders():
            try:
                for k in range(12):
                    r = client.post(f"/api/projects/{pid}/recompute?preview_only={'true' if k % 2 else 'false'}",
                                    json={"layering": 0.3 + 0.05 * k})
                    assert r.status_code == 200, r.text
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=reorders), threading.Thread(target=sliders)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=300)
        assert not errors, errors
        _settle(pid)
        cache = _cache(pid)
        pa = _pa(pid)
        expected = sculpt_overrides.solve(pa, sculpt_overrides.stored(_meta(isolated_env, pid)))
        assert np.array_equal(cache.full_layers["sculpt_basis"]["layer"], expected)
        H = _full(client, pid)
        params = db.get_project(pid)["params"]
        assert np.array_equal(H, api_main.recompute_relief(cache.full_layers, params, physical_width_mm=1000.0, relief_z_mm=15.0))
        api_main._RAM_CACHE.clear()
        assert np.array_equal(_full(client, pid), H)


def test_job_superseded_by_a_click_is_discarded(isolated_env, monkeypatch):
    """A click that lands while the job works off the lock (here: held in its staging write)
    supersedes it: the job's copy is discarded, the job re-armed, and the end state holds the
    later click, on disk too (byte-identical reload)."""
    monkeypatch.setattr(api_main, "SCULPT_SAVE_DELAY_S", 0.05)
    started, release = threading.Event(), threading.Event()
    real_stage = sculpt_cache.stage_basis
    calls = []

    def gated(cache_dir, basis, pa):
        calls.append(basis["layer"].copy())
        if len(calls) == 1:
            started.set()
            release.wait(30)
        return real_stage(cache_dir, basis, pa)

    monkeypatch.setattr(sculpt_cache, "stage_basis", gated)
    with TestClient(api_main.app) as client:
        pid = _create(client, res=1536)["id"]
        _order(client, pid, "forward", TORSO_UV)
        assert started.wait(30)  # the job is off the lock with the first click's copy
        lay = _layers(_order(client, pid, "forward", ARM_UV))  # does not wait for it
        release.set()
        cache = _cache(pid)
        man = _cache_dir(isolated_env, pid) / sculpt_cache.MANIFEST
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline and not (len(calls) >= 2 and man.exists() and cache.sculpt_unsaved_basis is None):
            time.sleep(0.05)
        _settle(pid)
        assert len(calls) >= 2 and calls[-1].tolist()[1:] == [lay[p] for p in sorted(lay)]
        assert cache.full_layers["sculpt_basis"]["layer"].tolist()[1:] == [lay[p] for p in sorted(lay)]
        assert not list(_cache_dir(isolated_env, pid).glob(sculpt_cache.STAGING_PREFIX + "*"))
        H = _full(client, pid)
        api_main._RAM_CACHE.clear()
        assert np.array_equal(_full(client, pid), H)


def test_copy_for_reorder_leaves_the_original_alone():
    """The job rebuilds and recomputes a private copy while the live basis stays in use: the
    original is unchanged, and the copy's results equal an in-place rebuild."""
    from backend.pipeline.sculpt_engine import copy_for_reorder, rebuild_parts
    from backend.pipeline.relief_engine import recompute_relief
    import copy as copy_mod
    import hashlib

    def digest(v):
        if isinstance(v, np.ndarray):
            return (v.dtype.str, v.shape, hashlib.sha1(np.ascontiguousarray(v).view(np.uint8)).hexdigest())
        if isinstance(v, dict):
            return tuple(sorted((str(k), digest(x)) for k, x in v.items()))
        if isinstance(v, (list, tuple)):
            return tuple(digest(x) for x in v)
        return repr(v) if not hasattr(v, "acquire") else "lock"

    white, sil = _figure(1024)
    L = (white & sil).astype(np.float32)
    S = sil.astype(np.float32)
    pa = analyze_parts(L, S, L)
    lay = auto_layers(pa)
    basis = build_sculpt_basis(pa, lay, S, S.shape, L_clean=L)
    d_sil = cv2.distanceTransform((S > 0.5).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    params = db.default_relief_params()
    recompute_relief({"M_sil": S, "d_sil": d_sil, "sculpt_basis": basis}, params, physical_width_mm=1000.0, relief_z_mm=15.0)
    torso = part_at(pa, *TORSO_UV)
    new = sculpt_overrides.solve(pa, [{"anchor": [float(x) for x in pa.anchors[torso]], "relation": "front"}])
    assert not np.array_equal(new, lay)
    ref = copy_mod.deepcopy({k: v for k, v in basis.items() if k != "_lock"})
    before = {k: digest(v) for k, v in basis.items()}
    work = copy_for_reorder(basis)
    rebuild_parts(work, [torso], new)
    H_work = recompute_relief({"M_sil": S, "d_sil": d_sil, "sculpt_basis": work}, params, physical_width_mm=1000.0, relief_z_mm=15.0)
    assert {k: digest(v) for k, v in basis.items()} == before
    rebuild_parts(ref, [torso], new)
    H_ref = recompute_relief({"M_sil": S, "d_sil": d_sil, "sculpt_basis": ref}, params, physical_width_mm=1000.0, relief_z_mm=15.0)
    assert np.array_equal(H_work, H_ref)


# --------------------------------------------------------------------------- speed


def _timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def _sleep_until(t: float) -> None:
    time.sleep(max(0.0, t - time.perf_counter()))


def test_reorder_under_1s_at_4k(isolated_env):
    """Reorder clicks on the 4K figure answer within 1 s (design: a reorder click stays under 1 s
    at 4K), including a click and a preview slider drag that land while the background job
    rebuilds the full basis (~1 s after the previous click). The real debounced job runs: nothing
    is settled between the timed requests."""
    with TestClient(api_main.app) as client:
        pid = _create(client, res=4096, data=_figure_png(4096))["id"]
        _parts(client, pid)  # the order inputs are computed once per analysis, not per click
        delay = api_main.SCULPT_SAVE_DELAY_S
        times = {}
        body, times["first"] = _timed(lambda: _order(client, pid, "back", ARM_UV))
        assert body["compute_ms"] <= times["first"] * 1000.0
        t1 = time.perf_counter()
        _sleep_until(t1 + delay + 0.05)  # the job has just started on the first click
        _, times["second"] = _timed(lambda: _order(client, pid, "forward", TORSO_UV))
        t2 = time.perf_counter()
        _sleep_until(t2 + delay + 0.1)  # ... and on the second
        r, times["slider"] = _timed(lambda: client.post(f"/api/projects/{pid}/recompute?preview_only=true", json={"layering": 0.7}))
        assert r.status_code == 200
        _sleep_until(time.perf_counter() + 0.3)
        _, times["reset_all"] = _timed(lambda: _order(client, pid, "reset_all"))
        if os.environ.get("MONOLITH_CHECK_TIMINGS") == "1":
            assert max(times.values()) < 1.0 * PERF_SLACK, times
        # and the end state is consistent: the full relief is the automatic order's
        _settle(pid)
        H = _full(client, pid)
        cache = _cache(pid)
        assert np.array_equal(cache.full_layers["sculpt_basis"]["layer"], sculpt_overrides.order_inputs(_pa(pid)).auto)
        params = db.get_project(pid)["params"]
        assert np.array_equal(H, api_main.recompute_relief(cache.full_layers, params, physical_width_mm=1000.0, relief_z_mm=15.0))


def test_full_recompute_during_pending_reorder_does_not_block_clicks(isolated_env):
    """Final review I5: the frontend's automatic full /recompute (260 ms after a slider touch) can
    land while a reorder's full rebuild is still pending. It answers from the preview basis and
    leaves the full relief dirty instead of rebuilding the full basis in place under the lock, so
    a click sent 50 ms after it does not wait seconds behind it. The real debounced job runs and
    nothing is settled between the timed requests; an export afterwards has the new order."""
    with TestClient(api_main.app) as client:
        pid = _create(client, res=4096, data=_figure_png(4096))["id"]
        _parts(client, pid)
        delay = api_main.SCULPT_SAVE_DELAY_S
        _order(client, pid, "back", ARM_UV)
        _sleep_until(time.perf_counter() + delay + 0.1)  # the job is rebuilding its copy
        times: dict[str, float] = {}
        result: dict[str, object] = {}

        def full_recompute():
            r, times["recompute"] = _timed(lambda: client.post(f"/api/projects/{pid}/recompute", json={"layering": 0.7}))
            result["recompute"] = r

        th = threading.Thread(target=full_recompute)
        th.start()
        time.sleep(0.05)
        body, times["click"] = _timed(lambda: _order(client, pid, "forward", TORSO_UV))
        th.join(timeout=60)
        r = result["recompute"]
        assert r.status_code == 200 and r.json()["params"]["layering"] == 0.7
        if os.environ.get("MONOLITH_CHECK_TIMINGS") == "1":
            assert times["click"] < 1.0 * PERF_SLACK and times["recompute"] < 1.0 * PERF_SLACK, times
        lay = _layers(body)
        # export: the full basis takes the new order first, and the heights are its relief
        exp = client.post(f"/api/projects/{pid}/export", json={"format": "png16"})
        assert exp.status_code == 200
        cache = _cache(pid)
        expected = sculpt_overrides.solve(_pa(pid), sculpt_overrides.stored(_meta(isolated_env, pid)))
        assert cache.full_layers["sculpt_basis"]["layer"].tolist()[1:] == [lay[p] for p in sorted(lay)] == expected.tolist()[1:]
        params = db.get_project(pid)["params"]
        assert params["layering"] == 0.7
        H = api_main.recompute_relief(cache.full_layers, params, physical_width_mm=1000.0, relief_z_mm=15.0)
        assert exp.content == api_main.encode_png16(np.ascontiguousarray(H, dtype=np.float32))
        _settle(pid)
        assert np.array_equal(_full(client, pid), H)
