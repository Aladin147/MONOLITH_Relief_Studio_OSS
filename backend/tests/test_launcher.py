from __future__ import annotations

import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from backend import launcher

REPO_ROOT = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_pick_port_skips_busy_port():
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        port = busy.getsockname()[1]
        picked = launcher.pick_port(port)
        assert picked != port
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", picked))


def test_read_version(tmp_path):
    assert launcher.read_version(tmp_path) == "dev"
    (tmp_path / "VERSION.txt").write_text("abc1234 2026-11-14\nmore\n", encoding="utf-8")
    assert launcher.read_version(tmp_path) == "abc1234 2026-11-14"


def test_stale_run_file_is_ignored(tmp_path):
    run_file = tmp_path / "instance.json"
    run_file.write_text(json.dumps({"port": _free_port(), "pid": 1}), encoding="utf-8")
    assert launcher.running_instance(tmp_path) is None
    assert not run_file.exists()


def test_logging_writes_file(tmp_path):
    root = logging.getLogger()
    saved = (list(root.handlers), root.level)
    try:
        log_path = launcher.configure_logging(tmp_path)
        logging.getLogger("uvicorn.error").info("hello-launcher-log")
        for h in logging.getLogger().handlers:
            h.flush()
        assert log_path == tmp_path / "logs" / "monolith.log"
        assert "hello-launcher-log" in log_path.read_text(encoding="utf-8")
    finally:
        for h in list(root.handlers):
            if h not in saved[0]:
                root.removeHandler(h)
                h.close()
        root.setLevel(saved[1])


def test_second_launch_reuses_running_instance(tmp_path, monkeypatch):
    for key in ("MONOLITH_DATA_DIR", "MONOLITH_DB_PATH", "MONOLITH_UI_DIR"):
        monkeypatch.setenv(key, "x")  # registered so main()'s env writes are undone
    port = _free_port()
    env = {k: v for k, v in os.environ.items() if not k.startswith("MONOLITH_")}
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    proc = subprocess.Popen(
        [sys.executable, "-m", "backend.launcher", "--no-browser",
         "--port", str(port), "--data-dir", str(tmp_path)],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=flags,
    )
    try:
        run_file = tmp_path / "instance.json"
        deadline = time.time() + 60
        while time.time() < deadline and not run_file.exists():
            assert proc.poll() is None, "launcher exited early"
            time.sleep(0.2)
        assert run_file.exists()
        info = json.loads(run_file.read_text())
        assert info["port"] == port
        assert info["version"] == launcher.read_version(REPO_ROOT)

        t0 = time.time()
        assert launcher.main(["--no-browser", "--data-dir", str(tmp_path)]) == 0
        assert time.time() - t0 < 10

        proc.send_signal(signal.CTRL_BREAK_EVENT if sys.platform == "win32" else signal.SIGINT)
        assert proc.wait(timeout=15) == 0
        assert not run_file.exists()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)


def test_run_file_only_removed_by_owner(tmp_path):
    run_file = tmp_path / "instance.json"
    run_file.write_text(json.dumps({"port": 1, "pid": os.getpid() + 12345}), encoding="utf-8")
    launcher.remove_run_file_if_mine(tmp_path)
    assert run_file.exists()
    run_file.write_text(json.dumps({"port": 1, "pid": os.getpid()}), encoding="utf-8")
    launcher.remove_run_file_if_mine(tmp_path)
    assert not run_file.exists()


def test_startup_failure_returns_1(tmp_path, capsys):
    not_a_dir = tmp_path / "file.txt"
    not_a_dir.write_text("x")
    assert launcher.main(["--no-browser", "--data-dir", str(not_a_dir)]) == 1
    assert "MONOLITH could not start." in capsys.readouterr().out


def test_reserve_port_holds_port(tmp_path):
    with launcher.reserve_port(_free_port()) as s:
        port = s.getsockname()[1]
        assert launcher.pick_port(port) != port


def _fake_server(port_holder):
    """A tiny /api/health responder standing in for a started server."""
    import http.server

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = b'{"status": "ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port_holder.append(srv.server_address[1])
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _isolate_env(monkeypatch):
    for key in ("MONOLITH_DATA_DIR", "MONOLITH_DB_PATH", "MONOLITH_UI_DIR"):
        monkeypatch.setenv(key, "x")


def test_instance_lock_is_exclusive(tmp_path):
    first = launcher.acquire_instance_lock(tmp_path)
    assert first is not None
    try:
        assert launcher.acquire_instance_lock(tmp_path) is None
    finally:
        launcher.release_instance_lock(first)
    again = launcher.acquire_instance_lock(tmp_path)
    assert again is not None
    launcher.release_instance_lock(again)


def test_lock_held_second_main_waits_then_reuses(tmp_path, monkeypatch, capsys):
    _isolate_env(monkeypatch)
    version = launcher.read_version(REPO_ROOT)
    held = launcher.acquire_instance_lock(tmp_path)
    assert held is not None
    ports: list[int] = []
    srv = _fake_server(ports)
    opened: list[str] = []
    monkeypatch.setattr(launcher.webbrowser, "open", lambda url: opened.append(url))

    def publish_later():
        time.sleep(1.5)  # the "other launcher" finishes its slow start
        (tmp_path / "instance.json").write_text(
            json.dumps({"port": ports[0], "pid": 1, "version": version}), encoding="utf-8")

    threading.Thread(target=publish_later, daemon=True).start()
    try:
        t0 = time.time()
        assert launcher.main(["--data-dir", str(tmp_path)]) == 0
        assert time.time() - t0 >= 1.0  # it really waited
        out = capsys.readouterr().out
        assert "Starting MONOLITH, please wait" in out
        assert "already starting or running in another window" in out
        assert opened == [f"http://127.0.0.1:{ports[0]}"]
    finally:
        srv.shutdown()
        launcher.release_instance_lock(held)


def test_lock_held_but_never_healthy_returns_1(tmp_path, monkeypatch, capsys):
    _isolate_env(monkeypatch)
    monkeypatch.setattr(launcher, "wait_for_instance", lambda *a, **k: None)
    held = launcher.acquire_instance_lock(tmp_path)
    try:
        assert launcher.main(["--no-browser", "--data-dir", str(tmp_path)]) == 1
        assert "did not finish starting" in capsys.readouterr().out
    finally:
        launcher.release_instance_lock(held)


def test_stale_file_removed_when_lock_is_ours(tmp_path, monkeypatch):
    _isolate_env(monkeypatch)
    run_file = tmp_path / "instance.json"
    run_file.write_text(json.dumps({"port": _free_port(), "pid": 1, "version": "x"}), encoding="utf-8")

    class Stop(Exception):
        pass

    def boom(*a, **k):
        raise Stop

    monkeypatch.setattr(launcher, "configure_logging", boom)  # stop right after the check
    assert launcher.main(["--no-browser", "--data-dir", str(tmp_path)]) == 1
    assert not run_file.exists()


def test_stale_file_kept_when_lock_not_ours(tmp_path):
    run_file = tmp_path / "instance.json"
    run_file.write_text(json.dumps({"port": _free_port(), "pid": 1}), encoding="utf-8")
    assert launcher.running_instance_info(tmp_path, remove_stale=False, timeout=0.5) is None
    assert run_file.exists()


def test_older_running_version_is_refused(tmp_path, monkeypatch, capsys):
    _isolate_env(monkeypatch)
    ports: list[int] = []
    srv = _fake_server(ports)
    (tmp_path / "instance.json").write_text(
        json.dumps({"port": ports[0], "pid": 1, "version": "old-build"}), encoding="utf-8")
    try:
        assert launcher.main(["--no-browser", "--data-dir", str(tmp_path)]) == 1
        assert "An older MONOLITH is still running" in capsys.readouterr().out
    finally:
        srv.shutdown()


def test_announce_timeout_prints_log_hint(tmp_path, capsys, monkeypatch):
    class Srv:
        should_exit = False

    monkeypatch.setattr(launcher, "_health_ok", lambda *a, **k: False)
    real_time = time.time
    calls = {"n": 0}

    def fake_time():
        calls["n"] += 1
        return real_time() + (0 if calls["n"] == 1 else 1000)  # deadline passes at once

    monkeypatch.setattr(launcher.time, "time", fake_time)
    launcher._announce(Srv(), 1, tmp_path, "v", tmp_path / "logs" / "monolith.log", False)
    out = capsys.readouterr().out
    assert "taking longer than expected to start" in out
    assert "monolith.log" in out
