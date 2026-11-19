"""End-to-end smoke test: start the launcher, drive the API, stop it gracefully.

    python smoke_test.py --python PATH --app-root DIR [--include-8k] [--port N]

--python    interpreter that runs `-m backend.launcher` (the bundle's python\\python.exe,
            or .venv/bin/python for a checkout).
--app-root  directory used as the launcher's working directory. For a checkout this is the
            repo root (so `backend` is importable from cwd). For the bundle pass the
            unzipped MONOLITH folder: the embeddable Python finds app\\ through its
            ._pth file and does not depend on the cwd.

Stdlib only. Prints "SMOKE OK" and exits 0, or prints the reason plus the last 80 lines of
the log and exits 1.
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HEALTH_TIMEOUT_S = 120
STOP_TIMEOUT_S = 20
REQUEST_TIMEOUT_S = 600


class SmokeFailure(Exception):
    pass


def request(base: str, path: str, *, data: bytes | None = None, headers: dict | None = None):
    req = urllib.request.Request(base + path, data=data, headers=headers or {})
    if data is not None:
        req.method = "POST"
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:
        body = exc.read()[:500].decode("utf-8", "replace")
        raise SmokeFailure(f"{path} returned HTTP {exc.code}: {body}") from None
    except (urllib.error.URLError, OSError) as exc:
        raise SmokeFailure(f"{path} failed: {exc}") from None


def post_json(base: str, path: str, payload: dict):
    return request(base, path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})


def header(headers: dict, name: str) -> str | None:
    for key, value in headers.items():
        if key.lower() == name.lower():
            return value
    return None


def create_benchmark(base: str, res: int, style: str = "sculpt") -> str:
    form = urllib.parse.urlencode({"use_benchmark": "true", "target_res": str(res), "style": style}).encode()
    _, _, body = request(
        base, "/api/projects/create", data=form, headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    doc = json.loads(body)
    pid = doc.get("id") or (doc.get("project") or {}).get("id")
    if not pid:
        raise SmokeFailure(f"create response has no project id: {list(doc)}")
    return pid


def export_stl(base: str, pid: str, label: str) -> None:
    t0 = time.perf_counter()
    status, headers, body = post_json(base, f"/api/projects/{pid}/export", {"format": "stl"})
    dt = time.perf_counter() - t0
    mode = header(headers, "X-Mesh-Mode")
    tris = header(headers, "X-Triangles")
    if status != 200:
        raise SmokeFailure(f"{label}: stl export status {status}")
    if mode != "adaptive":
        raise SmokeFailure(f"{label}: X-Mesh-Mode is {mode!r}, expected 'adaptive'")
    if not tris or int(tris) <= 0:
        raise SmokeFailure(f"{label}: X-Triangles missing or zero ({tris!r})")
    if len(body) <= 84:
        raise SmokeFailure(f"{label}: stl body too small ({len(body)} bytes)")
    print(
        f"  {label}: stl ok, {tris} triangles, {len(body) / 1048576:.1f} MB, "
        f"max error {header(headers, 'X-Max-Error-Mm')} mm, {dt:.1f} s"
    )


def check_layers(base: str, pid: str, label: str) -> None:
    """Sculpt Layers tool: parts map, one bring-forward click (order solver + background
    rebuild), then an export, which must first sync the full relief to the new order."""
    _, _, body = request(base, f"/api/projects/{pid}/parts")
    doc = json.loads(body)
    parts = doc.get("parts") or []
    if not parts:
        raise SmokeFailure(f"{label}: GET /parts returned no parts")
    if len(doc.get("labels", "")) == 0:
        raise SmokeFailure(f"{label}: GET /parts returned no label map")
    u, v = parts[0]["anchor"]
    t0 = time.perf_counter()
    _, _, body = post_json(base, f"/api/projects/{pid}/parts/order", {"u": u, "v": v, "action": "forward"})
    click_s = time.perf_counter() - t0
    after = {p["id"]: p for p in json.loads(body).get("parts") or []}
    if after.get(parts[0]["id"], {}).get("override") != "front":
        raise SmokeFailure(f"{label}: bring-forward did not record a 'front' override")
    print(f"  {label}: {len(parts)} parts, bring-forward answered in {click_s:.2f} s")
    export_stl(base, pid, f"{label} after reorder")


def run_resolution(base: str, res: int, label: str, compare_classic: bool = False) -> None:
    t0 = time.perf_counter()
    pid = create_benchmark(base, res)
    sculpt_s = time.perf_counter() - t0
    print(f"  {label}: created {res}px Sculpt project in {sculpt_s:.1f} s")
    if compare_classic:
        t0 = time.perf_counter()
        create_benchmark(base, res, style="classic")
        classic_s = time.perf_counter() - t0
        print(f"  {label}: created {res}px Classic project in {classic_s:.1f} s (Sculpt adds {sculpt_s - classic_s:+.1f} s)")
    export_stl(base, pid, label)


def drive(base: str, include_8k: bool) -> None:
    print("checking UI")
    status, _, body = request(base, "/")
    if status != 200 or b'<div id="root">' not in body:
        raise SmokeFailure('GET / did not return the UI (no <div id="root">)')
    # Windows can map .js to text/plain via the registry; a module script then never runs.
    match = re.search(r'src="(/assets/[^"]+\.js)"', body.decode("utf-8", "replace"))
    if not match:
        raise SmokeFailure("index.html references no /assets/*.js script")
    _, js_headers, _ = request(base, match.group(1))
    js_type = header(js_headers, "Content-Type") or ""
    if "javascript" not in js_type.lower():
        raise SmokeFailure(f"{match.group(1)} is served as {js_type!r}, expected a javascript type")
    print(f"  script content-type ok ({js_type})")

    print("1024 project: create, recompute, exports")
    pid = create_benchmark(base, 1024)
    status, _, _ = post_json(base, f"/api/projects/{pid}/recompute", {})
    if status != 200:
        raise SmokeFailure(f"recompute status {status}")
    export_stl(base, pid, "1024")
    _, _, body = post_json(base, f"/api/projects/{pid}/export", {"format": "png16"})
    if not body.startswith(b"\x89PNG\r\n\x1a\n"):
        raise SmokeFailure("png16 export is not a PNG")
    _, _, body = post_json(base, f"/api/projects/{pid}/export", {"format": "svg"})
    if b"<svg" not in body:
        raise SmokeFailure("svg export does not contain <svg")
    print("  1024: png16 and svg ok")
    check_layers(base, pid, "1024")

    print("1024 Classic project")
    cid = create_benchmark(base, 1024, style="classic")
    status, _, _ = post_json(base, f"/api/projects/{cid}/recompute", {})
    if status != 200:
        raise SmokeFailure(f"classic recompute status {status}")
    export_stl(base, cid, "1024 classic")

    print("4096 project")
    run_resolution(base, 4096, "4096", compare_classic=True)
    if include_8k:
        print("8192 project")
        run_resolution(base, 8192, "8192")


def wait_healthy(data_dir: Path, proc: subprocess.Popen) -> str:
    """Wait for instance.json (written once healthy; it holds the real port), then check health."""
    deadline = time.time() + HEALTH_TIMEOUT_S
    run_file = data_dir / "instance.json"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SmokeFailure(f"launcher exited early with code {proc.returncode}")
        try:
            port = int(json.loads(run_file.read_text(encoding="utf-8"))["port"])
            base = f"http://127.0.0.1:{port}"
            with urllib.request.urlopen(base + "/api/health", timeout=3) as resp:
                if resp.status == 200:
                    return base
        except (OSError, ValueError, KeyError, urllib.error.URLError):
            pass
        time.sleep(0.5)
    raise SmokeFailure(f"no healthy instance within {HEALTH_TIMEOUT_S} s")


def stop(proc: subprocess.Popen, data_dir: Path) -> None:
    if not (data_dir / "instance.json").exists():
        raise SmokeFailure("instance.json missing before the stop signal")
    if proc.poll() is None:
        if sys.platform == "win32":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.send_signal(signal.SIGINT)
    try:
        code = proc.wait(timeout=STOP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise SmokeFailure(f"launcher did not exit within {STOP_TIMEOUT_S} s of the stop signal") from None
    if code != 0:
        raise SmokeFailure(f"launcher exit code {code} after stop signal (expected 0)")
    if (data_dir / "instance.json").exists():
        raise SmokeFailure("instance.json still present after shutdown")
    print("stopped cleanly (exit 0, instance.json removed)")


def tail_log(data_dir: Path, console: Path) -> None:
    for label, path in (("monolith.log", data_dir / "logs" / "monolith.log"), ("launcher console", console)):
        print(f"--- last 80 lines of {label} ---")
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            print("\n".join(lines[-80:]) if lines else "(empty)")
        except OSError:
            print("(not found)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--python", required=True, help="interpreter that runs -m backend.launcher")
    ap.add_argument("--app-root", required=True, help="working directory for the launcher (see above)")
    ap.add_argument("--include-8k", action="store_true", help="also create and export at 8192")
    ap.add_argument("--port", type=int, default=8777)
    args = ap.parse_args(argv)

    # absolute() not resolve(): a venv's python is a symlink and must keep its venv identity.
    python = str(Path(args.python).absolute())
    app_root = Path(args.app_root).resolve()

    with tempfile.TemporaryDirectory(prefix="monolith-smoke-") as tmp:
        data_dir = Path(tmp) / "data"
        console = Path(tmp) / "console.txt"
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        failure: str | None = None
        with open(console, "wb") as console_fh:
            proc = subprocess.Popen(
                [python, "-m", "backend.launcher", "--no-browser", "--port", str(args.port), "--data-dir", str(data_dir)],
                cwd=str(app_root),
                stdout=console_fh,
                stderr=subprocess.STDOUT,
                creationflags=flags,
            )
            try:
                base = wait_healthy(data_dir, proc)
                drive(base, args.include_8k)
                stop(proc, data_dir)
                console_fh.flush()
                text = console.read_text(encoding="utf-8", errors="replace")
                for bad in ("Traceback", "Error processing line"):
                    if bad in text:
                        raise SmokeFailure(f"launcher console output contains {bad!r} during a normal run")
            except SmokeFailure as exc:
                failure = str(exc)
            except Exception as exc:  # noqa: BLE001
                failure = f"unexpected {type(exc).__name__}: {exc}"
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
        if failure is not None:
            print(f"SMOKE FAILED: {failure}")
            tail_log(data_dir, console)
            return 1
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
