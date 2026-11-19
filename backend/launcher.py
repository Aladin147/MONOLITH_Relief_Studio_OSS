"""Desktop launcher for MONOLITH Relief Studio.

Run as ``python -m backend.launcher``. Stdlib only at import time: backend.main
is imported lazily inside main() so the environment can be configured first.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import logging
import logging.handlers
import os
import signal
import socket
import sys
import threading
import time
import traceback
import urllib.request
import webbrowser
from pathlib import Path

HOST = "127.0.0.1"
RUN_FILE = "instance.json"
_STOP_WAIT_SECONDS = 4.5

# Kept at module level so the ctypes callback is never garbage-collected.
_console_handler = None


def default_data_dir() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "MONOLITH"
    return Path.home() / ".local" / "share" / "MONOLITH"


def reserve_port(preferred: int = 8765, host: str = HOST, attempts: int = 50) -> socket.socket:
    """Bind and return a socket on the first free port (kept open, so no race)."""
    for port in range(preferred, preferred + attempts):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((host, port))
        except OSError:
            s.close()
            continue
        return s
    raise RuntimeError(f"No free port found in {preferred}-{preferred + attempts - 1}")


def pick_port(preferred: int = 8765, host: str = HOST, attempts: int = 50) -> int:
    with reserve_port(preferred, host, attempts) as s:
        return s.getsockname()[1]


def read_version(app_root: Path) -> str:
    try:
        with open(Path(app_root) / "VERSION.txt", encoding="utf-8") as fh:
            line = fh.readline().strip()
        return line or "dev"
    except OSError:
        return "dev"


def _health_ok(port: int, timeout: float) -> bool:
    try:
        with urllib.request.urlopen(f"http://{HOST}:{port}/api/health", timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8")).get("status") == "ok"
    except Exception:
        return False


def _remove(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def running_instance_info(data_dir: Path, remove_stale: bool = True, timeout: float = 5.0) -> dict | None:
    """Return instance.json's contents if that instance answers /api/health, else None.

    A stale file is only deleted when ``remove_stale`` is set, which callers do only
    while holding the single-instance lock (otherwise it may belong to a starting server).
    """
    run_file = Path(data_dir) / RUN_FILE
    try:
        info = json.loads(run_file.read_text(encoding="utf-8"))
        port = int(info["port"])
    except FileNotFoundError:
        return None
    except Exception:
        if remove_stale:
            _remove(run_file)
        return None
    if _health_ok(port, timeout):
        info["port"] = port
        return info
    if remove_stale:
        _remove(run_file)
    return None


def running_instance(data_dir: Path) -> int | None:
    info = running_instance_info(data_dir)
    return None if info is None else info["port"]


# Kept at module level for the life of the process (see acquire_instance_lock).
_instance_lock = None


def acquire_instance_lock(data_dir: Path):
    """Take the process-lifetime single-instance lock for ``data_dir``.

    Returns an opaque handle (keep it alive) or None when another process holds it.
    The lock exists from the very start of a launch, long before instance.json is
    written, so a second double-click during the slow first start is detected.
    """
    data_dir = Path(data_dir)
    if sys.platform == "win32":
        key = hashlib.sha1(str(data_dir).lower().encode("utf-8", "replace")).hexdigest()[:16]
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.restype = ctypes.c_void_p
        kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        ctypes.set_last_error(0)
        handle = kernel32.CreateMutexW(None, False, "Local\\MONOLITH-" + key)
        err = ctypes.get_last_error()
        if not handle:
            raise OSError(f"CreateMutexW failed (error {err})")
        if err == 183:  # ERROR_ALREADY_EXISTS
            kernel32.CloseHandle(handle)
            return None
        return ("mutex", handle)
    import fcntl

    fd = os.open(str(data_dir / "instance.lock"), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return ("flock", fd)


def release_instance_lock(lock) -> None:
    if lock is None:
        return
    kind, handle = lock
    try:
        if kind == "mutex":
            ctypes.WinDLL("kernel32").CloseHandle(ctypes.c_void_p(handle))
        else:
            os.close(handle)  # closing the descriptor drops the flock
    except Exception:
        pass


def wait_for_instance(data_dir: Path, timeout: float = 180.0, interval: float = 1.0) -> dict | None:
    """Poll instance.json + health until the other launcher's server answers."""
    deadline = time.time() + timeout
    while True:
        info = running_instance_info(data_dir, remove_stale=False, timeout=2.0)
        if info is not None:
            return info
        if time.time() >= deadline:
            return None
        time.sleep(interval)


def _reuse_instance(info: dict, version: str, open_browser: bool) -> int:
    if info.get("version") != version:
        print("An older MONOLITH is still running. Close its black window, then start this one again.")
        return 1
    url = f"http://{HOST}:{info['port']}"
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    print(f"MONOLITH is already running at {url}")
    return 0


def _disable_quick_edit() -> None:
    """Windows 10 console: a stray click selects text and freezes the process (QuickEdit)."""
    log = logging.getLogger("monolith.launcher")
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetStdHandle.restype = ctypes.c_void_p
        kernel32.GetStdHandle.argtypes = [ctypes.c_ulong]
        kernel32.GetConsoleMode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel32.SetConsoleMode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        handle = kernel32.GetStdHandle(ctypes.c_ulong(-10 & 0xFFFFFFFF))  # STD_INPUT_HANDLE
        mode = ctypes.c_ulong(0)
        if not handle or not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            log.warning("QuickEdit: GetConsoleMode failed (error %s)", ctypes.get_last_error())
            return
        new_mode = (mode.value & ~0x40) | 0x80  # clear QUICK_EDIT, set EXTENDED_FLAGS
        if not kernel32.SetConsoleMode(handle, new_mode):
            log.warning("QuickEdit: SetConsoleMode failed (error %s)", ctypes.get_last_error())
    except Exception:
        log.warning("QuickEdit: could not adjust console mode:\n%s", traceback.format_exc())


def remove_run_file_if_mine(data_dir: Path) -> None:
    """Delete instance.json only if it records this process's pid."""
    run_file = Path(data_dir) / RUN_FILE
    try:
        pid = json.loads(run_file.read_text(encoding="utf-8")).get("pid")
    except Exception:
        return
    if pid == os.getpid():
        _remove(run_file)


def _safe(value) -> str:
    return str(value).encode("ascii", "backslashreplace").decode("ascii")


_OWNED = "_monolith_launcher_handler"


def configure_logging(data_dir: Path) -> Path:
    log_dir = Path(data_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "monolith.log"

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(fmt)
    for h in (file_handler, console):
        setattr(h, _OWNED, True)

    root = logging.getLogger()
    for h in list(root.handlers):  # drop handlers from an earlier call
        if getattr(h, _OWNED, False):
            root.removeHandler(h)
            h.close()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(console)

    for name, level in (("uvicorn", logging.INFO), ("uvicorn.error", logging.INFO),
                        ("uvicorn.access", logging.WARNING)):
        lg = logging.getLogger(name)
        lg.setLevel(level)
        lg.propagate = True
    return log_path


def _install_stop_handler(server, stopped: threading.Event) -> None:
    """Make Ctrl+C / Ctrl+Break / window close / SIGTERM stop the server cleanly.

    Handlers are installed BEFORE server.run(): uvicorn's capture_signals swaps
    in its own handlers, then restores ours and re-raises any captured signal.
    With ours in place that re-raise is harmless (no KeyboardInterrupt / kill).
    """
    global _console_handler

    def request_stop(*_args) -> None:
        server.should_exit = True

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)

    if sys.platform == "win32":
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, request_stop)

        handler_type = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_ulong)

        def console_ctrl(event: int) -> bool:
            # 0 = CTRL_C, 1 = CTRL_BREAK, 2 = CTRL_CLOSE
            if event in (0, 1, 2):
                server.should_exit = True
                # Windows kills the process shortly after this returns for
                # CTRL_CLOSE, so block until the server has shut down.
                stopped.wait(_STOP_WAIT_SECONDS)
                return True
            return False

        _console_handler = handler_type(console_ctrl)
        if not ctypes.windll.kernel32.SetConsoleCtrlHandler(_console_handler, True):
            logging.getLogger("monolith.launcher").warning("SetConsoleCtrlHandler failed")


def _announce_when_ready(*args) -> None:
    try:
        _announce(*args)
    except Exception:
        logging.getLogger("monolith.launcher").error("Announcer failed:\n%s", traceback.format_exc())


def _announce(server, port, data_dir, version, log_path, open_browser) -> None:
    url = f"http://{HOST}:{port}"
    deadline = time.time() + 120
    while time.time() < deadline and not server.should_exit:
        if _health_ok(port, 1.0):
            break
        time.sleep(0.3)
    else:
        if not server.should_exit:
            print("")
            print("MONOLITH is taking longer than expected to start. "
                  f"If nothing happens, send the log file: {_safe(log_path)}", flush=True)
        return
    run_file = Path(data_dir) / RUN_FILE
    try:
        tmp = run_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"port": port, "pid": os.getpid(), "version": version}), encoding="utf-8")
        os.replace(tmp, run_file)
    except OSError:
        pass
    print("")
    print("  ==============================================")
    print("   MONOLITH Relief Studio")
    print("  ==============================================")
    print(f"   Version:      {version}")
    print(f"   Address:      {url}")
    print(f"   Data folder:  {_safe(data_dir)}")
    print(f"   Log file:     {_safe(log_path)}")
    print("")
    print("   Keep this window open while you use MONOLITH.")
    print("   Close it to quit.")
    print("  ==============================================")
    print("", flush=True)
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    global _instance_lock
    print("Starting MONOLITH, please wait (the first start can take a minute)...", flush=True)
    parser = argparse.ArgumentParser(prog="monolith-launcher")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args(argv)

    log_path: Path | None = None
    data_dir: Path | None = None
    lock = None
    stopped = threading.Event()
    try:
        data_dir = Path(args.data_dir).expanduser() if args.data_dir else default_data_dir()
        data_dir.mkdir(parents=True, exist_ok=True)
        app_root = Path(__file__).resolve().parent.parent
        version = read_version(app_root)

        lock = acquire_instance_lock(data_dir)
        if lock is None:
            print("MONOLITH is already starting or running in another window...", flush=True)
            info = wait_for_instance(data_dir, 180)
            if info is None:
                print("The other MONOLITH window did not finish starting. "
                      "Close it, then start MONOLITH again.")
                return 1
            return _reuse_instance(info, version, not args.no_browser)
        _instance_lock = lock

        os.environ["MONOLITH_DATA_DIR"] = str(data_dir)
        os.environ["MONOLITH_DB_PATH"] = str(data_dir / "monolith_vault.db")
        ui_dir = app_root / "ui"
        if ui_dir.is_dir():
            os.environ["MONOLITH_UI_DIR"] = str(ui_dir)

        # We hold the lock, so no other launcher is mid-start: a stale file is safe to delete.
        existing = running_instance_info(data_dir, remove_stale=True)
        if existing is not None:
            return _reuse_instance(existing, version, not args.no_browser)

        log_path = configure_logging(data_dir)
        if sys.platform == "win32":
            _disable_quick_edit()
        import uvicorn

        from backend.main import app  # after env is set

        sock = reserve_port(args.port)  # stays bound until uvicorn takes it over
        port = sock.getsockname()[1]
        server = uvicorn.Server(
            uvicorn.Config(app, host=HOST, port=port, log_config=None, timeout_graceful_shutdown=2)
        )
        _install_stop_handler(server, stopped)
        threading.Thread(
            target=_announce_when_ready,
            args=(server, port, data_dir, version, log_path, not args.no_browser),
            daemon=True,
        ).start()
        server.run(sockets=[sock])
        return 0
    except BaseException as exc:  # noqa: BLE001
        if isinstance(exc, SystemExit) and not exc.code:
            return 0
        try:
            logging.getLogger("monolith.launcher").error("Launcher failed:\n%s", traceback.format_exc())
        except Exception:
            pass
        print("MONOLITH could not start.")
        if log_path is not None:
            print("Details were saved to the log file:")
            print(f"  {_safe(log_path)}")
        else:
            print(f"Reason: {_safe(exc)}")
        return 1
    finally:
        if data_dir is not None:
            remove_run_file_if_mine(data_dir)
        stopped.set()
        if lock is not None:
            release_instance_lock(lock)
            _instance_lock = None


if __name__ == "__main__":
    sys.exit(main())
