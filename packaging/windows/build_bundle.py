"""Build the portable Windows bundle (MONOLITH/ folder + zip).

    python packaging/windows/build_bundle.py --out dist-win --version "abc1234 2026-11-14"

Layout produced under --out:

    MONOLITH/
      Start MONOLITH.bat, Open MONOLITH data folder.bat, READ ME FIRST.txt
      python/   embeddable CPython + runtime packages      (skipped with --skip-python)
      app/      backend/ (no tests), benchmarks/<png>, ui/ (frontend/dist), VERSION.txt
    MONOLITH-<yyyymmdd>-<sha7>-windows.zip

Stdlib only. The Python download / pip steps are meant to run on a Windows
machine (GitHub Actions windows-latest); --skip-python builds just the layout
and works anywhere.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

PYTHON_ZIP_URL = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
BENCHMARK = "geometric-study.png"
ROOT_FILES = ("Start MONOLITH.bat", "Open MONOLITH data folder.bat", "READ ME FIRST.txt")
PTH_LINES = ["python310.zip", ".", "..\\app", "Lib\\site-packages", "import site"]
# Packages pip itself brings in; removed afterwards to keep the bundle small.
PIP_LEFTOVERS = ("pip", "setuptools", "wheel", "pkg_resources", "_distutils_hack")
# A stray .pth from setuptools would print a traceback on every launch once pip is gone.
PTH_LEFTOVERS = ("distutils-precedence.pth",)
# cv2.pyd (and other wheels) link the MSVC runtime dynamically but do not ship it, and the
# target PC may lack the VC++ 2015-2022 redistributable. DLLs placed next to python.exe are
# found by extension modules, so we copy the redistributable set there at build time.
VC_RUNTIME_DLLS = (
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
    "msvcp140_atomic_wait.dll",
    "msvcp140_codecvt_ids.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "concrt140.dll",
    "vcomp140.dll",
)


def download(url: str, dest: Path) -> None:
    print(f"downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "monolith-bundle-builder"})
    with urllib.request.urlopen(req, timeout=300) as resp, open(dest, "wb") as fh:
        shutil.copyfileobj(resp, fh)


def copy_crlf(src: Path, dest: Path) -> None:
    """Copy a text file with CRLF line endings (what cmd.exe and Notepad expect)."""
    text = src.read_bytes().decode("utf-8").replace("\r\n", "\n").replace("\n", "\r\n")
    dest.write_bytes(text.encode("utf-8"))


def stamp_from_version(version: str) -> tuple[str, str]:
    """'abc1234def 2026-11-14' -> ('20261114', 'abc1234')."""
    parts = version.split()
    if len(parts) < 2:
        raise SystemExit(f'--version must look like "SHA YYYY-MM-DD", got {version!r}')
    sha7 = parts[0][:7]
    date = re.sub(r"\D", "", parts[1])[:8]
    if len(parts[0]) < 7 or len(date) != 8:
        raise SystemExit(f'--version must look like "SHA YYYY-MM-DD", got {version!r}')
    return date, sha7


def build_app(app: Path, version: str) -> None:
    dist = REPO / "frontend" / "dist"
    if not (dist / "index.html").is_file():
        raise SystemExit("frontend/dist/index.html is missing: run `npm run build` in frontend/ first")
    bench = REPO / "benchmarks" / BENCHMARK
    if not bench.is_file():
        raise SystemExit(f"benchmark image missing: {bench}")

    shutil.copytree(
        REPO / "backend",
        app / "backend",
        ignore=shutil.ignore_patterns("tests", "__pycache__", "*.pyc", "*.lock", "requirements*.txt"),
    )
    (app / "benchmarks").mkdir(parents=True)
    shutil.copy2(bench, app / "benchmarks" / BENCHMARK)
    shutil.copytree(dist, app / "ui")
    (app / "VERSION.txt").write_text(version.strip() + "\n", encoding="utf-8")


def build_python(bundle: Path, python_zip_url: str) -> None:
    py_dir = bundle / "python"
    py_dir.mkdir()
    zip_path = bundle.parent / "_python-embed.zip"
    download(python_zip_url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(py_dir)
    zip_path.unlink()

    pth = py_dir / "python310._pth"
    pth.write_bytes(("\r\n".join(PTH_LINES) + "\r\n").encode("ascii"))
    site_packages = py_dir / "Lib" / "site-packages"
    site_packages.mkdir(parents=True, exist_ok=True)

    exe = py_dir / "python.exe"
    get_pip = bundle.parent / "_get-pip.py"
    download(GET_PIP_URL, get_pip)
    subprocess.run([str(exe), str(get_pip), "--no-warn-script-location", "--no-setuptools", "--no-wheel", "pip<27"], check=True)
    get_pip.unlink()

    subprocess.run(
        [
            str(exe),
            "-m",
            "pip",
            "install",
            "--no-warn-script-location",
            "--no-cache-dir",
            "--only-binary=:all:",
            "-r",
            str(REPO / "backend" / "requirements-runtime.txt"),
            "-c",
            str(REPO / "backend" / "requirements.lock"),
        ],
        check=True,
    )

    # Drop pip & friends (and their dist-info) plus the Scripts folder.
    for entry in list(site_packages.iterdir()):
        name = entry.name.lower()
        if any(name == p or name.startswith(p + "-") for p in PIP_LEFTOVERS):
            shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
        elif name in PTH_LEFTOVERS:
            entry.unlink()
    shutil.rmtree(py_dir / "Scripts", ignore_errors=True)

    bundle_vc_runtime(py_dir)


def bundle_vc_runtime(py_dir: Path) -> list[str]:
    """App-local deployment of the VC++ runtime (allowed by Microsoft's redistribution terms)."""
    system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    copied = []
    for name in VC_RUNTIME_DLLS:
        src, dest = system32 / name, py_dir / name
        if src.is_file() and not dest.exists():
            shutil.copy2(src, dest)
            copied.append(name)
    print("VC++ runtime DLLs copied into python/: " + (", ".join(copied) or "none (already present or not found)"))
    return copied


def make_zip(out: Path, bundle: Path, zip_name: str) -> Path:
    zip_path = out / zip_name
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(out).as_posix())
    return zip_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the MONOLITH portable Windows bundle.")
    ap.add_argument("--out", required=True, help="output directory (MONOLITH/ and the zip go here)")
    ap.add_argument("--version", required=True, help='"SHA YYYY-MM-DD", written to app/VERSION.txt')
    ap.add_argument("--skip-python", action="store_true", help="layout only: no Python download or pip install")
    ap.add_argument("--python-zip", default=PYTHON_ZIP_URL, help="URL of the embeddable Python zip")
    args = ap.parse_args(argv)

    date, sha7 = stamp_from_version(args.version)
    out = Path(args.out).resolve()
    bundle = out / "MONOLITH"
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)

    shutil.copy2(REPO / "LICENSE", bundle / "LICENSE")
    build_app(bundle / "app", args.version)
    for name in ROOT_FILES:
        copy_crlf(HERE / name, bundle / name)
    if not args.skip_python:
        build_python(bundle, args.python_zip)

    zip_path = make_zip(out, bundle, f"MONOLITH-{date}-{sha7}-windows.zip")
    print(f"bundle: {bundle}")
    print(f"zip:    {zip_path} ({zip_path.stat().st_size / (1024 * 1024):.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
