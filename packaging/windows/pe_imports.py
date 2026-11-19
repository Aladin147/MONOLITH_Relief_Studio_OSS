"""CI helper: list the DLLs a PE file imports, and say which are not satisfied.

    python pe_imports.py BUNDLE_PYTHON_DIR FILE_OR_DIR [...]

Stdlib only. For each .pyd/.dll found, prints its imported DLL names. An import is
"resolved" if the DLL sits next to python.exe, in the file's own folder, anywhere in the
bundle's site-packages (*.libs folders included), is an api-ms-win-* / ext-ms-* forwarder,
or exists in the host's System32 (a stand-in for "ships with Windows"). Informational: the
script always exits 0.
"""
from __future__ import annotations

import os
import struct
import sys
from pathlib import Path


def rva_to_off(sections, rva):
    for va, vsize, raw, rsize in sections:
        if va <= rva < va + max(vsize, rsize):
            return rva - va + raw
    return None


def cstr(data, off):
    end = data.index(b"\0", off)
    return data[off:end].decode("ascii", "replace")


def imports(path: Path) -> list[str]:
    data = path.read_bytes()
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        return []
    nsec, = struct.unpack_from("<H", data, pe + 6)
    opt_size, = struct.unpack_from("<H", data, pe + 20)
    opt = pe + 24
    magic, = struct.unpack_from("<H", data, opt)
    dd = opt + (112 if magic == 0x20B else 96)
    names = []
    # Directory 1 = import table, 13 = delay-load imports.
    sec_off = opt + opt_size
    sections = []
    for i in range(nsec):
        s = sec_off + 40 * i
        vsize, va, rsize, raw = struct.unpack_from("<IIII", data, s + 8)
        sections.append((va, vsize, raw, rsize))
    for idx, stride, name_at in ((1, 20, 12), (13, 32, 4)):
        rva, size = struct.unpack_from("<II", data, dd + 8 * idx)
        off = rva_to_off(sections, rva) if rva else None
        while off is not None:
            entry = data[off:off + stride]
            if len(entry) < stride or not any(entry):
                break
            nrva, = struct.unpack_from("<I", entry, name_at)
            noff = rva_to_off(sections, nrva)
            if noff is not None:
                names.append(cstr(data, noff))
            off += stride
    return names


def main() -> int:
    py_dir = Path(sys.argv[1])
    bundle_files = {p.name.lower() for p in py_dir.rglob("*") if p.suffix.lower() in (".dll", ".pyd")}
    system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    unresolved = {}
    targets = []
    for arg in sys.argv[2:]:
        p = Path(arg)
        targets += [p] if p.is_file() else [q for q in sorted(p.rglob("*")) if q.suffix.lower() in (".pyd", ".dll")]
    for t in targets:
        for dll in sorted(set(imports(t)), key=str.lower):
            low = dll.lower()
            if low in bundle_files:
                where = "bundle"
            elif low.startswith(("api-ms-win-", "ext-ms-")):
                where = "windows (api set)"
            elif (system32 / dll).is_file():
                where = "system32 (not app-local)"
            else:
                where = "UNRESOLVED"
            print(f"{t.name}: {dll}  [{where}]")
            if where != "bundle" and not where.startswith("windows"):
                unresolved.setdefault(dll, where)
    print("--- imports not satisfied by the bundle itself ---")
    for dll, where in sorted(unresolved.items(), key=lambda kv: kv[0].lower()):
        print(f"{dll}  [{where}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
