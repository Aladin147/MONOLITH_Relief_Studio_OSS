"""Tests for the sub-pixel SVG / DXF silhouette outline export."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import main as api_main
from backend.pipeline import preprocess_lineart
from backend.pipeline.outline_exporter import encode_dxf, encode_svg, extract_outline
from backend.pipeline.preprocessor import HD_SIGMOID_K
from backend.tests.conftest import make_lineart_png

S = 512


def _antialiased_ring(r_out: int = 180, r_in: int = 70, up: int = 8) -> np.ndarray:
    big = np.zeros((S * up, S * up), np.uint8)
    cv2.circle(big, (S * up // 2, S * up // 2), r_out * up, 255, -1)
    if r_in:
        cv2.circle(big, (S * up // 2, S * up // 2), r_in * up, 0, -1)
    return cv2.resize(big, (S, S), interpolation=cv2.INTER_AREA)


def _signed_area_yup(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _parse_dxf(data: bytes) -> dict:
    """Minimal R12 reader: group-code pairs -> polylines per layer plus header values."""
    lines = data.decode("ascii").splitlines()
    assert len(lines) % 2 == 0
    pairs = [(int(lines[i].strip()), lines[i + 1].strip()) for i in range(0, len(lines), 2)]
    assert pairs[-1] == (0, "EOF")
    header = {}
    polylines = []
    i = 0
    while i < len(pairs):
        code, val = pairs[i]
        if code == 9:
            header[val] = pairs[i + 1][1]
        if code == 0 and val == "POLYLINE":
            layer, closed, verts = None, False, []
            i += 1
            while pairs[i] != (0, "VERTEX") and pairs[i] != (0, "SEQEND"):
                if pairs[i][0] == 8:
                    layer = pairs[i][1]
                if pairs[i][0] == 70:
                    closed = int(pairs[i][1]) & 1 == 1
                i += 1
            while pairs[i] == (0, "VERTEX"):
                i += 1
                x = y = None
                while pairs[i][0] != 0:
                    if pairs[i][0] == 10:
                        x = float(pairs[i][1])
                    if pairs[i][0] == 20:
                        y = float(pairs[i][1])
                    i += 1
                verts.append((x, y))
            assert pairs[i] == (0, "SEQEND")
            polylines.append({"layer": layer, "closed": closed, "pts": np.array(verts)})
        i += 1
    return {"header": header, "polylines": polylines}


def test_outline_is_subpixel_accurate_and_beats_pixel_contour():
    prep = preprocess_lineart(_antialiased_ring(), target_res=S)
    loops = extract_outline(prep["L_clean"], prep["M_sil"], physical_width_mm=S,  # 1 mm == 1 px
                            tolerance_mm=0.0, sigmoid_k=HD_SIGMOID_K)
    assert len(loops) == 2 and sum(lp.is_hole for lp in loops) == 1
    worst = 0.0
    for lp in loops:
        r = np.hypot(lp.points[:, 0] - S / 2, lp.points[:, 1] - S / 2)
        err = np.abs(r - (70 if lp.is_hole else 180))
        assert float(err.mean()) < 0.2
        worst = max(worst, float(err.max()))
    # A pixel-quantized trace (cv2.findContours) is off by up to ~1 px here.
    assert worst < 0.45


def test_outline_topology_matches_silhouette():
    # Interior contour lines must not become extra loops.
    prep = preprocess_lineart(make_lineart_png(256), target_res=256)
    loops = extract_outline(prep["L_clean"], prep["M_sil"], physical_width_mm=256, tolerance_mm=0.0)
    assert len(loops) == 1 and not loops[0].is_hole
    area = abs(_signed_area_yup(loops[0].points))
    assert area == pytest.approx(float((prep["M_sil"] > 0.5).sum()), rel=0.01)


def test_outline_closes_along_canvas_border():
    img = np.zeros((128, 128), np.uint8)
    img[:, :64] = 255  # touches left/top/bottom edges
    prep = preprocess_lineart(img, target_res=128)
    loops = extract_outline(prep["L_clean"], prep["M_sil"], physical_width_mm=128, tolerance_mm=0.0)
    assert len(loops) == 1
    pts = loops[0].points
    assert float(pts[:, 0].min()) == pytest.approx(0.0, abs=0.05)
    assert float(pts[:, 1].min()) == pytest.approx(0.0, abs=0.05)
    assert float(pts[:, 1].max()) == pytest.approx(128.0, abs=0.05)


def test_dxf_structure_units_layers_and_orientation():
    prep = preprocess_lineart(_antialiased_ring(), target_res=S)
    loops = extract_outline(prep["L_clean"], prep["M_sil"], physical_width_mm=1000.0)
    dxf = _parse_dxf(encode_dxf(loops, prep["M_sil"].shape, 1000.0))
    assert dxf["header"]["$ACADVER"] == "AC1009"
    assert dxf["header"]["$INSUNITS"] == "4"
    polys = dxf["polylines"]
    assert len(polys) == 2 and all(p["closed"] for p in polys)
    outer = [p for p in polys if p["layer"] == "OUTLINE_OUTER"]
    holes = [p for p in polys if p["layer"] == "OUTLINE_HOLES"]
    assert len(outer) == 1 and len(holes) == 1
    # CAD axes (y up): outer counter-clockwise, holes clockwise.
    assert _signed_area_yup(outer[0]["pts"]) > 0 > _signed_area_yup(holes[0]["pts"])
    # True millimetres: outer radius 180/512 of a 1000 mm canvas.
    r = np.hypot(outer[0]["pts"][:, 0] - 500.0, outer[0]["pts"][:, 1] - 500.0)
    assert float(np.abs(r - 180 * 1000 / S).max()) < 1.0


def test_svg_structure_and_size():
    prep = preprocess_lineart(_antialiased_ring(), target_res=S)
    loops = extract_outline(prep["L_clean"], prep["M_sil"], physical_width_mm=600.0)
    root = ET.fromstring(encode_svg(loops, prep["M_sil"].shape, 600.0))
    assert root.get("width") == "600.0000mm" and root.get("height") == "600.0000mm"
    assert root.get("viewBox") == "0 0 600.0000 600.0000"
    ns = {"s": "http://www.w3.org/2000/svg"}
    assert len(root.findall(".//s:g[@id='outline_outer']/s:path", ns)) == 1
    assert len(root.findall(".//s:g[@id='outline_holes']/s:path", ns)) == 1


def test_simplification_tolerance_is_respected():
    prep = preprocess_lineart(_antialiased_ring(), target_res=S)
    full = extract_outline(prep["L_clean"], prep["M_sil"], physical_width_mm=S, tolerance_mm=0.0)
    simp = extract_outline(prep["L_clean"], prep["M_sil"], physical_width_mm=S, tolerance_mm=0.25)
    n_full = sum(len(lp.points) for lp in full)
    n_simp = sum(len(lp.points) for lp in simp)
    assert n_simp < 0.6 * n_full
    for lp in simp:
        r = np.hypot(lp.points[:, 0] - S / 2, lp.points[:, 1] - S / 2)
        assert float(np.abs(r - (70 if lp.is_hole else 180)).max()) < 0.45 + 0.25


@pytest.mark.parametrize("fmt,ctype", [("svg", "image/svg+xml"), ("dxf", "application/dxf")])
def test_outline_export_endpoint(isolated_env, fmt, ctype):
    with TestClient(api_main.app) as client:
        r = client.post("/api/projects/create",
                        files={"file": ("ring.png", cv2.imencode(".png", _antialiased_ring())[1].tobytes(), "image/png")},
                        data={"target_res": "512"})
        pid = r.json()["id"]
        client.post(f"/api/projects/{pid}/recompute", json={"physical_width_mm": 800.0})
        exp = client.post(f"/api/projects/{pid}/export", json={"format": fmt, "width_mm": 800.0})
        assert exp.status_code == 200
        assert exp.headers["content-type"].startswith(ctype)
        assert f"_outline.{fmt}" in exp.headers["content-disposition"]
        if fmt == "dxf":
            assert len(_parse_dxf(exp.content)["polylines"]) == 2
        bad = client.post(f"/api/projects/{pid}/export", json={"format": fmt, "outline_tolerance_mm": -1})
        assert bad.status_code == 422
