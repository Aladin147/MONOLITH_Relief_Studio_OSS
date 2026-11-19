"""Tests for the Sculpt height engine (backend.pipeline.sculpt_engine) and its dispatch."""

from __future__ import annotations

import copy
import hashlib
import os
import platform
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.pipeline import sculpt_engine as se
from backend.pipeline.relief_engine import compute_basis_layers, recompute_relief
from backend.pipeline.sculpt_engine import (
    SCULPT_DEFAULTS,
    SCULPT_LATEST_VERSION,
    build_sculpt_basis,
    cushion,
    cushion2,
    rebuild_parts,
    sculpt_profile,
    sculpt_radii,
    seam_mismatch,
)
from backend.pipeline.sculpt_order import auto_layers
from backend.pipeline.sculpt_parts import analyze_parts, part_at

PERF_SLACK = float(os.environ.get("MONOLITH_PERF_SLACK", "1"))
ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "benchmarks" / "geometric-study.png"
PW, RZ = 1000.0, 15.0  # physical width / relief depth (mm) used where the draft clamp is off
# slider grid of the seam sweep (gate G8): layering x fullness (down to its floor) x edge
# roundness, plus the crease extremes
SWEEP = [dict(layering=L, fullness=F, edge_roundness_pct=R) for L in (0.3, 0.6, 1.0) for F in (0.2, 0.55, 0.9) for R in (3.0, 6.0)]
SWEEP += [dict(crease_depth=0.5), dict(crease_width_pct=3.0), dict(crease_depth=0.5, crease_width_pct=3.0, layering=1.0, fullness=0.2)]


# --------------------------------------------------------------------------- synthetic scenes


def _ring(n, cx, cy, r, half):
    yy, xx = np.mgrid[:n, :n]
    return np.abs(np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) - r) <= half


def _disc(n, cx, cy, r):
    yy, xx = np.mgrid[:n, :n]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def _disc_over_rect(n: int):
    """White-on-black: a disc (contour fully drawn) in front of a rectangle whose contour stops at it."""
    s = n / 1024.0
    disc = _disc(n, 400 * s, 512 * s, 190 * s)
    rect = np.zeros((n, n), bool)
    rect[int(435 * s) : int(590 * s), int(400 * s) : int(875 * s)] = True
    sil = disc | rect
    white = sil & ~_ring(n, 400 * s, 512 * s, 190 * s, max(1.0, 1.5 * s))
    return white, sil


def _figure(n: int):
    """A 'figure': torso, a head and an arm drawn in front of it, and two open strokes."""
    t = max(2, n // 512)  # line thickness
    sil = np.zeros((n, n), np.uint8)
    lines = np.zeros((n, n), np.uint8)
    torso = (int(0.25 * n), int(0.22 * n), int(0.75 * n), int(0.88 * n))  # x0, y0, x1, y1
    cv2.rectangle(sil, torso[:2], torso[2:], 1, -1)
    head = (int(0.5 * n), int(0.16 * n)), int(0.12 * n)
    cv2.circle(sil, *head, 1, -1)
    cv2.circle(lines, *head, 1, t)
    arm = np.array([[0.12, 0.40], [0.60, 0.33], [0.63, 0.42], [0.15, 0.50]]) * n
    cv2.fillPoly(sil, [arm.astype(np.int32)], 1)
    cv2.polylines(lines, [arm.astype(np.int32)], True, 1, t)
    cv2.line(lines, (int(0.35 * n), int(0.62 * n)), (int(0.62 * n), int(0.66 * n)), 1, t)  # open stroke
    cv2.line(lines, (int(0.5 * n), int(0.70 * n)), (int(0.5 * n), int(0.84 * n)), 1, t)  # open stroke
    silb = sil > 0
    white = silb & (lines == 0)
    return white, silb


def _build(white: np.ndarray, sil: np.ndarray, constraints=(), L=None):
    if L is None:
        L = (white & sil).astype(np.float32)
    S = sil.astype(np.float32)
    pa = analyze_parts(L, S, L * S)
    layers = auto_layers(pa, list(constraints))
    basis = build_sculpt_basis(pa, layers, S, S.shape, L_clean=L)
    return pa, layers, basis, L, S


def _params(**kw):
    p = {"style": "sculpt", "h_min": 0.08, "invert_depth": False, "draft_angle_deg": 0.0}
    p.update(SCULPT_DEFAULTS)
    p.update(kw)
    return p


def _profile(basis, params, pw=PW, rz=RZ):
    out = np.zeros(basis["labels"].shape, np.float32)
    sculpt_profile(basis, params, pw, rz, out)
    return out


def _copy(basis):
    return {k: (copy.deepcopy(v) if k != "_lock" else threading.Lock()) for k, v in basis.items()}


def _seam_steps(H, basis, params):
    """Level difference across every seam: per adjacent pair of parts, the median height on each
    side in a band from just past the crease to one edge roundness away from the seam, away from
    the outline (so neither side samples its outer wall)."""
    labels = basis["labels"].astype(np.int32)
    R, Wc, _ = sculpt_radii(basis, params, PW, RZ)
    d_open = basis["d_open_px"].astype(np.float32)
    pairs = set()
    for la, lb in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        m = (la > 0) & (lb > 0) & (la != lb)
        pairs |= {(min(a, b), max(a, b)) for a, b in zip(la[m].tolist(), lb[m].tolist())}
    steps = {}
    for a, b in sorted(pairs):
        med = []
        for p, q in ((a, b), (b, a)):
            d = cv2.distanceTransform((labels != q).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            band = (labels == p) & (d > Wc + 2) & (d <= Wc + 2 + R) & (d_open > R)
            med.append(float(np.median(H[band])))
        steps[(a, b)] = abs(med[0] - med[1])
    return steps


def _seam_jumps(H, labels):
    """Jump across each seam pixel pair beyond the slope just either side of it (a step, not a wall)."""
    labels = labels.astype(np.int32)
    out = []
    for ax in (0, 1):
        def at(a, i, n=3):
            s = [slice(None)] * 2
            s[ax] = slice(i, a.shape[ax] - n + i)
            return a[tuple(s)]

        h = [at(H, i) for i in range(4)]
        lab = [at(labels, i) for i in range(4)]
        m = (lab[1] > 0) & (lab[2] > 0) & (lab[1] != lab[2]) & (lab[0] == lab[1]) & (lab[3] == lab[2])
        jump = np.abs(h[2] - h[1]) - np.maximum(np.abs(h[1] - h[0]), np.abs(h[3] - h[2]))
        out.append(jump[m])
    return np.concatenate(out)


def _sculpt_layers(basis, S):
    sil = (S > 0.5).astype(np.uint8)
    d_sil = cv2.distanceTransform(sil, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return {"M_sil": S, "d_sil": d_sil, "sculpt_basis": basis}


def _interior_slope_excess(H, S, d_sil, pw, rz, theta):
    """Largest interior slope (mm/mm) / cot(theta), over pixels > 2 px inside the outline that the
    shared tail's exact outer envelope does not cap (that envelope is Classic's, unchanged)."""
    px = pw / S.shape[1]
    gy, gx = np.gradient(H.astype(np.float64) * rz, px)
    slope = np.hypot(gx, gy)
    env = (d_sil - 0.5) * (px / (np.tan(np.radians(theta)) * rz))
    capped = cv2.dilate((H >= env - 1e-6).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    inner = (S > 0.5) & (d_sil > 2) & ~capped
    return float(slope[inner].max() * np.tan(np.radians(theta))), int(inner.sum())


@pytest.fixture(scope="module")
def dor():
    return _build(*_disc_over_rect(1024))


@pytest.fixture(scope="module")
def fig():
    return _build(*_figure(1024))


# --------------------------------------------------------------------------- unit behaviour


def test_cushion_curves():
    u = np.array([-1.0, 0.0, 0.5, 1.0, 2.0], np.float32)
    c = cushion(u)
    assert c.dtype == np.float32
    np.testing.assert_allclose(c, [0.0, 0.0, np.sin(np.pi / 4), 1.0, 1.0], atol=1e-6)
    e = 1e-3
    assert abs((cushion(np.float32(1.0)) - cushion(np.float32(1.0 - e))) / e) < 1e-2
    assert abs((cushion(np.float32(e)) - cushion(np.float32(0.0))) / e - np.pi / 2) < 1e-2
    buf = u.copy()
    assert cushion(buf, out=buf) is buf  # usable in place
    # the two-scale wall cushion: 0 -> 1, C1 (flat at u = 1 and continuous slope at u = rho),
    # steepest at the foot (slope _FMAX), and the lookup table used by the kernel matches it
    f = lambda x: float(cushion2(np.float64(x)))  # noqa: E731
    assert f(0.0) == 0.0 and abs(f(1.0) - 1.0) < 1e-6 and f(2.0) == pytest.approx(1.0)
    assert abs((f(1.0) - f(1.0 - e)) / e) < 1e-2
    rho = se.CUSHION_RHO
    left, right = (f(rho) - f(rho - e)) / e, (f(rho + e) - f(rho)) / e
    assert abs(left - right) < 2e-2
    assert abs((f(e) - f(0.0)) / e - se._FMAX) < 1e-2
    uu = np.linspace(0, 1, 997)
    np.testing.assert_allclose(se._LUT[np.minimum((uu * se._LUT_N).astype(int), se._LUT_N)], cushion2(uu), atol=4e-4)
    # the seam cushion: a sharper foot (rho_s), same C1 shape, its own table and foot slope
    rs, ws = se.CUSHION_RHO_SEAM, se.CUSHION_W1_SEAM
    fs = lambda x: float(cushion2(np.float64(x), rs, ws))  # noqa: E731
    assert fs(0.0) == 0.0 and abs(fs(1.0) - 1.0) < 1e-6
    assert abs((fs(e * 0.1) - fs(0.0)) / (e * 0.1) - se._FMAX_S) < 2e-2
    es = 1e-4  # the inner arc curves hard (f'' ~ -290 at rho_s): a short step for the C1 check
    assert abs((fs(rs) - fs(rs - es)) / es - (fs(rs + es) - fs(rs)) / es) < 5e-2
    np.testing.assert_allclose(se._LUT_S[np.minimum((uu * se._LUT_N).astype(int), se._LUT_N)], cushion2(uu, rs, ws), atol=2e-3)
    assert se._FMAX_S > se._FMAX
    assert SCULPT_LATEST_VERSION == 1


def _sine_cushion_lut(rho: float, w1: float) -> np.ndarray:
    """The pre-fix (C1) two-scale sine cushion's lookup table (final review I1)."""
    u = np.arange(se._LUT_N + 1, dtype=np.float64) / se._LUT_N
    return (w1 * np.sin(np.pi / 2 * np.minimum(u / rho, 1.0)) + (1 - w1) * np.sin(np.pi / 2 * u)).astype(np.float32)


def _lut_curvature_jump(lut: np.ndarray) -> float:
    """Largest change of the second difference between neighbouring samples of a cushion table,
    extended flat beyond u = 1 as the kernel reads it, over its largest |second difference|.
    Samples every 64th entry (h = 1/256): fine enough to resolve the knots, coarse enough that
    float32 rounding stays negligible."""
    f = np.concatenate([lut[::64].astype(np.float64), np.full(8, float(lut[-1]))])
    h = 64.0 / se._LUT_N
    d2 = (f[2:] - 2 * f[1:-1] + f[:-2]) / h**2
    return float(np.abs(np.diff(d2)).max() / np.abs(d2).max())


def test_cushion_luts_are_c2():
    """Final review I1: the wall and front-seam cushions are C2 (c(t) = 1 - (1 - t)^3), so a lit
    relief shows no Mach line parallel to the walls at the knots (u = rho and u = 1). The second
    derivative of each lookup table changes by at most 3 % of its range between samples, at the
    knots included (the old sine cushion: 93 % / 100 %); one-sided second derivatives agree at
    each knot; the foot slope of the steep term is the sine cushion's ((pi/2) / 0.4)."""
    for lut, rho, w1 in ((se._LUT, se.CUSHION_RHO, se.CUSHION_W1), (se._LUT_S, se.CUSHION_RHO_SEAM, se.CUSHION_W1_SEAM)):
        assert _lut_curvature_jump(lut) <= 0.03, (rho, _lut_curvature_jump(lut))
        f = lambda x: float(w1 * se._c2(x / rho) + (1 - w1) * se._c2(x))  # noqa: E731  (float64)
        np.testing.assert_allclose(cushion2(np.linspace(0, 1.2, 50), rho, w1), [f(x) for x in np.linspace(0, 1.2, 50)], atol=1e-6)
        e = 1e-4 * rho
        for k in (rho, 1.0):
            left = (f(k) - 2 * f(k - e) + f(k - 2 * e)) / e**2
            right = (f(k + 2 * e) - 2 * f(k + e) + f(k)) / e**2
            scale = 6 * (w1 / rho**2 + (1 - w1))  # |f''| at the wall, the largest
            assert abs(left - right) <= 0.01 * scale, (rho, k, left, right)
    assert _lut_curvature_jump(_sine_cushion_lut(0.4, se.CUSHION_W1)) > 0.5
    assert _lut_curvature_jump(_sine_cushion_lut(0.08, se.CUSHION_W1_SEAM)) > 0.5
    assert 3.0 * se.CUSHION_W1 / se.CUSHION_RHO == pytest.approx((np.pi / 2) * se.CUSHION_W1 / 0.4)
    assert 3.0 * se.CUSHION_W1_SEAM / se.CUSHION_RHO_SEAM == pytest.approx((np.pi / 2) * se.CUSHION_W1_SEAM / 0.08)


def test_basis_contents(dor):
    pa, layers, basis, L, S = dor
    H, W = S.shape
    for key in ("d_wall_px", "d_seam_px", "d_open_px", "d_gate_px", "d_crease_px", "d_crease_all_px", "d_thick_px"):
        assert basis[key].shape == (H, W) and basis[key].dtype == np.float16, key
    assert basis["support"].dtype == np.float32 and basis["dome"].dtype == np.float32
    assert basis["ink"].dtype == np.float16 and 0.0 <= float(basis["ink"].max()) <= 1.0
    assert basis["labels"].dtype == np.uint16 and basis["labels"].shape == (H, W)
    assert basis["layer"].dtype == np.int32
    assert np.array_equal(basis["layer"], layers.astype(np.int32))
    assert isinstance(basis["_sculpt_order_key"], str)
    sil = S > 0.5
    assert (basis["labels"][sil] > 0).all() and (basis["labels"][~sil] == 0).all()
    assert 0.0 <= float(basis["dome"].min()) and float(basis["dome"].max()) <= 1.0 + 1e-3


def test_basis_bytes_per_pixel(dor):
    """I4: about 30 bytes per working pixel (the fixed 2048-grid analysis arrays aside; round 2
    added the front-seam distance and the int8 sub-pixel correction)."""
    pa, layers, basis, L, S = dor
    total = sum(v.nbytes for v in basis.values() if isinstance(v, np.ndarray) and v.shape == S.shape and v.base is None)
    assert total / S.size <= 31.0, total / S.size


def test_disc_in_front_has_support(dor):
    pa, layers, basis, L, S = dor
    disc, rect = part_at(pa, 400 / 1024, 512 / 1024), part_at(pa, 800 / 1024, 512 / 1024)
    assert layers[disc] > layers[rect]
    sup = basis["support"]
    assert float(sup[basis["labels"] == rect].max()) == 0.0
    # the disc rests on the rectangle: support near the seam is about one part height
    assert float(sup[int(512), int(570)]) > 0.5


def test_sculpt_blob_is_cushion():
    n = 1024
    blob = _disc(n, 512, 512, 360)
    pa, layers, basis, L, S = _build(blob, blob)
    R, _, _ = sculpt_radii(basis, _params(), PW, RZ)
    # the wall alone (no dome): a cushion, at least 60 % up at R/2 and 95 % up at R
    row = _profile(basis, _params(fullness=1.0))[512]
    inside = np.nonzero(row > 0)[0]
    x0, x1 = int(inside[0]), int(inside[-1])
    top, lo = float(row.max()), float(row[x0])
    for px, need in ((R / 2, 0.6), (R, 0.95)):
        for x in (x0 + int(px), x1 - int(px)):
            assert (row[x] - lo) / (top - lo) >= need, (px, x, row[x], top)
    # broad top at the defaults (with the dome): >= 30 % of the diameter within 5 % of the maximum
    row = _profile(basis, _params())[512]
    top, lo = float(row.max()), float(row[x0])
    assert (row >= top - 0.05 * (top - lo)).sum() >= 0.30 * (x1 - x0)


def test_background_hmin_and_range(dor):
    pa, layers, basis, L, S = dor
    H = _profile(basis, _params(h_min=0.1))
    sil = S > 0.5
    assert (H[~sil] == 0).all()
    assert abs(float(H[sil].min()) - 0.1) < 1e-5 and abs(float(H[sil].max()) - 1.0) < 1e-5
    Hi = _profile(basis, _params(h_min=0.1, invert_depth=True))
    np.testing.assert_allclose(Hi[sil], 1.0 - H[sil], atol=1e-6)
    assert (Hi[~sil] == 0).all()


def test_brightness_contrast_gamma_ignored(dor):
    pa, layers, basis, L, S = dor
    a = _profile(basis, _params())
    b = _profile(basis, _params(brightness=0.3, contrast=1.7, gamma=0.4))
    assert np.array_equal(a, b)


def test_layering_raises_front_part(dor):
    pa, layers, basis, L, S = dor
    disc, rect = part_at(pa, 400 / 1024, 512 / 1024), part_at(pa, 800 / 1024, 512 / 1024)
    lab = basis["labels"]
    plateau = basis["d_wall_px"] > 60
    d = {}
    for lay in (0.0, 0.5, 1.0):
        H = _profile(basis, _params(layering=lay, fullness=1.0))  # no dome: layering alone
        d[lay] = float(np.median(H[(lab == disc) & plateau]) - np.median(H[(lab == rect) & plateau]))
    # at 0 % the plateaus are level; layering lifts the front part monotonically
    assert abs(d[0.0]) < 0.05 and d[0.0] < d[0.5] < d[1.0] and d[1.0] - d[0.0] > 0.2, d


@pytest.mark.parametrize("scene", ["dor", "fig"])
def test_layering_zero_has_no_steps(scene, request):
    pa, layers, basis, L, S = request.getfixturevalue(scene)
    H0 = _profile(basis, _params(layering=0.0))
    # no step anywhere along a seam (2 % of the relief) ...
    assert _seam_jumps(H0, basis["labels"]).max() < 0.02
    # ... and the parts are level across every seam, where 100 % layering steps them (measured
    # without the dome, whose slope would otherwise differ between the two sampling bands)
    p0, p1 = _params(layering=0.0, fullness=1.0), _params(layering=1.0, fullness=1.0)
    level0 = _seam_steps(_profile(basis, p0), basis, p0)
    level1 = _seam_steps(_profile(basis, p1), basis, p1)
    stepped = [k for k in level1 if basis["layer"][k[0]] != basis["layer"][k[1]]]
    assert stepped and min(level1[k] for k in stepped) > 0.2, level1
    assert max(level0.values()) < 0.05, level0


def test_seam_mismatch_measures_steps(dor):
    pa, layers, basis, L, S = dor
    p = _params()
    H = _profile(basis, p)
    assert 0.0 <= seam_mismatch(basis, p, H) <= 0.02
    # the measure is sensitive: lifting the front part by a few % of the relief fails it
    disc = part_at(pa, 400 / 1024, 512 / 1024)
    bad = H.copy()
    bad[basis["labels"] == disc] += 0.03
    assert seam_mismatch(basis, p, bad) > 0.02


@pytest.mark.parametrize("scene", ["dor", "fig"])
def test_seam_mismatch_slider_sweep(scene, request):
    """Gate G8 on the synthetic scenes: the foot follows the surface behind at every slider value."""
    pa, layers, basis, L, S = request.getfixturevalue(scene)
    assert layers.max() >= 1
    worst = 0.0
    for kw in SWEEP:
        p = _params(**kw)
        worst = max(worst, seam_mismatch(basis, p, _profile(basis, p)))
    assert worst <= 0.02, worst


def test_draft_clamp_raises_R(dor):
    pa, layers, basis, L, S = dor
    p0 = _params(draft_angle_deg=0.0)
    p1 = _params(draft_angle_deg=30.0)
    R0, _, c0 = sculpt_radii(basis, p0, 100.0, 15.0)
    R1, _, c1 = sculpt_radii(basis, p1, 100.0, 15.0)
    assert not c0 and c1 and R1 > R0
    # the steepest wall of the two-scale cushion leans at least the draft angle:
    # FMAX * h_mm / R_mm <= cot(theta)  (h = the whole relief, worst case)
    px_mm = 100.0 / S.shape[1]
    assert se._FMAX * 15.0 / (R1 * px_mm) <= 1.0 / np.tan(np.radians(30.0)) + 1e-6
    H0 = _profile(basis, p0, 100.0, 15.0)
    assert basis["_last_edge_clamped"] is False
    H1 = _profile(basis, p1, 100.0, 15.0)
    assert basis["_last_edge_clamped"] is True
    assert not np.array_equal(H0, H1)


def test_thin_part_reaches_full_height():
    """A bar narrower than twice the edge roundness still gets a full-height top (the radius
    adapts to the local thickness) instead of a low ridge."""
    n = 1024
    disc = _disc(n, 300, 512, 250)
    bar = np.zeros((n, n), bool)
    bar[200:820, 700:740] = True  # 40 px wide; R = 7 % of 1024 = 72 px
    sil = disc | bar
    pa, layers, basis, L, S = _build(sil, sil)
    R, _, _ = sculpt_radii(basis, _params(), PW, RZ)
    assert R > 40
    H = _profile(basis, _params(fullness=1.0))
    assert float(H[bar].max()) >= 0.95 * float(H[disc].max())
    # ... and the bar's crest is flat, not a ridge: within 2 % of its top across its middle
    mid = H[500, 716:725]
    assert float(mid.max() - mid.min()) <= 0.02, mid


def test_faint_stroke_makes_faint_crease():
    """Crease depth follows the stroke's ink weight (I2): a faint grey stroke notches less."""
    n = 1024
    sil = np.zeros((n, n), bool)
    sil[200:820, 150:870] = True
    L = sil.astype(np.float32)
    L[400:404, 250:770] = 0.0  # a solid black stroke
    L[620:623, 250:770] = 0.55  # a faint grey one (still a line: darker than the white threshold)
    pa, layers, basis, _L, S = _build(sil, sil, L=L)
    H = _profile(basis, _params(fullness=1.0))

    def notch(y):
        side = 0.5 * (H[y - 30, 300:720].mean() + H[y + 30, 300:720].mean())
        return side - H[y - 3 : y + 4, 300:720].min(axis=0).mean()

    deep, faint = notch(402), notch(621)
    assert deep > 0.03 and faint < 0.6 * deep, (deep, faint)


def test_draft_clamp_bounds_every_slope(fig):
    """With the draft clamp active, no interior slope (thin parts and seams included) exceeds it."""
    pa, layers, basis, L, S = fig
    d_sil = cv2.distanceTransform((S > 0.5).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    lay = _sculpt_layers(basis, S)
    for pw, rz, theta in ((300.0, 15.0, 30.0), (100.0, 15.0, 20.0)):
        H = recompute_relief(lay, _params(draft_angle_deg=theta), physical_width_mm=pw, relief_z_mm=rz)
        assert lay["sculpt_edge_clamped"] is True
        excess, n = _interior_slope_excess(H, S, d_sil, pw, rz, theta)
        assert n > 1000 and excess <= 1.0 + 1e-6, (pw, theta, excess)


def test_rebuild_parts_matches_fresh_build(dor):
    pa, layers, basis, L, S = dor
    disc, rect = part_at(pa, 400 / 1024, 512 / 1024), part_at(pa, 800 / 1024, 512 / 1024)
    new = auto_layers(pa, [(rect, disc)])
    assert new[rect] > new[disc]
    b2 = _copy(basis)
    _profile(b2, _params(layering=0.9))  # a recompute's seam correction must not leak into the rebuild
    rebuild_parts(b2, [rect], new)
    fresh = build_sculpt_basis(pa, new, S, S.shape, L_clean=L)
    assert b2["_sculpt_order_key"] == fresh["_sculpt_order_key"] != basis["_sculpt_order_key"]
    assert np.array_equal(b2["layer"], fresh["layer"])
    for key in ("d_wall_px", "d_seam_px", "d_gate_px", "d_crease_px", "d_thick_px", "support"):
        np.testing.assert_allclose(b2[key].astype(np.float32), fresh[key].astype(np.float32), atol=1e-4, err_msg=key)
    p = _params()
    np.testing.assert_allclose(_profile(b2, p), _profile(fresh, p), atol=1e-5)
    assert seam_mismatch(b2, p, _profile(b2, p)) <= 0.02


def _small_on_top(n: int = 1024):
    """`_disc_over_rect` plus a small disc (1.7 % of the silhouette) in front of both parts."""
    white, sil = _disc_over_rect(n)
    s = n / 1024.0
    small = _disc(n, 590 * s, 470 * s, 30 * s)
    sil2 = sil | small
    white2 = (white & ~small) | (small & ~_ring(n, 590 * s, 470 * s, 30 * s, max(1.0, 1.5 * s)))
    L = (white2 & sil2).astype(np.float32)
    S = sil2.astype(np.float32)
    pa = analyze_parts(L, S, L * S)
    sm, disc, rect = part_at(pa, 590 / 1024, 470 / 1024), part_at(pa, 400 / 1024, 512 / 1024), part_at(pa, 800 / 1024, 512 / 1024)
    return pa, L, S, sm, disc, rect


def test_top_layer_step_factor():
    """N1 (review round 4): the topmost layers covering < 100 - KNEE_PCT % of the silhouette step
    up by Layering x TOP_LAYER_LAMBDA; their feet stay on the surface behind at every slider, and
    a reorder that moves a part across the rule re-solves exactly what a fresh build gives."""
    pa, L, S, sm, disc, rect = _small_on_top()
    layers = auto_layers(pa, [(sm, disc), (sm, rect)])
    basis = build_sculpt_basis(pa, layers, S, S.shape, L_clean=L)
    lam = basis["_lam_part"]
    assert lam[sm] == np.float32(se.TOP_LAYER_LAMBDA) and lam[disc] == 1.0 and lam[rect] == 1.0
    assert basis["_lam_layer"] == {int(layers[sm]): se.TOP_LAYER_LAMBDA}
    worst = 0.0
    for kw in SWEEP:
        p = _params(**kw)
        worst = max(worst, seam_mismatch(basis, p, _profile(basis, p)))
    assert worst <= 0.02, worst
    # the small part stands lower on the disc than it does without the factor
    lab = basis["labels"]
    core = (lab == sm) & (basis["d_wall_px"].astype(np.float32) > 12)
    ring = (lab == disc) & (np.hypot(*np.mgrid[:1024, :1024] - np.array([470, 590])[:, None, None]) < 50)

    def lift(b):
        H = _profile(b, _params(layering=1.0, fullness=1.0))
        return float(np.median(H[core]) - np.median(H[ring & ~(lab == sm)]))

    lam0 = se.TOP_LAYER_LAMBDA
    try:
        se.TOP_LAYER_LAMBDA = 1.0
        full = build_sculpt_basis(pa, layers, S, S.shape, L_clean=L)
    finally:
        se.TOP_LAYER_LAMBDA = lam0
    assert full["_lam_layer"] == {int(layers[sm]): 1.0}
    assert lift(full) - lift(basis) > 0.05, (lift(basis), lift(full))
    # reorders across the rule: the small disc behind (no factor anywhere), then in front again
    for cons in ([(disc, sm), (sm, rect)], [(sm, disc), (sm, rect)]):
        new = auto_layers(pa, cons)
        rebuild_parts(basis, [sm], new)
        fresh = build_sculpt_basis(pa, new, S, S.shape, L_clean=L)
        np.testing.assert_array_equal(basis["_lam_part"], fresh["_lam_part"])
        np.testing.assert_allclose(basis["support"], fresh["support"], atol=1e-4)
        p = _params()
        np.testing.assert_allclose(_profile(basis, p), _profile(fresh, p), atol=1e-5)


def test_fullness_floor(fig):
    """Minor D: Fullness is floored at FULLNESS_MIN (0.2); lower values give the floor's relief."""
    pa, layers, basis, L, S = fig
    assert se.FULLNESS_MIN == 0.2
    H_floor = _profile(basis, _params(fullness=0.2))
    for F in (0.0, 0.1, -1.0):
        assert np.array_equal(_profile(basis, _params(fullness=F)), H_floor), F
    assert not np.array_equal(_profile(basis, _params(fullness=0.3)), H_floor)


@pytest.mark.parametrize("scene", ["dor", "fig"])
def test_recompute_is_deterministic(scene, request):
    """Important C: a recompute depends only on the basis and its own arguments, never on the
    previous call: P after any Q equals P on a fresh basis, bit for bit."""
    pa, layers, basis, L, S = request.getfixturevalue(scene)
    settings = [
        (_params(), PW, RZ),
        (_params(draft_angle_deg=20.0), 100.0, RZ),  # the draft clamp binds
        (_params(layering=1.0, fullness=0.2, edge_roundness_pct=3.0), PW, RZ),
        (_params(crease_depth=0.5, crease_width_pct=3.0, draft_angle_deg=5.0), 300.0, RZ),
    ]
    for P in settings:
        ref = _profile(_copy(basis), P[0], *P[1:])
        for Q in settings:
            b = _copy(basis)
            _profile(b, Q[0], *Q[1:])
            assert np.array_equal(_profile(b, P[0], *P[1:]), ref), (P, Q)


@pytest.mark.parametrize("scene", ["dor", "fig"])
def test_wall_seam_cushions_join_smoothly(scene, request):
    """R1: where a part's outline cushion meets its own front-seam cushion the relief has no
    crease: the second difference on the locus where f(d_wall) and f_s(d_seam) cross is at most
    1.5x its value 3 px beside it (a min of the two gives 5x on the disc-over-rectangle)."""
    pa, layers, basis, L, S = request.getfixturevalue(scene)
    H = _profile(basis, _params())
    R, _Wc, _c = sculpt_radii(basis, _params(), PW, RZ)
    Re = se.edge_radius(basis["d_thick_px"].astype(np.float32), R, 1e-3)
    fw = cushion2(basis["d_wall_px"].astype(np.float32) / Re)
    fs = cushion2(basis["d_seam_px"].astype(np.float32) / Re, se.CUSHION_RHO_SEAM, se.CUSHION_W1_SEAM)
    a = (fw < fs).astype(np.int8)
    sw = np.zeros(a.shape, bool)
    sw[:, :-1] |= a[:, :-1] != a[:, 1:]
    sw[:-1, :] |= a[:-1, :] != a[1:, :]
    zone = (fw < 0.97) & (fs < 0.97) & (basis["labels"] > 0)
    zone &= (basis["d_open_px"].astype(np.float32) > 3) & (basis["d_seam_px"].astype(np.float32) > 3)
    loc = sw & zone
    assert loc.sum() > 50
    Hd = H.astype(np.float64)
    c = np.zeros_like(Hd)
    c[1:-1, 1:-1] = np.maximum.reduce([
        np.abs(Hd[1:-1, 2:] - 2 * Hd[1:-1, 1:-1] + Hd[1:-1, :-2]),
        np.abs(Hd[2:, 1:-1] - 2 * Hd[1:-1, 1:-1] + Hd[:-2, 1:-1]),
        np.abs(Hd[2:, 2:] - Hd[2:, :-2] - Hd[:-2, 2:] + Hd[:-2, :-2]) / 4,
    ])
    dl = cv2.distanceTransform((~loc).view(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    beside = zone & (np.abs(dl - 3) <= 0.5)
    on, off = float(np.median(c[loc])), float(np.median(c[beside]))
    assert on <= 1.5 * off, (on, off)


def _second_diff(H):
    Hd = H.astype(np.float64)
    c = np.zeros_like(Hd)
    c[1:-1, 1:-1] = np.maximum.reduce([
        np.abs(Hd[1:-1, 2:] - 2 * Hd[1:-1, 1:-1] + Hd[1:-1, :-2]),
        np.abs(Hd[2:, 1:-1] - 2 * Hd[1:-1, 1:-1] + Hd[:-2, 1:-1]),
        np.abs(Hd[2:, 2:] - Hd[2:, :-2] - Hd[:-2, 2:] + Hd[:-2, :-2]) / 4,
    ])
    return c


def _switch(a):
    a = a.astype(np.int8)
    e = np.zeros(a.shape, bool)
    e[:, :-1] |= a[:, :-1] != a[:, 1:]
    e[:-1, :] |= a[:-1, :] != a[1:, :]
    return e


def _min_loci_ratios(basis, H, L_clean, crop=(slice(None), slice(None))):
    """Second difference on the loci where a hard min would switch (d_gate = Re, d_thick = R),
    over its value 3 px beside them; > 6 px from lines and the outline. {locus: (n, ratio)}."""
    R = sculpt_radii(basis, _params(), PW, RZ)[0]
    T = basis["d_thick_px"].astype(np.float32)[crop]
    dg = basis["d_gate_px"].astype(np.float32)[crop]
    sil = basis["_sil"][crop]
    inner = cv2.erode(sil.view(np.uint8), np.ones((9, 9), np.uint8)) > 0
    zone = inner & (basis["d_open_px"].astype(np.float32)[crop] > 6)
    zone &= cv2.distanceTransform((~(np.asarray(L_clean)[crop] < 0.5)).view(np.uint8), cv2.DIST_L2, 5) > 6
    c = _second_diff(H[crop])
    out = {}
    for name, loc in (("d_gate=Re", _switch(dg < np.minimum(T, R)) & (dg < 1e4)), ("d_thick=R", _switch(T < R))):
        loc &= zone
        if loc.sum() < 30:
            continue
        dl = cv2.distanceTransform((~loc).view(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        beside = zone & (np.abs(dl - 3) <= 0.5)
        out[name] = (int(loc.sum()), float(np.median(c[loc]) / np.median(c[beside])))
    return out


@pytest.mark.parametrize("scene", ["dor", "fig"])
def test_smooth_min_loci_join_smoothly(scene, request):
    """P1 (review round 4): the edge radius max(smin(d_thick, R), R_lo) and the gate denominator
    smin(d_gate, Re) are smooth (p-norm) minimums, so the relief has no contour where a hard min
    would switch: the second difference there is <= 1.5x its value 3 px beside it (hard mins:
    1.85 / 1.82 on the gate locus here)."""
    pa, layers, basis, L, S = request.getfixturevalue(scene)
    r = _min_loci_ratios(basis, _profile(basis, _params()), L)
    assert "d_gate=Re" in r and "d_thick=R" in r, r
    assert all(v <= 1.5 for _n, v in r.values()), r


def test_smooth_min_is_a_bounded_lower_min():
    """smin(a, b) <= min(a, b) (the draft floors still bound it) and >= 2^(-1/p) min; the kernel's
    lookup table matches the exact p-norm."""
    rng = np.random.default_rng(5)
    a = rng.uniform(0.5, 500, 20000).astype(np.float32)
    b = rng.uniform(0.5, 500, 20000).astype(np.float32)
    e = se.smin(a, b)
    m = np.minimum(a, b)
    assert (e <= m * (1 + 1e-12)).all() and (e >= se._SMIN_K * m * (1 - 1e-12)).all()
    x = a.copy()
    se._smin_lut(x, b, np.empty_like(a), np.empty(a.shape, np.int32))
    np.testing.assert_allclose(x, e, rtol=5e-5)
    x = a.copy()
    se._smin_lut(x, np.float32(72.0), np.empty_like(a), np.empty(a.shape, np.int32))
    np.testing.assert_allclose(x, se.smin(a, 72.0), rtol=5e-5)


def test_tone_curve_monotone_c1_and_bounded_gain():
    """R4: the soft-knee tone is monotone and C1, maps [zmin, zmax] onto [h_min, 1], and never
    steepens the plain [zmin, zmax] normalisation by more than `_KNEE_GAIN`."""
    rng = np.random.default_rng(3)
    for _ in range(50):
        zmin = float(rng.uniform(-1, 1))
        span = float(rng.uniform(0.2, 5))
        zp = zmin + span * float(rng.uniform(0.3, 1.0))
        p = {"h_min": 0.08}
        tone = se._tone(p, zmin, zmin + span, zp)
        z = np.linspace(zmin, zmin + span, 2001).astype(np.float32).reshape(1, -1)
        basis = {"_sil_u8": np.ones(z.shape, np.uint8), "_row_ext": (np.array([0]), np.array([z.shape[1]]))}
        out = z.copy()
        se._finish_rows(basis, out, 0, 1, 0, z.shape[1], tone)
        y = out[0].astype(np.float64)
        assert abs(y[0] - 0.08) < 1e-5 and abs(y[-1] - 1.0) < 1e-5
        dy = np.diff(y)
        assert (dy >= -1e-7).all()
        plain = (1 - 0.08) / span * (span / (z.shape[1] - 1))
        assert dy.max() <= plain * se._KNEE_GAIN * (1 + 1e-3)
        assert np.abs(np.diff(dy)).max() <= 1e-2 * dy.max() + 5e-7  # no slope jump at the knee


def test_draft_holds_unclamped(fig):
    """With a draft angle the clamp does not bind for, the construction bound alone (tone gain
    included) keeps every interior slope within the angle (the exact limiter does not run)."""
    pa, layers, basis, L, S = fig
    d_sil = cv2.distanceTransform((S > 0.5).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    lay = _sculpt_layers(basis, S)
    n_unclamped = 0
    for pw, theta in ((2000.0, 5.0), (5000.0, 20.0), (20000.0, 45.0), (1000.0, 2.0)):
        for extra in ({}, {"layering": 1.0}, {"edge_roundness_pct": 3.0}):
            H = recompute_relief(lay, _params(draft_angle_deg=theta, **extra), physical_width_mm=pw, relief_z_mm=RZ)
            if lay["sculpt_edge_clamped"]:
                continue
            n_unclamped += 1
            excess, n = _interior_slope_excess(H, S, d_sil, pw, RZ, theta)
            assert excess <= 1.0 + 1e-6, (pw, theta, extra, excess)
    assert n_unclamped >= 6


def test_lipschitz_limit_is_exact():
    """The limiter equals the brute-force min-plus convolution with the chamfer cone, and its
    result is never steeper than k between any two pixels (Euclidean)."""
    rng = np.random.default_rng(7)
    for _ in range(20):
        h, w = (int(v) for v in rng.integers(2, 18, 2))
        a = rng.random((h, w)).astype(np.float32)
        sil = rng.random((h, w)) > 0.25
        k = float(rng.uniform(0.01, 0.3))
        got = a.copy()
        se._lipschitz_limit(got, (0, h, 0, w), sil, k)
        ka = k / se._CHAMFER_FIX
        yy, xx = np.mgrid[:h, :w]
        sy, sx = np.nonzero(sil)
        dy = np.abs(yy[..., None] - sy)
        dx = np.abs(xx[..., None] - sx)
        cham = ka * (np.maximum(dy, dx) - np.minimum(dy, dx)) + ka * np.sqrt(2.0) * np.minimum(dy, dx)
        want = np.where(sil, (a[sy, sx] + cham).min(axis=-1), 0.0)
        np.testing.assert_allclose(got, want, atol=2e-6)
        v = got[sil]
        py, px = sy.astype(float), sx.astype(float)
        dist = np.hypot(py[:, None] - py, px[:, None] - px)
        assert ((v[:, None] - v) <= k * dist + 1e-5).all()


@pytest.mark.parametrize("scene", ["dor", "fig"])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_small_point_gathers_match_full_grid(scene, n, request):
    """Point gathers of 1-4 pixels (seam ends, a click pick) give the same z and (mix, g, q) as a
    gather over the whole silhouette: OpenCV reads a 1-D array of <= 4 elements as a Scalar, so
    unpadded they came back silently wrong, by up to 1.5 part heights (task-4 review round 4).
    (Equal up to float rounding: SIMD and scalar tails of the same op may differ in the last bit.)"""
    pa, layers, basis, L, S = request.getfixturevalue(scene)
    P = se._defaults_P(basis)
    sil_idx = np.flatnonzero(basis["_sil"].ravel())
    z_all = se._z_at(basis, sil_idx, P)
    parts_all = se._parts_at(basis, sil_idx, P)
    raw = np.zeros(S.shape, np.float32)
    se._raw_profile(basis, P, raw)
    for seed in range(3):
        pick = np.random.default_rng(100 * n + seed).choice(len(sil_idx), n, replace=False)
        idx = sil_idx[pick]
        z = se._z_at(basis, idx, P)
        np.testing.assert_allclose(z, z_all[pick], rtol=0, atol=1e-6)
        np.testing.assert_allclose(z, raw.ravel()[idx], rtol=0, atol=2e-6)
        for got, want in zip(se._parts_at(basis, idx, P), parts_all):
            np.testing.assert_allclose(got, want[pick], rtol=0, atol=1e-6)
    assert se._z_at(basis, sil_idx[:0], P).shape == (0,)


def test_r_lo_shortcut_keeps_a_binding_floor():
    """The "R_lo cannot bind" shortcut may drop R_lo only when K min(thick_min, R_user) >= R_lo
    (smin(T, R) >= K min(T, R), R >= R_user): with a thick part but a small Edge roundness the edge
    radius must still never fall below the floor the draft clamp asked for (task-4 review 4)."""
    m = 1000.0
    base = {"_shape": (int(m), int(m)), "_g_dome": 0.0, "_thick_p05": 400.0}
    checked = 0
    for pct in (1.0, 2.0, 3.0, 5.0, 7.0):
        for pw in np.geomspace(100.0, 4000.0, 40):
            p = _params(edge_roundness_pct=pct, draft_angle_deg=10.0)
            need = se._radii({**base, "_thick_min": 0.0}, p, float(pw), RZ)[0][1]  # no shortcut
            if need <= 1e-3:
                continue
            P, _c = se._radii({**base, "_thick_min": 400.0}, p, float(pw), RZ)
            T = np.geomspace(400.0, 4000.0, 50)
            assert float(se.edge_radius(T, P[0], P[1]).min()) >= need * (1 - 1e-9), (pct, pw, need, P)
            checked += 1
    assert checked > 50


def test_layer_lambda_counts_analysis_grid_areas():
    """The per-layer step rule counts part areas on the analysis-grid labels, so a preview build
    (1024) and an export build (4K) of the same drawing pick the same step factors. The small disc
    here covers 2.99 % of the 1024 grid's silhouette and 3.01 % of the 4K grid's (the rule's
    threshold is 3 %), so counted on the working grid the two builds disagreed."""
    from backend.pipeline.sculpt_parts import labels_on_grid

    def scene(n, r=39.56):
        white, sil = _disc_over_rect(n)
        s = n / 1024.0
        small = _disc(n, 590 * s, 470 * s, r * s)
        sil2 = sil | small
        white2 = (white & ~small) | (small & ~_ring(n, 590 * s, 470 * s, r * s, max(1.0, 1.5 * s)))
        return (white2 & sil2).astype(np.float32), sil2.astype(np.float32)

    L2, S2 = scene(2048)
    pa = analyze_parts(L2, S2, L2 * S2)
    sm, disc, rect = (part_at(pa, u / 1024, v / 1024) for u, v in ((590, 470), (400, 512), (800, 512)))
    layers = auto_layers(pa, [(sm, disc), (sm, rect)])
    lam = {}
    for n in (1024, 4096):
        L, S = scene(n)
        lab = labels_on_grid(pa, S.shape, S > 0.5)
        frac = float((lab == sm).sum()) / float((lab > 0).sum())
        assert (frac < 0.03) == (n == 1024), (n, frac)  # the working grids straddle the threshold
        lam[n] = build_sculpt_basis(pa, layers, S, S.shape, L_clean=L)
    np.testing.assert_array_equal(lam[1024]["_lam_part"], lam[4096]["_lam_part"])
    assert lam[1024]["_lam_layer"] == lam[4096]["_lam_layer"] == {int(layers[sm]): se.TOP_LAYER_LAMBDA}


# --------------------------------------------------------------------------- dispatch


def test_recompute_relief_dispatches_sculpt(dor):
    pa, layers, basis, L, S = dor
    lay = _sculpt_layers(basis, S)
    p = _params()
    H = recompute_relief(lay, p, physical_width_mm=PW, relief_z_mm=RZ)
    assert np.array_equal(H, _profile(basis, p))
    assert lay["sculpt_edge_clamped"] is False
    # the shared tail still runs: the exact draft limit caps the outer wall
    pd = _params(draft_angle_deg=40.0)
    Hd = recompute_relief(lay, pd, physical_width_mm=50.0, relief_z_mm=RZ)
    assert lay["sculpt_edge_clamped"] is True
    Hp = _profile(basis, pd, 50.0, RZ)
    assert (Hd <= Hp + 1e-7).all() and (Hd < Hp - 1e-3).any()
    # brush delta is added after the profile
    lay["H_brush_delta"] = np.zeros_like(S)
    lay["H_brush_delta"][500:520, 700:720] = 0.05
    lay["_has_brush_delta"] = True
    Hb = recompute_relief(lay, p, physical_width_mm=PW, relief_z_mm=RZ)
    assert float(Hb[510, 710] - H[510, 710]) == pytest.approx(min(0.05, 1 - H[510, 710]), abs=1e-6)
    no_brush = recompute_relief(lay, p, physical_width_mm=PW, relief_z_mm=RZ, include_brush=False)
    assert np.array_equal(no_brush, H)
    with pytest.raises(ValueError):
        recompute_relief({"M_sil": S, "d_sil": lay["d_sil"]}, p)


@pytest.mark.parametrize("deg", [3.0, 7.0, 17.0, 31.0, 44.0, 60.0])
def test_subpixel_reduces_staircase_ripple(deg):
    """Finding B, directly: on an anti-aliased slanted half-plane the corrected distance's error
    4-15 px from the edge has at most 0.6x the binary EDT's standard deviation."""
    n = 400
    yy, xx = np.mgrid[:n, :n].astype(np.float64)
    a = np.radians(deg)
    nx, ny = np.cos(a), np.sin(a)
    o = (np.arange(8) + 0.5) / 8 - 0.5
    cov = np.zeros((n, n))
    for u in o:
        for v in o:
            cov += (xx + u - 200) * nx + (yy + v - 200) * ny > 0
    L = (cov / 64).astype(np.float32)
    wall = L < 0.5
    corr = se._subpixel(L, wall, (0, n, 0, n), 1024.0)  # smoothing / reach of a 1024 build
    basis = {"_sub_corr": corr, "_role": np.where(wall, 1, 0).astype(np.uint8)}
    d = cv2.distanceTransform(np.ascontiguousarray(~wall).view(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dc = se._apply_subpixel(d.copy(), basis, slice(0, n), slice(0, n))
    true = (xx - 200) * nx + (yy - 200) * ny
    m = (true > 4) & (true < 15) & (xx > 60) & (xx < 340) & (yy > 60) & (yy < 340)
    e_bin = (np.maximum(d - 0.5, 0) - true)[m].std()
    e_sub = (dc - true)[m].std()
    assert e_sub <= 0.6 * e_bin, (deg, e_sub, e_bin)


def _plain_tone(params, zmin, zmax, zp):
    h = max(0.0, min(float(params.get("h_min", 0.08)), 0.95))
    return 1.0 / (zmax - zmin), -zmin / (zmax - zmin), False, 1.0 - h, h


# the three fingers of the hand resting on the left shoulder (4K: y 420-1000, x 950-1500), as
# normalised (u, v) points inside each finger on the analysis grid; the measured finger crest
# with the narrow thickness blend (THICK_BLUR_FRAC 0.006), at 1024 and at 4096
SHOULDER_FINGERS = ((0.317, 0.171), (0.333, 0.177), (0.337, 0.160))
SHOULDER_FINGER_CREST = {1024: (0.805, 0.844, 0.851), 4096: (0.810, 0.838, 0.850)}


def _finger_crest(H, labels, p):
    """A finger's crest height: the median over the part's rows of the row's highest pixel."""
    m = labels == p
    return float(np.median([H[r][m[r]].max() for r in np.flatnonzero(m.any(1))]))


def _shoulder_fingers(pa):
    ids = [part_at(pa, u, v) for u, v in SHOULDER_FINGERS]
    assert len(set(ids)) == 3 and 0 not in ids, ids
    for p in ids:  # each anchor lands on a finger: a small part inside the hand's box
        yy, xx = np.nonzero(pa.labels == p)
        assert yy.size < 1e-3 * pa.labels.size, (p, yy.size)
        h, w = pa.shape
        assert 420 / 4096 < yy.mean() / h < 1000 / 4096 and 950 / 3543 < xx.mean() / w < 1500 / 3543, p
    return ids


# --------------------------------------------------------------------------- speed and threads (4K)


@pytest.fixture(scope="module")
def figure_4k():
    white, sil = _figure(4096)
    t0 = time.perf_counter()
    L = (white & sil).astype(np.float32)
    S = sil.astype(np.float32)
    pa = analyze_parts(L, S, L * S)
    layers = auto_layers(pa)
    t1 = time.perf_counter()
    basis = build_sculpt_basis(pa, layers, S, S.shape, L_clean=L)
    t2 = time.perf_counter()
    return {"pa": pa, "layers": layers, "basis": basis, "S": S, "build_s": t2 - t1, "analysis_s": t1 - t0}


def test_sculpt_basis_build_4k_within_budget(figure_4k):
    assert figure_4k["layers"].max() >= 1
    if os.environ.get("MONOLITH_CHECK_TIMINGS") == "1":
        assert figure_4k["build_s"] < 3.0 * PERF_SLACK, figure_4k["build_s"]


def test_basis_4k_bytes_per_pixel(figure_4k):
    b, S = figure_4k["basis"], figure_4k["S"]
    total = sum(v.nbytes for v in b.values() if isinstance(v, np.ndarray) and v.size >= S.size and v.base is None)
    assert total / S.size <= 31.0, total / S.size


def test_sculpt_recompute_4k_under_80ms(figure_4k):
    from backend import db

    basis, S = figure_4k["basis"], figure_4k["S"]
    lay = _sculpt_layers(basis, S)
    base = {**db.default_relief_params(), **SCULPT_DEFAULTS, "style": "sculpt"}
    recompute_relief(lay, base)  # warm-up (draft envelope cache, buffers)
    times = []
    for i in range(5):
        p = {**base, "edge_roundness_pct": SCULPT_DEFAULTS["edge_roundness_pct"] * (1.0 + 0.05 * i)}
        t0 = time.perf_counter()
        recompute_relief(lay, p, physical_width_mm=1000.0, relief_z_mm=15.0)
        times.append(time.perf_counter() - t0)
    med = float(np.median(times))
    if os.environ.get("MONOLITH_CHECK_TIMINGS") == "1":
        assert med < 0.080 * PERF_SLACK, times


def test_preview_not_queued_behind_4k(figure_4k):
    """I3: a 1024 preview computed while a 4K recompute runs on another thread finishes within
    3x its solo time (previews run inline, not on the 4K worker pool)."""
    white, sil = _figure(1024)
    pa, layers, pb, L, S1 = _build(white, sil)
    p = _params()
    out = np.zeros(S1.shape, np.float32)
    sculpt_profile(pb, p, PW, RZ, out)
    solo = []
    for _ in range(5):
        t0 = time.perf_counter()
        sculpt_profile(pb, p, PW, RZ, out)
        solo.append(time.perf_counter() - t0)
    solo_t = float(np.median(solo))
    b4, S4 = figure_4k["basis"], figure_4k["S"]
    stop = threading.Event()

    def busy():
        o = np.zeros(S4.shape, np.float32)
        while not stop.is_set():
            sculpt_profile(b4, {**p, "fullness": 0.5}, PW, RZ, o)

    th = threading.Thread(target=busy)
    th.start()
    try:
        time.sleep(0.05)
        loaded = []
        for _ in range(5):
            t0 = time.perf_counter()
            sculpt_profile(pb, p, PW, RZ, out)
            loaded.append(time.perf_counter() - t0)
    finally:
        stop.set()
        th.join()
    assert float(np.median(loaded)) <= 3.0 * solo_t + 0.005, (solo_t, loaded)


def test_rebuild_parts_4k_under_1s(figure_4k):
    pa, layers, basis = figure_4k["pa"], figure_4k["layers"], figure_4k["basis"]
    torso = part_at(pa, 0.40, 0.75)
    nbrs = sorted({p for p in range(1, int(pa.labels.max()) + 1) if p != torso})
    new = auto_layers(pa, [(torso, q) for q in nbrs])  # bring the torso forward
    assert new[torso] > layers[torso]
    t0 = time.perf_counter()
    rebuild_parts(basis, [torso], new)
    dt = time.perf_counter() - t0
    if os.environ.get("MONOLITH_CHECK_TIMINGS") == "1":
        assert dt < 1.0 * PERF_SLACK, dt
    assert np.array_equal(basis["layer"], new.astype(np.int32))
