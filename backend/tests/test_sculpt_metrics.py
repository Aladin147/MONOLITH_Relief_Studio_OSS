"""Tests for the relief-vs-competitor measurement helpers (synthetic arrays only)."""

from __future__ import annotations

import numpy as np
import pytest

from sculpt_metrics import (
    align_to,
    find_centre_notch,
    height_correlation,
    notch_depth,
    part_order_agreement,
    row_profile,
    wall_rise,
)


def test_wall_rise_step_profile_is_one():
    p = np.zeros(1000, np.float32)
    p[300:700] = 1.0
    assert wall_rise(p, 10) == pytest.approx(1.0)


def test_wall_rise_linear_ramp_is_about_a_tenth():
    p = np.zeros(1000, np.float32)
    p[100:201] = np.linspace(0.0, 1.0, 101)  # ramp over 100 px, first pixel is 0 (outside)
    p[201:400] = 1.0
    p[400:] = 0.0
    assert wall_rise(p, 10) == pytest.approx(0.1, abs=0.02)


def test_wall_rise_takes_the_minimum_over_crossings():
    p = np.zeros(1000, np.float32)
    p[100:400] = 1.0  # vertical walls
    p[600:900] = np.linspace(0.01, 1.0, 300)  # sloped rising wall, vertical falling wall
    assert wall_rise(p, 10) < 0.2


def test_notch_depth_of_v_dipping_to_half():
    x = np.arange(200, dtype=np.float32)
    p = 0.5 + 0.5 * np.abs(x - 100) / 100.0  # dips to 0.5 at x=100, reaches 1.0 at the ends
    assert notch_depth(p, 100, 100) == pytest.approx(0.5, abs=0.01)


def test_find_centre_notch_locates_dip_near_centre_not_at_walls():
    p = np.zeros(400, np.float32)
    p[50:350] = 1.0
    p[196:204] = 0.5
    p[50:60] = 0.2  # low values near the outline must not win
    assert find_centre_notch(p) == 199  # centre of the tied flat bottom 196..203


def test_part_order_agreement_of_field_with_itself_is_one():
    rng = np.random.default_rng(0)
    labels = np.zeros((60, 60), np.int32)
    labels[:20, :20] = 1
    labels[:20, 30:50] = 2
    labels[30:55, :25] = 3
    labels[35:55, 35:58] = 4
    field = rng.random((60, 60)).astype(np.float32)
    for k, bump in enumerate((0.0, 0.3, 0.6, 0.9), start=1):
        field[labels == k] += bump
    assert part_order_agreement(field, field, labels, min_area=100) == pytest.approx(1.0)


def test_part_order_agreement_ignores_small_parts_and_detects_reversal():
    labels = np.zeros((60, 60), np.int32)
    labels[:20, :20] = 1
    labels[:20, 30:50] = 2
    labels[30:55, :25] = 3
    labels[58:60, 58:60] = 4  # tiny, below min_area
    ref = np.zeros((60, 60), np.float32)
    for k, v in ((1, 0.1), (2, 0.5), (3, 0.9), (4, 0.3)):
        ref[labels == k] = v
    rev = 1.0 - ref
    assert part_order_agreement(rev, ref, labels, min_area=100) == pytest.approx(-1.0)


def test_align_to_recovers_shifted_box():
    mask = np.zeros((200, 300), bool)
    mask[40:160, 100:200] = True
    depth = np.zeros((400, 400), np.float32)
    depth[200:300, 50:110] = 0.8  # same figure, other place, other scale
    out = align_to(mask, depth)
    assert out.shape == mask.shape and out.dtype == np.float32
    box = out > 0.06
    ys, xs = np.nonzero(box)
    assert (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1) == (40, 160, 100, 200)
    assert out[100, 150] == pytest.approx(0.8, abs=1e-3)
    assert out[~mask].max() == 0.0


def test_row_profile_is_min_max_normalised_inside_figure_only():
    mask = np.zeros((100, 100), bool)
    mask[10:90, 20:80] = True
    H = np.zeros((100, 100), np.float32)
    H[mask] = 5.0
    H[50, 30:70] = np.linspace(2.0, 8.0, 40)
    prof = row_profile(H, mask, 0.5)  # row 10 + 0.5 * 80 = 50
    assert prof.shape == (100,)
    assert prof[:20].max() == 0.0 and prof[80:].max() == 0.0
    inside = prof[20:80]
    assert inside.min() > 0.0 and inside.max() == pytest.approx(1.0)


def test_find_centre_notch_raises_when_window_has_no_inside_pixels():
    p = np.zeros(400, np.float32)
    p[50:190] = 1.0
    p[210:350] = 1.0  # 20 px gap at the centre
    with pytest.raises(ValueError):
        find_centre_notch(p)


def test_find_centre_notch_raises_without_a_true_local_minimum():
    p = np.zeros(400, np.float32)
    p[50:350] = np.linspace(0.2, 1.0, 300)  # monotone through the window
    with pytest.raises(ValueError):
        find_centre_notch(p)


def test_find_centre_notch_single_pixel_minimum():
    p = np.zeros(400, np.float32)
    p[50:350] = 1.0
    p[203] = 0.4
    assert find_centre_notch(p) == 203


def test_row_profile_accepts_uint8_mask():
    mask = np.zeros((100, 100), np.uint8)
    mask[10:90, 20:80] = 1
    H = np.random.default_rng(1).random((100, 100)).astype(np.float32)
    a = row_profile(H, mask, 0.5)
    b = row_profile(H, mask.astype(bool), 0.5)
    np.testing.assert_array_equal(a, b)


def test_notch_depth_clamps_window_at_zero_and_guards_zero_max():
    p = np.array([1.0, 0.5, 1.0, 1.0, 1.0], np.float32)
    assert notch_depth(p, 1, 3) == pytest.approx(0.5)  # x-hw < 0 must not wrap
    assert np.isnan(notch_depth(np.zeros(10, np.float32), 5, 2))


def test_align_to_non_rectangular_mask_and_uint8_rgb_depth():
    mask = np.zeros((120, 120), bool)
    yy, xx = np.mgrid[:120, :120]
    mask[(yy - 60) ** 2 + (xx - 60) ** 2 < 40**2] = True  # disc, bbox 21..100
    depth = np.zeros((300, 200, 3), np.uint8)
    depth[50:250, 20:180] = 200
    out = align_to(mask, depth)
    ys, xs = np.nonzero(out > 0.06)
    assert (ys.min(), ys.max() + 1) == (21, 100)
    assert (xs.min(), xs.max() + 1) == (21, 100)
    assert out.max() == pytest.approx(200 / 255, abs=1e-3)


def test_align_to_rejects_empty_inputs():
    good = np.zeros((50, 50), bool)
    good[10:20, 10:20] = True
    with pytest.raises(ValueError):
        align_to(np.zeros((50, 50), bool), np.ones((20, 20), np.float32))
    with pytest.raises(ValueError):
        align_to(good, np.zeros((20, 20), np.float32))


def test_wall_rise_nan_without_crossings():
    assert np.isnan(wall_rise(np.zeros(100, np.float32), 5))
    p = np.zeros(100, np.float32)
    p[10:14] = 1.0  # run no longer than px
    assert np.isnan(wall_rise(p, 5))


def test_wall_rise_counts_canvas_border_as_crossing():
    p = np.zeros(1000, np.float32)
    p[0:500] = np.linspace(0.01, 1.0, 500)  # slope starts at the canvas edge
    assert wall_rise(p, 10) < 0.2


def test_part_order_agreement_nan_with_fewer_than_three_parts():
    labels = np.zeros((40, 40), np.int32)
    labels[:15, :15] = 1
    labels[20:40, 20:40] = 2
    f = np.random.default_rng(2).random((40, 40)).astype(np.float32)
    assert np.isnan(part_order_agreement(f, f, labels, min_area=10))


def test_height_correlation_self_and_inverse():
    mask = np.zeros((80, 80), bool)
    mask[10:70, 10:70] = True
    rng = np.random.default_rng(3)
    ref = np.where(mask, 0.1 + 0.9 * rng.random((80, 80)), 0).astype(np.float32)
    assert height_correlation(ref, ref, mask) == pytest.approx(1.0)
    assert height_correlation(2.0 * ref + 5.0, ref, mask) == pytest.approx(1.0)
    inv = np.where(mask, 1.1 - ref, 0).astype(np.float32)
    assert height_correlation(inv, ref, mask) == pytest.approx(-1.0)
