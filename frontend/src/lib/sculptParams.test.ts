import { describe, expect, it } from 'vitest';
import { edgeRoundnessMm, creaseWidthMm, normalizeSculpt } from './sculptParams';
import { DEFAULT_RELIEF_PARAMS } from '../types/monolith';

describe('edgeRoundnessMm', () => {
  it('uses the shorter side for a landscape canvas (aspect = H/W < 1)', () => {
    const p = { ...DEFAULT_RELIEF_PARAMS, physical_width_mm: 1000, edge_roundness_pct: 2.5 };
    expect(edgeRoundnessMm(p, 0.5)).toBeCloseTo(12.5, 6);
  });
  it('uses the width for a portrait canvas (aspect > 1)', () => {
    const p = { ...DEFAULT_RELIEF_PARAMS, physical_width_mm: 1000, edge_roundness_pct: 2.5 };
    expect(edgeRoundnessMm(p, 1.5)).toBeCloseTo(25, 6);
  });
  it('crease width uses the same shorter side', () => {
    const p = { ...DEFAULT_RELIEF_PARAMS, physical_width_mm: 800, crease_width_pct: 1.0 };
    expect(creaseWidthMm(p, 0.5)).toBeCloseTo(4, 6);
  });
});

describe('normalizeSculpt', () => {
  it('missing style becomes classic', () => {
    const { style: _s, ...rest } = DEFAULT_RELIEF_PARAMS;
    expect(normalizeSculpt(rest).style).toBe('classic');
  });
  it('keeps an explicit sculpt style and fills sculpt defaults', () => {
    const out = normalizeSculpt({ style: 'sculpt' });
    expect(out.style).toBe('sculpt');
    expect(out.edge_roundness_pct).toBe(7.0);
    expect(out.fullness).toBe(0.45);
    expect(out.crease_depth).toBe(0.18);
    expect(out.crease_width_pct).toBe(1.0);
    expect(out.layering).toBe(0.35);
    expect(out.sculpt_version).toBeGreaterThanOrEqual(1);
  });
  it('fills frame defaults for empty input', () => {
    const out = normalizeSculpt({});
    expect(out.style).toBe('classic');
    expect(out.frame.bbox).toHaveLength(4);
  });
});

describe('style defaults', () => {
  it('new projects default to sculpt, legacy null params load as classic', () => {
    expect(DEFAULT_RELIEF_PARAMS.style).toBe('sculpt');
    expect(normalizeSculpt(null).style).toBe('classic');
  });
});
