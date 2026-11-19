import { describe, expect, it } from 'vitest';
import { applyPresetKeepingDimensions, presetStyleChangeMessage } from './presets';
import { DEFAULT_RELIEF_PARAMS } from '../types/monolith';
import { normalizeSculpt } from './sculptParams';

describe('applyPresetKeepingDimensions', () => {
  it('keeps current physical dimensions and takes the rest from the preset', () => {
    const current = {
      ...DEFAULT_RELIEF_PARAMS,
      physical_width_mm: 1800,
      relief_z_mm: 22,
      base_z_mm: 6,
    };
    const preset = {
      ...DEFAULT_RELIEF_PARAMS,
      gamma: 1.7,
      physical_width_mm: 400,
      relief_z_mm: 5,
      base_z_mm: 1,
      frame: { ...DEFAULT_RELIEF_PARAMS.frame, enabled: !DEFAULT_RELIEF_PARAMS.frame.enabled },
    };
    const out = applyPresetKeepingDimensions(preset, current);
    expect(out.physical_width_mm).toBe(1800);
    expect(out.relief_z_mm).toBe(22);
    expect(out.base_z_mm).toBe(6);
    expect(out.gamma).toBe(1.7);
    expect(out.frame).toEqual(preset.frame);
  });
});

describe('presetStyleChangeMessage', () => {
  it('names the new style when a preset switches it', () => {
    const sculpt = { ...DEFAULT_RELIEF_PARAMS, style: 'sculpt' as const };
    const classic = { ...DEFAULT_RELIEF_PARAMS, style: 'classic' as const };
    expect(presetStyleChangeMessage(classic, sculpt)).toBe(
      'This preset uses the Classic style: the project switches to Classic.'
    );
    expect(presetStyleChangeMessage(sculpt, classic)).toBe(
      'This preset uses the Sculpt style: the project switches to Sculpt.'
    );
  });
  it('says nothing when the style stays the same', () => {
    expect(presetStyleChangeMessage(DEFAULT_RELIEF_PARAMS, DEFAULT_RELIEF_PARAMS)).toBeNull();
  });
  it('an old preset without a style switches a Sculpt project to Classic', () => {
    const { style: _s, ...old } = DEFAULT_RELIEF_PARAMS;
    const next = normalizeSculpt(
      applyPresetKeepingDimensions(old as typeof DEFAULT_RELIEF_PARAMS, DEFAULT_RELIEF_PARAMS)
    );
    expect(DEFAULT_RELIEF_PARAMS.style).toBe('sculpt');
    expect(next.style).toBe('classic');
    expect(presetStyleChangeMessage(next, DEFAULT_RELIEF_PARAMS)).toMatch(/Classic style/);
  });
});
