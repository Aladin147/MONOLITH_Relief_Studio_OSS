import type { ReliefParams } from '../types/monolith';

/** Apply a preset's look, but keep the user's physical size (mm) settings. */
export function applyPresetKeepingDimensions(
  preset: ReliefParams,
  current: ReliefParams,
): ReliefParams {
  return {
    ...preset,
    physical_width_mm: current.physical_width_mm,
    relief_z_mm: current.relief_z_mm,
    base_z_mm: current.base_z_mm,
  };
}

const STYLE_LABEL: Record<string, string> = { sculpt: 'Sculpt', classic: 'Classic' };

/**
 * The notice to show when applying a preset changes the project's relief style, else null.
 * Presets saved before the Sculpt style existed carry no style and load as Classic, so applying
 * one to a Sculpt project switches it (final review I8): say so instead of switching silently.
 */
export function presetStyleChangeMessage(
  next: Pick<ReliefParams, 'style'>,
  current: Pick<ReliefParams, 'style'>,
): string | null {
  if (next.style === current.style) return null;
  const label = STYLE_LABEL[next.style] ?? next.style;
  return `This preset uses the ${label} style: the project switches to ${label}.`;
}
