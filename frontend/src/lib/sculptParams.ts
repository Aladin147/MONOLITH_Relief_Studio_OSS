import { DEFAULT_RELIEF_PARAMS, type ReliefParams } from '../types/monolith';

/** Shorter canvas side in mm. `aspect` is height / width of the project. */
export function shorterSideMm(params: ReliefParams, aspect: number): number {
  return Math.min(params.physical_width_mm, params.physical_width_mm * aspect);
}

/** Edge roundness radius in mm: pct/100 * shorter canvas side. */
export function edgeRoundnessMm(params: ReliefParams, aspect: number): number {
  return (params.edge_roundness_pct / 100) * shorterSideMm(params, aspect);
}

/** Crease width in mm: pct/100 * shorter canvas side. */
export function creaseWidthMm(params: ReliefParams, aspect: number): number {
  return (params.crease_width_pct / 100) * shorterSideMm(params, aspect);
}

/**
 * Fill defaults for a (possibly partial / stored) params object.
 * A missing `style` means a legacy project or preset, so it becomes "classic".
 */
export function normalizeSculpt(raw?: Partial<ReliefParams> | null): ReliefParams {
  const d = DEFAULT_RELIEF_PARAMS;
  return {
    ...d,
    ...(raw ?? {}),
    style: raw?.style === 'sculpt' ? 'sculpt' : 'classic',
    frame: {
      ...d.frame,
      ...(raw?.frame ?? {}),
      bbox:
        raw?.frame?.bbox && raw.frame.bbox.length === 4
          ? (raw.frame.bbox as [number, number, number, number])
          : [...d.frame.bbox],
    },
  };
}
