export interface StrokePoint {
  x: number;
  y: number;
}

export interface Stamp extends StrokePoint {
  weight: number;
}

/**
 * Evenly spaced dabs from `prev` (exclusive) to `next` (inclusive). Each dab
 * carries weight = actualSpacing / spacing, so the summed weight equals
 * dist / spacing regardless of how the path is split into events.
 */
export function strokeStamps(
  prev: StrokePoint | null,
  next: StrokePoint,
  spacingPx: number
): Stamp[] {
  if (!prev) return [{ x: next.x, y: next.y, weight: 1 }];
  const spacing = Math.max(1e-6, spacingPx);
  const dx = next.x - prev.x;
  const dy = next.y - prev.y;
  const dist = Math.hypot(dx, dy);
  const n = Math.max(1, Math.ceil(dist / spacing));
  const weight = dist / n / spacing;
  const out: Stamp[] = [];
  for (let i = 1; i <= n; i++) {
    const t = i / n;
    out.push({ x: prev.x + dx * t, y: prev.y + dy * t, weight });
  }
  return out;
}

/**
 * Stroke continuity: the anchor for the next stamp run. A pick that missed the
 * mesh (pointer left it, or no hit) breaks the stroke, so re-entry starts a
 * fresh stamp instead of interpolating a streak across the gap.
 */
export function anchorAfterPick(
  prev: StrokePoint | null,
  pickHit: boolean
): StrokePoint | null {
  return pickHit ? prev : null;
}
