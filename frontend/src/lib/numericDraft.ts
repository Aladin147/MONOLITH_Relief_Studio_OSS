/**
 * Parse a typed numeric draft and clamp it to [min, max].
 * Returns null for empty / non-finite input (caller should revert).
 */
export function commitNumericDraft(
  draft: string,
  min: number,
  max: number,
): number | null {
  if (draft.trim() === '') return null;
  const n = Number(draft);
  if (!Number.isFinite(n)) return null;
  return Math.min(max, Math.max(min, n));
}

/**
 * Decide what a blur/Enter should commit. Returns null (no change) when the
 * draft is untouched (equals the string it was seeded with), invalid, or
 * clamps to the current value.
 */
export function resolveDraftCommit(
  draft: string,
  seeded: string,
  value: number,
  min: number,
  max: number,
): number | null {
  if (draft === seeded) return null;
  const next = commitNumericDraft(draft, min, max);
  if (next === null || next === value) return null;
  return next;
}
