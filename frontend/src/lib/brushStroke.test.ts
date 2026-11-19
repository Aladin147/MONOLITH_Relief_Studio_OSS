import { describe, expect, it } from 'vitest';
import { anchorAfterPick, strokeStamps } from './brushStroke';

const sum = (s: { weight: number }[]) => s.reduce((a, b) => a + b.weight, 0);

describe('strokeStamps', () => {
  it('returns a single full-weight stamp without a previous point', () => {
    expect(strokeStamps(null, { x: 3, y: 4 }, 5)).toEqual([
      { x: 3, y: 4, weight: 1 },
    ]);
  });

  it('deposits the same total for 1 event and 10 events along 100 px', () => {
    const spacing = 4;
    const one = strokeStamps({ x: 0, y: 0 }, { x: 100, y: 0 }, spacing);
    let prev = { x: 0, y: 0 };
    const many: { x: number; y: number; weight: number }[] = [];
    for (let i = 1; i <= 10; i++) {
      const next = { x: i * 10, y: 0 };
      many.push(...strokeStamps(prev, next, spacing));
      prev = next;
    }
    expect(Math.abs(sum(one) - sum(many))).toBeLessThan(1e-9);
    expect(sum(one)).toBeCloseTo(100 / spacing, 9);
  });

  it('places stamps evenly after prev, ending at next', () => {
    const s = strokeStamps({ x: 0, y: 0 }, { x: 0, y: 9 }, 3);
    expect(s.map((p) => p.y)).toEqual([3, 6, 9]);
    const last = strokeStamps({ x: 0, y: 0 }, { x: 1, y: 0 }, 4);
    expect(last).toEqual([{ x: 1, y: 0, weight: 0.25 }]);
  });
});

describe('anchorAfterPick', () => {
  it('keeps the anchor while the pick hits the mesh', () => {
    expect(anchorAfterPick({ x: 1, y: 2 }, true)).toEqual({ x: 1, y: 2 });
  });
  it('breaks the stroke on a miss so re-entry does not streak across the gap', () => {
    const anchor = anchorAfterPick({ x: 1, y: 2 }, false);
    expect(anchor).toBeNull();
    expect(strokeStamps(anchor, { x: 500, y: 2 }, 3)).toEqual([
      { x: 500, y: 2, weight: 1 },
    ]);
  });
});
