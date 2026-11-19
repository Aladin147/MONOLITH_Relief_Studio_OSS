import { describe, expect, it } from 'vitest';
import { createLatestGate } from './latestGate';

describe('createLatestGate', () => {
  it('hands out strictly increasing tickets', () => {
    const gate = createLatestGate();
    const a = gate.next();
    const b = gate.next();
    const c = gate.next();
    expect(b).toBeGreaterThan(a);
    expect(c).toBeGreaterThan(b);
  });

  it('treats only the most recent ticket as latest', () => {
    const gate = createLatestGate();
    const a = gate.next();
    expect(gate.isLatest(a)).toBe(true);
    const b = gate.next();
    expect(gate.isLatest(a)).toBe(false);
    expect(gate.isLatest(b)).toBe(true);
  });

  it('keeps gates independent', () => {
    const g1 = createLatestGate();
    const g2 = createLatestGate();
    const t1 = g1.next();
    g2.next();
    g2.next();
    expect(g1.isLatest(t1)).toBe(true);
  });
});
