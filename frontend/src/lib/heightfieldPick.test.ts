import { describe, expect, it } from 'vitest';
import { pickHeightfield } from './heightfieldPick';

const geom = { planeW: 10, planeH: 8, baseZ: 0.5, reliefZ: 2 };
const step = 0.01;

function normalize(d: [number, number, number]): [number, number, number] {
  const n = Math.hypot(...d);
  return [d[0] / n, d[1] / n, d[2] / n];
}

describe('pickHeightfield', () => {
  it('hits the displaced plane of a flat field, not the z=0 plane', () => {
    const origin: [number, number, number] = [-3, -2, 8];
    const dir = normalize([0.4, 0.3, -1]);
    const flat = () => 0.5;
    const hit = pickHeightfield(origin, dir, geom, flat, step)!;
    expect(hit).not.toBeNull();
    const zPlane = geom.baseZ + 0.5 * geom.reliefZ;
    const t = (zPlane - origin[2]) / dir[2];
    const x = origin[0] + dir[0] * t;
    const y = origin[1] + dir[1] * t;
    expect(hit.u).toBeCloseTo(x / geom.planeW + 0.5, 3);
    expect(hit.v).toBeCloseTo(y / geom.planeH + 0.5, 3);
    const t0 = -origin[2] / dir[2];
    const u0 = (origin[0] + dir[0] * t0) / geom.planeW + 0.5;
    expect(Math.abs(hit.u - u0)).toBeGreaterThan(0.01);
  });

  it('returns the occluding bump in front of the plane hit', () => {
    // Bump around u=0.4, v=0.5; ray comes from the -x side at a shallow angle.
    const bump = (u: number, v: number) =>
      Math.hypot(u - 0.4, v - 0.5) < 0.08 ? 1 : 0;
    const origin: [number, number, number] = [-6, 0, 3.2];
    const dir = normalize([1, 0, -0.3]);
    const hit = pickHeightfield(origin, dir, geom, bump, step)!;
    expect(hit).not.toBeNull();
    expect(Math.hypot(hit.u - 0.4, hit.v - 0.5)).toBeLessThan(0.09);
    // The flat floor would have been hit much further along +x.
    const floor = pickHeightfield(origin, dir, geom, () => 0, step)!;
    expect(floor.u).toBeGreaterThan(hit.u);
  });

  it('maps a vertical ray straight to u, v', () => {
    const hit = pickHeightfield([2, -1, 9], [0, 0, -1], geom, () => 0.3, step)!;
    expect(hit.u).toBeCloseTo(2 / 10 + 0.5, 6);
    expect(hit.v).toBeCloseTo(-1 / 8 + 0.5, 6);
  });

  it('returns null for rays that point away or pass outside the plane', () => {
    expect(pickHeightfield([0, 0, 9], [0, 0, 1], geom, () => 0, step)).toBeNull();
    expect(
      pickHeightfield([20, 0, 9], normalize([0, 0, -1]), geom, () => 0, step)
    ).toBeNull();
    expect(
      pickHeightfield([20, 0, 3], normalize([0.1, 0, -1]), geom, () => 0, step)
    ).toBeNull();
  });
});
