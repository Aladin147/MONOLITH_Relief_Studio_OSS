export interface HeightfieldGeom {
  planeW: number;
  planeH: number;
  baseZ: number;
  reliefZ: number;
}

type Vec3 = [number, number, number];

/**
 * Intersect a mesh-local ray with the displaced relief surface
 * z = baseZ + h(u, v) * reliefZ. Returns the uv of the first hit or null.
 */
export function pickHeightfield(
  origin: Vec3,
  dir: Vec3,
  geom: HeightfieldGeom,
  sampleH: (u: number, v: number) => number,
  stepWorld: number
): { u: number; v: number } | null {
  const { planeW, planeH, baseZ, reliefZ } = geom;
  const lo: Vec3 = [-planeW / 2, -planeH / 2, 0];
  const hi: Vec3 = [planeW / 2, planeH / 2, baseZ + reliefZ];

  // Slab clip against the bounding box.
  let tEnter = 0;
  let tExit = Infinity;
  for (let i = 0; i < 3; i++) {
    if (Math.abs(dir[i]) < 1e-12) {
      if (origin[i] < lo[i] || origin[i] > hi[i]) return null;
    } else {
      let t0 = (lo[i] - origin[i]) / dir[i];
      let t1 = (hi[i] - origin[i]) / dir[i];
      if (t0 > t1) [t0, t1] = [t1, t0];
      tEnter = Math.max(tEnter, t0);
      tExit = Math.min(tExit, t1);
      if (tEnter > tExit) return null;
    }
  }
  if (!Number.isFinite(tExit)) return null;

  const f = (t: number): number => {
    const x = origin[0] + dir[0] * t;
    const y = origin[1] + dir[1] * t;
    const z = origin[2] + dir[2] * t;
    const u = Math.min(1, Math.max(0, x / planeW + 0.5));
    const v = Math.min(1, Math.max(0, y / planeH + 0.5));
    return z - (baseZ + sampleH(u, v) * reliefZ);
  };

  const dt = Math.max(1e-9, stepWorld) / Math.hypot(dir[0], dir[1], dir[2]);
  let tPrev = tEnter;
  let fPrev = f(tPrev);
  if (fPrev < 0) return null; // entered the box from the side already inside the solid
  let t = tEnter;
  while (t < tExit) {
    t = Math.min(tExit, t + dt);
    const fCur = f(t);
    if (fCur < 0) {
      let a = tPrev;
      let b = t;
      for (let i = 0; i < 10; i++) {
        const m = 0.5 * (a + b);
        if (f(m) < 0) b = m;
        else a = m;
      }
      const tHit = 0.5 * (a + b);
      return {
        u: Math.min(1, Math.max(0, (origin[0] + dir[0] * tHit) / planeW + 0.5)),
        v: Math.min(1, Math.max(0, (origin[1] + dir[1] * tHit) / planeH + 0.5)),
      };
    }
    tPrev = t;
    fPrev = fCur;
  }
  return null;
}
