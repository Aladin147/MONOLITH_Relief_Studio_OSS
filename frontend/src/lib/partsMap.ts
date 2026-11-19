/** Part-label map for the Sculpt Layers tool (backend GET /parts and POST /parts/order). */

export interface PartInfo {
  id: number;
  layer: number;
  auto_layer: number;
  override: 'front' | 'back' | null;
  /** Normalised point well inside the part, v measured downward (row 0 = top). */
  anchor: [number, number];
}

export interface PartsPayload {
  width: number;
  height: number;
  /** Base64 little-endian uint16 label map, row 0 = top. */
  labels: string;
  parts: PartInfo[];
  dropped_overrides?: number;
  compute_ms?: number;
}

export interface PartsMap {
  width: number;
  height: number;
  labels: Uint16Array;
  parts: PartInfo[];
}

export function decodeParts(payload: PartsPayload): PartsMap {
  const { width, height } = payload;
  const bin = atob(payload.labels);
  if (bin.length !== width * height * 2) {
    throw new Error(
      `Part label map has ${bin.length} bytes, expected ${width * height * 2}`
    );
  }
  const labels = new Uint16Array(width * height);
  for (let i = 0; i < labels.length; i++) {
    labels[i] = bin.charCodeAt(i * 2) | (bin.charCodeAt(i * 2 + 1) << 8);
  }
  return { width, height, labels, parts: payload.parts };
}

/** Raw label under (u, v); v is measured upward as in the viewer. Out-of-range clamps. */
export function partAtUv(map: PartsMap, u: number, v: number): number {
  const cu = Math.min(1, Math.max(0, u));
  const cv = Math.min(1, Math.max(0, v));
  const col = Math.round(cu * (map.width - 1));
  const row = Math.round((1 - cv) * (map.height - 1));
  return map.labels[row * map.width + col];
}

const BACK_RGB: [number, number, number] = [59, 130, 246]; // blue
const FRONT_RGB: [number, number, number] = [249, 115, 22]; // orange

/** t = 0 is the back layer (blue), t = 1 the front layer (orange). */
export function layerRampColor(t: number): [number, number, number] {
  const k = Math.min(1, Math.max(0, t));
  return [
    Math.round(BACK_RGB[0] + (FRONT_RGB[0] - BACK_RGB[0]) * k),
    Math.round(BACK_RGB[1] + (FRONT_RGB[1] - BACK_RGB[1]) * k),
    Math.round(BACK_RGB[2] + (FRONT_RGB[2] - BACK_RGB[2]) * k),
  ];
}

const TINT_ALPHA = Math.round(0.35 * 255);
const OUTLINE_RGBA = [255, 236, 64, 255] as const;

export interface OverlayOptions {
  showLayers: boolean;
  selectedId: number | null;
}

/** RGBA8 overlay, same size and row order as the label map (row 0 = top). */
export function buildOverlayRGBA(map: PartsMap, opts: OverlayOptions): Uint8Array {
  const { width: w, height: h, labels } = map;
  const out = new Uint8Array(w * h * 4);

  if (opts.showLayers && map.parts.length > 0) {
    let maxLayer = 0;
    for (const p of map.parts) maxLayer = Math.max(maxLayer, p.layer);
    const colorById = new Map<number, [number, number, number]>();
    for (const p of map.parts) {
      colorById.set(p.id, layerRampColor(maxLayer > 0 ? p.layer / maxLayer : 0));
    }
    for (let i = 0; i < labels.length; i++) {
      const c = colorById.get(labels[i]);
      if (!c) continue;
      const o = i * 4;
      out[o] = c[0];
      out[o + 1] = c[1];
      out[o + 2] = c[2];
      out[o + 3] = TINT_ALPHA;
    }
  }

  const sel = opts.selectedId;
  if (sel !== null) {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        if (labels[y * w + x] !== sel) continue;
        let edge = false;
        for (let dy = -1; dy <= 1 && !edge; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const nx = x + dx;
            const ny = y + dy;
            if (nx < 0 || ny < 0 || nx >= w || ny >= h || labels[ny * w + nx] !== sel) {
              edge = true;
              break;
            }
          }
        }
        if (edge) {
          const o = (y * w + x) * 4;
          out[o] = OUTLINE_RGBA[0];
          out[o + 1] = OUTLINE_RGBA[1];
          out[o + 2] = OUTLINE_RGBA[2];
          out[o + 3] = OUTLINE_RGBA[3];
        }
      }
    }
  }
  return out;
}
