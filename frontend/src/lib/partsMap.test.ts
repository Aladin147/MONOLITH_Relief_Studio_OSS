import { describe, expect, it } from 'vitest';
import {
  buildOverlayRGBA,
  decodeParts,
  layerRampColor,
  partAtUv,
  type PartInfo,
  type PartsPayload,
} from './partsMap';

function encodeU16LE(values: number[]): string {
  const bytes = new Uint8Array(values.length * 2);
  values.forEach((v, i) => {
    bytes[i * 2] = v & 0xff;
    bytes[i * 2 + 1] = (v >> 8) & 0xff;
  });
  let bin = '';
  for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin);
}

const part = (id: number, layer: number): PartInfo => ({
  id,
  layer,
  auto_layer: layer,
  override: null,
  anchor: [0.5, 0.5],
});

function payload(width: number, height: number, labels: number[], parts: PartInfo[]): PartsPayload {
  return { width, height, labels: encodeU16LE(labels), parts, dropped_overrides: 0 };
}

describe('decodeParts', () => {
  it('round-trips little-endian uint16 labels, including values above 255', () => {
    const labels = [0, 1, 255, 256, 1000, 65535];
    const out = decodeParts(payload(3, 2, labels, [part(1, 0)]));
    expect(out.width).toBe(3);
    expect(out.height).toBe(2);
    expect(Array.from(out.labels)).toEqual(labels);
    expect(out.parts).toHaveLength(1);
  });

  it('rejects a label buffer whose size does not match width*height', () => {
    expect(() => decodeParts(payload(2, 2, [1, 2, 3], []))).toThrow();
  });
});

describe('partAtUv (v measured upward, row 0 = top)', () => {
  // 3x3, row 0 = top:
  //  1 2 3
  //  4 5 6
  //  7 8 9
  const map = decodeParts(payload(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9], []));

  it('top-left is (u=0, v=1)', () => {
    expect(partAtUv(map, 0, 1)).toBe(1);
  });
  it('top-right is (u=1, v=1)', () => {
    expect(partAtUv(map, 1, 1)).toBe(3);
  });
  it('bottom-left is (u=0, v=0)', () => {
    expect(partAtUv(map, 0, 0)).toBe(7);
  });
  it('bottom-right is (u=1, v=0)', () => {
    expect(partAtUv(map, 1, 0)).toBe(9);
  });
  it('centre', () => {
    expect(partAtUv(map, 0.5, 0.5)).toBe(5);
  });
  it('clamps out-of-range uv to the border', () => {
    expect(partAtUv(map, -3, 4)).toBe(1);
    expect(partAtUv(map, 7, -2)).toBe(9);
  });
});

describe('layerRampColor', () => {
  it('back (0) is blue and front (1) is orange', () => {
    const back = layerRampColor(0);
    const front = layerRampColor(1);
    expect(back[2]).toBeGreaterThan(back[0]);
    expect(front[0]).toBeGreaterThan(front[2]);
    expect(back).toEqual([59, 130, 246]);
    expect(front).toEqual([249, 115, 22]);
  });
  it('clamps outside 0..1', () => {
    expect(layerRampColor(-1)).toEqual(layerRampColor(0));
    expect(layerRampColor(2)).toEqual(layerRampColor(1));
  });
});

describe('buildOverlayRGBA', () => {
  // 5x5 with a 3x3 part (id 2) in the middle, layer 1; id 1 on the left column, layer 0.
  const labels = [
    1, 0, 0, 0, 0,
    1, 2, 2, 2, 0,
    1, 2, 2, 2, 0,
    1, 2, 2, 2, 0,
    1, 0, 0, 0, 0,
  ];
  const map = decodeParts(payload(5, 5, labels, [part(1, 0), part(2, 1)]));
  const px = (data: Uint8Array, x: number, y: number) =>
    Array.from(data.slice((y * 5 + x) * 4, (y * 5 + x) * 4 + 4));

  it('is fully transparent with layers off and nothing selected', () => {
    const data = buildOverlayRGBA(map, { showLayers: false, selectedId: null });
    expect(data.every((v) => v === 0)).toBe(true);
  });

  it('tints parts at 35% opacity with the ramp ends', () => {
    const data = buildOverlayRGBA(map, { showLayers: true, selectedId: null });
    expect(px(data, 0, 1)).toEqual([59, 130, 246, 89]); // back = blue
    expect(px(data, 2, 2)).toEqual([249, 115, 22, 89]); // front = orange
    expect(px(data, 4, 4)[3]).toBe(0); // background untouched
  });

  it('outlines only the edge pixels of the selected part', () => {
    const data = buildOverlayRGBA(map, { showLayers: false, selectedId: 2 });
    expect(px(data, 1, 1)[3]).toBe(255); // corner edge
    expect(px(data, 2, 1)[3]).toBe(255); // top edge
    expect(px(data, 2, 2)[3]).toBe(0); // interior
    expect(px(data, 0, 2)[3]).toBe(0); // another part
  });
});
