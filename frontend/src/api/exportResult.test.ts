import { describe, expect, it } from 'vitest';
import { parseExportResult } from './exportResult';

describe('parseExportResult', () => {
  it('reads all three mesh headers', () => {
    const h = new Headers({
      'X-Mesh-Mode': 'adaptive',
      'X-Triangles': '1240000',
      'X-Max-Error-Mm': '0.018',
    });
    expect(parseExportResult(h, 'a.stl')).toEqual({
      filename: 'a.stl',
      meshMode: 'adaptive',
      triangles: 1240000,
      maxErrorMm: 0.018,
    });
  });

  it('tolerates missing headers', () => {
    const r = parseExportResult(new Headers(), 'a.png');
    expect(r.filename).toBe('a.png');
    expect(r.meshMode).toBeUndefined();
    expect(r.triangles).toBeUndefined();
    expect(r.maxErrorMm).toBeUndefined();
  });

  it('ignores garbage values and unknown modes', () => {
    const h = new Headers({
      'X-Mesh-Mode': 'weird',
      'X-Triangles': 'abc',
      'X-Max-Error-Mm': '',
    });
    const r = parseExportResult(h, 'a.obj');
    expect(r.meshMode).toBeUndefined();
    expect(r.triangles).toBeUndefined();
    expect(r.maxErrorMm).toBeUndefined();
  });

  it('keeps grid mode without a max error', () => {
    const h = new Headers({ 'X-Mesh-Mode': 'grid', 'X-Triangles': '500' });
    const r = parseExportResult(h, 'a.stl');
    expect(r.meshMode).toBe('grid');
    expect(r.triangles).toBe(500);
    expect(r.maxErrorMm).toBeUndefined();
  });
});
