import { describe, expect, it } from 'vitest';
import { exportErrorMessage, exportSuccessMessage } from './exportToast';

const base = { label: 'STL', format: 'stl', requestedMode: 'adaptive' as const };

describe('exportSuccessMessage', () => {
  it('reports triangles and error for an adaptive mesh', () => {
    const msg = exportSuccessMessage({
      ...base,
      result: { filename: 'a.stl', meshMode: 'adaptive', triangles: 2_345_000, maxErrorMm: 0.0181 },
    });
    expect(msg).toBe('Exported STL \u2014 2.35 M triangles, max error 0.018 mm');
  });

  it('says so when adaptive was requested but the server used a grid mesh', () => {
    const msg = exportSuccessMessage({
      ...base,
      result: { filename: 'a.stl', meshMode: 'grid', triangles: 500_000 },
    });
    expect(msg).toContain('Cut-out (base 0 mm) exported as grid mesh');
    expect(msg).toContain('500,000 triangles');
  });

  it('does not flag a grid mesh the user asked for', () => {
    const msg = exportSuccessMessage({
      ...base,
      requestedMode: 'grid',
      result: { filename: 'a.stl', meshMode: 'grid', triangles: 500_000 },
    });
    expect(msg).not.toContain('Cut-out');
  });

  it('falls back to the plain message when no mesh headers arrived', () => {
    expect(exportSuccessMessage({ ...base, result: { filename: 'a.stl' } })).toBe(
      'Exported STL (zero watermarks).'
    );
    expect(
      exportSuccessMessage({ label: 'PNG 16-bit', format: 'png16', requestedMode: 'adaptive', result: { filename: 'a.png' } })
    ).toBe('Exported PNG 16-bit (zero watermarks).');
  });

  it('writes sub-million triangle counts with separators', () => {
    const msg = exportSuccessMessage({
      ...base,
      result: { filename: 'a.stl', meshMode: 'adaptive', triangles: 999_999, maxErrorMm: 0.02 },
    });
    expect(msg).toContain('999,999 triangles');
  });
});

describe('exportErrorMessage', () => {
  it('translates the triangle-cap detail without API field names', () => {
    const msg = exportErrorMessage(
      "Adaptive mesh needs at least 18,000,000 surface triangles (cap 15,000,000); use a larger mesh_tolerance_mm or mesh_mode 'grid'"
    );
    expect(msg).toContain('18,000,000');
    expect(msg).toContain('15,000,000');
    expect(msg).not.toMatch(/mesh_tolerance_mm|mesh_mode/);
  });

  it('passes other errors through', () => {
    expect(exportErrorMessage('Export failed (HTTP 500)')).toBe('Export failed (HTTP 500)');
  });
});
