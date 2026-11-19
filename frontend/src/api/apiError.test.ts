import { describe, expect, it } from 'vitest';
import { readApiError } from './apiError';

describe('readApiError', () => {
  it('returns a string detail as-is', async () => {
    const res = new Response(JSON.stringify({ detail: "Project 'x' not found" }), { status: 404 });
    expect(await readApiError(res, 'Failed')).toBe("Project 'x' not found");
  });

  it('formats a FastAPI 422 detail array as "<last loc>: <msg>"', async () => {
    const body = {
      detail: [
        { loc: ['body', 'physical_width_mm'], msg: 'Input should be greater than or equal to 1' },
      ],
    };
    const res = new Response(JSON.stringify(body), { status: 422 });
    expect(await readApiError(res, 'Failed')).toBe(
      'physical_width_mm: Input should be greater than or equal to 1'
    );
  });

  it('joins multiple detail items with "; "', async () => {
    const body = {
      detail: [
        { loc: ['body', 'a'], msg: 'bad a' },
        { loc: ['query', 'b'], msg: 'bad b' },
      ],
    };
    const res = new Response(JSON.stringify(body), { status: 422 });
    expect(await readApiError(res, 'Failed')).toBe('a: bad a; b: bad b');
  });

  it('returns non-JSON text unchanged', async () => {
    expect(await readApiError(new Response('boom', { status: 500 }), 'Failed')).toBe('boom');
  });

  it('falls back with the HTTP status for an empty body', async () => {
    const res = new Response('', { status: 500 });
    expect(await readApiError(res, 'Export failed')).toBe('Export failed (HTTP 500)');
  });
});
