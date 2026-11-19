export interface ExportResult {
  filename: string;
  meshMode?: 'adaptive' | 'grid';
  triangles?: number;
  maxErrorMm?: number;
}

function finiteHeader(headers: Headers, name: string): number | undefined {
  const raw = headers.get(name);
  if (raw === null || raw.trim() === '') return undefined;
  const n = Number(raw);
  return Number.isFinite(n) ? n : undefined;
}

/** Reads the mesh statistics the server reports for STL/OBJ exports; every field is optional. */
export function parseExportResult(headers: Headers, filename: string): ExportResult {
  const mode = headers.get('X-Mesh-Mode');
  return {
    filename,
    meshMode: mode === 'adaptive' || mode === 'grid' ? mode : undefined,
    triangles: finiteHeader(headers, 'X-Triangles'),
    maxErrorMm: finiteHeader(headers, 'X-Max-Error-Mm'),
  };
}
