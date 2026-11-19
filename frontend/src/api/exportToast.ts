import type { ExportResult } from './exportResult';

export interface ExportToastInput {
  label: string;
  format: string;
  requestedMode: 'adaptive' | 'grid';
  result: ExportResult;
}

function formatTriangles(n: number): string {
  return n >= 1e6 ? `${(n / 1e6).toFixed(2)} M triangles` : `${n.toLocaleString('en-US')} triangles`;
}

/** Success toast text for an export; mesh statistics are shown only for STL/OBJ. */
export function exportSuccessMessage({ label, format, requestedMode, result }: ExportToastInput): string {
  const isMesh = format === 'stl' || format === 'obj';
  if (!isMesh || (result.triangles === undefined && result.maxErrorMm === undefined && result.meshMode === undefined)) {
    return `Exported ${label} (zero watermarks).`;
  }
  const parts: string[] = [];
  if (result.triangles !== undefined) parts.push(formatTriangles(result.triangles));
  if (result.maxErrorMm !== undefined) parts.push(`max error ${result.maxErrorMm.toFixed(3)} mm`);
  const stats = parts.length > 0 ? ` \u2014 ${parts.join(', ')}` : '';
  if (requestedMode === 'adaptive' && result.meshMode === 'grid') {
    return `Exported ${label}. Cut-out (base 0 mm) exported as grid mesh${stats}`;
  }
  return `Exported ${label}${stats}`;
}

/** Rewrites server errors that mention API field names into user wording. */
export function exportErrorMessage(message: string): string {
  const m = /Adaptive mesh needs at least ([\d,]+) surface triangles \(cap ([\d,]+)\)/.exec(message);
  if (m) {
    return (
      `The mesh is too detailed to export (at least ${m[1]} triangles, limit ${m[2]}). ` +
      'Allow a larger surface tolerance or switch to the grid mesh.'
    );
  }
  return message;
}
