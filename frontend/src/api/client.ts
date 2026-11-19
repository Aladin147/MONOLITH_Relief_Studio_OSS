import { readApiError } from './apiError';
import type { PartsPayload } from '../lib/partsMap';
import { parseExportResult, type ExportResult } from './exportResult';
import type {
  AiMode,
  BitDepthOption,
  ExportFormat,
  Float32TexturePayload,
  LinePolarity,
  PresetRecord,
  ProjectRecord,
  RecomputeResponse,
  ReliefParams,
  TargetResolution,
} from '../types/monolith';

async function parseJsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    throw new Error(await readApiError(res, 'Request failed'));
  }
  return res.json() as Promise<T>;
}

export async function checkHealth(): Promise<{ status: string }> {
  const res = await fetch('/api/health');
  return parseJsonOrThrow(res);
}

export async function listProjects(): Promise<ProjectRecord[]> {
  const res = await fetch('/api/projects');
  return parseJsonOrThrow(res);
}

export async function getProject(projectId: string): Promise<ProjectRecord> {
  const res = await fetch(`/api/projects/${encodeURIComponent(projectId)}`);
  return parseJsonOrThrow(res);
}

export interface CreateProjectOptions {
  file?: File | null;
  use_benchmark?: boolean;
  name?: string;
  target_res: TargetResolution;
  bg_removal: boolean;
  hd_restore: boolean;
  ai_mode: AiMode;
  bit_depth: BitDepthOption;
  polarity?: LinePolarity;
}

export async function createProject(
  opts: CreateProjectOptions
): Promise<ProjectRecord> {
  const form = new FormData();
  if (opts.file) {
    form.append('file', opts.file);
  } else if (opts.use_benchmark) {
    form.append('use_benchmark', 'true');
  }
  if (opts.name) {
    form.append('name', opts.name);
  }
  form.append('target_res', String(opts.target_res));
  form.append('bg_removal', opts.bg_removal ? 'true' : 'false');
  form.append('hd_restore', opts.hd_restore ? 'true' : 'false');
  form.append('ai_mode', opts.ai_mode);
  form.append('bit_depth', opts.bit_depth);
  if (opts.polarity) form.append('polarity', opts.polarity);

  const res = await fetch('/api/projects/create', {
    method: 'POST',
    body: form,
  });
  return parseJsonOrThrow(res);
}

export async function preprocessProject(
  projectId: string,
  payload: {
    bg_removal: boolean;
    hd_restore: boolean;
    target_res: TargetResolution;
    ai_mode: AiMode;
    polarity?: LinePolarity;
  }
): Promise<ProjectRecord> {
  const res = await fetch(
    `/api/projects/${encodeURIComponent(projectId)}/preprocess`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }
  );
  return parseJsonOrThrow(res);
}

export async function recomputeRelief(
  projectId: string,
  params: ReliefParams,
  signal?: AbortSignal,
  previewOnly = false,
  seq?: number
): Promise<RecomputeResponse> {
  const query = new URLSearchParams();
  if (previewOnly) query.set('preview_only', 'true');
  if (seq !== undefined) query.set('seq', String(seq));
  const qs = query.toString() ? `?${query.toString()}` : '';
  const res = await fetch(
    `/api/projects/${encodeURIComponent(projectId)}/recompute${qs}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
      signal,
    }
  );
  return parseJsonOrThrow(res);
}

export async function fetchFloat32Texture(
  projectId: string,
  maxDim: 512 | 1024 = 1024,
  signal?: AbortSignal
): Promise<Float32TexturePayload> {
  const res = await fetch(
    `/api/projects/${encodeURIComponent(projectId)}/float32-texture?max_dim=${maxDim}&t=${Date.now()}`,
    { signal }
  );
  if (!res.ok) {
    throw new Error(await readApiError(res, 'Failed to load Float32 texture'));
  }
  const buffer = await res.arrayBuffer();
  const width = Number(res.headers.get('X-Width'));
  const height = Number(res.headers.get('X-Height'));
  if (
    !Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0 ||
    buffer.byteLength !== width * height * 4
  ) {
    // Never guess dimensions: a wrong guess renders a plausible-looking but wrong relief.
    throw new Error(
      `Float32 texture shape mismatch (X-Width=${res.headers.get('X-Width')}, ` +
        `X-Height=${res.headers.get('X-Height')}, bytes=${buffer.byteLength})`
    );
  }
  const levels = Number(res.headers.get('X-Unique-Levels'));
  const uniqueLevels = Number.isInteger(levels) && levels > 0 ? levels : undefined;
  return { data: new Float32Array(buffer), width, height, uniqueLevels };
}

/**
 * Persist Local Edit brush strokes.
 * mode 'delta' (preferred): `data` is the stroke delta (edited - pre-stroke), zero elsewhere.
 * mode 'absolute' (legacy): `data` is the full edited texture.
 */
export async function uploadFloat32Texture(
  projectId: string,
  data: Float32Array,
  width: number,
  height: number,
  mode: 'delta' | 'absolute' = 'delta'
): Promise<void> {
  const copy = new Float32Array(data);
  const res = await fetch(
    `/api/projects/${encodeURIComponent(projectId)}/float32-texture?width=${width}&height=${height}&mode=${mode}`,
    {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/octet-stream',
        'X-Width': String(width),
        'X-Height': String(height),
      },
      body: copy.buffer,
    }
  );
  if (!res.ok) {
    throw new Error(await readApiError(res, 'Failed to save brush edits'));
  }
}

export async function resetBrushEdits(projectId: string): Promise<void> {
  const res = await fetch(
    `/api/projects/${encodeURIComponent(projectId)}/clear-brush`,
    { method: 'POST' }
  );
  if (!res.ok) {
    throw new Error(await readApiError(res, 'Failed to reset brush edits'));
  }
}

export async function upscaleProject(
  projectId: string,
  targetRes: 4096 | 8192
): Promise<ProjectRecord> {
  const res = await fetch(
    `/api/projects/${encodeURIComponent(projectId)}/upscale`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target_res: targetRes }),
    }
  );
  return parseJsonOrThrow(res);
}

export async function exportProjectFile(
  projectId: string,
  options: {
    format: ExportFormat;
    mesh_mode: 'adaptive' | 'grid';
    mesh_tolerance_mm: number;
    mesh_resolution: number;
  },
  fallbackFilename: string
): Promise<ExportResult> {
  const res = await fetch(
    `/api/projects/${encodeURIComponent(projectId)}/export`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(options),
    }
  );
  if (!res.ok) {
    throw new Error(await readApiError(res, 'Export failed'));
  }
  const blob = await res.blob();
  const disposition = res.headers.get('Content-Disposition') || '';
  const match = disposition.match(/filename="?([^"]+)"?/i);
  const filename = match?.[1] || fallbackFilename;

  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  // Revoking immediately can cancel the download in some browsers.
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
  return parseExportResult(res.headers, filename);
}

export async function deleteProject(projectId: string): Promise<void> {
  const res = await fetch(
    `/api/projects/${encodeURIComponent(projectId)}`,
    { method: 'DELETE' }
  );
  if (!res.ok) {
    throw new Error(await readApiError(res, 'Delete failed'));
  }
}

export function previewPngUrl(projectId: string, cacheBuster: number): string {
  return `/api/projects/${encodeURIComponent(projectId)}/preview-png?t=${cacheBuster}`;
}

export function originalImageUrl(projectId: string, cacheBuster = 0): string {
  return `/api/projects/${encodeURIComponent(projectId)}/original-image?t=${cacheBuster}`;
}

export async function listPresets(): Promise<PresetRecord[]> {
  const res = await fetch('/api/presets');
  return parseJsonOrThrow(res);
}

export async function createPreset(
  name: string,
  params: ReliefParams
): Promise<PresetRecord> {
  const res = await fetch('/api/presets', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, params }),
  });
  return parseJsonOrThrow(res);
}

export async function deletePreset(presetId: string): Promise<void> {
  const res = await fetch(
    `/api/presets/${encodeURIComponent(presetId)}`,
    { method: 'DELETE' }
  );
  if (!res.ok) {
    throw new Error(await readApiError(res, 'Delete preset failed'));
  }
}

export type PartOrderAction = 'forward' | 'back' | 'reset' | 'reset_all';

/** Part-label map and layer info (Sculpt projects only; 409 for Classic). */
export class PartsUnavailableError extends Error {}

export async function getParts(projectId: string): Promise<PartsPayload> {
  const res = await fetch(`/api/projects/${encodeURIComponent(projectId)}/parts`);
  if (res.status === 409) throw new PartsUnavailableError('Parts are only available in Sculpt');
  return parseJsonOrThrow(res);
}

/** (u, v) are normalised image coordinates with v measured downward (row 0 = top). */
export async function setPartOrder(
  projectId: string,
  u: number,
  v: number,
  action: PartOrderAction
): Promise<PartsPayload> {
  const res = await fetch(
    `/api/projects/${encodeURIComponent(projectId)}/parts/order`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ u, v, action }),
    }
  );
  return parseJsonOrThrow(res);
}
