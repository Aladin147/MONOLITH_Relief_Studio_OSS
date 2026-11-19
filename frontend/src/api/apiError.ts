interface ValidationItem {
  loc?: unknown;
  msg?: unknown;
}

function formatDetailItem(item: unknown): string {
  if (typeof item === 'string') return item;
  if (item && typeof item === 'object') {
    const { loc, msg } = item as ValidationItem;
    if (typeof msg === 'string') {
      const field = Array.isArray(loc) && loc.length > 0 ? String(loc[loc.length - 1]) : '';
      return field ? `${field}: ${msg}` : msg;
    }
  }
  return JSON.stringify(item);
}

/** Turn a failed API response into a human-readable message. */
export async function readApiError(res: Response, fallback: string): Promise<string> {
  const text = (await res.text().catch(() => '')).trim();
  if (!text) return `${fallback} (HTTP ${res.status})`;

  try {
    const body: unknown = JSON.parse(text);
    const detail = body && typeof body === 'object' ? (body as { detail?: unknown }).detail : undefined;
    if (typeof detail === 'string' && detail) return detail;
    if (Array.isArray(detail) && detail.length > 0) {
      return detail.map(formatDetailItem).join('; ');
    }
  } catch {
    // Not JSON: fall through and surface the raw text.
  }
  return text;
}
