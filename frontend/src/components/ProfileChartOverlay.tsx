import React, { useMemo, useState } from 'react';
import { Ruler, X } from 'lucide-react';
import type {
  Float32TexturePayload,
  ProfilePoint,
  ProfileSample,
} from '../types/monolith';

interface ProfileChartOverlayProps {
  texture: Float32TexturePayload | null;
  pointA: ProfilePoint | null;
  pointB: ProfilePoint | null;
  physicalWidthMm: number;
  reliefZMm: number;
  baseZMm: number;
  onClear: () => void;
  onClose: () => void;
}

export function sampleFloat32Bilinear(
  tex: Float32TexturePayload,
  u: number,
  v: number
): number {
  const { data, width, height } = tex;
  if (width <= 0 || height <= 0 || data.length === 0) return 0;

  const cu = Math.min(1, Math.max(0, u));
  const cv = Math.min(1, Math.max(0, v));

  // v=1 is top (row=0), v=0 is bottom (row=height-1)
  const fx = cu * (width - 1);
  const fy = (1 - cv) * (height - 1);

  const x0 = Math.floor(fx);
  const y0 = Math.floor(fy);
  const x1 = Math.min(width - 1, x0 + 1);
  const y1 = Math.min(height - 1, y0 + 1);

  const tx = fx - x0;
  const ty = fy - y0;

  const v00 = data[y0 * width + x0] ?? 0;
  const v10 = data[y0 * width + x1] ?? 0;
  const v01 = data[y1 * width + x0] ?? 0;
  const v11 = data[y1 * width + x1] ?? 0;

  const a = v00 * (1 - tx) + v10 * tx;
  const b = v01 * (1 - tx) + v11 * tx;
  return a * (1 - ty) + b * ty;
}

export const ProfileChartOverlay: React.FC<ProfileChartOverlayProps> = ({
  texture,
  pointA,
  pointB,
  physicalWidthMm,
  reliefZMm,
  baseZMm,
  onClear,
  onClose,
}) => {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  const { samples, totalLengthMm, minZ, maxZ } = useMemo(() => {
    if (!texture || !pointA || !pointB) {
      return {
        samples: [] as ProfileSample[],
        totalLengthMm: 0,
        minZ: 0,
        maxZ: baseZMm + reliefZMm,
      };
    }

    const aspect = texture.height / Math.max(1, texture.width);
    const physicalHeightMm = physicalWidthMm * aspect;
    const dxMm = (pointB.u - pointA.u) * physicalWidthMm;
    const dyMm = (pointB.v - pointA.v) * physicalHeightMm;
    const lenMm = Math.hypot(dxMm, dyMm);

    const count = 180;
    const out: ProfileSample[] = [];
    let lo = Infinity;
    let hi = -Infinity;

    for (let i = 0; i <= count; i++) {
      const t = i / count;
      const u = pointA.u + (pointB.u - pointA.u) * t;
      const v = pointA.v + (pointB.v - pointA.v) * t;
      const h = sampleFloat32Bilinear(texture, u, v);
      const zMm = baseZMm + h * reliefZMm;
      if (zMm < lo) lo = zMm;
      if (zMm > hi) hi = zMm;
      out.push({
        distanceMm: t * lenMm,
        zMm,
        normalized: h,
      });
    }

    return {
      samples: out,
      totalLengthMm: lenMm,
      minZ: Number.isFinite(lo) ? lo : 0,
      maxZ: Number.isFinite(hi) ? hi : baseZMm + reliefZMm,
    };
  }, [texture, pointA, pointB, physicalWidthMm, reliefZMm, baseZMm]);

  const svgW = 540;
  const svgH = 148;
  const padL = 46;
  const padR = 16;
  const padT = 14;
  const padB = 24;
  const plotW = svgW - padL - padR;
  const plotH = svgH - padT - padB;
  const zCap = Math.max(0.1, baseZMm + reliefZMm * 1.05);

  const pathData = useMemo(() => {
    if (samples.length < 2) return { line: '', area: '' };
    const pts = samples.map((s, idx) => {
      const x = padL + (idx / (samples.length - 1)) * plotW;
      const y = padT + plotH - (s.zMm / zCap) * plotH;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });
    const line = `M ${pts.join(' L ')}`;
    const area = `${line} L ${(padL + plotW).toFixed(1)},${(padT + plotH).toFixed(1)} L ${padL},${(padT + plotH).toFixed(1)} Z`;
    return { line, area };
  }, [samples, plotW, plotH, zCap]);

  const hovered =
    hoverIdx !== null && samples[hoverIdx] ? samples[hoverIdx] : null;

  const baseLineY = padT + plotH - (baseZMm / zCap) * plotH;

  return (
    <div className="pointer-events-auto w-[580px] max-w-[calc(100vw-400px)] rounded-2xl border border-[#2C3240] bg-[#161920]/95 p-3.5 text-slate-100 shadow-2xl backdrop-blur-md">
      <div className="mb-2 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Ruler className="h-4 w-4 text-cyan-400" />
          <span className="text-xs font-semibold text-white">
            Cross-Section Elevation Profile
          </span>
          {samples.length > 0 ? (
            <span className="rounded bg-[#222733] px-2 py-0.5 font-mono text-[10px] text-cyan-300">
              L: {totalLengthMm.toFixed(1)} mm • ΔZ: {(maxZ - minZ).toFixed(2)} mm
            </span>
          ) : (
            <span className="text-[11px] text-slate-400">
              Click &amp; drag across the 3D relief surface to slice a millimeter section
            </span>
          )}
        </div>
        <div className="flex items-center gap-1.5">
          {samples.length > 0 && (
            <button
              type="button"
              onClick={onClear}
              className="rounded-md bg-[#232834] px-2 py-1 text-[10px] font-medium text-slate-300 hover:bg-[#2E3444]"
            >
              Reset Line
            </button>
          )}
          <button
            type="button"
            onClick={onClose}
            className="rounded-md p-1 text-slate-400 hover:bg-[#232834] hover:text-white"
            aria-label="Close profile graph"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {samples.length > 0 ? (
        <div className="relative">
          <svg
            viewBox={`0 0 ${svgW} ${svgH}`}
            className="h-36 w-full select-none overflow-visible rounded-lg bg-[#101218]"
            onMouseMove={(e) => {
              const rect = e.currentTarget.getBoundingClientRect();
              const relX = ((e.clientX - rect.left) / rect.width) * svgW;
              const t = Math.min(1, Math.max(0, (relX - padL) / plotW));
              setHoverIdx(Math.round(t * (samples.length - 1)));
            }}
            onMouseLeave={() => setHoverIdx(null)}
          >
            <defs>
              <linearGradient id="profileAreaGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.45" />
                <stop offset="100%" stopColor="#38bdf8" stopOpacity="0.03" />
              </linearGradient>
            </defs>

            {/* Horizontal grid lines */}
            {[0, 0.25, 0.5, 0.75, 1].map((frac) => {
              const y = padT + plotH * (1 - frac);
              const val = zCap * frac;
              return (
                <g key={frac}>
                  <line
                    x1={padL}
                    x2={padL + plotW}
                    y1={y}
                    y2={y}
                    stroke="#242936"
                    strokeWidth="1"
                  />
                  <text
                    x={padL - 6}
                    y={y + 3}
                    textAnchor="end"
                    fill="#7c879b"
                    fontSize="9"
                    fontFamily="monospace"
                  >
                    {val.toFixed(1)}
                  </text>
                </g>
              );
            })}

            {/* Base plaque datum line */}
            {baseZMm > 0 && (
              <line
                x1={padL}
                x2={padL + plotW}
                y1={baseLineY}
                y2={baseLineY}
                stroke="#f59e0b"
                strokeOpacity="0.55"
                strokeDasharray="4 3"
                strokeWidth="1"
              />
            )}

            {/* Area & Elevation Curve */}
            <path d={pathData.area} fill="url(#profileAreaGrad)" />
            <path
              d={pathData.line}
              fill="none"
              stroke="#38bdf8"
              strokeWidth="2"
              strokeLinejoin="round"
            />

            {/* X-axis distance ticks */}
            {[0, 0.25, 0.5, 0.75, 1].map((frac) => {
              const x = padL + plotW * frac;
              return (
                <text
                  key={frac}
                  x={x}
                  y={svgH - 6}
                  textAnchor="middle"
                  fill="#7c879b"
                  fontSize="9"
                  fontFamily="monospace"
                >
                  {(totalLengthMm * frac).toFixed(0)} mm
                </text>
              );
            })}

            <text
              x={10}
              y={padT + 8}
              fill="#94a3b8"
              fontSize="9"
              fontFamily="monospace"
            >
              mm
            </text>

            {/* Hover crosshair */}
            {hovered && hoverIdx !== null && (
              <g>
                <line
                  x1={padL + (hoverIdx / (samples.length - 1)) * plotW}
                  x2={padL + (hoverIdx / (samples.length - 1)) * plotW}
                  y1={padT}
                  y2={padT + plotH}
                  stroke="#e2e8f0"
                  strokeOpacity="0.45"
                  strokeDasharray="2 2"
                />
                <circle
                  cx={padL + (hoverIdx / (samples.length - 1)) * plotW}
                  cy={padT + plotH - (hovered.zMm / zCap) * plotH}
                  r="4"
                  fill="#38bdf8"
                  stroke="#ffffff"
                  strokeWidth="1.5"
                />
              </g>
            )}
          </svg>

          <div className="mt-1.5 flex items-center justify-between text-[11px] font-mono text-slate-400">
            <span>
              Min Z: <strong className="text-slate-200">{minZ.toFixed(2)} mm</strong> • Max Z:{' '}
              <strong className="text-slate-200">{maxZ.toFixed(2)} mm</strong>
            </span>
            {hovered ? (
              <span className="text-cyan-300">
                Pos: {hovered.distanceMm.toFixed(1)} mm → Z: {hovered.zMm.toFixed(3)} mm (
                {(hovered.normalized * 100).toFixed(1)}%)
              </span>
            ) : (
              <span>Base Z: {baseZMm.toFixed(1)} mm (amber line)</span>
            )}
          </div>
        </div>
      ) : (
        <div className="flex h-20 items-center justify-center rounded-lg border border-dashed border-[#2C3240] bg-[#101218] text-xs text-slate-400">
          Drag a measurement line across any limb or crease on the 3D mesh to inspect millimeter smoothness.
        </div>
      )}
    </div>
  );
};
