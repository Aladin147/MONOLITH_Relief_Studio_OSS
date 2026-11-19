import React from 'react';
import type { MeshSettings, ProjectRecord, ReliefParams } from '../types/monolith';

interface MeshExportSettingsProps {
  value: MeshSettings;
  onChange: (v: MeshSettings) => void;
  params: ReliefParams;
  project: ProjectRecord | null;
}

const TOLERANCES: MeshSettings['toleranceMm'][] = [0.01, 0.02, 0.05, 0.1];
const GRID_RESOLUTIONS: MeshSettings['gridResolution'][] = [512, 1024, 2048];

/** Grid pitch in mm: cols = round(W * min(1, res / max(W, H))), pitch = physical_width / cols. */
export function gridPitchMm(
  project: ProjectRecord | null,
  physicalWidthMm: number,
  gridResolution: number
): number | null {
  if (!project || project.width < 1 || project.height < 1) return null;
  const longest = Math.max(project.width, project.height);
  const cols = Math.max(1, Math.round(project.width * Math.min(1, gridResolution / longest)));
  return physicalWidthMm / cols;
}

const chip = (active: boolean) =>
  `rounded px-2 py-0.5 font-mono text-[11px] font-semibold ${
    active
      ? 'bg-blue-600 text-white'
      : 'bg-[#232732] text-slate-400 hover:text-slate-200'
  }`;

export const MeshExportSettings: React.FC<MeshExportSettingsProps> = ({
  value,
  onChange,
  params,
  project,
}) => {
  return (
    <div className="rounded-lg bg-[#12141A] px-3 py-2 text-left">
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-medium text-slate-300">STL / OBJ mesh</span>
        <div className="flex gap-1">
          {(['adaptive', 'grid'] as const).map((mode) => (
            <button
              key={mode}
              type="button"
              onClick={() => onChange({ ...value, mode })}
              className={chip(value.mode === mode)}
            >
              {mode === 'adaptive' ? 'Adaptive' : 'Grid'}
            </button>
          ))}
        </div>
      </div>

      {value.mode === 'adaptive' ? (
        <div className="mt-2 flex items-center justify-between">
          <span className="text-[11px] text-slate-400">Max error</span>
          <div className="flex gap-1">
            {TOLERANCES.map((t) => (
              <button
                key={t}
                type="button"
                onClick={() => onChange({ ...value, toleranceMm: t })}
                className={chip(value.toleranceMm === t)}
              >
                {t} mm
              </button>
            ))}
          </div>
        </div>
      ) : (
        <div className="mt-2 flex items-center justify-between">
          <span className="text-[11px] text-slate-400">Grid samples</span>
          <div className="flex gap-1">
            {GRID_RESOLUTIONS.map((res) => {
              const pitch = gridPitchMm(project, params.physical_width_mm, res);
              return (
                <button
                  key={res}
                  type="button"
                  onClick={() => onChange({ ...value, gridResolution: res })}
                  className={chip(value.gridResolution === res)}
                >
                  {res}
                  {pitch !== null && ` · ${pitch.toFixed(2)} mm`}
                </button>
              );
            })}
          </div>
        </div>
      )}

      <p className="mt-2 font-mono text-[10px] leading-snug text-slate-500">
        Cut-out (base 0 mm) always exports as grid. For CNC, the 16-bit PNG / 32-bit EXR heightmap is the lossless master.
      </p>
    </div>
  );
};
