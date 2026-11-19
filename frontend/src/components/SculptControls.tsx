import React from 'react';
import type { ReliefParams, ReliefStyle } from '../types/monolith';
import { creaseWidthMm, edgeRoundnessMm } from '../lib/sculptParams';
import { CadSliderInput, StudioSlider } from './UiControls';

type Theme = 'light' | 'dark';

interface StyleToggleProps {
  params: ReliefParams;
  onChange: (p: ReliefParams, immediate?: boolean) => void;
  isBusy: boolean;
  theme?: Theme;
}

/** Sculpt / Classic switch. A switch is an immediate full recompute. */
export const StyleToggle: React.FC<StyleToggleProps> = ({
  params,
  onChange,
  isBusy,
  theme = 'light',
}) => {
  const dark = theme === 'dark';
  const options: { id: ReliefStyle; label: string }[] = [
    { id: 'sculpt', label: 'Sculpt' },
    { id: 'classic', label: 'Classic' },
  ];
  return (
    <div className="space-y-1.5">
      <div className={`text-xs font-medium ${dark ? 'text-slate-300' : 'text-slate-700'}`}>
        Relief Style
      </div>
      <div
        role="group"
        aria-label="Relief style"
        className={`grid grid-cols-2 gap-1 rounded-xl border p-1 ${
          dark ? 'border-[#2A2F3D] bg-[#161920]' : 'border-slate-200 bg-slate-50'
        }`}
      >
        {options.map((o) => {
          const active = params.style === o.id;
          return (
            <button
              key={o.id}
              type="button"
              aria-pressed={active}
              disabled={isBusy}
              onClick={() => {
                if (!active) onChange({ ...params, style: o.id }, true);
              }}
              className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition-colors disabled:opacity-60 ${
                active
                  ? 'bg-blue-600 text-white shadow-xs'
                  : dark
                  ? 'text-slate-300 hover:bg-[#232834]'
                  : 'text-slate-600 hover:bg-white'
              }`}
            >
              {o.label}
            </button>
          );
        })}
      </div>
      {isBusy && (
        <div className={`text-[11px] ${dark ? 'text-slate-400' : 'text-slate-500'}`}>
          Computing relief... the first switch can take a few seconds.
        </div>
      )}
    </div>
  );
};

interface SculptControlsProps {
  params: ReliefParams;
  onChange: (p: ReliefParams, immediate?: boolean) => void;
  isBusy: boolean;
  /** Canvas height / width (from the project). Defaults to square. */
  aspect?: number;
  /** Newest Sculpt version the server offers (project.sculpt_latest_version). */
  latestVersion?: number;
  /** The server raised Edge roundness to keep the draft angle. */
  edgeClamped?: boolean;
  theme?: Theme;
}

export const SculptControls: React.FC<SculptControlsProps> = ({
  params,
  onChange,
  isBusy,
  aspect = 1,
  latestVersion,
  edgeClamped = false,
  theme = 'light',
}) => {
  const dark = theme === 'dark';
  const set = <K extends keyof ReliefParams>(key: K, val: ReliefParams[K]) =>
    onChange({ ...params, [key]: val });
  const needsUpdate =
    latestVersion !== undefined && params.sculpt_version < latestVersion;

  const edgeMm = edgeRoundnessMm(params, aspect);
  const creaseMm = creaseWidthMm(params, aspect);

  const slider = (
    label: string,
    key: 'edge_roundness_pct' | 'fullness' | 'crease_depth' | 'crease_width_pct' | 'layering',
    min: number,
    max: number,
    step: number,
    opts: { unit?: string; decimals?: number; mm?: number; tooltip?: string; pct?: boolean } = {}
  ) => {
    if (dark) {
      return (
        <div>
          <CadSliderInput
            label={label}
            value={params[key]}
            min={min}
            max={max}
            step={step}
            unit={opts.unit}
            decimals={opts.decimals}
            tooltip={opts.tooltip}
            onChange={(v) => set(key, v)}
          />
          {opts.mm !== undefined && (
            <div className="mt-0.5 text-right font-mono text-[10px] text-slate-500">
              ≈ {opts.mm.toFixed(1)} mm
            </div>
          )}
        </div>
      );
    }
    return (
      <StudioSlider
        label={label}
        value={params[key]}
        min={min}
        max={max}
        step={step}
        formatValue={(v) => {
          const base = `${v.toFixed(opts.decimals ?? 2)}${opts.unit ? ` ${opts.unit}` : ''}`;
          return opts.mm !== undefined ? `${base} ≈ ${opts.mm.toFixed(1)} mm` : base;
        }}
        onChange={(v) => set(key, v)}
      />
    );
  };

  return (
    <div className="space-y-3" data-testid="sculpt-controls">
      {/* A native fieldset disables every slider and number field inside it while busy. */}
      <fieldset
        disabled={isBusy}
        className={`m-0 min-w-0 space-y-3 border-0 p-0 ${isBusy ? 'opacity-60' : ''}`}
      >
      {slider('Edge Roundness', 'edge_roundness_pct', 0.2, 15, 0.1, {
        unit: '%',
        decimals: 1,
        mm: edgeMm,
        tooltip: 'Rounding radius of the silhouette edge, relative to the shorter canvas side.',
      })}
      {edgeClamped && (
        <div
          data-testid="edge-clamped-note"
          className={`-mt-1 text-[11px] ${dark ? 'text-amber-300' : 'text-amber-600'}`}
        >
          Raised to keep the draft angle
        </div>
      )}
      {slider('Fullness', 'fullness', 0.2, 1, 0.01, {
        tooltip: 'How full and pillowy each form is.',
      })}
      {slider('Crease Depth', 'crease_depth', 0, 0.6, 0.01, {
        tooltip: 'Depth of the carved line creases.',
      })}
      {slider('Crease Width', 'crease_width_pct', 0.1, 5, 0.1, {
        unit: '%',
        decimals: 1,
        mm: creaseMm,
        tooltip: 'Width of line creases, relative to the shorter canvas side.',
      })}
      {/* Layering is stored 0..1; shown and edited as a percentage. */}
      {dark ? (
        <CadSliderInput
          label="Layering"
          value={params.layering * 100}
          min={0}
          max={100}
          step={1}
          decimals={0}
          unit="%"
          tooltip="How strongly front parts rise above the ones behind them."
          onChange={(v) => set('layering', v / 100)}
        />
      ) : (
        <StudioSlider
          label="Layering"
          value={params.layering * 100}
          min={0}
          max={100}
          step={1}
          formatValue={(v) => `${Math.round(v)} %`}
          onChange={(v) => set('layering', v / 100)}
        />
      )}
      </fieldset>

      {needsUpdate && (
        <button
          type="button"
          disabled={isBusy}
          onClick={() =>
            onChange({ ...params, sculpt_version: latestVersion as number }, true)
          }
          className={`w-full rounded-lg border py-1.5 text-xs font-semibold disabled:opacity-60 ${
            dark
              ? 'border-blue-500/50 bg-blue-950/40 text-blue-200 hover:bg-blue-900/50'
              : 'border-blue-300 bg-blue-50 text-blue-700 hover:bg-blue-100'
          }`}
        >
          Update to latest Sculpt
        </button>
      )}
    </div>
  );
};
