import React, { useRef, useState } from 'react';
import { ChevronDown, ChevronUp, HelpCircle } from 'lucide-react';
import { resolveDraftCommit } from '../lib/numericDraft';

interface InfoTipProps {
  text: string;
  dark?: boolean;
}

export const InfoTip: React.FC<InfoTipProps> = ({ text, dark = false }) => {
  const [open, setOpen] = useState(false);

  return (
    <span
      className="relative inline-flex items-center"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          setOpen((v) => !v);
        }}
        className={`inline-flex items-center justify-center rounded-full transition-colors ${
          dark
            ? 'text-slate-400 hover:text-slate-200'
            : 'text-slate-400 hover:text-slate-600'
        }`}
        aria-label={text}
      >
        <HelpCircle className="h-3.5 w-3.5" />
      </button>
      {open && (
        <span
          className={`pointer-events-none absolute bottom-full left-1/2 z-50 mb-1.5 w-56 -translate-x-1/2 rounded-lg px-2.5 py-1.5 text-[11px] leading-snug shadow-lg ${
            dark
              ? 'border border-slate-700 bg-[#101217] text-slate-200'
              : 'border border-slate-200 bg-slate-900 text-white'
          }`}
        >
          {text}
        </span>
      )}
    </span>
  );
};

interface CadSliderInputProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit?: string;
  decimals?: number;
  tooltip?: string;
  onChange: (next: number) => void;
}

export const CadSliderInput: React.FC<CadSliderInputProps> = ({
  label,
  value,
  min,
  max,
  step,
  unit = '',
  decimals = 2,
  tooltip,
  onChange,
}) => {
  const displayVal = Number.isInteger(step) && decimals === 0
    ? Math.round(value)
    : Number(value.toFixed(decimals));
  // Local draft while the numeric field is focused; null = show the prop value.
  const [draft, setDraft] = useState<string | null>(null);

  const revertRef = useRef(false);

  const commitDraft = () => {
    if (revertRef.current) {
      revertRef.current = false;
      setDraft(null);
      return;
    }
    if (draft === null) return;
    const next = resolveDraftCommit(draft, String(displayVal), value, min, max);
    setDraft(null);
    if (next !== null) onChange(next);
  };

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 text-xs font-medium text-slate-300">
          <span>{label}</span>
          {tooltip && <InfoTip text={tooltip} dark />}
        </div>
        <div className="flex items-center gap-1">
          <input
            type="number"
            value={draft ?? displayVal}
            min={min}
            max={max}
            step={step}
            onFocus={() => setDraft(String(displayVal))}
            onChange={(e) => setDraft(e.target.value)}
            onBlur={commitDraft}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.currentTarget.blur(); // blur commits
              } else if (e.key === 'Escape') {
                revertRef.current = true;
                e.currentTarget.blur();
              }
            }}
            className="w-16 rounded-md border border-[#2D323E] bg-[#232732] px-2 py-1 text-right text-xs font-mono text-slate-100 outline-none focus:border-blue-500"
          />
          {unit && <span className="text-[11px] text-slate-400">{unit}</span>}
        </div>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={Math.min(max, Math.max(min, value))}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="cad-slider w-full"
      />
    </div>
  );
};

interface StudioSliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  formatValue?: (val: number) => string;
  onChange: (next: number) => void;
}

export const StudioSlider: React.FC<StudioSliderProps> = ({
  label,
  value,
  min,
  max,
  step,
  formatValue,
  onChange,
}) => (
  <div className="space-y-1">
    <div className="flex items-center justify-between text-xs">
      <span className="font-medium text-slate-700">{label}</span>
      <span className="font-mono text-[11px] text-slate-500">
        {formatValue ? formatValue(value) : value.toFixed(2)}
      </span>
    </div>
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={Math.min(max, Math.max(min, value))}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      className="studio-slider w-full"
    />
  </div>
);

interface AccordionSectionProps {
  title: string;
  subtitle?: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}

export const AccordionSection: React.FC<AccordionSectionProps> = ({
  title,
  subtitle,
  defaultOpen = true,
  children,
}) => {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="border-t border-slate-200/80 pt-3.5">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between text-left"
      >
        <span className="text-sm font-semibold text-slate-800">{title}</span>
        {open ? (
          <ChevronUp className="h-4 w-4 text-slate-400" />
        ) : (
          <ChevronDown className="h-4 w-4 text-slate-400" />
        )}
      </button>
      {subtitle && open && (
        <p className="mt-1 text-xs font-medium text-blue-600">{subtitle}</p>
      )}
      {open && <div className="mt-3">{children}</div>}
    </div>
  );
};
