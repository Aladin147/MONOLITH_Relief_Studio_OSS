import React from 'react';
import {
  Archive,
  Box,
  Layers,
  ShieldCheck,
  Sparkles,
  Zap,
} from 'lucide-react';
import type { ProjectRecord, WorkspaceView } from '../types/monolith';

interface TopBarProps {
  workspace: WorkspaceView;
  onChangeWorkspace: (ws: WorkspaceView) => void;
  activeProject: ProjectRecord | null;
  projectCount: number;
  onOpenVault: () => void;
  lastComputeMs: number | null;
  uniqueLevels: number | null;
  isBusy: boolean;
}

export const TopBar: React.FC<TopBarProps> = ({
  workspace,
  onChangeWorkspace,
  activeProject,
  projectCount,
  onOpenVault,
  lastComputeMs,
  uniqueLevels,
  isBusy,
}) => {
  const isDark = workspace === 'editor3d';

  return (
    <header
      className={`flex h-13 shrink-0 items-center justify-between border-b px-4 transition-colors ${
        isDark
          ? 'border-[#252935] bg-[#14161C] text-slate-100'
          : 'border-slate-200 bg-white text-slate-900'
      }`}
    >
      {/* Left: Brand & Zero-Watermark Guarantee */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-600 text-white shadow-sm">
            <Layers className="h-4 w-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-sm font-bold tracking-wider">MONOLITH</span>
              <span
                className={`hidden items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold sm:inline-flex ${
                  isDark
                    ? 'bg-emerald-500/15 text-emerald-300 ring-1 ring-emerald-500/30'
                    : 'bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200'
                }`}
              >
                <ShieldCheck className="h-3 w-3" />
                Zero Watermark • 100% Local
              </span>
            </div>
          </div>
        </div>

        {activeProject && (
          <div
            className={`hidden items-center gap-2 rounded-lg px-2.5 py-1 text-xs lg:flex ${
              isDark
                ? 'bg-[#1D2029] text-slate-300 ring-1 ring-[#2B303E]'
                : 'bg-slate-100 text-slate-600'
            }`}
          >
            <span className="max-w-[180px] truncate font-medium">
              {activeProject.name}
            </span>
            <span className={isDark ? 'text-slate-500' : 'text-slate-400'}>•</span>
            <span className="font-mono text-[11px]">
              {activeProject.width}×{activeProject.height}
            </span>
            {uniqueLevels !== null && (
              <>
                <span className={isDark ? 'text-slate-500' : 'text-slate-400'}>•</span>
                <span className="font-mono text-[11px] text-emerald-500">
                  {uniqueLevels.toLocaleString()} levels
                </span>
              </>
            )}
          </div>
        )}
      </div>

      {/* Center: Workspace Switcher */}
      <div
        className={`flex items-center rounded-xl p-1 ${
          isDark
            ? 'bg-[#1D2029] ring-1 ring-[#2B303E]'
            : 'bg-slate-100 ring-1 ring-slate-200/70'
        }`}
      >
        <button
          type="button"
          onClick={() => onChangeWorkspace('studio')}
          className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-semibold transition-all ${
            workspace === 'studio'
              ? 'bg-white text-blue-600 shadow-sm'
              : isDark
              ? 'text-slate-400 hover:text-slate-200'
              : 'text-slate-600 hover:text-slate-900'
          }`}
        >
          <Sparkles className="h-3.5 w-3.5" />
          Studio Generator
        </button>
        <button
          type="button"
          onClick={() => onChangeWorkspace('editor3d')}
          className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-semibold transition-all ${
            workspace === 'editor3d'
              ? 'bg-blue-600 text-white shadow-sm'
              : isDark
              ? 'text-slate-400 hover:text-slate-200'
              : 'text-slate-600 hover:text-slate-900'
          }`}
        >
          <Box className="h-3.5 w-3.5" />
          3D Relief &amp; Frame Editor
        </button>
      </div>

      {/* Right: Latency & Local Vault Trigger */}
      <div className="flex items-center gap-2.5">
        {isBusy ? (
          <span
            className={`hidden items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs font-medium sm:inline-flex ${
              isDark ? 'bg-blue-500/15 text-blue-300' : 'bg-blue-50 text-blue-700'
            }`}
          >
            <span className="h-2 w-2 animate-pulse rounded-full bg-blue-500" />
            Computing...
          </span>
        ) : lastComputeMs !== null ? (
          <span
            className={`hidden items-center gap-1 rounded-lg px-2.5 py-1 font-mono text-[11px] sm:inline-flex ${
              isDark
                ? 'bg-[#1D2029] text-emerald-400 ring-1 ring-[#2B303E]'
                : 'bg-slate-100 text-slate-600'
            }`}
          >
            <Zap className="h-3 w-3 text-amber-500" />
            {Math.round(lastComputeMs)} ms
          </span>
        ) : null}

        <button
          type="button"
          onClick={onOpenVault}
          className={`flex items-center gap-2 rounded-lg px-3 py-1.5 text-xs font-semibold transition-colors ${
            isDark
              ? 'bg-[#222632] text-slate-200 ring-1 ring-[#303647] hover:bg-[#2A2F3E]'
              : 'bg-slate-100 text-slate-700 ring-1 ring-slate-200 hover:bg-slate-200/70'
          }`}
        >
          <Archive className="h-3.5 w-3.5 text-blue-500" />
          <span>Local Vault</span>
          <span
            className={`rounded-full px-1.5 py-0.2 text-[10px] font-bold ${
              isDark
                ? 'bg-blue-500/20 text-blue-300'
                : 'bg-blue-100 text-blue-700'
            }`}
          >
            {projectCount}
          </span>
        </button>
      </div>
    </header>
  );
};
