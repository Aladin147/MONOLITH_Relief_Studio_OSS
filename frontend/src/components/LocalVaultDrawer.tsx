import React, { useEffect, useState } from 'react';
import {
  Archive,
  CheckCircle2,
  FolderOpen,
  Plus,
  Trash2,
  X,
} from 'lucide-react';
import type { ProjectRecord } from '../types/monolith';
import { originalImageUrl } from '../api/client';

interface LocalVaultDrawerProps {
  open: boolean;
  onClose: () => void;
  projects: ProjectRecord[];
  activeProjectId: string | null;
  onSelectProject: (project: ProjectRecord, openEditor?: boolean) => void;
  onDeleteProject: (projectId: string) => void;
  onCreateBenchmark: () => void;
  isBusy: boolean;
}

export const LocalVaultDrawer: React.FC<LocalVaultDrawerProps> = ({
  open,
  onClose,
  projects,
  activeProjectId,
  onSelectProject,
  onDeleteProject,
  onCreateBenchmark,
  isBusy,
}) => {
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);

  // Disarm a pending "Delete forever?" when the drawer closes or the list changes.
  useEffect(() => {
    setConfirmDeleteId(null);
  }, [open, projects]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <div
        className="fixed inset-0 bg-black/50 backdrop-blur-[2px]"
        onClick={onClose}
      />
      <aside className="relative z-10 flex h-full w-full max-w-md flex-col border-l border-[#282D3A] bg-[#181A20] text-slate-100 shadow-2xl">
        <div className="flex items-center justify-between border-b border-[#252934] px-5 py-4">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-600/20 text-blue-400">
              <Archive className="h-4 w-4" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-white">Local SQLite Vault</h2>
              <p className="text-[11px] text-slate-400">
                100% private • Re-open &amp; re-export forever at $0
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg p-1.5 text-slate-400 transition-colors hover:bg-[#232732] hover:text-white"
            aria-label="Close vault"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="border-b border-[#252934] px-5 py-3">
          <button
            type="button"
            disabled={isBusy}
            onClick={() => {
              onCreateBenchmark();
              onClose();
            }}
            className="flex w-full items-center justify-center gap-2 rounded-lg bg-blue-600 px-3 py-2 text-xs font-semibold text-white transition-colors hover:bg-blue-500 disabled:opacity-50"
          >
            <Plus className="h-3.5 w-3.5" />
            New Piece from Example Drawing
          </button>
        </div>

        <div className="cad-scroll flex-1 space-y-2.5 overflow-y-auto p-4">
          {projects.length === 0 ? (
            <div className="rounded-xl border border-dashed border-[#2C313F] bg-[#1E2129] p-8 text-center">
              <p className="text-xs font-medium text-slate-300">
                No pieces in vault yet.
              </p>
              <p className="mt-1 text-[11px] text-slate-500">
                Upload line art or load the example to begin.
              </p>
            </div>
          ) : (
            projects.map((proj) => {
              const isActive = proj.id === activeProjectId;
              return (
                <div
                  key={proj.id}
                  className={`group rounded-xl border p-3 transition-all ${
                    isActive
                      ? 'border-blue-500/70 bg-blue-500/10'
                      : 'border-[#262B37] bg-[#1E212A] hover:border-[#373E50]'
                  }`}
                >
                  <div className="flex gap-3">
                    <div className="h-16 w-16 shrink-0 overflow-hidden rounded-lg border border-[#2C3240] bg-black">
                      <img
                        src={originalImageUrl(proj.id)}
                        alt={proj.name}
                        className="h-full w-full object-contain"
                      />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-start justify-between gap-2">
                        <div className="min-w-0">
                          <div className="flex items-center gap-1.5">
                            <h3 className="truncate text-xs font-semibold text-white">
                              {proj.name}
                            </h3>
                            {isActive && (
                              <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-blue-400" />
                            )}
                          </div>
                          <p className="truncate text-[11px] text-slate-400">
                            {proj.original_filename}
                          </p>
                        </div>
                        {confirmDeleteId !== proj.id && (
                          <button
                            type="button"
                            disabled={isBusy}
                            onClick={(e) => {
                              e.stopPropagation();
                              setConfirmDeleteId(proj.id);
                            }}
                            title="Delete project from vault"
                            className="rounded-md p-1 text-slate-500 transition-colors hover:bg-red-500/15 hover:text-red-400 disabled:opacity-40"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </button>
                        )}
                      </div>

                      {confirmDeleteId === proj.id && (
                        <div
                          className="mt-1.5 rounded-md border border-red-500/30 bg-red-500/10 p-2"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <p className="text-[11px] font-semibold text-red-300">
                            Delete forever?
                          </p>
                          <p className="text-[10px] text-slate-400">
                            The project and all brush edits are lost.
                          </p>
                          <div className="mt-1.5 flex gap-1.5">
                            <button
                              type="button"
                              disabled={isBusy}
                              onClick={() => {
                                setConfirmDeleteId(null);
                                onDeleteProject(proj.id);
                              }}
                              className="rounded-md bg-red-600 px-2 py-1 text-[11px] font-semibold text-white hover:bg-red-500 disabled:opacity-40"
                            >
                              Confirm
                            </button>
                            <button
                              type="button"
                              disabled={isBusy}
                              onClick={() => setConfirmDeleteId(null)}
                              className="rounded-md bg-[#262B36] px-2 py-1 text-[11px] font-semibold text-slate-300 hover:bg-[#2C313F] disabled:opacity-40"
                            >
                              Cancel
                            </button>
                          </div>
                        </div>
                      )}

                      <div className="mt-1.5 flex flex-wrap items-center gap-1.5 text-[10px] font-mono text-slate-400">
                        <span className="rounded bg-[#262B36] px-1.5 py-0.5 text-slate-300">
                          {proj.width}×{proj.height}
                        </span>
                        <span className="rounded bg-[#262B36] px-1.5 py-0.5">
                          AI: {proj.ai_mode === 'off' ? 'Off' : 'Guided'}
                        </span>
                        {proj.params.frame.enabled && (
                          <span className="rounded bg-blue-500/20 px-1.5 py-0.5 text-blue-300">
                            Frame
                          </span>
                        )}
                      </div>

                      <div className="mt-2.5 flex items-center gap-2">
                        <button
                          type="button"
                          onClick={() => {
                            onSelectProject(proj, false);
                            onClose();
                          }}
                          className="flex flex-1 items-center justify-center gap-1 rounded-md bg-[#282D3A] px-2 py-1 text-[11px] font-semibold text-slate-200 transition-colors hover:bg-[#323848]"
                        >
                          <FolderOpen className="h-3 w-3" />
                          Open Studio
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            onSelectProject(proj, true);
                            onClose();
                          }}
                          className="flex flex-1 items-center justify-center gap-1 rounded-md bg-blue-600/90 px-2 py-1 text-[11px] font-semibold text-white transition-colors hover:bg-blue-500"
                        >
                          3D Editor
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </aside>
    </div>
  );
};
