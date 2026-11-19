import { exportErrorMessage, exportSuccessMessage } from './api/exportToast';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { AlertCircle, CheckCircle2, X } from 'lucide-react';
import {
  createPreset,
  createProject,
  deletePreset,
  deleteProject,
  exportProjectFile,
  fetchFloat32Texture,
  getParts,
  PartsUnavailableError,
  getProject,
  listPresets,
  listProjects,
  preprocessProject,
  recomputeRelief,
  resetBrushEdits,
  setPartOrder,
  type PartOrderAction,
  uploadFloat32Texture,
  upscaleProject,
} from './api/client';
import {
  DEFAULT_MESH_SETTINGS,
  DEFAULT_RELIEF_PARAMS,
  EXPORT_FORMAT_META,
  type AiMode,
  type BitDepthOption,
  type ExportFormat,
  type Float32TexturePayload,
  type LinePolarity,
  type MeshSettings,
  type PresetRecord,
  type ProjectRecord,
  type ReliefParams,
  type TargetResolution,
  type WorkspaceView,
} from './types/monolith';
import { TopBar } from './components/TopBar';
import { StudioGenerator } from './components/StudioGenerator';
import { LocalVaultDrawer } from './components/LocalVaultDrawer';
import { createLatestGate } from './lib/latestGate';
import { BrushUploadQueue, type BrushJob } from './lib/brushUploadQueue';
import { applyPresetKeepingDimensions, presetStyleChangeMessage } from './lib/presets';
import { decodeParts, type PartsMap, type PartsPayload } from './lib/partsMap';
import { normalizeSculpt } from './lib/sculptParams';
import { createThrottle } from './lib/throttle';
import { createLatestRunner } from './lib/latestRunner';

const Editor3DWorkspace = React.lazy(() =>
  import('./components/Editor3DWorkspace').then((m) => ({
    default: m.Editor3DWorkspace,
  }))
);

const normalizeParams = normalizeSculpt;

interface ToastState {
  kind: 'error' | 'success';
  message: string;
}

/** Float32 texture tagged with the project it belongs to (guards cross-project mixups). */
type ProjectTexture = Float32TexturePayload & { projectId: string };

const MAX_BRUSH_UNDO = 15;

/** dst = clamp(dst + sign * delta, 0, 1) in place. */
function addDeltaClamped(dst: Float32Array, delta: Float32Array, sign: 1 | -1): void {
  for (let i = 0; i < dst.length; i++) {
    const v = dst[i] + sign * delta[i];
    dst[i] = v < 0 ? 0 : v > 1 ? 1 : v;
  }
}

export const App: React.FC = () => {
  const [workspace, setWorkspace] = useState<WorkspaceView>('studio');
  const [vaultOpen, setVaultOpen] = useState(false);

  const [projects, setProjects] = useState<ProjectRecord[]>([]);
  const [activeProject, setActiveProject] = useState<ProjectRecord | null>(null);
  const [params, setParams] = useState<ReliefParams>(DEFAULT_RELIEF_PARAMS);
  const [meshSettings, setMeshSettings] = useState<MeshSettings>(DEFAULT_MESH_SETTINGS);
  const [floatTex, setFloatTex] = useState<ProjectTexture | null>(null);
  const [textureError, setTextureError] = useState<string | null>(null);
  const [presets, setPresets] = useState<PresetRecord[]>([]);

  // Workspace 1 generation settings
  const [targetRes, setTargetRes] = useState<TargetResolution>(4096);
  const [bgRemoval, setBgRemoval] = useState(true);
  const [hdRestore, setHdRestore] = useState(true);
  const [polarity, setPolarity] = useState<LinePolarity>('auto');
  const [aiMode, setAiMode] = useState<AiMode>('off');
  const [bitDepth, setBitDepth] = useState<BitDepthOption>('16bit');

  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [pendingFilePreviewUrl, setPendingFilePreviewUrl] = useState<string | null>(
    null
  );

  const [lastComputeMs, setLastComputeMs] = useState<number | null>(null);
  const [uniqueLevels, setUniqueLevels] = useState<number | null>(null);
  const [isBusy, setIsBusy] = useState(false);
  // Style / Sculpt-version switches can take several seconds on the first recompute.
  const [styleBusy, setStyleBusy] = useState(false);
  const [layersBusy, setLayersBusy] = useState(false);
  // Sculpt part-label map for the Layers tool, tagged with its project.
  const [partsState, setPartsState] = useState<{ projectId: string; map: PartsMap } | null>(null);
  const [edgeClamped, setEdgeClamped] = useState(false);
  const [isSavingBrush, setIsSavingBrush] = useState(false);
  const [canUndoBrush, setCanUndoBrush] = useState(false);
  const [toast, setToast] = useState<ToastState | null>(null);

  // Undo stack of stroke *deltas* (not absolute snapshots) so undo stays correct after
  // slider changes: the server keeps brush edits as a delta over the procedural relief.
  const brushUndoRef = useRef<Float32Array[]>([]);
  // Live preview recompute: leading+trailing 50 ms throttle (stable across renders).
  // previewPendingRef is true while a trailing preview call is still queued.
  const previewPendingRef = useRef(false);
  // Previews run in-flight + latest-pending: a slow round trip is never aborted by the
  // next throttle tick, so the picture updates during a drag.
  const previewExecRef = useRef<(projectId: string, p: ReliefParams) => Promise<unknown>>(
    async () => {}
  );
  const previewRunnerRef = useRef(
    createLatestRunner((a: { id: string; p: ReliefParams }) => previewExecRef.current(a.id, a.p))
  );
  const previewAbortRef = useRef<AbortController | null>(null);
  const previewThrottleRef = useRef(
    createThrottle((projectId: string, p: ReliefParams) => {
      previewPendingRef.current = false;
      previewRunnerRef.current.submit({ id: projectId, p });
    }, 50)
  );
  const fullFlushTimerRef = useRef<number | null>(null);
  const recomputeAbortRef = useRef<AbortController | null>(null);

  // Synchronous mirrors of state used by async callbacks to detect stale results.
  const activeProjectIdRef = useRef<string | null>(null);
  const floatTexRef = useRef<ProjectTexture | null>(null);
  const projectsRef = useRef<ProjectRecord[]>([]);
  // Generation token for the parts map: bumped whenever the map is cleared or replaced, so a
  // stale in-flight response (older project, cleared map) is dropped.
  const partsGenRef = useRef(0);
  const partsLoadingRef = useRef<{ projectId: string; gen: number } | null>(null);
  const layersBusyRef = useRef(false);
  const clearParts = useCallback(() => {
    partsGenRef.current += 1;
    setPartsState(null);
  }, []);
  const droppedShownRef = useRef<Record<string, number>>({});
  // Shared ordering guard: only the newest texture request may be applied (load or recompute).
  const textureGateRef = useRef(createLatestGate());
  const paramsEditRef = useRef(0);
  const lastSeqRef = useRef(0);

  // Brush stroke bookkeeping.
  const strokeRef = useRef<{ projectId: string; snapshot: Float32Array } | null>(null);
  const deferredTexRef = useRef<ProjectTexture | null>(null);
  const strokeEpochRef = useRef(0);
  // Error handler is swapped in below (it needs helpers defined later); the queue itself is stable.
  const brushErrorRef = useRef<(err: unknown, job: BrushJob) => void>(() => undefined);
  const brushQueueRef = useRef<BrushUploadQueue | null>(null);
  if (brushQueueRef.current === null) {
    brushQueueRef.current = new BrushUploadQueue(
      (job) => uploadFloat32Texture(job.projectId, job.delta, job.width, job.height, 'delta'),
      (err, job) => brushErrorRef.current(err, job)
    );
  }
  const brushQueue = brushQueueRef.current;

  useEffect(() => {
    brushQueue.onPendingChange = (n) => setIsSavingBrush(n > 0);
    return () => {
      brushQueue.onPendingChange = undefined;
    };
  }, [brushQueue]);

  useEffect(() => {
    projectsRef.current = projects;
  }, [projects]);

  const showToast = useCallback((kind: 'error' | 'success', message: string) => {
    setToast({ kind, message });
  }, []);

  useEffect(() => {
    if (!toast) return;
    const id = window.setTimeout(() => setToast(null), 4500);
    return () => window.clearTimeout(id);
  }, [toast]);

  const commitTexture = useCallback((tex: ProjectTexture | null) => {
    floatTexRef.current = tex;
    setFloatTex(tex);
  }, []);

  const clearBrushUndo = useCallback(() => {
    brushUndoRef.current = [];
    setCanUndoBrush(false);
  }, []);

  /** The newest full (non-preview) recompute still running; export waits for it. */
  const fullInFlightRef = useRef<Promise<boolean> | null>(null);

  const cancelPendingRecompute = useCallback(() => {
    previewThrottleRef.current.cancel();
    previewRunnerRef.current.cancel();
    previewAbortRef.current?.abort();
    previewAbortRef.current = null;
    previewPendingRef.current = false;
    if (fullFlushTimerRef.current !== null) {
      window.clearTimeout(fullFlushTimerRef.current);
      fullFlushTimerRef.current = null;
    }
    recomputeAbortRef.current?.abort();
    recomputeAbortRef.current = null;
  }, []);

  const activateProject = useCallback((proj: ProjectRecord | null) => {
    activeProjectIdRef.current = proj?.id ?? null;
    setActiveProject(proj);
  }, []);

  /** Apply a server texture only if it still belongs to the active project. */
  const applyServerTexture = useCallback(
    (projectId: string, tex: Float32TexturePayload): boolean => {
      if (activeProjectIdRef.current !== projectId) return false;
      const next: ProjectTexture = { ...tex, projectId };
      if (strokeRef.current) {
        // Never swap the buffer being painted; it is merged when the stroke ends.
        deferredTexRef.current = next;
        return true;
      }
      commitTexture(next);
      setTextureError(null);
      return true;
    },
    [commitTexture]
  );

  /** Fetch a texture that already contains every completed brush stroke. */
  const fetchConsistentTexture = useCallback(
    async (projectId: string, signal?: AbortSignal) => {
      for (;;) {
        const epoch = strokeEpochRef.current;
        await brushQueue.whenIdle();
        const tex = await fetchFloat32Texture(projectId, 1024, signal);
        if (epoch === strokeEpochRef.current) return tex;
      }
    },
    []
  );

  const loadTextureForProject = useCallback(
    async (projectId: string) => {
      const gate = textureGateRef.current;
      const ticket = gate.next();
      setTextureError(null);
      try {
        const tex = await fetchConsistentTexture(projectId);
        if (!gate.isLatest(ticket)) return;
        if (applyServerTexture(projectId, tex) && tex.uniqueLevels !== undefined) {
          setUniqueLevels(tex.uniqueLevels);
        }
      } catch (err) {
        if (gate.isLatest(ticket) && activeProjectIdRef.current === projectId) {
          // Never leave the previous project's heights on screen under this project.
          commitTexture(null);
          setTextureError(err instanceof Error ? err.message : 'Failed to load Float32 buffer');
        }
        throw err;
      }
    },
    [applyServerTexture, commitTexture, fetchConsistentTexture]
  );

  const openProject = useCallback(
    async (proj: ProjectRecord, switchToEditor = false) => {
      cancelPendingRecompute();
      if (activeProjectIdRef.current !== proj.id) {
        commitTexture(null);
        clearBrushUndo();
        deferredTexRef.current = null;
      }
      activateProject(proj);
      setEdgeClamped(false);
      clearParts();
      setParams(normalizeParams(proj.params));
      setAiMode(proj.ai_mode === 'guided_overlap' ? 'guided_overlap' : 'off');
      setBgRemoval(proj.bg_removal ?? true);
      setHdRestore(proj.hd_restore ?? true);
      setPolarity(
        proj.polarity === 'white_on_black' || proj.polarity === 'black_on_white'
          ? proj.polarity
          : 'auto'
      );
      if (proj.bit_depth === '8bit' || proj.bit_depth === '32bit_exr') {
        setBitDepth(proj.bit_depth);
      } else {
        setBitDepth('16bit');
      }
      if (typeof proj.compute_ms === 'number') setLastComputeMs(proj.compute_ms);
      if (typeof proj.unique_levels === 'number') setUniqueLevels(proj.unique_levels);

      const maxEdge = proj.target_res ?? Math.max(proj.width, proj.height);
      if (maxEdge >= 7000) setTargetRes(8192);
      else if (maxEdge >= 3500) setTargetRes(4096);
      else setTargetRes(2048);

      setPendingFile(null);
      setPendingFilePreviewUrl((url) => {
        if (url) URL.revokeObjectURL(url);
        return null;
      });

      if (switchToEditor) setWorkspace('editor3d');

      // The cached record may be stale (e.g. params saved by another session): refresh it
      // from the server while the texture loads, unless the user already edited params.
      const editsAtOpen = paramsEditRef.current;
      getProject(proj.id)
        .then((fresh) => {
          if (activeProjectIdRef.current !== proj.id) return;
          setProjects((prev) => prev.map((p) => (p.id === fresh.id ? fresh : p)));
          if (paramsEditRef.current !== editsAtOpen) return;
          setActiveProject(fresh);
          setParams(normalizeParams(fresh.params));
        })
        .catch(() => {
          /* keep the cached record; the texture load reports connectivity problems */
        });

      try {
        await loadTextureForProject(proj.id);
      } catch (err) {
        showToast(
          'error',
          err instanceof Error ? err.message : 'Failed to load Float32 buffer'
        );
      }
    },
    [
      activateProject,
      cancelPendingRecompute,
      clearBrushUndo,
      clearParts,
      commitTexture,
      loadTextureForProject,
      showToast,
    ]
  );

  const openProjectRef = useRef(openProject);
  openProjectRef.current = openProject;

  // Fetch vault projects & presets and open the most recent project.
  const reloadVault = useCallback(
    async (isCancelled: () => boolean = () => false) => {
      setIsBusy(true);
      setTextureError(null);
      try {
        const [projList, presetList] = await Promise.all([
          listProjects(),
          listPresets().catch(() => [] as PresetRecord[]),
        ]);
        if (isCancelled()) return;
        setPresets(presetList);

        if (projList.length > 0) {
          setProjects(projList);
          projectsRef.current = projList;
          await openProjectRef.current(projList[0], false);
        } else {
          // Auto-create from example if DB is completely empty
          const created = await createProject({
            use_benchmark: true,
            name: 'Geometric Study',
            target_res: 4096,
            bg_removal: true,
            hd_restore: true,
            ai_mode: 'off',
            bit_depth: '16bit',
          });
          if (isCancelled()) return;
          setProjects([created]);
          projectsRef.current = [created];
          await openProjectRef.current(created, false);
        }
      } catch (err) {
        if (!isCancelled()) {
          const message =
            err instanceof Error && err.message
              ? err.message
              : 'Waiting for MONOLITH backend on port 8000';
          setTextureError(message);
          showToast('error', message);
        }
      } finally {
        if (!isCancelled()) setIsBusy(false);
      }
    },
    [showToast]
  );

  useEffect(() => {
    let cancelled = false;
    reloadVault(() => cancelled);
    return () => {
      cancelled = true;
    };
  }, [reloadVault]);

  const handleRetryLoad = useCallback(() => {
    const pid = activeProjectIdRef.current;
    if (pid) {
      loadTextureForProject(pid).catch((err) =>
        showToast('error', err instanceof Error ? err.message : 'Failed to load Float32 buffer')
      );
    } else {
      reloadVault();
    }
  }, [loadTextureForProject, reloadVault, showToast]);

  /** Returns false only when the recompute failed (superseded requests count as success). */
  const runRecompute = useCallback(
    async (projectId: string, nextParams: ReliefParams, previewOnly = false): Promise<boolean> => {
      // Previews never abort an in-flight request (the runner serialises them); full
      // recomputes supersede everything in flight.
      const ctrl = new AbortController();
      if (previewOnly) {
        previewAbortRef.current = ctrl;
      } else {
        recomputeAbortRef.current?.abort();
        previewAbortRef.current?.abort();
        previewAbortRef.current = null;
        previewRunnerRef.current.cancel();
        recomputeAbortRef.current = ctrl;
      }
      // Monotonic stamp (epoch ms survives reloads) so the server drops out-of-order requests.
      const seq = Math.max(Date.now(), lastSeqRef.current + 1);
      lastSeqRef.current = seq;

      try {
        const t0 = performance.now();
        const res = await recomputeRelief(projectId, nextParams, ctrl.signal, previewOnly, seq);
        if (ctrl.signal.aborted || res.stale) return true;

        const gate = textureGateRef.current;
        const ticket = gate.next();
        const tex = await fetchConsistentTexture(projectId, ctrl.signal);
        if (ctrl.signal.aborted || !gate.isLatest(ticket)) return true;
        if (!applyServerTexture(projectId, tex)) return true;

        const elapsed = res.compute_ms ?? performance.now() - t0;
        setLastComputeMs(elapsed);
        setUniqueLevels(res.unique_levels ?? tex.uniqueLevels ?? null);
        setEdgeClamped(res.sculpt_edge_clamped === true);
        setActiveProject((prev) =>
          prev && prev.id === projectId ? { ...prev, params: nextParams } : prev
        );
        setProjects((prev) =>
          prev.map((p) => (p.id === projectId ? { ...p, params: nextParams } : p))
        );
        return true;
      } catch (err) {
        if ((err as Error).name === 'AbortError') return true;
        showToast(
          'error',
          err instanceof Error ? err.message : 'Recompute failed'
        );
        return false;
      }
    },
    [applyServerTexture, fetchConsistentTexture, showToast]
  );

  const executeRecompute = useCallback(
    (projectId: string, nextParams: ReliefParams, previewOnly = false): Promise<boolean> => {
      const run = runRecompute(projectId, nextParams, previewOnly);
      if (!previewOnly) {
        fullInFlightRef.current = run;
        void run.finally(() => {
          if (fullInFlightRef.current === run) fullInFlightRef.current = null;
        });
      }
      return run;
    },
    [runRecompute]
  );

  const handleParamsChange = useCallback(
    (next: ReliefParams, immediate = false) => {
      const prevParams = params;
      setParams(next);
      paramsEditRef.current += 1;
      const editId = paramsEditRef.current;
      if (!activeProject) return;
      const projectId = activeProject.id;
      const prevRecordParams =
        projectsRef.current.find((p) => p.id === projectId)?.params ?? activeProject.params;
      // Optimistic: reopening this project from the vault must not show pre-edit params.
      setProjects((prev) => prev.map((p) => (p.id === projectId ? { ...p, params: next } : p)));

      if (fullFlushTimerRef.current !== null) {
        window.clearTimeout(fullFlushTimerRef.current);
        fullFlushTimerRef.current = null;
      }

      if (immediate) {
        previewThrottleRef.current.cancel();
        previewPendingRef.current = false;
        const run = executeRecompute(projectId, next, false); // also cancels the preview runner
        if (next.style !== params.style || next.sculpt_version !== params.sculpt_version) {
          clearParts();
          setStyleBusy(true);
          void run
            .then((ok) => {
              // A failed switch must not leave the UI showing a style the server never applied.
              if (ok || paramsEditRef.current !== editId) return;
              if (activeProjectIdRef.current !== projectId) return;
              setParams(prevParams);
              setActiveProject((prev) =>
                prev && prev.id === projectId ? { ...prev, params: prevRecordParams } : prev
              );
              setProjects((prev) =>
                prev.map((p) => (p.id === projectId ? { ...p, params: prevRecordParams } : p))
              );
              clearParts();
            })
            .finally(() => setStyleBusy(false));
        }
      } else {
        previewExecRef.current = (id, p) => executeRecompute(id, p, true);
        // The throttle's leading call may run synchronously and clear the flag.
        previewPendingRef.current = true;
        previewThrottleRef.current.call(projectId, next);
        fullFlushTimerRef.current = window.setTimeout(() => {
          fullFlushTimerRef.current = null;
          executeRecompute(projectId, next, false);
        }, 260);
      }
    },
    [activeProject, executeRecompute, params]
  );

  const handleSelectFile = (file: File) => {
    if (pendingFilePreviewUrl) {
      URL.revokeObjectURL(pendingFilePreviewUrl);
    }
    setPendingFile(file);
    setPendingFilePreviewUrl(URL.createObjectURL(file));
  };

  const handleLoadBenchmark = async () => {
    setIsBusy(true);
    try {
      const created = await createProject({
        use_benchmark: true,
        name: 'Geometric Study',
        target_res: targetRes,
        bg_removal: bgRemoval,
        hd_restore: hdRestore,
        ai_mode: aiMode,
        bit_depth: bitDepth,
      });
      setProjects((prev) => [created, ...prev.filter((p) => p.id !== created.id)]);
      await openProject(created, false);
      showToast('success', 'Loaded example line drawing into vault.');
    } catch (err) {
      showToast(
        'error',
        err instanceof Error ? err.message : 'Failed to load benchmark'
      );
    } finally {
      setIsBusy(false);
    }
  };

  const runCreateFromPendingOrPreprocess = async (alsoRecompute: boolean) => {
    setIsBusy(true);
    try {
      if (pendingFile) {
        const baseName = pendingFile.name.replace(/\.[^.]+$/, '') || 'Untitled Relief';
        const created = await createProject({
          file: pendingFile,
          name: baseName,
          target_res: targetRes,
          bg_removal: bgRemoval,
          hd_restore: hdRestore,
          ai_mode: aiMode,
          bit_depth: bitDepth,
          polarity,
        });
        setProjects((prev) => [created, ...prev]);
        await openProject(created, false);
        showToast('success', `Generated relief for "${created.name}".`);
        return;
      }

      if (!activeProject) {
        await handleLoadBenchmark();
        return;
      }

      const projectId = activeProject.id;
      cancelPendingRecompute();
      await brushQueue.whenIdle();
      const updated = await preprocessProject(projectId, {
        bg_removal: bgRemoval,
        hd_restore: hdRestore,
        target_res: targetRes,
        ai_mode: aiMode,
        polarity,
      });
      clearBrushUndo();
      clearParts();
      if (activeProjectIdRef.current === updated.id) setActiveProject(updated);
      setProjects((prev) =>
        prev.map((p) => (p.id === updated.id ? updated : p))
      );

      if (alsoRecompute) {
        const ok = await executeRecompute(updated.id, params);
        if (ok) showToast('success', 'Relief regenerated cleanly with zero hallucinations.');
        if (ok) checkDroppedAfterReprocess(updated.id);
      } else {
        await loadTextureForProject(updated.id);
        showToast('success', 'Pre-processing applied locally at $0.');
        checkDroppedAfterReprocess(updated.id);
      }
    } catch (err) {
      showToast(
        'error',
        err instanceof Error ? err.message : 'Processing failed'
      );
    } finally {
      setIsBusy(false);
    }
  };

  const handleUpscale = async (target: 4096 | 8192) => {
    if (!activeProject) return;
    setIsBusy(true);
    try {
      cancelPendingRecompute();
      await brushQueue.whenIdle();
      const updated = await upscaleProject(activeProject.id, target);
      clearBrushUndo();
      clearParts();
      setTargetRes(target);
      if (activeProjectIdRef.current === updated.id) setActiveProject(updated);
      setProjects((prev) =>
        prev.map((p) => (p.id === updated.id ? updated : p))
      );
      await loadTextureForProject(updated.id);
      checkDroppedAfterReprocess(updated.id);
      showToast(
        'success',
        `Upscaled to ${updated.width}×${updated.height} with float32 precision.`
      );
    } catch (err) {
      showToast('error', err instanceof Error ? err.message : 'Upscale failed');
    } finally {
      setIsBusy(false);
    }
  };

  const handleExport = async (format: ExportFormat) => {
    if (!activeProject) return;
    setIsBusy(true);
    try {
      // Make sure every stroke and the latest slider state are on the server first.
      await brushQueue.whenIdle();
      // A full recompute already in flight (its debounce timer has fired) must land first,
      // otherwise the export could go out with the previous parameters.
      if (fullInFlightRef.current) await fullInFlightRef.current;
      if (
        previewPendingRef.current ||
        previewRunnerRef.current.busy ||
        fullFlushTimerRef.current !== null
      ) {
        cancelPendingRecompute();
        const ok = await executeRecompute(activeProject.id, params, false);
        if (!ok) throw new Error('Could not apply the latest settings before exporting');
      }
      const ext = EXPORT_FORMAT_META[format].ext;
      const safeName = activeProject.name
        .toLowerCase()
        .replace(/[^a-z0-9-_]+/g, '_');
      const result = await exportProjectFile(
        activeProject.id,
        {
          format,
          mesh_mode: meshSettings.mode,
          mesh_tolerance_mm: meshSettings.toleranceMm,
          mesh_resolution: meshSettings.gridResolution,
        },
        `${safeName}_${format}.${ext}`
      );
      showToast(
        'success',
        exportSuccessMessage({
          label: EXPORT_FORMAT_META[format].label,
          format,
          requestedMode: meshSettings.mode,
          result,
        })
      );
    } catch (err) {
      showToast('error', err instanceof Error ? exportErrorMessage(err.message) : 'Export failed');
    } finally {
      setIsBusy(false);
    }
  };

  const handleDeleteProject = async (projectId: string) => {
    try {
      if (activeProjectIdRef.current === projectId) cancelPendingRecompute();
      await deleteProject(projectId);
      const remaining = projectsRef.current.filter((p) => p.id !== projectId);
      projectsRef.current = remaining;
      setProjects((prev) => prev.filter((p) => p.id !== projectId));
      if (activeProjectIdRef.current === projectId) {
        if (remaining.length > 0) {
          await openProject(remaining[0], false);
        } else {
          activateProject(null);
          commitTexture(null);
          clearBrushUndo();
        }
      }
      showToast('success', 'Removed project from local vault.');
    } catch (err) {
      showToast('error', err instanceof Error ? err.message : 'Delete failed');
    }
  };

  /** Queue a brush delta; queued-but-unstarted deltas for the same project coalesce. */
  const enqueueBrushUpload = useCallback(
    (projectId: string, delta: Float32Array, width: number, height: number) => {
      brushQueue.enqueue({ projectId, delta, width, height });
    },
    [brushQueue]
  );

  useEffect(() => {
    brushErrorRef.current = (err, job) => {
      showToast('error', err instanceof Error ? err.message : 'Failed to save brush edit');
      // Resync with the server so the screen never shows unsaved sculpting.
      if (activeProjectIdRef.current === job.projectId) {
        clearBrushUndo();
        loadTextureForProject(job.projectId).catch(() => undefined);
      }
    };
  }, [clearBrushUndo, loadTextureForProject, showToast]);

  const handleBrushStrokeStart = (snapshot: Float32Array) => {
    const projectId = activeProjectIdRef.current;
    const tex = floatTexRef.current;
    if (!projectId || !tex || tex.projectId !== projectId) return;
    strokeRef.current = { projectId, snapshot };
  };

  const handleBrushDataChange = () => {
    // WebGL DataTexture is mutated & flagged needsUpdate directly inside ReliefViewport3D for 60 FPS sculpting.
  };

  const handleBrushStrokeEnd = (finalData: Float32Array) => {
    const stroke = strokeRef.current;
    strokeRef.current = null;
    const tex = floatTexRef.current;
    const deferred = deferredTexRef.current;
    deferredTexRef.current = null;
    if (!stroke || !tex || tex.projectId !== stroke.projectId) return;
    if (finalData.length !== stroke.snapshot.length) return;

    // Upload only what this stroke changed: untouched pixels are exactly zero.
    const delta = new Float32Array(finalData.length);
    let changed = false;
    for (let i = 0; i < delta.length; i++) {
      const d = finalData[i] - stroke.snapshot[i];
      delta[i] = d;
      if (d !== 0) changed = true;
    }

    if (changed) {
      strokeEpochRef.current += 1;
      brushUndoRef.current = [...brushUndoRef.current.slice(-(MAX_BRUSH_UNDO - 1)), delta];
      setCanUndoBrush(true);
      enqueueBrushUpload(stroke.projectId, delta, tex.width, tex.height);
    }

    // A recompute that landed mid-stroke: show it with this stroke merged on top.
    if (
      deferred &&
      deferred.projectId === stroke.projectId &&
      deferred.data.length === delta.length
    ) {
      if (changed) addDeltaClamped(deferred.data, delta, 1);
      commitTexture(deferred);
    }
  };

  const handleUndoBrush = () => {
    if (strokeRef.current) return;
    const projectId = activeProjectIdRef.current;
    const tex = floatTexRef.current;
    if (!projectId || !tex || tex.projectId !== projectId) return;
    const delta = brushUndoRef.current.pop();
    setCanUndoBrush(brushUndoRef.current.length > 0);
    if (!delta || delta.length !== tex.data.length) return;

    const data = new Float32Array(tex.data);
    addDeltaClamped(data, delta, -1);
    strokeEpochRef.current += 1;
    commitTexture({ ...tex, data });
    enqueueBrushUpload(projectId, delta.map((v) => -v), tex.width, tex.height);
  };

  const handleResetBrush = async () => {
    const projectId = activeProjectIdRef.current;
    if (!projectId) return;
    setIsSavingBrush(true);
    try {
      await brushQueue.whenIdle();
      await resetBrushEdits(projectId);
      clearBrushUndo();
      await loadTextureForProject(projectId);
      showToast('success', 'Brush edits reset to original mathematical relief.');
    } catch (err) {
      showToast(
        'error',
        err instanceof Error ? err.message : 'Failed to reset brush edits'
      );
    } finally {
      setIsSavingBrush(brushQueue.pending > 0);
    }
  };

  /** Store a parts payload for the project and surface dropped overrides once per count. */
  const applyPartsPayload = useCallback(
    (projectId: string, payload: PartsPayload, gen?: number) => {
      if (activeProjectIdRef.current !== projectId) return;
      if (gen !== undefined && gen !== partsGenRef.current) return; // stale response
      partsGenRef.current += 1;
      const dropped = payload.dropped_overrides ?? 0;
      if (dropped > 0 && droppedShownRef.current[projectId] !== dropped) {
        showToast(
          'error',
          `${dropped} layer choice${dropped === 1 ? '' : 's'} could not be re-applied after reprocessing`
        );
      }
      droppedShownRef.current[projectId] = dropped;
      setPartsState({ projectId, map: decodeParts(payload) });
    },
    [showToast]
  );

  /** Fetch parts; `quiet` hides non-actionable failures (used for background checks). */
  const fetchParts = useCallback(
    (projectId: string, quiet: boolean) => {
      const gen = partsGenRef.current;
      const inFlight = partsLoadingRef.current;
      if (inFlight && inFlight.projectId === projectId && inFlight.gen === gen) return;
      const key = { projectId, gen };
      partsLoadingRef.current = key;
      getParts(projectId)
        .then((payload) => applyPartsPayload(projectId, payload, gen))
        .catch((err) => {
          if (err instanceof PartsUnavailableError) return; // Classic on the server: nothing to load
          if (quiet || gen !== partsGenRef.current) return;
          showToast('error', err instanceof Error ? err.message : 'Failed to load parts');
        })
        .finally(() => {
          if (partsLoadingRef.current === key) partsLoadingRef.current = null;
        });
    },
    [applyPartsPayload, showToast]
  );

  const handleLoadParts = useCallback(() => {
    const projectId = activeProjectIdRef.current;
    if (projectId) fetchParts(projectId, false);
  }, [fetchParts]);

  /** After a reprocess of a Sculpt project, tell the artist about dropped layer choices now. */
  const checkDroppedAfterReprocess = (projectId: string) => {
    if (params.style === 'sculpt') fetchParts(projectId, true);
  };

  const handlePartOrder = async (
    action: PartOrderAction,
    uv: { u: number; v: number } | null
  ) => {
    const projectId = activeProjectIdRef.current;
    if (!projectId || layersBusyRef.current) return;
    // Reset all ignores (u, v) on the backend.
    const u = uv ? uv.u : 0.5;
    const vApi = uv ? 1 - uv.v : 0.5; // the viewer's v points up, the API's points down
    layersBusyRef.current = true;
    setLayersBusy(true);
    try {
      await brushQueue.whenIdle();
      // Land any pending slider edit first so the reorder sees the current params.
      if (fullInFlightRef.current) await fullInFlightRef.current;
      if (
        previewPendingRef.current ||
        previewRunnerRef.current.busy ||
        fullFlushTimerRef.current !== null
      ) {
        cancelPendingRecompute();
        const ok = await executeRecompute(projectId, params, false);
        if (!ok) throw new Error('Could not apply the latest settings before changing layers');
      }
      const res = await setPartOrder(projectId, u, vApi, action);
      if (typeof res.compute_ms === 'number') setLastComputeMs(res.compute_ms);
      try {
        applyPartsPayload(projectId, res);
      } catch (err) {
        showToast('error', err instanceof Error ? err.message : 'Could not read the part map');
      }
      await loadTextureForProject(projectId);
    } catch (err) {
      showToast('error', err instanceof Error ? err.message : 'Layer change failed');
    } finally {
      layersBusyRef.current = false;
      setLayersBusy(false);
    }
  };

  const handleSavePreset = async (name: string) => {
    try {
      const created = await createPreset(name, params);
      setPresets((prev) => [created, ...prev]);
      showToast('success', `Saved preset "${name}".`);
    } catch (err) {
      showToast(
        'error',
        err instanceof Error ? err.message : 'Failed to save preset'
      );
    }
  };

  const handleDeletePreset = async (id: string) => {
    try {
      await deletePreset(id);
      setPresets((prev) => prev.filter((p) => p.id !== id));
    } catch (err) {
      showToast(
        'error',
        err instanceof Error ? err.message : 'Failed to delete preset'
      );
    }
  };

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden">
      <TopBar
        workspace={workspace}
        onChangeWorkspace={setWorkspace}
        activeProject={activeProject}
        projectCount={projects.length}
        onOpenVault={() => setVaultOpen(true)}
        lastComputeMs={lastComputeMs}
        uniqueLevels={uniqueLevels}
        isBusy={isBusy}
      />

      {workspace === 'studio' ? (
        <StudioGenerator
          project={activeProject}
          params={params}
          floatTex={floatTex}
          textureError={textureError}
          onRetryLoad={handleRetryLoad}
          targetRes={targetRes}
          bgRemoval={bgRemoval}
          polarity={polarity}
          onChangePolarity={setPolarity}
          hdRestore={hdRestore}
          aiMode={aiMode}
          bitDepth={bitDepth}
          pendingFile={pendingFile}
          pendingFilePreviewUrl={pendingFilePreviewUrl}
          lastComputeMs={lastComputeMs}
          uniqueLevels={uniqueLevels}
          isBusy={isBusy || styleBusy}
          edgeClamped={edgeClamped && params.style === 'sculpt'}
          onChangeTargetRes={setTargetRes}
          onChangeBgRemoval={setBgRemoval}
          onChangeHdRestore={setHdRestore}
          onChangeAiMode={setAiMode}
          onChangeBitDepth={setBitDepth}
          onSelectFile={handleSelectFile}
          onLoadBenchmark={handleLoadBenchmark}
          onApplyPreprocess={() => runCreateFromPendingOrPreprocess(false)}
          onGenerateRelief={() => runCreateFromPendingOrPreprocess(true)}
          onParamsChange={handleParamsChange}
          onUpscale={handleUpscale}
          onExport={handleExport}
          meshSettings={meshSettings}
          onMeshSettingsChange={setMeshSettings}
          onOpenEditor3D={() => setWorkspace('editor3d')}
          onOpenVault={() => setVaultOpen(true)}
        />
      ) : (
        <React.Suspense
          fallback={
            <div className="flex flex-1 flex-col items-center justify-center gap-3 bg-[#181A20] text-slate-300">
              <div className="h-7 w-7 animate-spin rounded-full border-2 border-[#2C3240] border-t-blue-500" />
              <p className="text-xs font-medium tracking-wide text-slate-400">
                Loading 3D CAD Workspace...
              </p>
            </div>
          }
        >
          <Editor3DWorkspace
            project={activeProject}
            params={params}
            floatTex={floatTex}
            presets={presets}
            isBusy={isBusy || styleBusy || layersBusy}
            isSavingBrush={isSavingBrush}
            onParamsChange={handleParamsChange}
            onBrushStrokeStart={handleBrushStrokeStart}
            onBrushDataChange={handleBrushDataChange}
            onBrushStrokeEnd={handleBrushStrokeEnd}
            onUndoBrush={handleUndoBrush}
            onResetBrush={handleResetBrush}
            canUndoBrush={canUndoBrush}
            onSavePreset={handleSavePreset}
            onApplyPreset={(presetParams) => {
              const next = normalizeParams(applyPresetKeepingDimensions(presetParams, params));
              const notice = presetStyleChangeMessage(next, params);
              handleParamsChange(next, true);
              if (notice) showToast('success', notice);
            }}
            onDeletePreset={handleDeletePreset}
            onExport={handleExport}
            meshSettings={meshSettings}
            onMeshSettingsChange={setMeshSettings}
            partsMap={partsState && partsState.projectId === activeProject?.id ? partsState.map : null}
            onLoadParts={handleLoadParts}
            onPartOrder={handlePartOrder}
            edgeClamped={edgeClamped && params.style === 'sculpt'}
          />
        </React.Suspense>
      )}

      <LocalVaultDrawer
        open={vaultOpen}
        onClose={() => setVaultOpen(false)}
        projects={projects}
        activeProjectId={activeProject?.id ?? null}
        onSelectProject={(proj, openEditor) => openProject(proj, openEditor)}
        onDeleteProject={handleDeleteProject}
        onCreateBenchmark={handleLoadBenchmark}
        isBusy={isBusy}
      />

      {toast && (
        <div
          className={`fixed top-16 right-4 z-50 flex max-w-sm items-center gap-2.5 rounded-xl border px-3.5 py-2.5 text-xs font-medium shadow-xl backdrop-blur ${
            toast.kind === 'error'
              ? 'border-red-500/40 bg-red-950/90 text-red-100'
              : 'border-emerald-500/40 bg-emerald-950/90 text-emerald-100'
          }`}
        >
          {toast.kind === 'error' ? (
            <AlertCircle className="h-4 w-4 shrink-0 text-red-400" />
          ) : (
            <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-400" />
          )}
          <span className="flex-1">{toast.message}</span>
          <button
            type="button"
            onClick={() => setToast(null)}
            className="text-white/60 hover:text-white"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      )}
    </div>
  );
};

export default App;
