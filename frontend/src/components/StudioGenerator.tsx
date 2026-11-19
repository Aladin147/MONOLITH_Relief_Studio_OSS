import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Archive,
  Box,
  Check,
  ChevronDown,
  Contrast,
  Download,
  Eye,
  FlipVertical,
  Image as ImageIcon,
  Layers,
  Maximize2,
  RotateCcw,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Upload,
  Wand2,
  Zap,
} from 'lucide-react';
import {
  type LinePolarity,
  EXPORT_FORMAT_META,
  downloadFmtForBitDepth,
  type AiMode,
  type BitDepthOption,
  type ExportFormat,
  type Float32TexturePayload,
  type MeshSettings,
  type ProjectRecord,
  type ReliefParams,
  type TargetResolution,
} from '../types/monolith';
import { originalImageUrl } from '../api/client';
import { AccordionSection, InfoTip, StudioSlider } from './UiControls';
import { SculptControls, StyleToggle } from './SculptControls';
import { MeshExportSettings } from './MeshExportSettings';

const ReliefViewport3D = React.lazy(() =>
  import('./ReliefViewport3D').then((m) => ({ default: m.ReliefViewport3D }))
);

interface StudioGeneratorProps {
  /** Edge roundness was raised by the server to keep the draft angle. */
  edgeClamped?: boolean;
  project: ProjectRecord | null;
  params: ReliefParams;
  floatTex: Float32TexturePayload | null;
  /** Set when loading the depth buffer failed; shows a Retry action instead of a spinner. */
  textureError?: string | null;
  onRetryLoad?: () => void;
  targetRes: TargetResolution;
  bgRemoval: boolean;
  polarity: LinePolarity;
  onChangePolarity: (val: LinePolarity) => void;
  hdRestore: boolean;
  aiMode: AiMode;
  bitDepth: BitDepthOption;
  pendingFile: File | null;
  pendingFilePreviewUrl: string | null;
  lastComputeMs: number | null;
  uniqueLevels: number | null;
  isBusy: boolean;
  onChangeTargetRes: (res: TargetResolution) => void;
  onChangeBgRemoval: (val: boolean) => void;
  onChangeHdRestore: (val: boolean) => void;
  onChangeAiMode: (mode: AiMode) => void;
  onChangeBitDepth: (depth: BitDepthOption) => void;
  onSelectFile: (file: File) => void;
  onLoadBenchmark: () => void;
  onApplyPreprocess: () => void;
  onGenerateRelief: () => void;
  onParamsChange: (next: ReliefParams, immediate?: boolean) => void;
  onUpscale: (target: 4096 | 8192) => void;
  onExport: (format: ExportFormat) => void;
  meshSettings: MeshSettings;
  onMeshSettingsChange: (v: MeshSettings) => void;
  onOpenEditor3D: () => void;
  onOpenVault: () => void;
}

interface HoverInspection {
  px: number;
  py: number;
  norm: number;
  zMm: number;
  level16: number;
}

export const StudioGenerator: React.FC<StudioGeneratorProps> = ({
  edgeClamped = false,
  project,
  params,
  floatTex,
  textureError = null,
  onRetryLoad,
  targetRes,
  bgRemoval,
  polarity,
  onChangePolarity,
  hdRestore,
  aiMode,
  bitDepth,
  pendingFile,
  pendingFilePreviewUrl,
  lastComputeMs,
  uniqueLevels,
  isBusy,
  onChangeTargetRes,
  onChangeBgRemoval,
  onChangeHdRestore,
  onChangeAiMode,
  onChangeBitDepth,
  onSelectFile,
  onLoadBenchmark,
  onApplyPreprocess,
  onGenerateRelief,
  onParamsChange,
  onUpscale,
  onExport,
  meshSettings,
  onMeshSettingsChange,
  onOpenEditor3D,
  onOpenVault,
}) => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wheelContainerRef = useRef<HTMLDivElement | null>(null);

  const [inline3D, setInline3D] = useState(false);
  const [falseColor, setFalseColor] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const isPanningRef = useRef(false);
  const lastMouseRef = useRef({ x: 0, y: 0 });

  const [hoverInfo, setHoverInfo] = useState<HoverInspection | null>(null);
  const [selectedDownloadFmt, setSelectedDownloadFmt] =
    useState<ExportFormat>(() => downloadFmtForBitDepth(bitDepth));

  // Keep the primary Download button in sync when bit depth changes externally
  // (e.g. loading a project from the vault).
  useEffect(() => {
    setSelectedDownloadFmt(downloadFmtForBitDepth(bitDepth));
  }, [bitDepth]);

  // Explicit Bit Depth clicks always re-sync the Download format, even when the
  // clicked option is already active (e.g. after picking STL from the split menu).
  const handleSelectBitDepth = (depth: BitDepthOption) => {
    setSelectedDownloadFmt(downloadFmtForBitDepth(depth));
    onChangeBitDepth(depth);
  };
  const [downloadMenuOpen, setDownloadMenuOpen] = useState(false);
  const [noisePopoverOpen, setNoisePopoverOpen] = useState(false);
  const [upscaleMenuOpen, setUpscaleMenuOpen] = useState(false);
  const [watermarkInfoOpen, setWatermarkInfoOpen] = useState(false);
  const [isDraggingFile, setIsDraggingFile] = useState(false);

  // Render Float32Array depth map cleanly onto 2D HTML5 Canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !floatTex || floatTex.width <= 0 || floatTex.height <= 0) {
      return;
    }
    const { data, width, height } = floatTex;
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const imgData = ctx.createImageData(width, height);
    const out = imgData.data;

    for (let i = 0; i < width * height; i++) {
      const v = Math.min(1, Math.max(0, data[i] ?? 0));
      const p = i * 4;
      if (!falseColor) {
        const g = Math.round(v * 255);
        out[p] = g;
        out[p + 1] = g;
        out[p + 2] = g;
        out[p + 3] = 255;
      } else {
        // Viridis-like perceptual elevation inspect gradient (pure black at 0)
        if (v <= 0.0005) {
          out[p] = 8;
          out[p + 1] = 10;
          out[p + 2] = 16;
        } else if (v < 0.33) {
          const t = v / 0.33;
          out[p] = Math.round(24 + t * 35);
          out[p + 1] = Math.round(28 + t * 92);
          out[p + 2] = Math.round(95 + t * 80);
        } else if (v < 0.66) {
          const t = (v - 0.33) / 0.33;
          out[p] = Math.round(59 + t * 80);
          out[p + 1] = Math.round(120 + t * 88);
          out[p + 2] = Math.round(175 - t * 55);
        } else {
          const t = (v - 0.66) / 0.34;
          out[p] = Math.round(139 + t * 116);
          out[p + 1] = Math.round(208 + t * 44);
          out[p + 2] = Math.round(120 - t * 68);
        }
        out[p + 3] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }, [floatTex, falseColor]);

  // React registers wheel listeners as passive, so preventDefault only works natively.
  useEffect(() => {
    const el = wheelContainerRef.current;
    if (!el || inline3D) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.15 : 0.87;
      setZoom((z) => Math.min(12, Math.max(0.5, z * factor)));
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [inline3D]);

  const textureStats = useMemo(() => {
    if (!floatTex || floatTex.data.length === 0) {
      return { minPct: 0, maxPct: 0, bgZeroPct: 100 };
    }
    let min = 1;
    let max = 0;
    let zeroCount = 0;
    const len = floatTex.data.length;
    for (let i = 0; i < len; i++) {
      const v = floatTex.data[i];
      if (v < min) min = v;
      if (v > max) max = v;
      if (v <= 0.0001) zeroCount++;
    }
    return {
      minPct: min * 100,
      maxPct: max * 100,
      bgZeroPct: (zeroCount / len) * 100,
    };
  }, [floatTex]);

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (isPanningRef.current) {
      const dx = e.clientX - lastMouseRef.current.x;
      const dy = e.clientY - lastMouseRef.current.y;
      lastMouseRef.current = { x: e.clientX, y: e.clientY };
      setPan((p) => ({ x: p.x + dx, y: p.y + dy }));
    }

    const canvas = canvasRef.current;
    if (!canvas || !floatTex) return;
    const rect = canvas.getBoundingClientRect();
    if (
      e.clientX < rect.left ||
      e.clientX > rect.right ||
      e.clientY < rect.top ||
      e.clientY > rect.bottom
    ) {
      setHoverInfo(null);
      return;
    }

    const u = (e.clientX - rect.left) / rect.width;
    const v = (e.clientY - rect.top) / rect.height;
    const px = Math.min(floatTex.width - 1, Math.max(0, Math.floor(u * floatTex.width)));
    const py = Math.min(floatTex.height - 1, Math.max(0, Math.floor(v * floatTex.height)));
    const norm = Math.min(1, Math.max(0, floatTex.data[py * floatTex.width + px] ?? 0));
    const zMm = params.base_z_mm + norm * params.relief_z_mm;
    const level16 = Math.round(norm * 65535);
    setHoverInfo({ px, py, norm, zMm, level16 });
  };

  const thumbSrc =
    pendingFilePreviewUrl ?? (project ? originalImageUrl(project.id) : null);

  const bitDepthLabel =
    bitDepth === '32bit_exr'
      ? '32-bit Float'
      : bitDepth === '8bit'
      ? '8-bit Legacy'
      : '16-bit (65,536 levels)';

  const formatTimestamp = (ts?: string) => {
    if (!ts) return '—';
    const d = new Date(ts);
    return Number.isNaN(d.getTime())
      ? ts
      : d.toLocaleString(undefined, {
          year: 'numeric',
          month: '2-digit',
          day: '2-digit',
          hour: '2-digit',
          minute: '2-digit',
        });
  };

  const updateParam = <K extends keyof ReliefParams>(
    key: K,
    val: ReliefParams[K],
    immediate = false
  ) => {
    onParamsChange({ ...params, [key]: val }, immediate);
  };

  return (
    <div className="flex flex-1 flex-col overflow-hidden bg-[#F5F7FA] text-slate-900 lg:flex-row">
      {/* LEFT SIDEBAR: 380px matching ref_01 & ref_02 */}
      <aside className="flex w-full shrink-0 flex-col border-b border-slate-200 bg-white lg:w-[380px] lg:border-r lg:border-b-0">
        <div className="flex-1 space-y-4 overflow-y-auto p-5">
          {/* Upload Dropzone Card (ref_01) */}
          <div>
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setIsDraggingFile(true);
              }}
              onDragLeave={() => setIsDraggingFile(false)}
              onDrop={(e) => {
                e.preventDefault();
                setIsDraggingFile(false);
                const f = e.dataTransfer.files?.[0];
                if (f) onSelectFile(f);
              }}
              onClick={() => fileInputRef.current?.click()}
              className={`group relative flex h-48 cursor-pointer items-center justify-center overflow-hidden rounded-2xl border transition-all ${
                isDraggingFile
                  ? 'border-blue-500 bg-blue-50/70'
                  : 'border-slate-200 bg-[#F3F5F9] hover:border-blue-400'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/png,image/jpeg,image/webp,image/tiff"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) onSelectFile(f);
                  e.target.value = '';
                }}
              />
              {thumbSrc ? (
                <div className="relative flex h-full w-full items-center justify-center p-2">
                  <div className="h-full overflow-hidden rounded-lg bg-black shadow-sm">
                    <img
                      src={thumbSrc}
                      alt="Uploaded line art"
                      className="h-full w-auto object-contain"
                    />
                  </div>
                  <span className="absolute right-2.5 bottom-2.5 flex items-center gap-1 rounded-lg bg-white/90 px-2 py-1 text-[11px] font-medium text-slate-700 opacity-0 shadow-sm backdrop-blur transition-opacity group-hover:opacity-100">
                    <Upload className="h-3 w-3 text-blue-600" />
                    Replace Art
                  </span>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-2 px-4 text-center">
                  <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-white text-blue-600 shadow-sm">
                    <Upload className="h-5 w-5" />
                  </div>
                  <p className="text-xs font-semibold text-slate-700">
                    Drop hand-drawn line art here
                  </p>
                  <p className="text-[11px] text-slate-400">
                    PNG, JPEG, TIFF • Stays 100% on your machine
                  </p>
                </div>
              )}
            </div>

            <div className="mt-2.5 flex items-center justify-between gap-2">
              <button
                type="button"
                disabled={isBusy}
                onClick={onLoadBenchmark}
                className="flex flex-1 items-center justify-center gap-1.5 rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs font-semibold text-slate-700 transition-colors hover:border-blue-300 hover:bg-blue-50 hover:text-blue-700 disabled:opacity-50"
              >
                <ImageIcon className="h-3.5 w-3.5 text-blue-600" />
                Load Example Drawing
              </button>
              {pendingFile && (
                <span className="max-w-[140px] truncate rounded-md bg-blue-50 px-2 py-1 text-[11px] font-medium text-blue-700">
                  {pendingFile.name}
                </span>
              )}
            </div>
          </div>

          {/* Image Improvement Accordion (ref_01) */}
          <AccordionSection
            title="Image Improvement"
            subtitle="Clean scan noise & lock outer background to Z = 0"
          >
            <div className="space-y-3">
              <div>
                <div className="text-xs font-semibold text-slate-800">
                  Background Removal
                </div>
                <label className="mt-1.5 flex cursor-pointer items-center gap-2 select-none">
                  <button
                    type="button"
                    role="checkbox"
                    aria-checked={bgRemoval}
                    onClick={() => onChangeBgRemoval(!bgRemoval)}
                    className={`flex h-4 w-4 items-center justify-center rounded border transition-colors ${
                      bgRemoval
                        ? 'border-blue-500 bg-blue-500 text-white'
                        : 'border-slate-300 bg-white'
                    }`}
                  >
                    {bgRemoval && <Check className="h-3 w-3 stroke-[3]" />}
                  </button>
                  <span
                    className={`text-xs font-medium ${
                      bgRemoval ? 'text-blue-600' : 'text-slate-600'
                    }`}
                  >
                    Background Removal
                  </span>
                </label>
              </div>

              <div>
                <div className="text-xs font-semibold text-slate-800">
                  HD Restoration
                </div>
                <label className="mt-1.5 flex cursor-pointer items-center gap-2 select-none">
                  <button
                    type="button"
                    role="checkbox"
                    aria-checked={hdRestore}
                    onClick={() => onChangeHdRestore(!hdRestore)}
                    className={`flex h-4 w-4 items-center justify-center rounded border transition-colors ${
                      hdRestore
                        ? 'border-blue-500 bg-blue-500 text-white'
                        : 'border-slate-300 bg-white'
                    }`}
                  >
                    {hdRestore && <Check className="h-3 w-3 stroke-[3]" />}
                  </button>
                  <span
                    className={`text-xs font-medium ${
                      hdRestore ? 'text-blue-600' : 'text-slate-600'
                    }`}
                  >
                    HD Restoration
                  </span>
                </label>
              </div>

              <div>
                <div className="flex items-center gap-1 text-xs font-semibold text-slate-800">
                  <span>Drawing Style</span>
                  <InfoTip text="Auto detects from the canvas border. White shapes on black: the artist's cut-out style. Black lines on white: classic ink on paper (the paper becomes background)." />
                </div>
                <select
                  value={polarity}
                  onChange={(e) => onChangePolarity(e.target.value as LinePolarity)}
                  className="mt-1.5 w-full rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-xs font-medium text-slate-700"
                >
                  <option value="auto">Auto-detect</option>
                  <option value="white_on_black">White shapes on black</option>
                  <option value="black_on_white">Black lines on white paper</option>
                </select>
              </div>

              <button
                type="button"
                disabled={isBusy}
                onClick={onApplyPreprocess}
                className="flex items-center gap-2 rounded-lg bg-[#3897FF] px-3.5 py-2 text-xs font-semibold text-white shadow-sm transition-colors hover:bg-blue-600 disabled:opacity-50"
              >
                <Wand2 className="h-3.5 w-3.5" />
                <span>Apply Pre-Processing</span>
                <span className="rounded bg-white/20 px-1.5 py-0.5 text-[10px] font-medium">
                  100% Local • Free
                </span>
              </button>
            </div>
          </AccordionSection>

          {/* Image Dimensions Selector (ref_02) */}
          <div className="border-t border-slate-200/80 pt-3.5">
            <div className="mb-2.5 flex items-center justify-between">
              <span className="text-sm font-semibold text-slate-800">
                Image Dimensions
              </span>
              <span className="text-[11px] font-mono text-slate-400">
                {targetRes}px max edge
              </span>
            </div>
            <div className="grid grid-cols-3 gap-2">
              {(
                [
                  { res: 2048 as const, title: 'BASIC', sub: '2K' },
                  { res: 4096 as const, title: 'PRO', sub: '4K' },
                  { res: 8192 as const, title: 'ULTRA', sub: '8K' },
                ] as const
              ).map((item) => {
                const active = targetRes === item.res;
                return (
                  <button
                    key={item.res}
                    type="button"
                    onClick={() => onChangeTargetRes(item.res)}
                    className={`rounded-xl border px-2 py-2.5 text-center transition-all ${
                      active
                        ? 'border-blue-300 bg-[#DCEBFF] text-slate-900 shadow-xs'
                        : 'border-slate-200 bg-white text-slate-600 hover:border-slate-300 hover:bg-slate-50'
                    }`}
                  >
                    <div className="text-xs font-bold tracking-wide">
                      {item.title}
                    </div>
                    <div className="text-[11px] font-medium text-slate-500">
                      {item.sub}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Detailed Parameters Accordion (ref_02) */}
          <AccordionSection title="Detailed Parameters" defaultOpen>
            <div className="grid grid-cols-2 gap-3">
              {/* Background Cleanup */}
              <div className="space-y-1.5">
                <div className="flex items-center gap-1 text-xs font-semibold text-slate-700">
                  <span>Background Cleanup</span>
                  <InfoTip text="Locks outer dark components & enclosed gaps strictly to Z = 0.0 mm." />
                </div>
                <div className="space-y-1.5">
                  {[
                    { label: 'On', value: true },
                    { label: 'Off', value: false },
                  ].map((opt) => (
                    <button
                      key={opt.label}
                      type="button"
                      onClick={() => onChangeBgRemoval(opt.value)}
                      className={`w-full rounded-lg border py-2 text-xs font-semibold transition-all ${
                        bgRemoval === opt.value
                          ? 'border-blue-300 bg-[#DCEBFF] text-slate-900'
                          : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
                      }`}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* AI Optimization */}
              <div className="space-y-1.5">
                <div className="flex items-center gap-1 text-xs font-semibold text-slate-700">
                  <span>AI Optimization</span>
                  <InfoTip text="Default Off preserves pure mathematical line-locked cushions with zero AI hallucinations. Guided Overlap subtly layers overlapping limbs." />
                </div>
                <div className="space-y-1.5">
                  {[
                    { label: 'Off', value: 'off' as const },
                    { label: 'Optimize (Guided)', value: 'guided_overlap' as const },
                  ].map((opt) => (
                    <button
                      key={opt.value}
                      type="button"
                      onClick={() => onChangeAiMode(opt.value)}
                      className={`w-full rounded-lg border px-2 py-2 text-xs font-semibold transition-all ${
                        aiMode === opt.value
                          ? 'border-blue-300 bg-[#DCEBFF] text-slate-900'
                          : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
                      }`}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Bit Depth */}
            <div className="mt-4 space-y-1.5">
              <div className="flex items-center gap-1 text-xs font-semibold text-slate-700">
                <span>Bit Depth</span>
                <InfoTip text="Internal computations always run in 32-bit float. Choose 16bit (65,536 levels) or 32bit EXR for zero stair-stepping." />
              </div>
              <div className="grid grid-cols-3 gap-1.5">
                {(
                  [
                    { label: '8bit', value: '8bit' as const },
                    { label: '16bit', value: '16bit' as const },
                    { label: '32bit EXR', value: '32bit_exr' as const },
                  ] as const
                ).map((opt) => (
                  <button
                    key={opt.value}
                    type="button"
                    onClick={() => handleSelectBitDepth(opt.value)}
                    className={`rounded-lg border py-2 text-xs font-semibold transition-all ${
                      bitDepth === opt.value
                        ? 'border-blue-300 bg-[#DCEBFF] text-slate-900'
                        : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
                    }`}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            </div>
          </AccordionSection>
        </div>

        {/* Sticky Primary Action Button */}
        <div className="border-t border-slate-200 bg-white p-4">
          <button
            type="button"
            disabled={isBusy}
            onClick={onGenerateRelief}
            className="flex w-full items-center justify-between rounded-xl bg-[#2E89FF] px-4 py-3 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-blue-600 disabled:opacity-60"
          >
            <span className="flex items-center gap-2">
              <Sparkles className="h-4 w-4" />
              {isBusy ? 'Generating Relief...' : 'Generate Relief'}
            </span>
            <span className="flex items-center gap-1 rounded-md bg-white/20 px-2 py-0.5 font-mono text-xs">
              <Zap className="h-3 w-3" />
              {lastComputeMs !== null ? `${Math.round(lastComputeMs)} ms` : 'Local Free'}
            </span>
          </button>
        </div>
      </aside>

      {/* RIGHT HUB: Depth Map Preview + 2x4 Action Grid + Metadata Footer (ref_03) */}
      <main className="flex flex-1 flex-col overflow-y-auto p-5 lg:p-7">
        <div className="mx-auto grid w-full max-w-6xl grid-cols-1 gap-7 xl:grid-cols-12">
          {/* Depth Map / Inline 3D Preview Column */}
          <div className="flex flex-col xl:col-span-6">
            <div className="mb-2.5 flex items-center justify-between gap-2">
              <div className="flex items-center gap-1.5 rounded-lg bg-white p-1 shadow-xs ring-1 ring-slate-200">
                <button
                  type="button"
                  onClick={() => setInline3D(false)}
                  className={`rounded-md px-2.5 py-1 text-xs font-semibold transition-colors ${
                    !inline3D
                      ? 'bg-blue-600 text-white'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  2D Depth Map
                </button>
                <button
                  type="button"
                  onClick={() => setInline3D(true)}
                  className={`rounded-md px-2.5 py-1 text-xs font-semibold transition-colors ${
                    inline3D
                      ? 'bg-blue-600 text-white'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  Inline 3D Preview
                </button>
              </div>

              {!inline3D && (
                <div className="flex items-center gap-1.5">
                  <button
                    type="button"
                    onClick={() => setFalseColor((v) => !v)}
                    className={`rounded-lg px-2.5 py-1 text-xs font-medium ring-1 transition-colors ${
                      falseColor
                        ? 'bg-indigo-50 text-indigo-700 ring-indigo-300'
                        : 'bg-white text-slate-600 ring-slate-200 hover:bg-slate-50'
                    }`}
                  >
                    {falseColor ? 'Elevation Color' : 'Grayscale'}
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setZoom(1);
                      setPan({ x: 0, y: 0 });
                    }}
                    title="Reset zoom & pan"
                    className="rounded-lg bg-white p-1.5 text-slate-600 ring-1 ring-slate-200 hover:bg-slate-50"
                  >
                    <RotateCcw className="h-3.5 w-3.5" />
                  </button>
                </div>
              )}
            </div>

            {/* Canvas Container */}
            <div
              ref={wheelContainerRef}
              onMouseDown={(e) => {
                if (inline3D || e.button !== 0) return;
                isPanningRef.current = true;
                lastMouseRef.current = { x: e.clientX, y: e.clientY };
              }}
              onMouseUp={() => {
                isPanningRef.current = false;
              }}
              onMouseLeave={() => {
                isPanningRef.current = false;
                setHoverInfo(null);
              }}
              onMouseMove={handleCanvasMouseMove}
              onDoubleClick={() => {
                if (!inline3D) {
                  setZoom(1);
                  setPan({ x: 0, y: 0 });
                }
              }}
              className="relative flex h-[480px] items-center justify-center overflow-hidden rounded-2xl border border-slate-200 bg-[#0d1016] shadow-sm sm:h-[540px]"
            >
              {inline3D ? (
                <React.Suspense
                  fallback={
                    <div className="flex flex-col items-center gap-2 text-slate-400">
                      <div className="h-6 w-6 animate-spin rounded-full border-2 border-slate-700 border-t-blue-500" />
                      <p className="text-xs">Loading 3D Preview...</p>
                    </div>
                  }
                >
                  <ReliefViewport3D
                    texture={floatTex}
                    params={params}
                    displayMode="clay"
                    materialId="matte_clay"
                    lightAzimuthDeg={315}
                    lightElevationDeg={28}
                    lightIntensity={1.1}
                    sliceEnabled={false}
                    sliceIntervalMm={1.0}
                    twoSided={false}
                    cameraPreset="perspective"
                    cameraTrigger={0}
                    activeTool="orbit"
                    brushMode="smooth"
                    brushRadiusPx={20}
                    brushStrength={0.4}
                    profilePointA={null}
                    profilePointB={null}
                    segments={512}
                  />
                </React.Suspense>
              ) : floatTex ? (
                <div
                  className="flex h-full w-full items-center justify-center p-4 select-none"
                  style={{
                    transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                    transformOrigin: 'center center',
                  }}
                >
                  <canvas
                    ref={canvasRef}
                    className="max-h-full max-w-full cursor-crosshair object-contain shadow-lg"
                  />
                </div>
              ) : textureError ? (
                <div className="flex max-w-sm flex-col items-center gap-3 px-6 text-center text-slate-300">
                  <Layers className="h-8 w-8 text-red-400" />
                  <p className="text-xs font-semibold">Could not load the Float32 depth buffer</p>
                  <p className="text-[11px] break-words text-slate-500">{textureError}</p>
                  {onRetryLoad && (
                    <button
                      type="button"
                      onClick={onRetryLoad}
                      className="rounded-lg bg-blue-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-blue-500"
                    >
                      Retry
                    </button>
                  )}
                </div>
              ) : (
                <div className="flex flex-col items-center gap-2 text-slate-400">
                  <Layers className="h-8 w-8 animate-pulse text-blue-400" />
                  <p className="text-xs">Loading Float32 depth buffer...</p>
                </div>
              )}

              {/* Zero-watermark top-left subtle badge */}
              <div className="pointer-events-none absolute top-3 left-3 flex items-center gap-1.5 rounded-lg bg-black/65 px-2.5 py-1 text-[11px] font-medium text-emerald-300 backdrop-blur">
                <ShieldCheck className="h-3.5 w-3.5" />
                Zero Watermark • Float32
              </div>

              {/* Hover Cursor Z-Height Readout */}
              {!inline3D && (
                <div className="pointer-events-none absolute right-3 bottom-3 left-3 flex flex-wrap items-center justify-between gap-2 rounded-xl bg-black/75 px-3 py-1.5 font-mono text-[11px] text-slate-200 backdrop-blur">
                  {hoverInfo ? (
                    <>
                      <span>
                        XY: {hoverInfo.px}, {hoverInfo.py}
                      </span>
                      <span className="text-cyan-300">
                        ≈ (preview) Z: {(hoverInfo.norm * 100).toFixed(1)}% ({hoverInfo.zMm.toFixed(2)} mm)
                      </span>
                      <span className="text-amber-300">
                        16-bit: {hoverInfo.level16.toLocaleString()} / 65,535
                      </span>
                    </>
                  ) : (
                    <>
                      <span>Scroll to zoom ({zoom.toFixed(1)}×) • Drag to pan</span>
                      <span className="text-slate-400">
                        Hover image for approximate mm &amp; 16-bit level
                      </span>
                    </>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Right Action Hub & Metadata Column (ref_03) */}
          <div className="flex flex-col justify-between xl:col-span-6">
            <div>
              {/* 2x4 Action Grid */}
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                {/* 1. Download Split Button */}
                <div className="relative">
                  <div className="flex h-14 overflow-hidden rounded-xl bg-[#2E89FF] text-white shadow-sm">
                    <button
                      type="button"
                      disabled={!project || isBusy}
                      onClick={() => onExport(selectedDownloadFmt)}
                      className="flex flex-1 items-center justify-center gap-2 px-3 text-sm font-semibold transition-colors hover:bg-blue-600 disabled:opacity-50"
                    >
                      <Download className="h-4 w-4" />
                      <span>Download</span>
                    </button>
                    <button
                      type="button"
                      disabled={!project || isBusy}
                      onClick={() => setDownloadMenuOpen((v) => !v)}
                      className="flex items-center gap-1 border-l border-white/25 bg-blue-600/40 px-3 text-xs font-semibold transition-colors hover:bg-blue-600/80 disabled:opacity-50"
                    >
                      <span>{EXPORT_FORMAT_META[selectedDownloadFmt].short}</span>
                      <ChevronDown className="h-3.5 w-3.5" />
                    </button>
                  </div>

                  {downloadMenuOpen && (
                    <div className="absolute top-full left-0 z-30 mt-1.5 w-full min-w-[260px] rounded-xl border border-slate-200 bg-white p-1.5 shadow-xl">
                      {(Object.keys(EXPORT_FORMAT_META) as ExportFormat[]).map(
                        (fmt) => {
                          const meta = EXPORT_FORMAT_META[fmt];
                          return (
                            <button
                              key={fmt}
                              type="button"
                              onClick={() => {
                                setSelectedDownloadFmt(fmt);
                                setDownloadMenuOpen(false);
                                onExport(fmt);
                              }}
                              className={`flex w-full flex-col rounded-lg px-3 py-2 text-left transition-colors ${
                                selectedDownloadFmt === fmt
                                  ? 'bg-blue-50 text-blue-700'
                                  : 'hover:bg-slate-50 text-slate-800'
                              }`}
                            >
                              <span className="text-xs font-semibold">
                                {meta.label}
                              </span>
                              <span className="text-[11px] text-slate-500">
                                {meta.desc}
                              </span>
                            </button>
                          );
                        }
                      )}
                    </div>
                  )}
                  <div className="mt-2">
                    <MeshExportSettings
                      value={meshSettings}
                      onChange={onMeshSettingsChange}
                      params={params}
                      project={project}
                    />
                  </div>
                </div>

                {/* 2. Depth & 3D Editor */}
                <button
                  type="button"
                  disabled={!project}
                  onClick={onOpenEditor3D}
                  className="flex h-14 items-center gap-3 rounded-xl border border-slate-200 bg-white px-4 text-left shadow-xs transition-all hover:border-blue-400 hover:shadow-sm disabled:opacity-50"
                >
                  <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-slate-100 text-slate-700">
                    <Box className="h-4 w-4 text-blue-600" />
                  </div>
                  <div>
                    <div className="text-sm font-semibold text-slate-800">
                      Depth &amp; 3D Editor
                    </div>
                    <span className="inline-block rounded bg-blue-600 px-1.5 py-0.2 text-[9px] font-bold tracking-wide text-white uppercase">
                      CAD 3D
                    </span>
                  </div>
                </button>

                {/* 3. 3D Preview Toggle */}
                <button
                  type="button"
                  disabled={!project}
                  onClick={() => setInline3D((v) => !v)}
                  className={`flex h-14 items-center gap-3 rounded-xl border px-4 text-left shadow-xs transition-all disabled:opacity-50 ${
                    inline3D
                      ? 'border-blue-400 bg-blue-50 text-blue-800'
                      : 'border-slate-200 bg-white text-slate-800 hover:border-blue-300'
                  }`}
                >
                  <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-slate-100 text-slate-700">
                    <Eye className="h-4 w-4 text-blue-600" />
                  </div>
                  <div className="text-sm font-semibold">
                    {inline3D ? '2D Depth View' : '3D Preview'}
                  </div>
                </button>

                {/* 4. Zero-Watermark Guarantee Badge */}
                <div className="relative">
                  <button
                    type="button"
                    onClick={() => setWatermarkInfoOpen((v) => !v)}
                    className="flex h-14 w-full items-center gap-3 rounded-xl border border-emerald-200 bg-emerald-50/70 px-4 text-left shadow-xs transition-all hover:border-emerald-300 hover:bg-emerald-50"
                  >
                    <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-100 text-emerald-700">
                      <ShieldCheck className="h-4 w-4" />
                    </div>
                    <div>
                      <div className="text-xs font-bold text-emerald-900">
                        Zero-Watermark Guarantee
                      </div>
                      <div className="text-[11px] text-emerald-700">
                        {uniqueLevels !== null
                          ? `${uniqueLevels.toLocaleString()} levels • 0% hidden grid`
                          : 'Pure Float32 • Free Forever'}
                      </div>
                    </div>
                  </button>

                  {watermarkInfoOpen && (
                    <div className="absolute top-full right-0 z-30 mt-1.5 w-72 rounded-xl border border-emerald-200 bg-white p-3.5 text-xs shadow-xl">
                      <div className="flex items-center justify-between font-semibold text-emerald-800">
                        <span>Purity &amp; Bit-Depth Verification</span>
                        <button
                          type="button"
                          onClick={() => setWatermarkInfoOpen(false)}
                          className="text-slate-400 hover:text-slate-600"
                        >
                          ✕
                        </button>
                      </div>
                      <div className="mt-2 space-y-1 font-mono text-[11px] text-slate-600">
                        <div>Min Z: {textureStats.minPct.toFixed(2)}%</div>
                        <div>Max Z: {textureStats.maxPct.toFixed(2)}%</div>
                        <div>Flat Background: {textureStats.bgZeroPct.toFixed(1)}%</div>
                        <div>
                          Unique 16-bit levels:{' '}
                          {uniqueLevels !== null ? uniqueLevels.toLocaleString() : '—'}
                        </div>
                      </div>
                      <p className="mt-2 text-[11px] text-slate-500">
                        Zero baked watermarks, zero 8-bit downgrades, zero credits.
                      </p>
                    </div>
                  )}
                </div>

                {/* 5. Invert Depth-Map */}
                <button
                  type="button"
                  disabled={!project || isBusy}
                  onClick={() =>
                    updateParam('invert_depth', !params.invert_depth, true)
                  }
                  className={`flex h-14 items-center gap-3 rounded-xl border px-4 text-left shadow-xs transition-all disabled:opacity-50 ${
                    params.invert_depth
                      ? 'border-blue-400 bg-blue-50 text-blue-800'
                      : 'border-slate-200 bg-white text-slate-800 hover:border-blue-300'
                  }`}
                >
                  <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-slate-100 text-slate-700">
                    <FlipVertical className="h-4 w-4 text-slate-600" />
                  </div>
                  <div>
                    <div className="text-sm font-semibold">Invert Depth-Map</div>
                    {params.invert_depth && (
                      <div className="text-[10px] font-semibold text-blue-600">
                        Concave Mold Active
                      </div>
                    )}
                  </div>
                </button>

                {/* 6. Image Upscale */}
                <div className="relative">
                  <button
                    type="button"
                    disabled={!project || isBusy}
                    onClick={() => setUpscaleMenuOpen((v) => !v)}
                    className="flex h-14 w-full items-center gap-3 rounded-xl border border-slate-200 bg-white px-4 text-left shadow-xs transition-all hover:border-blue-300 disabled:opacity-50"
                  >
                    <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-slate-100 text-slate-700">
                      <Maximize2 className="h-4 w-4 text-slate-600" />
                    </div>
                    <div>
                      <div className="text-sm font-semibold text-slate-800">
                        Image Upscale
                      </div>
                      <div className="text-[11px] text-slate-400">
                        Super-sample 4K / 8K
                      </div>
                    </div>
                  </button>

                  {upscaleMenuOpen && (
                    <div className="absolute top-full right-0 z-30 mt-1.5 w-56 rounded-xl border border-slate-200 bg-white p-1.5 shadow-xl">
                      <button
                        type="button"
                        onClick={() => {
                          setUpscaleMenuOpen(false);
                          onUpscale(4096);
                        }}
                        className="flex w-full items-center justify-between rounded-lg px-3 py-2 text-xs font-semibold text-slate-700 hover:bg-blue-50 hover:text-blue-700"
                      >
                        <span>Upscale to PRO 4K</span>
                        <span className="font-mono text-[10px] text-slate-400">4096px</span>
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          setUpscaleMenuOpen(false);
                          onUpscale(8192);
                        }}
                        className="flex w-full items-center justify-between rounded-lg px-3 py-2 text-xs font-semibold text-slate-700 hover:bg-blue-50 hover:text-blue-700"
                      >
                        <span>Upscale to ULTRA 8K</span>
                        <span className="font-mono text-[10px] text-slate-400">8192px</span>
                      </button>
                    </div>
                  )}
                </div>

                {/* 7. Noise and Brightness Popover */}
                <div className="relative">
                  <button
                    type="button"
                    disabled={!project}
                    onClick={() => setNoisePopoverOpen((v) => !v)}
                    className={`flex h-14 w-full items-center gap-3 rounded-xl border px-4 text-left shadow-xs transition-all disabled:opacity-50 ${
                      noisePopoverOpen
                        ? 'border-blue-400 bg-blue-50 text-blue-800'
                        : 'border-slate-200 bg-white text-slate-800 hover:border-blue-300'
                    }`}
                  >
                    <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-slate-100 text-slate-700">
                      <SlidersHorizontal className="h-4 w-4 text-blue-600" />
                    </div>
                    <div>
                      <div className="text-sm font-semibold">
                        Noise and Brightness
                      </div>
                      <div className="text-[11px] text-slate-400">
                        Real-time &lt;80ms tuning
                      </div>
                    </div>
                  </button>

                  {noisePopoverOpen && (
                    <div className="absolute top-full left-0 z-30 mt-2 w-80 rounded-2xl border border-slate-200 bg-white p-4 shadow-2xl">
                      <div className="mb-3 flex items-center justify-between">
                        <div className="flex items-center gap-1.5 text-xs font-bold text-slate-800">
                          <Contrast className="h-3.5 w-3.5 text-blue-600" />
                          Relief Curves &amp; Smoothness
                        </div>
                        <button
                          type="button"
                          onClick={() => setNoisePopoverOpen(false)}
                          className="rounded px-1.5 py-0.5 text-xs text-slate-400 hover:bg-slate-100 hover:text-slate-700"
                        >
                          Done
                        </button>
                      </div>

                      <div className="mb-3">
                        <StyleToggle
                          params={params}
                          onChange={onParamsChange}
                          isBusy={isBusy}
                        />
                      </div>

                      {params.style === 'sculpt' ? (
                        <SculptControls
                          params={params}
                          onChange={onParamsChange}
                          isBusy={isBusy}
                          edgeClamped={edgeClamped}
                          aspect={project ? project.height / Math.max(1, project.width) : 1}
                          latestVersion={project?.sculpt_latest_version}
                        />
                      ) : (
                        <>
                      <div className="space-y-3">
                        <StudioSlider
                          label="Global Cushion"
                          value={params.w_global}
                          min={0}
                          max={1.5}
                          step={0.02}
                          onChange={(v) => updateParam('w_global', v)}
                        />
                        <StudioSlider
                          label="Limb Puffiness"
                          value={params.w_local}
                          min={0}
                          max={1.5}
                          step={0.02}
                          onChange={(v) => updateParam('w_local', v)}
                        />
                        <StudioSlider
                          label="Line Crease Depth"
                          value={params.w_groove}
                          min={0}
                          max={1.5}
                          step={0.02}
                          onChange={(v) => updateParam('w_groove', v)}
                        />
                        <StudioSlider
                          label="Line Crease Width"
                          value={params.groove_sigma}
                          min={0.5}
                          max={10}
                          step={0.1}
                          formatValue={(v) => `${v.toFixed(1)} px`}
                          onChange={(v) => updateParam('groove_sigma', v)}
                        />
                        <StudioSlider
                          label="Gamma"
                          value={params.gamma}
                          min={0.35}
                          max={2.2}
                          step={0.02}
                          onChange={(v) => updateParam('gamma', v)}
                        />
                        <StudioSlider
                          label="Brightness"
                          value={params.brightness}
                          min={-0.35}
                          max={0.35}
                          step={0.01}
                          formatValue={(v) => `${v >= 0 ? '+' : ''}${v.toFixed(2)}`}
                          onChange={(v) => updateParam('brightness', v)}
                        />
                        <StudioSlider
                          label="Contrast"
                          value={params.contrast}
                          min={0.5}
                          max={1.8}
                          step={0.02}
                          onChange={(v) => updateParam('contrast', v)}
                        />
                      </div>

                      <button
                        type="button"
                        onClick={() =>
                          onParamsChange(
                            {
                              ...params,
                              w_global: 0.45,
                              w_local: 0.65,
                              w_groove: 0.35,
                              groove_sigma: 2.5,
                              gamma: 0.85,
                              brightness: 0.0,
                              contrast: 1.0,
                            },
                            true
                          )
                        }
                        className="mt-3 w-full rounded-lg border border-slate-200 py-1.5 text-xs font-semibold text-slate-600 hover:bg-slate-50"
                      >
                        Reset Adjustments
                      </button>
                        </>
                      )}
                    </div>
                  )}
                </div>

                {/* 8. Local Vault Trigger */}
                <button
                  type="button"
                  onClick={onOpenVault}
                  className="flex h-14 items-center gap-3 rounded-xl border border-slate-200 bg-white px-4 text-left shadow-xs transition-all hover:border-blue-300"
                >
                  <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-slate-100 text-slate-700">
                    <Archive className="h-4 w-4 text-slate-600" />
                  </div>
                  <div>
                    <div className="text-sm font-semibold text-slate-800">
                      Local Vault
                    </div>
                    <div className="text-[11px] text-slate-400">
                      Saved SQLite Archive
                    </div>
                  </div>
                </button>
              </div>
            </div>

            {/* Metadata Footer matching ref_03 */}
            <div className="mt-8 border-t border-slate-200 pt-5">
              <div className="grid grid-cols-2 gap-x-6 gap-y-4">
                <div>
                  <div className="text-[11px] font-semibold tracking-wider text-slate-400 uppercase">
                    BIT DEPTH
                  </div>
                  <div className="mt-1 text-sm font-semibold text-slate-800">
                    {bitDepthLabel}
                  </div>
                </div>
                <div>
                  <div className="text-[11px] font-semibold tracking-wider text-slate-400 uppercase">
                    RESOLUTION
                  </div>
                  <div className="mt-1 font-mono text-sm font-semibold text-slate-800">
                    {project ? `${project.width} × ${project.height}` : '—'}
                  </div>
                </div>
                <div>
                  <div className="text-[11px] font-semibold tracking-wider text-slate-400 uppercase">
                    STYLE
                  </div>
                  <div className="mt-1 text-sm font-semibold text-slate-800">
                    Line-Locked Bas-Relief
                  </div>
                </div>
                <div>
                  <div className="text-[11px] font-semibold tracking-wider text-slate-400 uppercase">
                    CREATED
                  </div>
                  <div className="mt-1 text-sm font-semibold text-slate-800">
                    {formatTimestamp(project?.created_at)}
                  </div>
                </div>
              </div>

              <div className="mt-5 border-t border-slate-200/80 pt-4">
                <div className="text-[11px] font-semibold tracking-wider text-slate-400 uppercase">
                  ORIGINAL IMAGE
                </div>
                {project ? (
                  <div className="mt-2 flex items-center gap-3">
                    <div className="h-16 w-16 overflow-hidden rounded-lg border border-slate-200 bg-black">
                      <img
                        src={originalImageUrl(project.id)}
                        alt={project.original_filename}
                        className="h-full w-full object-contain"
                      />
                    </div>
                    <div>
                      <div className="text-xs font-semibold text-slate-800">
                        {project.original_filename}
                      </div>
                      <div className="text-[11px] text-emerald-600">
                        Verified • Stored locally in SQLite vault
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="mt-2 text-xs text-slate-400">
                    No project loaded yet
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};
