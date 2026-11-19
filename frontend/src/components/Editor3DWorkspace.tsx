import React, { useEffect, useMemo, useState } from 'react';
import {
  Activity,
  Bookmark,
  Box,
  Check,
  Combine,
  Download,
  Eye,
  FlipHorizontal2,
  Frame as FrameIcon,
  Layers,
  ListOrdered,
  MousePointer2,
  Lock,
  Paintbrush,
  Plus,
  RotateCcw,
  ShieldCheck,
  Sliders,
  Trash2,
  Triangle,
  Type,
  Undo2,
  X,
} from 'lucide-react';
import {
  EXPORT_FORMAT_META,
  FRAME_BBOX_PRESETS,
  MATERIAL_SPHERES,
  type BrushMode,
  type MeshSettings,
  type CameraViewPreset,
  type DisplayMode,
  type EditorTab,
  type ExportFormat,
  type Float32TexturePayload,
  type FramePreset,
  type MaterialId,
  type PresetRecord,
  type ProfilePoint,
  type ProjectRecord,
  type ReliefParams,
} from '../types/monolith';
import { MeshExportSettings } from './MeshExportSettings';
import { SculptControls, StyleToggle } from './SculptControls';
import { CadSliderInput, InfoTip } from './UiControls';
import { ReliefViewport3D } from './ReliefViewport3D';
import { ProfileChartOverlay } from './ProfileChartOverlay';
import type { PartOrderAction } from '../api/client';
import type { PartsMap } from '../lib/partsMap';

interface Editor3DWorkspaceProps {
  project: ProjectRecord | null;
  params: ReliefParams;
  floatTex: Float32TexturePayload | null;
  presets: PresetRecord[];
  isBusy: boolean;
  isSavingBrush: boolean;
  onParamsChange: (next: ReliefParams, immediate?: boolean) => void;
  onBrushStrokeStart: (snapshot: Float32Array) => void;
  onBrushDataChange: (updated: Float32Array) => void;
  onBrushStrokeEnd: (finalData: Float32Array) => void;
  onUndoBrush: () => void;
  onResetBrush: () => void;
  canUndoBrush: boolean;
  onSavePreset: (name: string) => Promise<void>;
  onApplyPreset: (params: ReliefParams) => void;
  onDeletePreset: (id: string) => Promise<void>;
  onExport: (format: ExportFormat) => void;
  meshSettings: MeshSettings;
  onMeshSettingsChange: (v: MeshSettings) => void;
  /** Sculpt part-label map (null until loaded). */
  partsMap: PartsMap | null;
  /** Ask App to (re)load the part map; must be a stable callback. */
  onLoadParts: () => void;
  /** Layers action; uv is the clicked point in viewer coordinates (v up), null for Reset all. */
  onPartOrder: (action: PartOrderAction, uv: { u: number; v: number } | null) => void;
  /** Edge roundness was raised by the server to keep the draft angle. */
  edgeClamped: boolean;
}

type HudPopover =
  | null
  | 'adjust'
  | 'view'
  | 'slices'
  | 'brush'
  | 'layers'
  | 'presets'
  | 'export';

export const Editor3DWorkspace: React.FC<Editor3DWorkspaceProps> = ({
  project,
  params,
  floatTex,
  presets,
  isBusy,
  isSavingBrush,
  onParamsChange,
  onBrushStrokeStart,
  onBrushDataChange,
  onBrushStrokeEnd,
  onUndoBrush,
  onResetBrush,
  canUndoBrush,
  onSavePreset,
  onApplyPreset,
  onDeletePreset,
  onExport,
  meshSettings,
  onMeshSettingsChange,
  partsMap,
  onLoadParts,
  onPartOrder,
  edgeClamped,
}) => {
  const [activeTab, setActiveTab] = useState<EditorTab>('frame');
  const [panelOpen, setPanelOpen] = useState(true);

  // Viewport inspection state
  const [displayMode, setDisplayMode] = useState<DisplayMode>('solid');
  const [materialId, setMaterialId] = useState<MaterialId>('matte_clay');
  const [lightAzimuthDeg, setLightAzimuthDeg] = useState(315);
  const [lightElevationDeg, setLightElevationDeg] = useState(28);
  const [lightIntensity, setLightIntensity] = useState(1.1);
  const [backdropDark, setBackdropDark] = useState(false);

  const [cameraPreset, setCameraPreset] =
    useState<CameraViewPreset>('perspective');
  const [cameraTrigger, setCameraTrigger] = useState(0);

  // Tools & Overlays
  const [activeTool, setActiveTool] = useState<'orbit' | 'profile' | 'brush' | 'layers'>(
    'orbit'
  );
  const [sliceEnabled, setSliceEnabled] = useState(false);
  const [sliceIntervalMm, setSliceIntervalMm] = useState(1.0);
  const [twoSided, setTwoSided] = useState(false);
  const [showFrameGuide, setShowFrameGuide] = useState(true);

  const [profileOpen, setProfileOpen] = useState(false);
  const [profileA, setProfileA] = useState<ProfilePoint | null>(null);
  const [profileB, setProfileB] = useState<ProfilePoint | null>(null);

  const [brushMode, setBrushMode] = useState<BrushMode>('smooth');
  const [brushRadiusPx, setBrushRadiusPx] = useState(24);
  const [brushStrength, setBrushStrength] = useState(0.45);

  const [showLayers, setShowLayers] = useState(false);
  const [selectedPartId, setSelectedPartId] = useState<number | null>(null);
  const [selectedUv, setSelectedUv] = useState<{ u: number; v: number } | null>(null);

  const [openPopover, setOpenPopover] = useState<HudPopover>(null);
  const [newPresetName, setNewPresetName] = useState('');

  // Optional non-destructive provenance note (Text tab)
  const [editionStampEnabled, setEditionStampEnabled] = useState(false);
  const [artistName, setArtistName] = useState('Studio Master');
  const [editionNote, setEditionNote] = useState('Edition 1/1 • Zero Watermark');

  const aspect = floatTex
    ? floatTex.height / Math.max(1, floatTex.width)
    : project
    ? project.height / Math.max(1, project.width)
    : 1.22;
  const physicalHeightMm = params.physical_width_mm * aspect;

  const isSculpt = params.style === 'sculpt';
  const layersWanted = activeTool === 'layers' && isSculpt;
  const selectedPart = partsMap?.parts.find((p) => p.id === selectedPartId) ?? null;

  // The Layers tool only exists for Sculpt projects.
  useEffect(() => {
    if (!isSculpt && activeTool === 'layers') {
      setActiveTool('orbit');
      setOpenPopover((cur) => (cur === 'layers' ? null : cur));
    }
  }, [isSculpt, activeTool]);

  // Load the part map when the tool is used; App clears it whenever the parts change.
  useEffect(() => {
    if (layersWanted && !isBusy && !partsMap && project) onLoadParts();
  }, [layersWanted, isBusy, partsMap, project, onLoadParts]);

  // A new project / part map invalidates the selection.
  useEffect(() => {
    setSelectedPartId(null);
    setSelectedUv(null);
  }, [project?.id, partsMap]);

  const updateParam = <K extends keyof ReliefParams>(
    key: K,
    val: ReliefParams[K],
    immediate = false
  ) => {
    onParamsChange({ ...params, [key]: val }, immediate);
  };

  const updateFrame = <K extends keyof ReliefParams['frame']>(
    key: K,
    val: ReliefParams['frame'][K],
    immediate = false
  ) => {
    onParamsChange(
      {
        ...params,
        frame: { ...params.frame, [key]: val },
      },
      immediate
    );
  };

  const setBboxIndex = (idx: number, val: number) => {
    const next: [number, number, number, number] = [...params.frame.bbox];
    next[idx] = Math.min(1, Math.max(0, val));
    if (idx === 0 && next[0] > next[2] - 0.05) next[0] = next[2] - 0.05;
    if (idx === 2 && next[2] < next[0] + 0.05) next[2] = next[0] + 0.05;
    if (idx === 1 && next[1] > next[3] - 0.05) next[1] = next[3] - 0.05;
    if (idx === 3 && next[3] < next[1] + 0.05) next[3] = next[1] + 0.05;
    updateFrame('bbox', next);
  };

  const togglePopover = (pop: HudPopover) => {
    setOpenPopover((cur) => (cur === pop ? null : pop));
  };

  const tabMeta: {
    id: EditorTab;
    label: string;
    short: string;
    icon: React.ReactNode;
  }[] = [
    {
      id: 'relief',
      label: 'Relief Settings',
      short: 'Relief',
      icon: <Layers className="h-4 w-4" />,
    },
    {
      id: 'draft_angle',
      label: 'Draft Angle',
      short: 'Draft',
      icon: <Triangle className="h-4 w-4" />,
    },
    {
      id: 'two_sided',
      label: '2-Sided Relief',
      short: '2-Sided',
      icon: <FlipHorizontal2 className="h-4 w-4" />,
    },
    {
      id: 'composite',
      label: 'Composite Layers',
      short: 'Composite',
      icon: <Combine className="h-4 w-4" />,
    },
    {
      id: 'frame',
      label: 'Frame Settings',
      short: 'Frame',
      icon: <FrameIcon className="h-4 w-4" />,
    },
    {
      id: 'text',
      label: 'Zero-Watermark & Notes',
      short: 'Text',
      icon: <Type className="h-4 w-4" />,
    },
  ];

  const activeTabTitle =
    tabMeta.find((t) => t.id === activeTab)?.label ?? 'Inspector';

  const framePresetOptions: { value: FramePreset; label: string }[] = [
    { value: 'gallery_classic', label: 'Gallery Classic' },
    { value: 'modern_flat', label: 'Modern Flat' },
    { value: 'stepped_bevel', label: 'Stepped Bevel' },
    { value: 'circular_medallion', label: 'Circular Medallion' },
  ];

  const displayButtons: {
    id: DisplayMode;
    title: string;
    swatch: React.ReactNode;
  }[] = [
    {
      id: 'solid',
      title: 'Solid Lit',
      swatch: <Box className="h-3.5 w-3.5 text-slate-100" />,
    },
    {
      id: 'clay',
      title: 'Matte Clay',
      swatch: (
        <span
          className="block h-5 w-5 rounded-full"
          style={{
            background:
              'radial-gradient(circle at 35% 30%, #ffffff 0%, #c8cdd6 55%, #69707d 100%)',
          }}
        />
      ),
    },
    {
      id: 'curvature',
      title: 'Curvature / Cavity Map',
      swatch: (
        <span
          className="block h-5 w-5 rounded-full"
          style={{
            background:
              'radial-gradient(circle at 35% 30%, #fce7bd 0%, #98b1c4 52%, #2b4564 100%)',
          }}
        />
      ),
    },
    {
      id: 'normal',
      title: 'Surface Normal Map (Rainbow)',
      swatch: (
        <span
          className="block h-5 w-5 rounded-full"
          style={{
            background:
              'conic-gradient(from 45deg, #ff5e95, #5ee7ff, #7dff8b, #b678ff, #ff5e95)',
          }}
        />
      ),
    },
  ];

  const isLowerRef04 = useMemo(() => {
    const ref = FRAME_BBOX_PRESETS.lower_ref_04.bbox;
    return params.frame.bbox.every((v, i) => Math.abs(v - ref[i]) < 0.02);
  }, [params.frame.bbox]);

  const isFullPiece = useMemo(() => {
    const full = FRAME_BBOX_PRESETS.full_piece.bbox;
    return params.frame.bbox.every((v, i) => Math.abs(v - full[i]) < 0.02);
  }, [params.frame.bbox]);

  return (
    <div className="relative flex flex-1 overflow-hidden bg-[#181A20] text-slate-100">
      {/* MAIN 3D VIEWPORT AREA */}
      <div className="relative flex-1 overflow-hidden">
        <ReliefViewport3D
          texture={floatTex}
          params={params}
          displayMode={displayMode}
          materialId={materialId}
          lightAzimuthDeg={lightAzimuthDeg}
          lightElevationDeg={lightElevationDeg}
          lightIntensity={lightIntensity}
          sliceEnabled={sliceEnabled}
          sliceIntervalMm={sliceIntervalMm}
          twoSided={twoSided}
          showFrameGuide={showFrameGuide && activeTab === 'frame'}
          cameraPreset={cameraPreset}
          cameraTrigger={cameraTrigger}
          backdropDark={backdropDark}
          activeTool={activeTool}
          brushMode={brushMode}
          brushRadiusPx={brushRadiusPx}
          brushStrength={brushStrength}
          brushLocked={isBusy}
          partsMap={layersWanted ? partsMap : null}
          showLayers={showLayers}
          selectedPartId={selectedPartId}
          onPartPick={(id, uv) => {
            setSelectedPartId(id);
            setSelectedUv(id === null ? null : uv);
          }}
          profilePointA={profileA}
          profilePointB={profileB}
          onProfileChange={(a, b) => {
            setProfileA(a);
            setProfileB(b);
          }}
          onBrushStrokeStart={onBrushStrokeStart}
          onBrushDataChange={onBrushDataChange}
          onBrushStrokeEnd={onBrushStrokeEnd}
          segments={1024}
        />

        {/* Top-left CAD Status Overlay */}
        <div className="pointer-events-none absolute top-3 left-3 flex flex-wrap items-center gap-2">
          <div className="pointer-events-auto flex items-center gap-2 rounded-xl border border-[#2C3240] bg-[#181A20]/90 px-3 py-1.5 text-[11px] text-slate-300 shadow-lg backdrop-blur">
            <span className="font-mono text-slate-100">
              {params.physical_width_mm.toFixed(0)} × {physicalHeightMm.toFixed(0)} mm
            </span>
            <span className="text-slate-600">•</span>
            <span className="font-mono text-cyan-300">
              Z: {params.relief_z_mm.toFixed(1)} mm
            </span>
            <span className="text-slate-600">•</span>
            <span className="font-mono text-amber-300">
              {params.base_z_mm <= 0.01
                ? 'Base: Cutout (0 mm)'
                : `Base: ${params.base_z_mm.toFixed(1)} mm`}
            </span>
            {params.draft_angle_deg > 0 && (
              <>
                <span className="text-slate-600">•</span>
                <span className="font-mono text-emerald-300">
                  Draft: {params.draft_angle_deg.toFixed(1)}°
                </span>
              </>
            )}
          </div>

          {activeTool !== 'orbit' && (
            <div className="pointer-events-auto flex items-center gap-2 rounded-xl border border-blue-500/50 bg-blue-600/20 px-3 py-1.5 text-xs font-semibold text-blue-200 shadow-lg backdrop-blur">
              <span>
                {activeTool === 'brush'
                  ? `Local Edit (${brushMode})`
                  : activeTool === 'layers'
                  ? 'Layers Active'
                  : 'Profile Slicer Active'}
              </span>
              <button
                type="button"
                onClick={() => setActiveTool('orbit')}
                className="flex items-center gap-1 rounded-md bg-blue-600 px-2 py-0.5 text-[10px] font-bold text-white hover:bg-blue-500"
              >
                <MousePointer2 className="h-3 w-3" />
                Back to Orbit
              </button>
            </div>
          )}

          {isSavingBrush && (
            <div className="rounded-xl border border-amber-500/40 bg-amber-500/15 px-3 py-1.5 text-[11px] font-medium text-amber-200 backdrop-blur">
              Syncing brush buffer to vault...
            </div>
          )}
        </div>

        {/* Optional Artist Provenance Watermark-Free Stamp */}
        {editionStampEnabled && (
          <div className="pointer-events-none absolute top-3 right-3 rounded-xl border border-[#2C3240] bg-[#181A20]/85 px-3 py-1.5 text-right text-[11px] text-slate-300 backdrop-blur">
            <div className="font-semibold text-white">{artistName}</div>
            <div className="text-[10px] text-slate-400">{editionNote}</div>
          </div>
        )}

        {/* BOTTOM FLOATING HUD TOOLBARS (ref_04) */}
        <div className="pointer-events-none absolute right-0 bottom-4 left-0 z-20 flex flex-col items-start gap-2.5 px-4 sm:items-center">
          {/* Active Profile Chart Overlay */}
          {profileOpen && (
            <ProfileChartOverlay
              texture={floatTex}
              pointA={profileA}
              pointB={profileB}
              physicalWidthMm={params.physical_width_mm}
              reliefZMm={params.relief_z_mm}
              baseZMm={params.base_z_mm}
              onClear={() => {
                setProfileA(null);
                setProfileB(null);
              }}
              onClose={() => {
                setProfileOpen(false);
                if (activeTool === 'profile') setActiveTool('orbit');
              }}
            />
          )}

          {/* Contextual Popovers anchored neatly above bottom HUD */}
          {openPopover === 'adjust' && (
            <div className="pointer-events-auto w-72 rounded-2xl border border-[#2E3444] bg-[#191C24]/95 p-4 shadow-2xl backdrop-blur-md">
              <div className="mb-3 flex items-center justify-between">
                <span className="text-xs font-semibold text-white">
                  Studio Grazing Light &amp; Backdrop
                </span>
                <button
                  type="button"
                  onClick={() => setOpenPopover(null)}
                  className="text-slate-400 hover:text-white"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </div>
              <div className="space-y-3">
                <CadSliderInput
                  label="Light Azimuth"
                  value={lightAzimuthDeg}
                  min={0}
                  max={360}
                  step={1}
                  decimals={0}
                  unit="°"
                  onChange={setLightAzimuthDeg}
                />
                <CadSliderInput
                  label="Light Elevation"
                  value={lightElevationDeg}
                  min={5}
                  max={85}
                  step={1}
                  decimals={0}
                  unit="°"
                  tooltip="Lower angles rake across the surface to reveal subtle ripples."
                  onChange={setLightElevationDeg}
                />
                <CadSliderInput
                  label="Intensity"
                  value={lightIntensity}
                  min={0.4}
                  max={2.0}
                  step={0.05}
                  onChange={setLightIntensity}
                />
                <div className="flex items-center justify-between pt-1">
                  <span className="text-xs font-medium text-slate-300">
                    Viewport Backdrop
                  </span>
                  <div className="flex rounded-lg bg-[#232732] p-0.5 text-[11px]">
                    <button
                      type="button"
                      onClick={() => setBackdropDark(false)}
                      className={`rounded-md px-2 py-1 font-medium ${
                        !backdropDark
                          ? 'bg-blue-600 text-white'
                          : 'text-slate-400'
                      }`}
                    >
                      Studio Light
                    </button>
                    <button
                      type="button"
                      onClick={() => setBackdropDark(true)}
                      className={`rounded-md px-2 py-1 font-medium ${
                        backdropDark
                          ? 'bg-blue-600 text-white'
                          : 'text-slate-400'
                      }`}
                    >
                      CAD Dark
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {openPopover === 'view' && (
            <div className="pointer-events-auto flex flex-wrap items-center gap-1.5 rounded-2xl border border-[#2E3444] bg-[#191C24]/95 p-2 shadow-2xl backdrop-blur-md">
              {(
                [
                  { id: 'perspective', label: 'Perspective' },
                  { id: 'front', label: 'Front' },
                  { id: 'top', label: 'Top' },
                  { id: 'side', label: 'Side' },
                  { id: 'isometric', label: 'Isometric' },
                  { id: 'reset', label: 'Reset' },
                ] as { id: CameraViewPreset; label: string }[]
              ).map((v) => (
                <button
                  key={v.id}
                  type="button"
                  onClick={() => {
                    setActiveTool('orbit');
                    setCameraPreset(v.id);
                    setCameraTrigger((t) => t + 1);
                  }}
                  className={`rounded-lg px-2.5 py-1.5 text-xs font-medium transition-colors ${
                    cameraPreset === v.id
                      ? 'bg-blue-600 text-white'
                      : 'bg-[#232732] text-slate-300 hover:bg-[#2C3240]'
                  }`}
                >
                  {v.label}
                </button>
              ))}
            </div>
          )}

          {openPopover === 'slices' && (
            <div className="pointer-events-auto w-72 rounded-2xl border border-[#2E3444] bg-[#191C24]/95 p-4 shadow-2xl backdrop-blur-md">
              <div className="mb-3 flex items-center justify-between">
                <span className="text-xs font-semibold text-white">
                  Elevation Contours (Slices)
                </span>
                <button
                  type="button"
                  onClick={() => setSliceEnabled((v) => !v)}
                  className={`rounded-md px-2.5 py-1 text-[11px] font-semibold ${
                    sliceEnabled
                      ? 'bg-cyan-600 text-white'
                      : 'bg-[#252A36] text-slate-300'
                  }`}
                >
                  {sliceEnabled ? 'Enabled' : 'Off'}
                </button>
              </div>
              <CadSliderInput
                label="Contour Interval"
                value={sliceIntervalMm}
                min={0.25}
                max={5.0}
                step={0.25}
                unit="mm"
                onChange={(val) => {
                  setSliceIntervalMm(val);
                  if (!sliceEnabled) setSliceEnabled(true);
                }}
              />
            </div>
          )}

          {openPopover === 'layers' && layersWanted && (
            <div className="pointer-events-auto w-80 rounded-2xl border border-[#2E3444] bg-[#191C24]/95 p-4 shadow-2xl backdrop-blur-md">
              <div className="mb-3 flex items-center justify-between gap-2">
                <span className="text-xs font-semibold text-white">Layers</span>
                <button
                  type="button"
                  onClick={() => setOpenPopover(null)}
                  className="text-slate-400 hover:text-white"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </div>
              <p className="mb-3 text-[11px] leading-relaxed text-slate-400" data-testid="layers-status">
                {!partsMap
                  ? 'Loading parts...'
                  : selectedPart
                  ? `Part ${selectedPart.id}: layer ${selectedPart.layer} of ${Math.max(
                      0,
                      ...partsMap.parts.map((p) => p.layer)
                    )} (${
                      selectedPart.override === 'front'
                        ? 'brought forward'
                        : selectedPart.override === 'back'
                        ? 'sent back'
                        : 'automatic'
                    })`
                  : 'Click a part on the relief to select it.'}
              </p>
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  disabled={isBusy || !selectedPart || !selectedUv}
                  onClick={() => onPartOrder('forward', selectedUv)}
                  className="rounded-lg bg-[#2E74F6] px-2 py-1.5 text-[11px] font-semibold text-white hover:bg-blue-500 disabled:opacity-40"
                >
                  Bring forward
                </button>
                <button
                  type="button"
                  disabled={isBusy || !selectedPart || !selectedUv}
                  onClick={() => onPartOrder('back', selectedUv)}
                  className="rounded-lg bg-[#2E74F6] px-2 py-1.5 text-[11px] font-semibold text-white hover:bg-blue-500 disabled:opacity-40"
                >
                  Send back
                </button>
                <button
                  type="button"
                  disabled={isBusy || !selectedPart || !selectedUv}
                  onClick={() => onPartOrder('reset', selectedUv)}
                  title="Return this part to the automatic order"
                  className="rounded-lg bg-[#242936] px-2 py-1.5 text-[11px] text-slate-200 hover:bg-[#2E3545] disabled:opacity-40"
                >
                  Reset
                </button>
                <button
                  type="button"
                  disabled={isBusy || !partsMap}
                  onClick={() => onPartOrder('reset_all', null)}
                  title="Clear every layer choice"
                  className="rounded-lg bg-[#242936] px-2 py-1.5 text-[11px] text-slate-200 hover:bg-[#2E3545] disabled:opacity-40"
                >
                  Reset all
                </button>
              </div>
              <label className="mt-3 flex cursor-pointer items-center gap-2 text-[11px] text-slate-300">
                <input
                  type="checkbox"
                  checked={showLayers}
                  onChange={(e) => setShowLayers(e.target.checked)}
                />
                Show layers
                <span className="ml-auto flex items-center gap-1 text-[10px] text-slate-500">
                  <span className="inline-block h-2 w-6 rounded-sm bg-gradient-to-r from-[#3b82f6] to-[#f97316]" />
                  back to front
                </span>
              </label>
            </div>
          )}

          {openPopover === 'brush' && (
            <div className="pointer-events-auto w-80 rounded-2xl border border-[#2E3444] bg-[#191C24]/95 p-4 shadow-2xl backdrop-blur-md">
              <div className="mb-3 flex items-center justify-between gap-2">
                <span className="text-xs font-semibold text-white">
                  Local Edit 3D Brush (60 FPS Float32)
                </span>
                <div className="flex shrink-0 items-center gap-1.5">
                  <button
                    type="button"
                    disabled={!canUndoBrush}
                    onClick={onUndoBrush}
                    title="Undo last brush stroke"
                    className="flex items-center gap-1 rounded-md bg-[#242936] px-2 py-1 text-[11px] text-slate-200 hover:bg-[#2E3545] disabled:opacity-40"
                  >
                    <Undo2 className="h-3 w-3" />
                    Undo
                  </button>
                  <button
                    type="button"
                    disabled={isSavingBrush || !project}
                    onClick={onResetBrush}
                    title="Clear all manual brush strokes (restores the procedural relief)"
                    className="flex items-center gap-1 rounded-md bg-[#242936] px-2 py-1 text-[11px] text-slate-200 hover:bg-[#2E3545] disabled:opacity-40"
                  >
                    <RotateCcw className="h-3 w-3" />
                    Reset Brush
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setOpenPopover(null);
                      setActiveTool('orbit');
                    }}
                    className="text-slate-400 hover:text-white"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>

              <div className="mb-3 grid grid-cols-4 gap-1.5">
                {(
                  [
                    { id: 'smooth', label: 'Smooth' },
                    { id: 'inflate', label: 'Inflate' },
                    { id: 'deflate', label: 'Deflate' },
                    { id: 'flatten', label: 'Flatten' },
                  ] as { id: BrushMode; label: string }[]
                ).map((b) => (
                  <button
                    key={b.id}
                    type="button"
                    onClick={() => {
                      setBrushMode(b.id);
                      setActiveTool('brush');
                    }}
                    className={`rounded-lg py-1.5 text-[11px] font-semibold transition-colors ${
                      brushMode === b.id && activeTool === 'brush'
                        ? 'bg-blue-600 text-white'
                        : 'bg-[#232732] text-slate-300 hover:bg-[#2C313F]'
                    }`}
                  >
                    {b.label}
                  </button>
                ))}
              </div>

              <div className="space-y-3">
                <CadSliderInput
                  label="Brush Radius"
                  value={brushRadiusPx}
                  min={4}
                  max={96}
                  step={1}
                  decimals={0}
                  unit="px"
                  onChange={setBrushRadiusPx}
                />
                <CadSliderInput
                  label="Brush Strength"
                  value={brushStrength}
                  min={0.05}
                  max={1.0}
                  step={0.05}
                  onChange={setBrushStrength}
                />
              </div>
            </div>
          )}

          {openPopover === 'presets' && (
            <div className="pointer-events-auto w-80 rounded-2xl border border-[#2E3444] bg-[#191C24]/95 p-4 shadow-2xl backdrop-blur-md">
              <div className="mb-2.5 flex items-center justify-between">
                <span className="text-xs font-semibold text-white">
                  Studio Parameter Presets
                </span>
                <button
                  type="button"
                  onClick={() => setOpenPopover(null)}
                  className="text-slate-400 hover:text-white"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </div>
              <form
                onSubmit={async (e) => {
                  e.preventDefault();
                  const trimmed = newPresetName.trim();
                  if (!trimmed) return;
                  await onSavePreset(trimmed);
                  setNewPresetName('');
                }}
                className="mb-3 flex gap-1.5"
              >
                <input
                  type="text"
                  value={newPresetName}
                  onChange={(e) => setNewPresetName(e.target.value)}
                  placeholder="New preset name..."
                  className="flex-1 rounded-lg border border-[#2E3443] bg-[#12141A] px-2.5 py-1.5 text-xs text-slate-100 outline-none focus:border-blue-500"
                />
                <button
                  type="submit"
                  className="flex items-center gap-1 rounded-lg bg-blue-600 px-2.5 py-1.5 text-xs font-semibold text-white hover:bg-blue-500"
                >
                  <Plus className="h-3.5 w-3.5" />
                  Save
                </button>
              </form>

              <div className="cad-scroll max-h-48 space-y-1.5 overflow-y-auto">
                {presets.length === 0 ? (
                  <p className="py-2 text-center text-[11px] text-slate-500">
                    No saved presets yet. Save your favorite relief balance above.
                  </p>
                ) : (
                  presets.map((pr) => (
                    <div
                      key={pr.id}
                      className="flex items-center justify-between rounded-lg bg-[#222631] px-2.5 py-1.5"
                    >
                      <button
                        type="button"
                        onClick={() => onApplyPreset(pr.params)}
                        className="flex-1 truncate text-left text-xs font-medium text-slate-200 hover:text-blue-400"
                      >
                        {pr.name}
                      </button>
                      <button
                        type="button"
                        onClick={() => onDeletePreset(pr.id)}
                        className="ml-2 text-slate-500 hover:text-red-400"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}

          {openPopover === 'export' && (
            <div className="pointer-events-auto w-96 max-w-[calc(100vw-2rem)] rounded-2xl border border-[#2E3444] bg-[#191C24]/95 p-4 shadow-2xl backdrop-blur-md">
              <div className="mb-3 flex items-center justify-between">
                <div>
                  <h3 className="text-xs font-bold text-white">
                    Lossless Zero-Watermark Export
                  </h3>
                  <p className="text-[11px] text-slate-400">
                    {params.physical_width_mm.toFixed(0)} ×{' '}
                    {physicalHeightMm.toFixed(0)} mm • Z:{' '}
                    {params.relief_z_mm.toFixed(1)} mm • Base:{' '}
                    {params.base_z_mm.toFixed(1)} mm
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => setOpenPopover(null)}
                  className="text-slate-400 hover:text-white"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </div>

              <div className="mb-3">
                <MeshExportSettings
                  value={meshSettings}
                  onChange={onMeshSettingsChange}
                  params={params}
                  project={project}
                />
              </div>

              <div className="space-y-1.5">
                {(Object.keys(EXPORT_FORMAT_META) as ExportFormat[]).map(
                  (fmt) => {
                    const meta = EXPORT_FORMAT_META[fmt];
                    return (
                      <button
                        key={fmt}
                        type="button"
                        disabled={!project || isBusy}
                        onClick={() => onExport(fmt)}
                        className="flex w-full items-center justify-between rounded-xl border border-[#282D3B] bg-[#20242F] px-3 py-2 text-left transition-colors hover:border-blue-500/60 hover:bg-[#262B38] disabled:opacity-50"
                      >
                        <div>
                          <div className="text-xs font-semibold text-white">
                            {meta.label}
                          </div>
                          <div className="text-[10px] text-slate-400">
                            {meta.desc}
                          </div>
                        </div>
                        <Download className="h-4 w-4 text-blue-400" />
                      </button>
                    );
                  }
                )}
              </div>
            </div>
          )}

          {/* ROW 1: Display Pill + Materials Pill (matching ref_04) */}
          <div className="pointer-events-auto flex flex-wrap items-center gap-2.5">
            <div className="flex items-center gap-2 rounded-2xl border border-[#2C313F] bg-[#181A20]/95 px-3 py-1.5 shadow-xl backdrop-blur-md">
              <span className="pr-1 text-[11px] font-medium text-slate-400">
                Display
              </span>
              {displayButtons.map((d) => {
                const active = displayMode === d.id;
                return (
                  <button
                    key={d.id}
                    type="button"
                    title={d.title}
                    onClick={() => setDisplayMode(d.id)}
                    className={`flex h-7 w-7 items-center justify-center rounded-full transition-all ${
                      active
                        ? 'ring-2 ring-blue-500 ring-offset-2 ring-offset-[#181A20]'
                        : 'opacity-85 hover:opacity-100'
                    }`}
                  >
                    {d.swatch}
                  </button>
                );
              })}
            </div>

            <div className="flex items-center gap-2 rounded-2xl border border-[#2C313F] bg-[#181A20]/95 px-3 py-1.5 shadow-xl backdrop-blur-md">
              <span className="pr-1 text-[11px] font-medium text-slate-400">
                Materials
              </span>
              {MATERIAL_SPHERES.map((m) => {
                const active = materialId === m.id && displayMode === 'solid';
                return (
                  <button
                    key={m.id}
                    type="button"
                    title={m.name}
                    onClick={() => {
                      setMaterialId(m.id);
                      if (displayMode !== 'solid') setDisplayMode('solid');
                    }}
                    className={`h-6 w-6 rounded-full transition-transform hover:scale-110 ${
                      active
                        ? 'ring-2 ring-blue-500 ring-offset-2 ring-offset-[#181A20]'
                        : ''
                    }`}
                    style={{ background: m.swatchGradient }}
                  />
                );
              })}
              <div className="mx-1 h-4 w-px bg-[#2C313F]" />
              <button
                type="button"
                onClick={() => togglePopover('adjust')}
                className={`flex items-center gap-1.5 rounded-lg px-2 py-1 text-[11px] font-medium transition-colors ${
                  openPopover === 'adjust'
                    ? 'bg-blue-600 text-white'
                    : 'text-slate-300 hover:bg-[#232732]'
                }`}
              >
                <Sliders className="h-3.5 w-3.5" />
                <span>Adjust</span>
              </button>
            </div>
          </div>

          {/* ROW 2: View | Profile | Slices | Local Edit | Presets | Export (matching ref_04) */}
          <div className="pointer-events-auto flex flex-wrap items-center gap-1 rounded-2xl border border-[#2C313F] bg-[#181A20]/95 p-1.5 shadow-xl backdrop-blur-md">
            <button
              type="button"
              onClick={() => togglePopover('view')}
              className={`flex items-center gap-1.5 rounded-xl px-3 py-1.5 text-xs font-medium transition-colors ${
                openPopover === 'view'
                  ? 'bg-[#262B38] text-white'
                  : 'text-slate-300 hover:bg-[#222631]'
              }`}
            >
              <Eye className="h-3.5 w-3.5 text-slate-400" />
              <span>View</span>
            </button>

            <div className="h-4 w-px bg-[#2A2F3D]" />

            <button
              type="button"
              onClick={() => {
                const nextOpen = !profileOpen;
                setProfileOpen(nextOpen);
                setOpenPopover(null);
                setActiveTool(nextOpen ? 'profile' : 'orbit');
              }}
              className={`flex items-center gap-1.5 rounded-xl px-3 py-1.5 text-xs font-medium transition-colors ${
                profileOpen || activeTool === 'profile'
                  ? 'bg-cyan-600/25 text-cyan-300 ring-1 ring-cyan-500/40'
                  : 'text-slate-300 hover:bg-[#222631]'
              }`}
            >
              <Activity className="h-3.5 w-3.5" />
              <span>Profile</span>
            </button>

            <div className="h-4 w-px bg-[#2A2F3D]" />

            <button
              type="button"
              onClick={() => {
                if (!sliceEnabled) setSliceEnabled(true);
                togglePopover('slices');
              }}
              className={`flex items-center gap-1.5 rounded-xl px-3 py-1.5 text-xs font-medium transition-colors ${
                sliceEnabled
                  ? 'bg-blue-600/25 text-blue-300 ring-1 ring-blue-500/40'
                  : 'text-slate-300 hover:bg-[#222631]'
              }`}
            >
              <Layers className="h-3.5 w-3.5" />
              <span>Slices</span>
            </button>

            <div className="h-4 w-px bg-[#2A2F3D]" />

            <button
              type="button"
              onClick={() => {
                const entering = activeTool !== 'brush';
                setActiveTool(entering ? 'brush' : 'orbit');
                setOpenPopover(entering ? 'brush' : null);
              }}
              title={isBusy ? 'Locked while processing' : 'Local Edit brush'}
              className={`flex items-center gap-1.5 rounded-xl px-3 py-1.5 text-xs font-medium transition-colors ${
                activeTool === 'brush'
                  ? 'bg-blue-600/25 text-blue-300 ring-1 ring-blue-500/40'
                  : 'text-slate-300 hover:bg-[#222631]'
              } ${isBusy ? 'opacity-60' : ''}`}
            >
              {isBusy ? (
                <Lock className="h-3.5 w-3.5" />
              ) : (
                <Paintbrush className="h-3.5 w-3.5" />
              )}
              <span>Local Edit</span>
            </button>

            <div className="h-4 w-px bg-[#2A2F3D]" />

            <button
              type="button"
              disabled={!isSculpt}
              onClick={() => {
                const entering = activeTool !== 'layers';
                setActiveTool(entering ? 'layers' : 'orbit');
                setOpenPopover(entering ? 'layers' : null);
              }}
              title={
                !isSculpt
                  ? 'Switch to Sculpt to use layers'
                  : isBusy
                  ? 'Locked while processing'
                  : 'Bring parts forward or send them back'
              }
              className={`flex items-center gap-1.5 rounded-xl px-3 py-1.5 text-xs font-medium transition-colors ${
                activeTool === 'layers'
                  ? 'bg-orange-600/25 text-orange-300 ring-1 ring-orange-500/40'
                  : 'text-slate-300 hover:bg-[#222631]'
              } ${isBusy ? 'opacity-60' : ''} disabled:cursor-not-allowed disabled:opacity-40`}
            >
              {isBusy ? (
                <Lock className="h-3.5 w-3.5" />
              ) : (
                <ListOrdered className="h-3.5 w-3.5" />
              )}
              <span>Layers</span>
            </button>

            <div className="h-4 w-px bg-[#2A2F3D]" />

            <button
              type="button"
              onClick={() => togglePopover('presets')}
              className={`flex items-center gap-1.5 rounded-xl px-3 py-1.5 text-xs font-medium transition-colors ${
                openPopover === 'presets'
                  ? 'bg-[#262B38] text-white'
                  : 'text-slate-300 hover:bg-[#222631]'
              }`}
            >
              <Bookmark className="h-3.5 w-3.5 text-slate-400" />
              <span>Presets</span>
            </button>

            <button
              type="button"
              onClick={() => togglePopover('export')}
              className="ml-1 flex items-center gap-1.5 rounded-xl bg-[#2E74F6] px-4 py-1.5 text-xs font-semibold text-white shadow-sm transition-colors hover:bg-blue-500"
            >
              <Download className="h-3.5 w-3.5" />
              <span>Export</span>
            </button>
          </div>
        </div>
      </div>

      {/* RIGHT DOCK CONTEXT PANEL (340px, matching ref_04) */}
      {panelOpen && (
        <aside className="flex w-[340px] shrink-0 flex-col border-l border-[#252935] bg-[#1D2028] text-slate-100">
          <div className="flex items-center justify-between border-b border-[#282D3A] px-4 py-3.5">
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setPanelOpen(false)}
                className="rounded-md p-1 text-slate-400 hover:bg-[#272C38] hover:text-white"
                aria-label="Collapse panel"
              >
                <X className="h-4 w-4" />
              </button>
              <h2 className="text-sm font-semibold text-white">
                {activeTabTitle}
              </h2>
            </div>
            <button
              type="button"
              onClick={() => onParamsChange({ ...params }, true)}
              title="Force recompute"
              className="rounded-md p-1.5 text-slate-400 hover:bg-[#272C38] hover:text-blue-400"
            >
              <RotateCcw className="h-3.5 w-3.5" />
            </button>
          </div>

          <div className="cad-scroll flex-1 space-y-5 overflow-y-auto p-4">
            {/* TAB 1: FRAME SETTINGS (matching ref_04) */}
            {activeTab === 'frame' && (
              <div className="space-y-4">
                <div>
                  <div className="mb-2 flex items-center gap-1.5 text-xs font-medium text-slate-300">
                    <span>Frame Presets</span>
                    <InfoTip
                      dark
                      text="Synthesizes a parametric architectural molding border around the selected region."
                    />
                  </div>
                  <label className="flex cursor-pointer items-center gap-2.5 select-none">
                    <button
                      type="button"
                      role="checkbox"
                      aria-checked={params.frame.enabled}
                      onClick={() =>
                        updateFrame('enabled', !params.frame.enabled, true)
                      }
                      className={`flex h-4 w-4 items-center justify-center rounded border transition-colors ${
                        params.frame.enabled
                          ? 'border-blue-500 bg-blue-600 text-white'
                          : 'border-[#3C4354] bg-[#14161C]'
                      }`}
                    >
                      {params.frame.enabled && (
                        <Check className="h-3 w-3 stroke-[3]" />
                      )}
                    </button>
                    <span className="text-xs font-medium text-slate-200">
                      Enable Frame
                    </span>
                  </label>
                </div>

                <div>
                  <label className="mb-1.5 block text-xs font-medium text-slate-300">
                    Frame Preset
                  </label>
                  <select
                    value={params.frame.preset}
                    onChange={(e) =>
                      updateFrame(
                        'preset',
                        e.target.value as FramePreset,
                        true
                      )
                    }
                    className="w-full rounded-lg border border-[#2F3545] bg-[#161920] px-3 py-2 text-xs font-medium text-slate-100 outline-none focus:border-blue-500"
                  >
                    {framePresetOptions.map((opt) => (
                      <option key={opt.value} value={opt.value}>
                        {opt.label}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="grid grid-cols-1 gap-4">
                  {/* Frame Width/Thickness/Back Protrusion are true physical millimetres on the
                      Physical Width canvas; thickness is capped at the Max Relief Z-Height. */}
                  <CadSliderInput
                    label="Frame Width"
                    value={params.frame.width_mm}
                    min={0.5}
                    max={Math.max(25, Math.round(params.physical_width_mm * 0.25))}
                    step={0.5}
                    decimals={1}
                    unit="mm"
                    onChange={(v) => updateFrame('width_mm', v)}
                  />
                  <CadSliderInput
                    label="Frame Thickness"
                    value={params.frame.thickness_mm}
                    min={0.2}
                    max={params.relief_z_mm}
                    step={0.1}
                    decimals={1}
                    unit="mm"
                    onChange={(v) => updateFrame('thickness_mm', v)}
                  />
                  <CadSliderInput
                    label="Back Protrusion"
                    value={params.frame.back_protrusion_mm}
                    min={0}
                    max={params.relief_z_mm}
                    step={0.1}
                    decimals={1}
                    unit="mm"
                    onChange={(v) => updateFrame('back_protrusion_mm', v)}
                  />
                </div>

                <div className="border-t border-[#2A2F3D] pt-4">
                  <div className="mb-2 flex items-center justify-between">
                    <span className="text-xs font-semibold text-slate-300">
                      Frame Bounding Box
                    </span>
                    <button
                      type="button"
                      onClick={() => setShowFrameGuide((v) => !v)}
                      className={`rounded px-2 py-0.5 text-[10px] font-semibold ${
                        showFrameGuide
                          ? 'bg-blue-500/20 text-blue-300'
                          : 'bg-[#262B38] text-slate-400'
                      }`}
                    >
                      {showFrameGuide ? 'Guide On' : 'Guide Off'}
                    </button>
                  </div>
                  <div className="mb-3 grid grid-cols-2 gap-2">
                    <button
                      type="button"
                      onClick={() =>
                        updateFrame(
                          'bbox',
                          [...FRAME_BBOX_PRESETS.lower_ref_04.bbox],
                          true
                        )
                      }
                      className={`rounded-lg border px-2.5 py-2 text-[11px] font-semibold transition-colors ${
                        isLowerRef04
                          ? 'border-blue-500 bg-blue-600/20 text-blue-300'
                          : 'border-[#2E3443] bg-[#222631] text-slate-300 hover:border-[#3E465B]'
                      }`}
                    >
                      Lower Figure Ref 04
                    </button>
                    <button
                      type="button"
                      onClick={() =>
                        updateFrame(
                          'bbox',
                          [...FRAME_BBOX_PRESETS.full_piece.bbox],
                          true
                        )
                      }
                      className={`rounded-lg border px-2.5 py-2 text-[11px] font-semibold transition-colors ${
                        isFullPiece
                          ? 'border-blue-500 bg-blue-600/20 text-blue-300'
                          : 'border-[#2E3443] bg-[#222631] text-slate-300 hover:border-[#3E465B]'
                      }`}
                    >
                      Full Piece
                    </button>
                  </div>

                  <div className="space-y-2.5">
                    <CadSliderInput
                      label="Left (u0)"
                      value={params.frame.bbox[0]}
                      min={0}
                      max={0.9}
                      step={0.01}
                      onChange={(v) => setBboxIndex(0, v)}
                    />
                    <CadSliderInput
                      label="Top (v0)"
                      value={params.frame.bbox[1]}
                      min={0}
                      max={0.9}
                      step={0.01}
                      onChange={(v) => setBboxIndex(1, v)}
                    />
                    <CadSliderInput
                      label="Right (u1)"
                      value={params.frame.bbox[2]}
                      min={0.1}
                      max={1}
                      step={0.01}
                      onChange={(v) => setBboxIndex(2, v)}
                    />
                    <CadSliderInput
                      label="Bottom (v1)"
                      value={params.frame.bbox[3]}
                      min={0.1}
                      max={1}
                      step={0.01}
                      onChange={(v) => setBboxIndex(3, v)}
                    />
                  </div>
                </div>
              </div>
            )}

            {/* TAB 2: RELIEF SETTINGS */}
            {activeTab === 'relief' && (
              <div className="space-y-4">
                <div className="rounded-xl border border-[#2A2F3D] bg-[#161920] p-3">
                  <div className="text-[11px] font-semibold text-slate-400">
                    Physical Fabrication Dimensions
                  </div>
                  <div className="mt-1 font-mono text-xs text-cyan-300">
                    {params.physical_width_mm.toFixed(0)} ×{' '}
                    {physicalHeightMm.toFixed(0)} ×{' '}
                    {(params.base_z_mm + params.relief_z_mm).toFixed(1)} mm
                  </div>
                </div>

                <CadSliderInput
                  label="Physical Width"
                  value={params.physical_width_mm}
                  min={100}
                  max={2500}
                  step={10}
                  decimals={0}
                  unit="mm"
                  onChange={(v) => updateParam('physical_width_mm', v)}
                />
                <CadSliderInput
                  label="Max Relief Z-Height"
                  value={params.relief_z_mm}
                  min={1}
                  max={80}
                  step={0.5}
                  decimals={1}
                  unit="mm"
                  onChange={(v) => updateParam('relief_z_mm', v)}
                />
                <CadSliderInput
                  label="Base Plaque Thickness"
                  value={params.base_z_mm}
                  min={0}
                  max={30}
                  step={0.5}
                  decimals={1}
                  unit="mm"
                  tooltip="Set to 0.0 mm to export a true silhouette-cutout mesh without a rectangular backing plaque."
                  onChange={(v) => updateParam('base_z_mm', v)}
                />

                <div className="border-t border-[#2A2F3D] pt-4 space-y-4">
                  <StyleToggle
                    params={params}
                    onChange={onParamsChange}
                    isBusy={isBusy}
                    theme="dark"
                  />
                  {params.style === 'sculpt' ? (
                    <SculptControls
                      params={params}
                      onChange={onParamsChange}
                      isBusy={isBusy}
                      edgeClamped={edgeClamped}
                      aspect={aspect}
                      latestVersion={project?.sculpt_latest_version}
                      theme="dark"
                    />
                  ) : (
                    <>
                  <CadSliderInput
                    label="Global Cushion Volume"
                    value={params.w_global}
                    min={0}
                    max={1.5}
                    step={0.02}
                    onChange={(v) => updateParam('w_global', v)}
                  />
                  <CadSliderInput
                    label="Limb Puffiness"
                    value={params.w_local}
                    min={0}
                    max={1.5}
                    step={0.02}
                    onChange={(v) => updateParam('w_local', v)}
                  />
                  <CadSliderInput
                    label="Line Crease Depth"
                    value={params.w_groove}
                    min={0}
                    max={1.5}
                    step={0.02}
                    onChange={(v) => updateParam('w_groove', v)}
                  />
                  <CadSliderInput
                    label="Line Crease Width"
                    value={params.groove_sigma}
                    min={0.5}
                    max={12}
                    step={0.1}
                    decimals={1}
                    unit="px"
                    onChange={(v) => updateParam('groove_sigma', v)}
                  />
                  <CadSliderInput
                    label="Gamma Curve"
                    value={params.gamma}
                    min={0.3}
                    max={2.4}
                    step={0.02}
                    onChange={(v) => updateParam('gamma', v)}
                  />
                    </>
                  )}

                  <label className="flex cursor-pointer items-center justify-between rounded-xl border border-[#2A2F3D] bg-[#161920] px-3 py-2.5">
                    <span className="text-xs font-medium text-slate-200">
                      Invert Depth (Concave Mold)
                    </span>
                    <input
                      type="checkbox"
                      checked={params.invert_depth}
                      onChange={(e) =>
                        updateParam('invert_depth', e.target.checked, true)
                      }
                      className="h-4 w-4 accent-blue-600"
                    />
                  </label>
                </div>
              </div>
            )}

            {/* TAB 3: DRAFT ANGLE */}
            {activeTab === 'draft_angle' && (
              <div className="space-y-4">
                <p className="text-xs leading-relaxed text-slate-400">
                  Limits relief height near the silhouette. Brush edits and frames can introduce steeper surfaces.
                </p>
                <CadSliderInput
                  label="Draft Angle"
                  value={params.draft_angle_deg}
                  min={0}
                  max={15}
                  step={0.5}
                  decimals={1}
                  unit="deg"
                  onChange={(v) => updateParam('draft_angle_deg', v)}
                />

              </div>
            )}

            {/* TAB 4: 2-SIDED */}
            {activeTab === 'two_sided' && (
              <div className="space-y-4">
                <p className="text-xs leading-relaxed text-slate-400">
                  Preview a double-sided bas-relief partition or freestanding
                  bronze medallion with mirrored front &amp; back sculpting.
                </p>
                <label className="flex cursor-pointer items-center justify-between rounded-xl border border-[#2A2F3D] bg-[#161920] px-3 py-3">
                  <span className="text-xs font-semibold text-slate-100">
                    2-Sided Mirror Preview
                  </span>
                  <input
                    type="checkbox"
                    checked={twoSided}
                    onChange={(e) => setTwoSided(e.target.checked)}
                    className="h-4 w-4 accent-blue-600"
                  />
                </label>
                <div className="rounded-xl border border-[#2A2F3D] bg-[#161920] p-3.5 font-mono text-[11px] text-slate-300 space-y-1">
                  <div>Single-Side Max Z: {(params.base_z_mm + params.relief_z_mm).toFixed(1)} mm</div>
                  <div>
                    Double-Sided Span:{' '}
                    {(params.base_z_mm + params.relief_z_mm * 2).toFixed(1)} mm
                  </div>
                </div>
              </div>
            )}

            {/* TAB 5: COMPOSITE */}
            {activeTab === 'composite' && (
              <div className="space-y-4">
                <p className="text-xs leading-relaxed text-slate-400">
                  {isSculpt
                    ? 'Sculpt builds each part as its own form and stacks them by layer. Use the Relief tab sliders and the Layers tool; the Classic basis blend is hidden in this style.'
                    : 'Blend cached float32 mathematical basis tensors directly in real time without re-running decomposition.'}
                </p>
                {params.style === 'classic' && (
                  <>
                <CadSliderInput
                  label="Global Dome (H_global)"
                  value={params.w_global}
                  min={0}
                  max={1.5}
                  step={0.02}
                  onChange={(v) => updateParam('w_global', v)}
                />
                <CadSliderInput
                  label="Local Limb Puff (H_local)"
                  value={params.w_local}
                  min={0}
                  max={1.5}
                  step={0.02}
                  onChange={(v) => updateParam('w_local', v)}
                />
                <CadSliderInput
                  label="Guided AI Overlap (H_ai)"
                  value={params.w_ai}
                  min={0}
                  max={1.0}
                  step={0.02}
                  tooltip="Keep at 0.00 for pure mathematical line-locked fidelity."
                  onChange={(v) => updateParam('w_ai', v)}
                />
                  </>
                )}
                <CadSliderInput
                  label="Silhouette Min Floor"
                  value={params.h_min}
                  min={0}
                  max={0.4}
                  step={0.01}
                  onChange={(v) => updateParam('h_min', v)}
                />
                {params.style === 'classic' && (
                  <>
                <CadSliderInput
                  label="Brightness Offset"
                  value={params.brightness}
                  min={-0.35}
                  max={0.35}
                  step={0.01}
                  onChange={(v) => updateParam('brightness', v)}
                />
                <CadSliderInput
                  label="Contrast Gain"
                  value={params.contrast}
                  min={0.5}
                  max={1.8}
                  step={0.02}
                  onChange={(v) => updateParam('contrast', v)}
                />
                  </>
                )}
              </div>
            )}

            {/* TAB 6: TEXT / ZERO-WATERMARK POLICY */}
            {activeTab === 'text' && (
              <div className="space-y-4">
                <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-3.5">
                  <div className="flex items-center gap-2 text-xs font-bold text-emerald-300">
                    <ShieldCheck className="h-4 w-4" />
                    MONOLITH Zero-Watermark Policy
                  </div>
                  <p className="mt-1.5 text-[11px] leading-relaxed text-emerald-100/80">
                    Unlike cloud services, MONOLITH never injects repeating
                    &ldquo;watermark&rdquo; grids or locks clean geometry behind
                    extra credit fees. Your exported 16-bit PNG, 32-bit EXR, and
                    watertight STL meshes are 100% pure geometry.
                  </p>
                </div>

                <label className="flex cursor-pointer items-center justify-between rounded-xl border border-[#2A2F3D] bg-[#161920] px-3 py-2.5">
                  <span className="text-xs font-medium text-slate-200">
                    Show Artist Edition Card in Viewport
                  </span>
                  <input
                    type="checkbox"
                    checked={editionStampEnabled}
                    onChange={(e) => setEditionStampEnabled(e.target.checked)}
                    className="h-4 w-4 accent-blue-600"
                  />
                </label>

                {editionStampEnabled && (
                  <div className="space-y-3 rounded-xl border border-[#2A2F3D] bg-[#161920] p-3">
                    <div>
                      <label className="mb-1 block text-[11px] text-slate-400">
                        Artist / Studio Name
                      </label>
                      <input
                        type="text"
                        value={artistName}
                        onChange={(e) => setArtistName(e.target.value)}
                        className="w-full rounded-lg border border-[#2E3443] bg-[#12141A] px-2.5 py-1.5 text-xs text-white outline-none focus:border-blue-500"
                      />
                    </div>
                    <div>
                      <label className="mb-1 block text-[11px] text-slate-400">
                        Edition Stamp (Viewport Overlay Only)
                      </label>
                      <input
                        type="text"
                        value={editionNote}
                        onChange={(e) => setEditionNote(e.target.value)}
                        className="w-full rounded-lg border border-[#2E3443] bg-[#12141A] px-2.5 py-1.5 text-xs text-white outline-none focus:border-blue-500"
                      />
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </aside>
      )}

      {/* FAR-RIGHT VERTICAL ICON STRIP (56px, matching ref_04) */}
      <nav className="flex w-14 shrink-0 flex-col items-center gap-1.5 border-l border-[#252935] bg-[#16181E] py-3">
        {tabMeta.map((tab) => {
          const active = activeTab === tab.id && panelOpen;
          return (
            <button
              key={tab.id}
              type="button"
              onClick={() => {
                if (activeTab === tab.id && panelOpen) {
                  setPanelOpen(false);
                } else {
                  setActiveTab(tab.id);
                  setPanelOpen(true);
                }
              }}
              className={`flex w-12 flex-col items-center gap-1 rounded-xl py-2 text-[10px] font-medium transition-colors ${
                active
                  ? 'bg-blue-600/20 text-blue-400 ring-1 ring-blue-500/40'
                  : 'text-slate-400 hover:bg-[#222631] hover:text-slate-200'
              }`}
            >
              {tab.icon}
              <span className="leading-none">{tab.short}</span>
            </button>
          );
        })}
      </nav>
    </div>
  );
};
