export type FramePreset =
  | 'gallery_classic'
  | 'modern_flat'
  | 'stepped_bevel'
  | 'circular_medallion';

export interface FrameConfig {
  enabled: boolean;
  preset: FramePreset;
  width_mm: number;
  thickness_mm: number;
  back_protrusion_mm: number;
  bbox: [number, number, number, number];
  /** Frame dimensions are true physical millimetres on the physical_width_mm canvas. */
  units?: 'mm';
}

export interface ReliefParams {
  w_global: number;
  w_local: number;
  w_ai: number;
  w_groove: number;
  /** Groove blur in pixels at a 4096 px longest edge (same look at 2K / 4K / 8K). */
  groove_sigma: number;
  groove_sigma_units?: 'px@4096';
  gamma: number;
  brightness: number;
  contrast: number;
  h_min: number;
  invert_depth: boolean;
  draft_angle_deg: number;
  physical_width_mm: number;
  relief_z_mm: number;
  base_z_mm: number;
  frame: FrameConfig;
  /** Relief style. Stored projects without it are classic; new projects default to sculpt. */
  style: ReliefStyle;
  sculpt_version: number;
  /** Edge rounding radius, % of the shorter canvas side. */
  edge_roundness_pct: number;
  fullness: number;
  crease_depth: number;
  /** Crease width, % of the shorter canvas side. */
  crease_width_pct: number;
  layering: number;
}

export type ReliefStyle = 'sculpt' | 'classic';

/** Drawing convention: auto-detect, white shapes on black, or ink lines on white paper. */
export type LinePolarity = 'auto' | 'white_on_black' | 'black_on_white';

export interface MeshSettings {
  mode: 'adaptive' | 'grid';
  toleranceMm: 0.01 | 0.02 | 0.05 | 0.1;
  gridResolution: 512 | 1024 | 2048;
}

export const DEFAULT_MESH_SETTINGS: MeshSettings = {
  mode: 'adaptive',
  toleranceMm: 0.02,
  gridResolution: 2048,
};

export interface ProjectRecord {
  id: string;
  name: string;
  original_filename: string;
  width: number;
  height: number;
  ai_mode: 'off' | 'guided_overlap' | string;
  params: ReliefParams;
  created_at: string;
  updated_at: string;
  bit_depth?: BitDepthOption | string;
  bg_removal?: boolean;
  hd_restore?: boolean;
  target_res?: number;
  polarity?: LinePolarity | string;
  compute_ms?: number;
  unique_levels?: number;
  /** Newest Sculpt algorithm version the server offers. */
  sculpt_latest_version?: number;
}

export interface RecomputeResponse {
  id: string;
  /** True when the server ignored this request because a newer one was already applied. */
  stale?: boolean;
  compute_ms: number;
  unique_levels: number;
  width: number;
  height: number;
  params: ReliefParams;
  /** Sculpt: Edge roundness was raised to keep the draft angle. */
  sculpt_edge_clamped?: boolean;
}

export interface PresetRecord {
  id: string;
  name: string;
  params: ReliefParams;
  created_at: string;
}

export type WorkspaceView = 'studio' | 'editor3d';

export type TargetResolution = 2048 | 4096 | 8192;
export type AiMode = 'off' | 'guided_overlap';
export type BitDepthOption = '8bit' | '16bit' | '32bit_exr';
export type ExportFormat =
  | 'png16'
  | 'png8'
  | 'exr32'
  | 'tiff16'
  | 'stl'
  | 'obj'
  | 'svg'
  | 'dxf';

/** Map the sidebar Bit Depth selection to its matching primary Download format. */
export function downloadFmtForBitDepth(depth: BitDepthOption | string): ExportFormat {
  if (depth === '32bit_exr') return 'exr32';
  if (depth === '8bit') return 'png8';
  return 'png16';
}

export type DisplayMode = 'solid' | 'clay' | 'curvature' | 'normal';

export type MaterialId =
  | 'silver_steel'
  | 'matte_clay'
  | 'aged_bronze'
  | 'polished_copper'
  | 'brushed_titanium'
  | 'gold_24k';

export type CameraViewPreset =
  | 'perspective'
  | 'front'
  | 'top'
  | 'side'
  | 'isometric'
  | 'reset';

export type EditorTab =
  | 'relief'
  | 'draft_angle'
  | 'two_sided'
  | 'composite'
  | 'frame'
  | 'text';

export type BrushMode = 'smooth' | 'inflate' | 'deflate' | 'flatten';

export interface Float32TexturePayload {
  data: Float32Array;
  width: number;
  height: number;
  /** Distinct height levels (from the X-Unique-Levels header), when the server provides it. */
  uniqueLevels?: number;
}

export interface ProfilePoint {
  u: number;
  v: number;
}

export interface ProfileSample {
  distanceMm: number;
  zMm: number;
  normalized: number;
}

export const FRAME_BBOX_PRESETS: Record<
  string,
  { label: string; bbox: [number, number, number, number] }
> = {
  lower_ref_04: {
    label: 'Lower Figure (Ref 04)',
    bbox: [0.16, 0.48, 0.84, 0.96],
  },
  full_piece: {
    label: 'Full Piece',
    bbox: [0.04, 0.04, 0.96, 0.96],
  },
};

export const DEFAULT_RELIEF_PARAMS: ReliefParams = {
  w_global: 0.45,
  w_local: 0.65,
  w_ai: 0.25,
  w_groove: 0.35,
  groove_sigma: 2.5,
  groove_sigma_units: 'px@4096',
  gamma: 0.85,
  brightness: 0.0,
  contrast: 1.0,
  h_min: 0.08,
  invert_depth: false,
  draft_angle_deg: 2.0,
  physical_width_mm: 1000.0,
  relief_z_mm: 15.0,
  base_z_mm: 5.0,
  frame: {
    enabled: false,
    preset: 'gallery_classic',
    width_mm: 22.0,
    thickness_mm: 5.5,
    back_protrusion_mm: 0.0,
    bbox: [0.16, 0.48, 0.84, 0.96],
    units: 'mm',
  },
  style: 'sculpt',
  sculpt_version: 1,
  edge_roundness_pct: 7.0,
  fullness: 0.45,
  crease_depth: 0.18,
  crease_width_pct: 1.0,
  layering: 0.35,
};

export const MATERIAL_SPHERES: {
  id: MaterialId;
  name: string;
  swatchGradient: string;
  baseColor: [number, number, number];
  specColor: [number, number, number];
  metalness: number;
  roughness: number;
}[] = [
  {
    id: 'silver_steel',
    name: 'Silver Steel',
    swatchGradient:
      'radial-gradient(circle at 32% 30%, #e6edf5 0%, #8c96a4 38%, #414854 75%, #1d2128 100%)',
    baseColor: [0.64, 0.68, 0.74],
    specColor: [0.92, 0.95, 1.0],
    metalness: 0.88,
    roughness: 0.24,
  },
  {
    id: 'matte_clay',
    name: 'Matte Plaster Clay',
    swatchGradient:
      'radial-gradient(circle at 34% 30%, #ffffff 0%, #e4e8ee 48%, #b6bec9 84%, #8e96a3 100%)',
    baseColor: [0.89, 0.90, 0.92],
    specColor: [0.28, 0.29, 0.31],
    metalness: 0.02,
    roughness: 0.78,
  },
  {
    id: 'aged_bronze',
    name: 'Aged Bronze',
    swatchGradient:
      'radial-gradient(circle at 32% 30%, #e2bd89 0%, #8f653b 42%, #4a3119 78%, #211509 100%)',
    baseColor: [0.56, 0.40, 0.24],
    specColor: [0.90, 0.72, 0.46],
    metalness: 0.84,
    roughness: 0.36,
  },
  {
    id: 'polished_copper',
    name: 'Polished Copper',
    swatchGradient:
      'radial-gradient(circle at 32% 30%, #ffd2ba 0%, #d4724a 40%, #7d361d 78%, #331308 100%)',
    baseColor: [0.82, 0.45, 0.30],
    specColor: [1.0, 0.76, 0.62],
    metalness: 0.92,
    roughness: 0.20,
  },
  {
    id: 'brushed_titanium',
    name: 'Brushed Titanium',
    swatchGradient:
      'radial-gradient(circle at 32% 30%, #eef2f6 0%, #a5adb8 45%, #5b636e 80%, #2a2f36 100%)',
    baseColor: [0.70, 0.73, 0.77],
    specColor: [0.90, 0.92, 0.95],
    metalness: 0.78,
    roughness: 0.38,
  },
  {
    id: 'gold_24k',
    name: '24K Gold',
    swatchGradient:
      'radial-gradient(circle at 32% 30%, #fff4b8 0%, #f0be3d 38%, #9b6c0e 78%, #422a03 100%)',
    baseColor: [0.95, 0.74, 0.24],
    specColor: [1.0, 0.92, 0.56],
    metalness: 0.95,
    roughness: 0.19,
  },
];

export const EXPORT_FORMAT_META: Record<
  ExportFormat,
  { label: string; short: string; ext: string; desc: string }
> = {
  png16: {
    label: 'PNG 16-bit',
    short: 'PNG 16',
    ext: 'png',
    desc: 'Lossless uint16 (65,536 depth levels)',
  },
  png8: {
    label: 'PNG 8-bit (Legacy)',
    short: 'PNG 8',
    ext: 'png',
    desc: 'Standard 256-level 8-bit grayscale preview',
  },
  exr32: {
    label: 'EXR 32-bit Float',
    short: 'EXR 32',
    ext: 'exr',
    desc: 'Linear IEEE float32 master heightmap',
  },
  tiff16: {
    label: 'TIFF 16-bit',
    short: 'TIFF 16',
    ext: 'tiff',
    desc: 'Uncompressed studio print/CAM archive',
  },
  stl: {
    label: 'Watertight STL',
    short: 'STL 3D',
    ext: 'stl',
    desc: 'Manifold binary mesh ready for CNC/print',
  },
  obj: {
    label: 'Wavefront OBJ',
    short: 'OBJ 3D',
    ext: 'obj',
    desc: 'Quad/triangulated mesh for ZBrush/Rhino',
  },
  svg: {
    label: 'Outline SVG',
    short: 'SVG',
    ext: 'svg',
    desc: 'Sub-pixel silhouette cut path in true mm',
  },
  dxf: {
    label: 'Outline DXF',
    short: 'DXF',
    ext: 'dxf',
    desc: 'CAM profile path (mm, outer/hole layers)',
  },
};
