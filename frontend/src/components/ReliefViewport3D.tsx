import React, { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { Canvas, ThreeEvent, useFrame, useThree } from '@react-three/fiber';
import { Line, OrbitControls } from '@react-three/drei';
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib';
import {
  MATERIAL_SPHERES,
  type BrushMode,
  type CameraViewPreset,
  type DisplayMode,
  type Float32TexturePayload,
  type MaterialId,
  type ProfilePoint,
  type ReliefParams,
} from '../types/monolith';
import { sampleFloat32Bilinear } from './ProfileChartOverlay';
import { pickHeightfield } from '../lib/heightfieldPick';
import { anchorAfterPick, strokeStamps } from '../lib/brushStroke';
import { buildOverlayRGBA, partAtUv, type PartsMap } from '../lib/partsMap';

const VERTEX_SHADER = `
  uniform sampler2D uHeightMap;
  uniform vec2 uTexSize;
  uniform float uBaseZ;
  uniform float uReliefZ;
  uniform float uFlipMirror;

  varying vec2 vUv;
  varying float vHeightNorm;

  float sampleBilinear(vec2 uv) {
    vec2 texUv = vec2(uFlipMirror > 0.5 ? (1.0 - uv.x) : uv.x, 1.0 - uv.y);
    vec2 clamped = clamp(texUv, vec2(0.0), vec2(1.0));
    vec2 px = clamped * (uTexSize - 1.0);
#ifdef USE_HW_LINEAR
    // Hardware-filtered float texture: identical to the 4-tap below (texel-centre aligned).
    return texture2D(uHeightMap, (px + 0.5) / uTexSize).r;
#else
    vec2 f = fract(px);
    vec2 i0 = floor(px);
    vec2 i1 = min(i0 + 1.0, uTexSize - 1.0);

    vec2 uv00 = (i0 + 0.5) / uTexSize;
    vec2 uv10 = (vec2(i1.x, i0.y) + 0.5) / uTexSize;
    vec2 uv01 = (vec2(i0.x, i1.y) + 0.5) / uTexSize;
    vec2 uv11 = (i1 + 0.5) / uTexSize;

    float h00 = texture2D(uHeightMap, uv00).r;
    float h10 = texture2D(uHeightMap, uv10).r;
    float h01 = texture2D(uHeightMap, uv01).r;
    float h11 = texture2D(uHeightMap, uv11).r;

    return mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
#endif
  }

  void main() {
    vUv = uv;
    float h = sampleBilinear(uv);
    vHeightNorm = h;
    vec3 displaced = position;
    displaced.z += uBaseZ + h * uReliefZ;
    vec4 worldPosition = modelMatrix * vec4(displaced, 1.0);
    gl_Position = projectionMatrix * viewMatrix * worldPosition;
  }
`;

const FRAGMENT_SHADER = `
  uniform sampler2D uHeightMap;
  uniform vec2 uTexSize;
  uniform float uPhysicalWidthMm;
  uniform float uReliefZMm;
  uniform float uBaseZMm;
  uniform float uPlaneWidth;
  uniform float uPlaneHeight;
  uniform int uDisplayMode;
  uniform vec3 uMatBaseColor;
  uniform vec3 uMatSpecColor;
  uniform float uMatMetalness;
  uniform float uMatRoughness;
  uniform vec3 uLightDir;
  uniform float uLightIntensity;
  uniform float uSliceEnabled;
  uniform float uSliceIntervalMm;
  uniform float uShowBrush;
  uniform vec2 uBrushUv;
  uniform float uBrushRadiusUv;
  uniform float uShowFrameBox;
  uniform vec4 uFrameBbox;
  uniform float uFlipMirror;
  uniform sampler2D uOverlayMap;
  uniform float uShowOverlay;
  uniform vec2 uOverlaySize;

  varying vec2 vUv;
  varying float vHeightNorm;

  float sampleBilinear(vec2 uv) {
    vec2 texUv = vec2(uFlipMirror > 0.5 ? (1.0 - uv.x) : uv.x, 1.0 - uv.y);
    vec2 clamped = clamp(texUv, vec2(0.0), vec2(1.0));
    vec2 px = clamped * (uTexSize - 1.0);
#ifdef USE_HW_LINEAR
    // Hardware-filtered float texture: identical to the 4-tap below (texel-centre aligned).
    return texture2D(uHeightMap, (px + 0.5) / uTexSize).r;
#else
    vec2 f = fract(px);
    vec2 i0 = floor(px);
    vec2 i1 = min(i0 + 1.0, uTexSize - 1.0);

    vec2 uv00 = (i0 + 0.5) / uTexSize;
    vec2 uv10 = (vec2(i1.x, i0.y) + 0.5) / uTexSize;
    vec2 uv01 = (vec2(i0.x, i1.y) + 0.5) / uTexSize;
    vec2 uv11 = (i1 + 0.5) / uTexSize;

    float h00 = texture2D(uHeightMap, uv00).r;
    float h10 = texture2D(uHeightMap, uv10).r;
    float h01 = texture2D(uHeightMap, uv01).r;
    float h11 = texture2D(uHeightMap, uv11).r;

    return mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
#endif
  }

  void main() {
    float du = 1.0 / max(uTexSize.x, 1.0);
    float dv = 1.0 / max(uTexSize.y, 1.0);

    float hC = sampleBilinear(vUv);
    float hL = sampleBilinear(vUv - vec2(du, 0.0));
    float hR = sampleBilinear(vUv + vec2(du, 0.0));
    float hD = sampleBilinear(vUv - vec2(0.0, dv));
    float hU = sampleBilinear(vUv + vec2(0.0, dv));

    float physicalHeightMm = uPhysicalWidthMm * (uPlaneHeight / max(uPlaneWidth, 0.001));
    float dxMm = max(0.001, 2.0 * du * uPhysicalWidthMm);
    float dyMm = max(0.001, 2.0 * dv * physicalHeightMm);

    float dzdx = ((hR - hL) * uReliefZMm) / dxMm;
    float dzdy = ((hU - hD) * uReliefZMm) / dyMm;

    vec3 n = normalize(vec3(-dzdx, -dzdy, 1.0));
    if (uFlipMirror > 0.5) {
      n = normalize(vec3(n.x, n.y, -n.z));
    }

    // Approximate mean curvature (Laplacian) for cavity & crevice shading
    float lap = (hL + hR + hD + hU - 4.0 * hC) * 40.0;
    float cavity = clamp(0.5 + lap * 0.45, 0.0, 1.0);

    vec3 viewDir = normalize(vec3(0.0, 0.0, 1.0));
    vec3 lKey = normalize(uLightDir);
    vec3 lFill = normalize(vec3(-uLightDir.x * 0.6, -uLightDir.y * 0.4 + 0.3, 0.8));
    vec3 halfVec = normalize(lKey + viewDir);

    float ndlKey = max(dot(n, lKey), 0.0);
    float ndlFill = max(dot(n, lFill), 0.0);
    float ndv = max(dot(n, viewDir), 0.0);

    vec3 color = vec3(0.85);

    if (uDisplayMode == 3) {
      // Surface Normal Map (Rainbow)
      color = normalize(vec3(-dzdx, -dzdy, 1.0)) * 0.5 + 0.5;
    } else if (uDisplayMode == 2) {
      // Curvature / Cavity Map
      float ridge = clamp(lap * 1.4, -1.0, 1.0);
      vec3 valleyCol = vec3(0.14, 0.25, 0.40);
      vec3 flatCol = vec3(0.76, 0.79, 0.83);
      vec3 ridgeCol = vec3(0.98, 0.89, 0.72);
      color = ridge < 0.0 ? mix(flatCol, valleyCol, -ridge) : mix(flatCol, ridgeCol, ridge);
      color *= (0.72 + 0.35 * ndlKey);
    } else if (uDisplayMode == 1) {
      // Matte Clay (Sculpting Inspection View)
      vec3 clayBase = vec3(0.88, 0.89, 0.91);
      float ao = mix(0.80, 1.0, cavity);
      float wrapDiff = clamp((dot(n, lKey) + 0.25) / 1.25, 0.0, 1.0);
      float spec = pow(max(dot(n, halfVec), 0.0), 22.0) * 0.12;
      color = clayBase * ao * (0.28 + 0.76 * wrapDiff * uLightIntensity + 0.18 * ndlFill) + vec3(spec);
    } else {
      // Solid Lit PBR-inspired Material Shading
      float fresnel = pow(1.0 - ndv, 3.0);
      float shininess = mix(140.0, 16.0, clamp(uMatRoughness, 0.05, 0.98));
      float specPow = pow(max(dot(n, halfVec), 0.0), shininess);
      float specStrength = mix(0.28, 1.35, uMatMetalness) * (1.15 - uMatRoughness * 0.75);
      float diffWeight = mix(0.92, 0.42, uMatMetalness);
      float ao = mix(0.76, 1.0, cavity);

      vec3 diffuse = uMatBaseColor * diffWeight * (0.24 + 0.78 * ndlKey * uLightIntensity + 0.22 * ndlFill);
      vec3 specular = uMatSpecColor * specPow * specStrength * uLightIntensity;
      vec3 rim = mix(vec3(0.12), uMatSpecColor, uMatMetalness) * fresnel * 0.25;
      color = diffuse * ao + specular + rim;
    }

    // Topological Elevation Slices (anti-aliased millimeter contours)
    if (uSliceEnabled > 0.5) {
      float zMm = uBaseZMm + hC * uReliefZMm;
      float interval = max(0.05, uSliceIntervalMm);
      float phase = zMm / interval;
      float distMinor = abs(fract(phase - 0.5) - 0.5) / max(fwidth(phase), 0.0001);
      float minorMask = 1.0 - smoothstep(0.6, 1.6, distMinor);

      float majorPhase = zMm / (interval * 5.0);
      float distMajor = abs(fract(majorPhase - 0.5) - 0.5) / max(fwidth(majorPhase), 0.0001);
      float majorMask = 1.0 - smoothstep(0.7, 1.8, distMajor);

      color = mix(color, vec3(0.08, 0.65, 0.98), minorMask * 0.52);
      color = mix(color, vec3(0.04, 0.16, 0.34), majorMask * 0.85);
    }

    // Optional Frame Bounding Box Preview Guide
    if (uShowFrameBox > 0.5 && uFlipMirror < 0.5) {
      vec2 imgUv = vec2(vUv.x, 1.0 - vUv.y);
      bool inside = imgUv.x >= uFrameBbox.x && imgUv.x <= uFrameBbox.z && imgUv.y >= uFrameBbox.y && imgUv.y <= uFrameBbox.w;
      float dxEdge = min(abs(imgUv.x - uFrameBbox.x), abs(imgUv.x - uFrameBbox.z));
      float dyEdge = min(abs(imgUv.y - uFrameBbox.y), abs(imgUv.y - uFrameBbox.w));
      float border = min(dxEdge, dyEdge);
      if (inside && border < 0.003) {
        color = mix(color, vec3(0.23, 0.51, 0.96), 0.65);
      }
    }

    // Layers tool: per-part tint and selected-part outline (same uv convention as the height map)
    if (uShowOverlay > 0.5 && uFlipMirror < 0.5) {
      // Texel-centre mapping, like the height map and partAtUv (nearest label under the point).
      vec2 ovT = clamp(vec2(vUv.x, 1.0 - vUv.y), vec2(0.0), vec2(1.0));
      vec2 ovP = floor(ovT * (uOverlaySize - 1.0) + 0.5);
      vec4 ov = texture2D(uOverlayMap, (ovP + 0.5) / uOverlaySize);
      color = mix(color, ov.rgb, ov.a);
    }

    // Local Edit Brush Ring Overlay
    if (uShowBrush > 0.5 && uFlipMirror < 0.5) {
      vec2 deltaUv = (vUv - uBrushUv) * vec2(1.0, uPlaneHeight / max(uPlaneWidth, 0.001));
      float dist = length(deltaUv);
      float ringWidth = 0.004;
      float ring = smoothstep(uBrushRadiusUv + ringWidth, uBrushRadiusUv, dist) *
                   smoothstep(uBrushRadiusUv - ringWidth, uBrushRadiusUv, dist);
      float innerTint = (1.0 - smoothstep(0.0, uBrushRadiusUv, dist)) * 0.14;
      color = mix(color, vec3(0.22, 0.74, 0.98), clamp(ring + innerTint, 0.0, 0.85));
    }

    gl_FragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
  }
`;

export interface ReliefViewport3DProps {
  texture: Float32TexturePayload | null;
  params: ReliefParams;
  displayMode: DisplayMode;
  materialId: MaterialId;
  lightAzimuthDeg: number;
  lightElevationDeg: number;
  lightIntensity: number;
  sliceEnabled: boolean;
  sliceIntervalMm: number;
  twoSided: boolean;
  showFrameGuide?: boolean;
  cameraPreset: CameraViewPreset;
  cameraTrigger: number;
  backdropDark?: boolean;
  activeTool: 'orbit' | 'profile' | 'brush' | 'layers';
  brushMode: BrushMode;
  brushRadiusPx: number;
  brushStrength: number;
  /** Brush input is ignored while true (e.g. a recompute is in flight). */
  brushLocked?: boolean;
  /** Sculpt part-label map for the Layers tool. */
  partsMap?: PartsMap | null;
  /** Tint each part by its layer (Layers tool only). */
  showLayers?: boolean;
  selectedPartId?: number | null;
  /** Layers tool click: the part id under the cursor (null = background) and the viewer uv (v up). */
  onPartPick?: (partId: number | null, uv: { u: number; v: number }) => void;
  profilePointA: ProfilePoint | null;
  profilePointB: ProfilePoint | null;
  onProfileChange?: (a: ProfilePoint | null, b: ProfilePoint | null) => void;
  onBrushStrokeStart?: (snapshot: Float32Array) => void;
  onBrushDataChange?: (updated: Float32Array) => void;
  onBrushStrokeEnd?: (finalData: Float32Array) => void;
  segments?: number;
}

interface SceneContentProps extends ReliefViewport3DProps {
  planeW: number;
  planeH: number;
}

const CameraRig: React.FC<{
  preset: CameraViewPreset;
  trigger: number;
  planeW: number;
  planeH: number;
  controlsRef: React.RefObject<OrbitControlsImpl | null>;
}> = ({ preset, trigger, planeW, planeH, controlsRef }) => {
  const { camera, invalidate } = useThree();

  useEffect(() => {
    const dist = Math.max(planeW, planeH) * 1.28;
    const pos = new THREE.Vector3(0, 0, dist);

    if (preset === 'front') {
      pos.set(0, 0, dist);
    } else if (preset === 'top') {
      pos.set(0, dist * 0.9, dist * 0.35);
    } else if (preset === 'side') {
      pos.set(dist * 0.92, 0, dist * 0.32);
    } else if (preset === 'isometric') {
      pos.set(-dist * 0.55, -dist * 0.32, dist * 0.85);
    } else {
      // Perspective default (sleek slight grazing perspective like ref_04)
      pos.set(-dist * 0.24, -dist * 0.08, dist * 0.92);
    }

    camera.position.copy(pos);
    camera.lookAt(0, 0, 0);
    if (controlsRef.current) {
      controlsRef.current.target.set(0, 0, 0);
      controlsRef.current.update();
    }
    invalidate();
  }, [preset, trigger, planeW, planeH, camera, controlsRef, invalidate]);

  return null;
};

const SceneContent: React.FC<SceneContentProps> = ({
  texture,
  params,
  displayMode,
  materialId,
  lightAzimuthDeg,
  lightElevationDeg,
  lightIntensity,
  sliceEnabled,
  sliceIntervalMm,
  twoSided,
  showFrameGuide = false,
  cameraPreset,
  cameraTrigger,
  activeTool,
  brushMode,
  brushRadiusPx,
  brushStrength,
  brushLocked = false,
  partsMap = null,
  showLayers = false,
  selectedPartId = null,
  onPartPick,
  profilePointA,
  profilePointB,
  onProfileChange,
  onBrushStrokeStart,
  onBrushDataChange,
  onBrushStrokeEnd,
  segments = 512,
  planeW,
  planeH,
}) => {
  const invalidate = useThree((st) => st.invalidate);
  const hasFloatLinear = useThree((st) => st.gl.extensions.has('OES_texture_float_linear'));
  const controlsRef = useRef<OrbitControlsImpl | null>(null);
  const frontMatRef = useRef<THREE.ShaderMaterial | null>(null);
  const backMatRef = useRef<THREE.ShaderMaterial | null>(null);
  const dataTexRef = useRef<THREE.DataTexture | null>(null);
  // True once the current DataTexture has been uploaded at least once. Before that
  // three allocates zeroed storage and honours update ranges, which would wipe the rest.
  const uploadedRef = useRef(false);

  const isPaintingRef = useRef(false);
  const isProfilingRef = useRef(false);
  const hoverUvRef = useRef<{ uv: THREE.Vector2; active: boolean }>({
    uv: new THREE.Vector2(0.5, 0.5),
    active: false,
  });
  const flattenTargetRef = useRef<number | null>(null);
  const lastPaintPxRef = useRef<{ x: number; y: number } | null>(null);

  const texWidth = texture?.width ?? 2;
  const texHeight = texture?.height ?? 2;

  // Create or rebuild GPU DataTexture when buffer dimensions change
  const gpuTexture = useMemo(() => {
    const buf =
      texture && texture.data.length === texWidth * texHeight
        ? texture.data
        : new Float32Array(texWidth * texHeight);
    const dt = new THREE.DataTexture(
      buf,
      texWidth,
      texHeight,
      THREE.RedFormat,
      THREE.FloatType
    );
    // Float32 linear filtering needs OES_texture_float_linear; otherwise the shader's
    // manual 4-tap bilinear runs on a nearest-filtered texture.
    const filter = hasFloatLinear ? THREE.LinearFilter : THREE.NearestFilter;
    dt.minFilter = filter;
    dt.magFilter = filter;
    dt.wrapS = THREE.ClampToEdgeWrapping;
    dt.wrapT = THREE.ClampToEdgeWrapping;
    dt.flipY = false;
    uploadedRef.current = false;
    dt.onUpdate = () => {
      uploadedRef.current = true;
    };
    dt.needsUpdate = true;
    dataTexRef.current = dt;
    return dt;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [texWidth, texHeight, texture, hasFloatLinear]);

  // Layers overlay: RGBA8 at label-map size, same uv convention as the height texture.
  const overlayActive = activeTool === 'layers' && partsMap !== null;
  const overlayTex = useMemo(() => {
    let dt: THREE.DataTexture;
    if (overlayActive && partsMap) {
      const data = buildOverlayRGBA(partsMap, {
        showLayers,
        selectedId: selectedPartId,
      });
      dt = new THREE.DataTexture(data, partsMap.width, partsMap.height, THREE.RGBAFormat);
    } else {
      dt = new THREE.DataTexture(new Uint8Array(4), 1, 1, THREE.RGBAFormat);
    }
    dt.minFilter = THREE.NearestFilter;
    dt.magFilter = THREE.NearestFilter;
    dt.wrapS = THREE.ClampToEdgeWrapping;
    dt.wrapT = THREE.ClampToEdgeWrapping;
    dt.flipY = false;
    dt.needsUpdate = true;
    return dt;
  }, [overlayActive, partsMap, showLayers, selectedPartId]);

  useEffect(() => {
    return () => {
      overlayTex.dispose();
    };
  }, [overlayTex]);

  const shaderDefines = useMemo(
    () => (hasFloatLinear ? { USE_HW_LINEAR: '' } : {}),
    [hasFloatLinear]
  );

  useEffect(() => {
    return () => {
      gpuTexture.dispose();
    };
  }, [gpuTexture]);

  const materialSpec = useMemo(
    () =>
      MATERIAL_SPHERES.find((m) => m.id === materialId) ?? MATERIAL_SPHERES[1],
    [materialId]
  );

  const displayInt =
    displayMode === 'solid'
      ? 0
      : displayMode === 'clay'
      ? 1
      : displayMode === 'curvature'
      ? 2
      : 3;

  const mmToWorld = planeW / Math.max(10, params.physical_width_mm);
  const baseWorld = Math.max(0, params.base_z_mm) * mmToWorld;
  const reliefWorld = Math.max(0.5, params.relief_z_mm) * mmToWorld;

  const lightVec = useMemo(() => {
    const az = (lightAzimuthDeg * Math.PI) / 180;
    const el = (lightElevationDeg * Math.PI) / 180;
    return new THREE.Vector3(
      Math.cos(el) * Math.cos(az),
      Math.cos(el) * Math.sin(az),
      Math.sin(el)
    ).normalize();
  }, [lightAzimuthDeg, lightElevationDeg]);

  const segW = Math.min(segments, Math.max(128, texWidth));
  const segH = Math.max(
    64,
    Math.round(segW * (texHeight / Math.max(1, texWidth)))
  );

  const brushRadiusUv = useMemo(() => {
    return Math.max(0.008, brushRadiusPx / Math.max(1, texWidth));
  }, [brushRadiusPx, texWidth]);

  const createUniforms = (flipMirror: number) => ({
    uHeightMap: { value: gpuTexture },
    uTexSize: { value: new THREE.Vector2(texWidth, texHeight) },
    uBaseZ: { value: baseWorld },
    uReliefZ: { value: reliefWorld },
    uPhysicalWidthMm: { value: params.physical_width_mm },
    uReliefZMm: { value: params.relief_z_mm },
    uBaseZMm: { value: params.base_z_mm },
    uPlaneWidth: { value: planeW },
    uPlaneHeight: { value: planeH },
    uDisplayMode: { value: displayInt },
    uMatBaseColor: { value: new THREE.Color(...materialSpec.baseColor) },
    uMatSpecColor: { value: new THREE.Color(...materialSpec.specColor) },
    uMatMetalness: { value: materialSpec.metalness },
    uMatRoughness: { value: materialSpec.roughness },
    uLightDir: { value: lightVec },
    uLightIntensity: { value: lightIntensity },
    uSliceEnabled: { value: sliceEnabled ? 1.0 : 0.0 },
    uSliceIntervalMm: { value: sliceIntervalMm },
    uShowBrush: { value: 0.0 },
    uBrushUv: { value: new THREE.Vector2(0.5, 0.5) },
    uBrushRadiusUv: { value: brushRadiusUv },
    uShowFrameBox: { value: showFrameGuide && params.frame.enabled ? 1.0 : 0.0 },
    uFrameBbox: { value: new THREE.Vector4(...params.frame.bbox) },
    uFlipMirror: { value: flipMirror },
    uOverlayMap: { value: overlayTex },
    uOverlaySize: { value: new THREE.Vector2(overlayTex.image.width, overlayTex.image.height) },
    uShowOverlay: { value: overlayActive ? 1.0 : 0.0 },
  });

  const frontUniforms = useMemo(
    () => createUniforms(0.0),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [gpuTexture]
  );
  const backUniforms = useMemo(
    () => createUniforms(1.0),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [gpuTexture]
  );

  // Sync dynamic shader uniforms smoothly every change without recompiling shaders
  useFrame(() => {
    const syncUniforms = (mat: THREE.ShaderMaterial | null, isMirror: boolean) => {
      if (!mat) return;
      const u = mat.uniforms;
      u.uHeightMap.value = gpuTexture;
      u.uTexSize.value.set(texWidth, texHeight);
      u.uBaseZ.value = baseWorld;
      u.uReliefZ.value = reliefWorld;
      u.uPhysicalWidthMm.value = params.physical_width_mm;
      u.uReliefZMm.value = params.relief_z_mm;
      u.uBaseZMm.value = params.base_z_mm;
      u.uPlaneWidth.value = planeW;
      u.uPlaneHeight.value = planeH;
      u.uDisplayMode.value = displayInt;
      u.uMatBaseColor.value.setRGB(...materialSpec.baseColor);
      u.uMatSpecColor.value.setRGB(...materialSpec.specColor);
      u.uMatMetalness.value = materialSpec.metalness;
      u.uMatRoughness.value = materialSpec.roughness;
      u.uLightDir.value.copy(lightVec);
      u.uLightIntensity.value = lightIntensity;
      u.uSliceEnabled.value = sliceEnabled ? 1.0 : 0.0;
      u.uSliceIntervalMm.value = sliceIntervalMm;
      u.uShowBrush.value =
        !isMirror && activeTool === 'brush' && hoverUvRef.current.active
          ? 1.0
          : 0.0;
      u.uBrushUv.value.copy(hoverUvRef.current.uv);
      u.uBrushRadiusUv.value = brushRadiusUv;
      u.uShowFrameBox.value =
        !isMirror && showFrameGuide && params.frame.enabled ? 1.0 : 0.0;
      u.uFrameBbox.value.set(...params.frame.bbox);
      u.uOverlayMap.value = overlayTex;
      u.uOverlaySize.value.set(overlayTex.image.width, overlayTex.image.height);
      u.uShowOverlay.value = !isMirror && overlayActive ? 1.0 : 0.0;
    };

    syncUniforms(frontMatRef.current, false);
    syncUniforms(backMatRef.current, true);
  });

  // frameloop="demand": every input that changes the picture must request a frame.
  // (OrbitControls and R3F's own prop reconciliation invalidate by themselves.)
  useEffect(() => {
    invalidate();
  }, [
    invalidate, gpuTexture, texture, texWidth, texHeight, baseWorld, reliefWorld, params,
    planeW, planeH, displayInt, materialSpec, lightVec, lightIntensity, sliceEnabled,
    sliceIntervalMm, showFrameGuide, brushRadiusUv, activeTool, twoSided, segW, segH,
    displayMode, profilePointA, profilePointB, hasFloatLinear, overlayTex, overlayActive,
  ]);

  // Apply 60 FPS Float32Array sculpting stroke directly onto buffer
  const applyBrushAt = (cx: number, cy: number, weight: number) => {
    if (!texture || !dataTexRef.current) return;
    const { data, width, height } = texture;
    const rad = Math.max(2, brushRadiusPx);
    const r2 = rad * rad;

    // Full texture is reachable: the smooth kernel reads an edge-clamped snapshot patch.
    const xMin = Math.max(0, Math.floor(cx - rad));
    const xMax = Math.min(width - 1, Math.ceil(cx + rad));
    const yMin = Math.max(0, Math.floor(cy - rad));
    const yMax = Math.min(height - 1, Math.ceil(cy + rad));

    if (brushMode === 'flatten' && flattenTargetRef.current === null) {
      const ix = Math.min(width - 1, Math.max(0, Math.round(cx)));
      const iy = Math.min(height - 1, Math.max(0, Math.round(cy)));
      flattenTargetRef.current = data[iy * width + ix] ?? 0;
    }

    const strength = Math.min(1, Math.max(0.02, brushStrength));
    let snapshotForSmooth: Float32Array | null = null;
    if (brushMode === 'smooth') {
      // Copy local bounding patch to avoid directional smearing
      const patchW = xMax - xMin + 3;
      const patchH = yMax - yMin + 3;
      snapshotForSmooth = new Float32Array(patchW * patchH);
      for (let py = 0; py < patchH; py++) {
        const sy = Math.min(height - 1, Math.max(0, yMin - 1 + py));
        const rowOff = sy * width;
        const dstOff = py * patchW;
        for (let px = 0; px < patchW; px++) {
          const sx = Math.min(width - 1, Math.max(0, xMin - 1 + px));
          snapshotForSmooth[dstOff + px] = data[rowOff + sx];
        }
      }
      for (let y = yMin; y <= yMax; y++) {
        const dy = y - cy;
        for (let x = xMin; x <= xMax; x++) {
          const dx = x - cx;
          const d2 = dx * dx + dy * dy;
          if (d2 > r2) continue;
          const falloff = Math.exp((-2.8 * d2) / r2);
          const py = y - (yMin - 1);
          const px = x - (xMin - 1);
          const avg =
            (snapshotForSmooth[(py - 1) * patchW + px] +
              snapshotForSmooth[(py + 1) * patchW + px] +
              snapshotForSmooth[py * patchW + (px - 1)] +
              snapshotForSmooth[py * patchW + (px + 1)] +
              snapshotForSmooth[py * patchW + px] * 2) /
            6;
          const idx = y * width + x;
          const k = Math.min(1, falloff * strength * 0.65 * weight);
          data[idx] = data[idx] + (avg - data[idx]) * k;
        }
      }
    } else {
      const targetFlat = flattenTargetRef.current ?? 0;
      for (let y = yMin; y <= yMax; y++) {
        const dy = y - cy;
        for (let x = xMin; x <= xMax; x++) {
          const dx = x - cx;
          const d2 = dx * dx + dy * dy;
          if (d2 > r2) continue;
          const falloff = Math.exp((-3.0 * d2) / r2);
          const idx = y * width + x;
          let nextVal = data[idx];
          if (brushMode === 'inflate') {
            nextVal += falloff * strength * 0.045 * weight;
          } else if (brushMode === 'deflate') {
            nextVal -= falloff * strength * 0.045 * weight;
          } else if (brushMode === 'flatten') {
            nextVal += (targetFlat - nextVal) * Math.min(1, falloff * strength * 0.38 * weight);
          }
          data[idx] = Math.min(1, Math.max(0, nextVal));
        }
      }
    }

    // Upload only the touched span of each row (three's range merge assumes RGBA, so
    // ranges are expressed in 4-component units: pixel p -> start 4p).
    const dt = dataTexRef.current;
    for (let y = yMin; uploadedRef.current && y <= yMax; y++) {
      dt.addUpdateRange((y * width + xMin) * 4, (xMax - xMin + 1) * 4);
    }
    dt.needsUpdate = true;
    invalidate();
    onBrushDataChange?.(data);
  };

  // Pick the displaced surface (not the flat plane the raycaster hits).
  const pickUv = (e: ThreeEvent<PointerEvent>): { u: number; v: number } | null => {
    if (!texture) return null;
    const inv = new THREE.Matrix4().copy(e.object.matrixWorld).invert();
    const o = e.ray.origin.clone().applyMatrix4(inv);
    const d = e.ray.direction.clone().transformDirection(inv);
    const stepWorld =
      Math.min(planeW / Math.max(1, texWidth), planeH / Math.max(1, texHeight)) * 0.5;
    return pickHeightfield(
      [o.x, o.y, o.z],
      [d.x, d.y, d.z],
      { planeW, planeH, baseZ: baseWorld, reliefZ: reliefWorld },
      (u, v) => sampleFloat32Bilinear(texture, u, v),
      stepWorld
    );
  };

  const paintTo = (uv: { u: number; v: number }) => {
    if (!texture) return;
    const next = {
      x: uv.u * (texture.width - 1),
      y: (1 - uv.v) * (texture.height - 1),
    };
    const spacing = Math.max(1, brushRadiusPx / 4);
    for (const st of strokeStamps(lastPaintPxRef.current, next, spacing)) {
      applyBrushAt(st.x, st.y, st.weight);
    }
    lastPaintPxRef.current = next;
  };

  const handlePointerDown = (e: ThreeEvent<PointerEvent>) => {
    if (activeTool === 'orbit') return;
    if (e.button !== 0) return;
    if ((activeTool === 'brush' || activeTool === 'layers') && brushLocked) return;
    const hit = pickUv(e);
    if (!hit) return;
    e.stopPropagation();

    if (activeTool === 'layers') {
      if (!partsMap) return;
      const label = partAtUv(partsMap, hit.u, hit.v);
      const known = partsMap.parts.some((p) => p.id === label);
      onPartPick?.(known ? label : null, hit);
      return;
    }

    if (activeTool === 'profile') {
      isProfilingRef.current = true;
      const pt: ProfilePoint = { u: hit.u, v: hit.v };
      onProfileChange?.(pt, pt);
    } else if (activeTool === 'brush' && texture) {
      isPaintingRef.current = true;
      flattenTargetRef.current = null;
      lastPaintPxRef.current = null;
      onBrushStrokeStart?.(new Float32Array(texture.data));
      hoverUvRef.current.uv.set(hit.u, hit.v);
      hoverUvRef.current.active = true;
      invalidate();
      paintTo(hit);
    }
  };

  const handlePointerMove = (e: ThreeEvent<PointerEvent>) => {
    const hit = pickUv(e);
    if (!hit) {
      hoverUvRef.current.active = false;
      lastPaintPxRef.current = anchorAfterPick(lastPaintPxRef.current, false);
      if (activeTool === 'brush') invalidate();
      return;
    }
    hoverUvRef.current.uv.set(hit.u, hit.v);
    hoverUvRef.current.active = true;
    if (activeTool === 'brush') invalidate(); // the brush ring is the only hover-dependent visual

    if (activeTool === 'profile' && isProfilingRef.current && profilePointA) {
      e.stopPropagation();
      onProfileChange?.(profilePointA, { u: hit.u, v: hit.v });
    } else if (activeTool === 'brush' && isPaintingRef.current) {
      e.stopPropagation();
      paintTo(hit);
    }
  };

  const handlePointerUp = () => {
    if (isProfilingRef.current) {
      isProfilingRef.current = false;
    }
    if (isPaintingRef.current) {
      isPaintingRef.current = false;
      flattenTargetRef.current = null;
      lastPaintPxRef.current = null;
      if (texture) {
        onBrushStrokeEnd?.(texture.data);
      }
    }
  };

  useEffect(() => {
    const onWinUp = () => handlePointerUp();
    window.addEventListener('pointerup', onWinUp);
    return () => window.removeEventListener('pointerup', onWinUp);
  });

  // 3D Profile laser line points lifted cleanly onto the displaced surface
  const profileLinePoints = useMemo(() => {
    if (!texture || !profilePointA || !profilePointB) return null;
    const pts: [number, number, number][] = [];
    const steps = 64;
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      const u = profilePointA.u + (profilePointB.u - profilePointA.u) * t;
      const v = profilePointA.v + (profilePointB.v - profilePointA.v) * t;
      const h = sampleFloat32Bilinear(texture, u, v);
      const x = (u - 0.5) * planeW;
      const y = (v - 0.5) * planeH;
      const z = baseWorld + h * reliefWorld + 0.035;
      pts.push([x, y, z]);
    }
    return pts;
  }, [texture, profilePointA, profilePointB, planeW, planeH, baseWorld, reliefWorld]);

  const slabColor = useMemo(() => {
    if (displayMode === 'normal') return '#6e72a8';
    if (displayMode === 'clay') return '#d4d8df';
    if (displayMode === 'curvature') return '#949ca8';
    const [r, g, b] = materialSpec.baseColor;
    return `rgb(${Math.round(r * 195)}, ${Math.round(g * 195)}, ${Math.round(b * 195)})`;
  }, [displayMode, materialSpec]);

  const slabDepth = Math.max(0.03, baseWorld);

  return (
    <>
      <CameraRig
        preset={cameraPreset}
        trigger={cameraTrigger}
        planeW={planeW}
        planeH={planeH}
        controlsRef={controlsRef}
      />

      <OrbitControls
        ref={controlsRef}
        makeDefault
        enableDamping
        dampingFactor={0.08}
        enabled={activeTool === 'orbit'}
        minDistance={Math.max(planeW, planeH) * 0.3}
        maxDistance={Math.max(planeW, planeH) * 4.5}
      />

      <ambientLight intensity={0.65} />
      <directionalLight position={[4, 6, 8]} intensity={0.8} />

      <group>
        {/* Front 32-bit Float Displaced Mesh */}
        <mesh
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerLeave={() => {
            hoverUvRef.current.active = false;
            lastPaintPxRef.current = anchorAfterPick(lastPaintPxRef.current, false);
            if (activeTool === 'brush') invalidate();
          }}
        >
          <planeGeometry args={[planeW, planeH, segW, segH]} />
          <shaderMaterial
            ref={frontMatRef}
            vertexShader={VERTEX_SHADER}
            fragmentShader={FRAGMENT_SHADER}
            defines={shaderDefines}
            uniforms={frontUniforms}
          />
        </mesh>

        {/* Solid Slab Perimeter / Backing Block */}
        <mesh position={[0, 0, baseWorld * 0.5 - 0.004]}>
          <boxGeometry args={[planeW, planeH, slabDepth]} />
          <meshStandardMaterial
            color={slabColor}
            roughness={materialSpec.roughness}
            metalness={materialSpec.metalness * 0.5}
          />
        </mesh>

        {/* Optional 2-Sided Back Mirror Relief */}
        {twoSided && (
          <mesh rotation={[0, Math.PI, 0]} position={[0, 0, -0.008]}>
            <planeGeometry args={[planeW, planeH, segW, segH]} />
            <shaderMaterial
              ref={backMatRef}
              vertexShader={VERTEX_SHADER}
              fragmentShader={FRAGMENT_SHADER}
              defines={shaderDefines}
              uniforms={backUniforms}
            />
          </mesh>
        )}

        {/* 3D Cross-Section Laser Ruler */}
        {profileLinePoints && profileLinePoints.length >= 2 && (
          <group>
            <Line
              points={profileLinePoints}
              color="#38bdf8"
              lineWidth={3}
            />
            <mesh position={profileLinePoints[0]}>
              <sphereGeometry args={[0.08, 16, 16]} />
              <meshBasicMaterial color="#38bdf8" />
            </mesh>
            <mesh position={profileLinePoints[profileLinePoints.length - 1]}>
              <sphereGeometry args={[0.08, 16, 16]} />
              <meshBasicMaterial color="#f59e0b" />
            </mesh>
          </group>
        )}
      </group>
    </>
  );
};

export const ReliefViewport3D: React.FC<ReliefViewport3DProps> = (props) => {
  const { texture, backdropDark = false } = props;
  const texW = texture?.width ?? 512;
  const texH = texture?.height ?? 625;
  const planeW = 10.0;
  const planeH = 10.0 * (texH / Math.max(1, texW));

  return (
    <div
      className={`relative h-full w-full select-none ${
        props.activeTool === 'brush' || props.activeTool === 'profile' || props.activeTool === 'layers'
          ? 'cursor-crosshair'
          : 'cursor-grab active:cursor-grabbing'
      }`}
      style={{
        background: backdropDark
          ? 'radial-gradient(circle at 50% 40%, #232834 0%, #181A20 75%)'
          : 'radial-gradient(circle at 50% 40%, #eef1f5 0%, #dde2ea 80%)',
      }}
    >
      <Canvas
        dpr={[1, 2]}
        frameloop="demand"
        camera={{ fov: 36, near: 0.1, far: 250, position: [-3.2, -1.0, 15.0] }}
        gl={{ antialias: true, powerPreference: 'high-performance' }}
      >
        <SceneContent {...props} planeW={planeW} planeH={planeH} />
      </Canvas>
    </div>
  );
};
