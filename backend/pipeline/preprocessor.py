"""Stage 1: Line-art super-sampling, HD restoration, and T-junction-safe Tri-Map extraction."""

from __future__ import annotations

import io
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

ImageSource = str | Path | bytes | bytearray | memoryview | np.ndarray | Image.Image

# Decompression-bomb guard: refuse images whose header declares more pixels than this.
# 1000 MP matches OpenCV's own decode limit; large sources in gray-capable modes are area-shrunk in
# uint8 (below) to keep memory bounded.
MAX_SOURCE_PIXELS = int(os.environ.get("MONOLITH_MAX_SOURCE_MEGAPIXELS", "1000")) * 1_000_000
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(MAX_SOURCE_PIXELS))

# Every other mode (RGBA, LA, P, 16-bit, ...) is decoded in full and converted to 4-channel float32
# (16-24 bytes/pixel at peak), so it gets a much lower cap.
MAX_FULL_DECODE_PIXELS = int(os.environ.get("MONOLITH_MAX_FULL_DECODE_MEGAPIXELS", "400")) * 1_000_000

_EXIF_ORIENTATION_TAG = 0x0112

# Sources above this many pixels are pre-shrunk in uint8 before the float32 conversion
# (which would otherwise cost 4 bytes/pixel on top of the decode).
LARGE_SOURCE_PIXELS = int(os.environ.get("MONOLITH_LARGE_SOURCE_MEGAPIXELS", "64")) * 1_000_000

# PIL modes that can be decoded straight to an 8-bit grayscale array. Alpha, 16-bit and
# palette modes (transparency) keep the full IMREAD_UNCHANGED path.
_REDUCIBLE_MODES = frozenset({"1", "L", "RGB", "CMYK"})

_pil_limit_lock = threading.Lock()


@dataclass(frozen=True)
class SourceInfo:
    orientation: int  # EXIF orientation, 1 = upright
    width: int  # 0 when the format could not be identified
    height: int
    mode: str | None  # PIL mode, None when unidentified


def _open_header(source: Path | bytes) -> Image.Image:
    """Open an image with PIL's own bomb check off (the size is checked by the caller).

    The MAX_IMAGE_PIXELS swap is only safe because every PIL open in backend/ goes
    through this lock-protected helper.
    """
    with _pil_limit_lock:
        previous = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = None
        try:
            return Image.open(source if isinstance(source, Path) else io.BytesIO(source))
        finally:
            Image.MAX_IMAGE_PIXELS = previous


def _probe_header(source: Path | bytes) -> SourceInfo:
    """Read size/mode/orientation from the header and enforce the declared pixel limit."""
    try:
        handle = _open_header(source)
    except Exception:
        return SourceInfo(1, 0, 0, None)  # Let OpenCV try (and report) formats PIL cannot open.
    with handle as img:
        w, h = img.size
        if w * h > MAX_SOURCE_PIXELS:
            raise ValueError(
                f"Image is {w}x{h} ({w * h / 1e6:.0f} MP), above the "
                f"{MAX_SOURCE_PIXELS / 1e6:.0f} MP safety limit"
            )
        if img.mode not in _REDUCIBLE_MODES and w * h > MAX_FULL_DECODE_PIXELS:
            raise ValueError(
                f"Image is {w}x{h} ({w * h / 1e6:.0f} MP) in mode {img.mode}, above the "
                f"{MAX_FULL_DECODE_PIXELS / 1e6:.0f} MP safety limit for this mode; "
                f"save the drawing as grayscale or RGB without alpha"
            )
        try:
            orientation = int(img.getexif().get(_EXIF_ORIENTATION_TAG, 1) or 1)
        except Exception:
            orientation = 1
        return SourceInfo(orientation, w, h, img.mode)


def _use_large_path(info: SourceInfo, target_res: int | None) -> bool:
    """True when the source is big enough to be pre-shrunk in uint8 (see _load_luminance_float32)."""
    return (
        target_res is not None
        and info.mode in _REDUCIBLE_MODES
        and info.width * info.height > LARGE_SOURCE_PIXELS
        and max(info.width, info.height) > target_res
    )


def _imread_flag(info: SourceInfo, target_res: int | None) -> int:
    if _use_large_path(info, target_res):
        # 8-bit grayscale (w*h bytes); orientation is applied once, explicitly, afterwards.
        return cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION
    return cv2.IMREAD_UNCHANGED  # keeps 16-bit depth and alpha


def _preshrink(arr: np.ndarray, target_res: int) -> np.ndarray:
    """Area-downsample a uint8 grayscale array so its long edge is target_res (never below)."""
    h, w = arr.shape[:2]
    scale = target_res / float(max(h, w))
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(arr, size, interpolation=cv2.INTER_AREA)


def _apply_exif_orientation(arr: np.ndarray, orientation: int) -> np.ndarray:
    """Rotate/flip a decoded (H, W[, C]) array upright, matching PIL.ImageOps.exif_transpose."""
    if orientation == 2:
        out = arr[:, ::-1]
    elif orientation == 3:
        out = arr[::-1, ::-1]
    elif orientation == 4:
        out = arr[::-1, :]
    elif orientation == 5:
        out = arr.swapaxes(0, 1)
    elif orientation == 6:
        out = np.rot90(arr, k=-1)
    elif orientation == 7:
        out = np.rot90(arr, k=2).swapaxes(0, 1)
    elif orientation == 8:
        out = np.rot90(arr, k=1)
    else:
        return arr
    return np.ascontiguousarray(out)


def _load_luminance_float32(image_source: ImageSource, target_res: int | None = None) -> np.ndarray:
    """Load any supported image source directly into [0.0, 1.0] float32 luminance."""
    arr: np.ndarray | None = None
    is_bgr = False

    if isinstance(image_source, (str, Path)):
        path = Path(image_source)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")
        info = _probe_header(path)
        orientation = info.orientation
        # cv2 UNCHANGED ignores EXIF orientation, so phone photos of drawings are rotated
        # upright explicitly. Large scans decode to uint8 grayscale and are area-shrunk
        # before the float32 conversion (peak memory is still about w*h bytes).
        arr = cv2.imread(str(path), _imread_flag(info, target_res))
        if arr is not None and _use_large_path(info, target_res):
            arr = _preshrink(arr, target_res)
        if arr is None:
            try:
                with _open_header(path) as pil_img:
                    arr = np.asarray(ImageOps.exif_transpose(pil_img))
            except OSError as exc:  # includes PIL.UnidentifiedImageError
                raise ValueError(f"Unable to decode image file {path.name}: {exc}") from exc
        else:
            is_bgr = True
            arr = _apply_exif_orientation(arr, orientation)
    elif isinstance(image_source, (bytes, bytearray, memoryview)):
        raw = bytes(image_source)
        info = _probe_header(raw)
        orientation = info.orientation
        buf = np.frombuffer(raw, dtype=np.uint8)
        arr = cv2.imdecode(buf, _imread_flag(info, target_res))
        if arr is None:
            raise ValueError("Unable to decode image bytes")
        if _use_large_path(info, target_res):
            arr = _preshrink(arr, target_res)
        is_bgr = True
        arr = _apply_exif_orientation(arr, orientation)
    elif isinstance(image_source, Image.Image):
        arr = np.asarray(image_source)
    elif isinstance(image_source, np.ndarray):
        arr = image_source
    else:
        raise TypeError(f"Unsupported image_source type: {type(image_source)!r}")

    if arr is None or arr.size == 0:
        raise ValueError("Loaded image is empty")

    # Normalize to float32 [0.0, 1.0] immediately; never keep uint8 inside computation.
    if arr.dtype == np.uint8:
        f = arr.astype(np.float32) / np.float32(255.0)
    elif arr.dtype == np.uint16:
        f = arr.astype(np.float32) / np.float32(65535.0)
    elif np.issubdtype(arr.dtype, np.integer):
        max_val = float(np.iinfo(arr.dtype).max)
        f = arr.astype(np.float32) / np.float32(max_val if max_val > 0 else 1.0)
    else:
        f = arr.astype(np.float32, copy=True)
        if f.max() > 1.0 + 1e-3:
            f /= np.float32(255.0)
    np.clip(f, 0.0, 1.0, out=f)

    if f.ndim == 2:
        return np.ascontiguousarray(f, dtype=np.float32)

    if f.ndim == 3:
        ch = f.shape[2]
        if ch == 1:
            return np.ascontiguousarray(f[:, :, 0], dtype=np.float32)
        if ch == 3:
            code = cv2.COLOR_BGR2GRAY if is_bgr else cv2.COLOR_RGB2GRAY
            gray = cv2.cvtColor(f, code)
            return np.ascontiguousarray(gray, dtype=np.float32)
        if ch == 4:
            alpha = f[:, :, 3]
            rgb = f[:, :, :3]
            code = cv2.COLOR_BGR2GRAY if is_bgr else cv2.COLOR_RGB2GRAY
            gray = cv2.cvtColor(rgb, code)
            # Composite transparent areas onto black background.
            gray = (gray * alpha).astype(np.float32, copy=False)
            return np.ascontiguousarray(gray, dtype=np.float32)

    raise ValueError(f"Unsupported image array shape: {f.shape}")


# Steepness of the HD-restore sigmoid (L_clean = sigmoid(k * (L_blur - 0.5))).
HD_SIGMOID_K = 18.0

POLARITY_AUTO = "auto"
POLARITY_WHITE_ON_BLACK = "white_on_black"  # white shapes, black lines, black background
POLARITY_BLACK_ON_WHITE = "black_on_white"  # classic ink lines on white paper
POLARITIES = (POLARITY_AUTO, POLARITY_WHITE_ON_BLACK, POLARITY_BLACK_ON_WHITE)


def detect_polarity(L: np.ndarray) -> str:
    """Classify the drawing style from the brightness of the canvas border ring."""
    H, W = L.shape
    ring = max(1, int(round(0.02 * min(H, W))))
    border = np.concatenate(
        [L[:ring, :].ravel(), L[-ring:, :].ravel(), L[:, :ring].ravel(), L[:, -ring:].ravel()]
    )
    return POLARITY_BLACK_ON_WHITE if float(np.median(border)) > 0.5 else POLARITY_WHITE_ON_BLACK


def _knock_out_paper(L_clean: np.ndarray, rad: int) -> np.ndarray:
    """
    Turn black-lines-on-white-paper art into the white-shapes-on-black convention by
    blackening the paper connected to the canvas border. Dark strokes are dilated by
    `rad` first so small gaps in the outer contour cannot let the paper flood inside.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
    dark = (L_clean <= np.float32(0.5)).astype(np.uint8)
    open_paper = (1 - cv2.dilate(dark, k)).astype(np.uint8)
    n, labels = cv2.connectedComponents(open_paper, connectivity=4)
    border_ids = np.unique(
        np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    )
    border_ids = border_ids[border_ids > 0]
    if border_ids.size == 0:
        return L_clean
    keep = np.zeros(n, dtype=bool)
    keep[border_ids] = True
    paper_core = keep[labels].astype(np.uint8)
    # Grow back over the band hidden by the stroke dilation, staying on bright pixels.
    paper = (cv2.dilate(paper_core, k) > 0) & (dark == 0)
    out = L_clean.copy()
    out[paper] = np.float32(0.0)
    return out


def preprocess_lineart(
    image_source: ImageSource,
    target_res: int = 4096,
    bg_removal: bool = True,
    hd_restore: bool = True,
    polarity: str = POLARITY_AUTO,
) -> dict[str, Any]:
    """
    Super-sample line art in float32 and extract the T-junction-safe Tri-Map.

    `polarity` selects the drawing convention: "white_on_black" (white shapes separated
    by black lines on a black background), "black_on_white" (ink lines on white paper;
    the border-connected paper becomes background), or "auto" (decided from the border).

    Returns a dictionary containing:
      - L_clean: float32 [0, 1] restored luminance
      - M_bg: float32 {0, 1} outer background + large enclosed holes mask
      - M_sil: float32 {0, 1} subject silhouette mask (1 - M_bg)
      - M_lines: float32 [0, 1] interior contour lines ((1 - L_clean) * M_sil)
      - M_regions: float32 [0, 1] enclosed white body regions (L_clean * M_sil)
      - original_shape: (h0, w0)
      - shape: (H, W)
      - polarity: the resolved drawing convention
    """
    if polarity not in POLARITIES:
        raise ValueError(f"Unsupported polarity {polarity!r}; expected one of {POLARITIES}")
    target_res_int = max(16, int(target_res))
    L_src = _load_luminance_float32(image_source, target_res_int)
    h0, w0 = int(L_src.shape[0]), int(L_src.shape[1])
    original_shape = (h0, w0)

    longest = max(h0, w0)
    scale = float(target_res_int) / float(longest)
    if h0 >= w0:
        H = target_res_int
        W = max(1, int(round(w0 * scale)))
    else:
        W = target_res_int
        H = max(1, int(round(h0 * scale)))

    upscale = float(max(H, W)) / float(longest)
    if (H, W) != (h0, w0):
        # Area resampling when shrinking (cubic aliases thin contour lines and can break
        # them, merging regions); cubic when enlarging.
        interp = cv2.INTER_AREA if upscale < 1.0 else cv2.INTER_CUBIC
        L = cv2.resize(L_src, (W, H), interpolation=interp).astype(np.float32, copy=False)
        np.clip(L, 0.0, 1.0, out=L)
    else:
        L = L_src.copy()

    if hd_restore:
        # Base smoothing scales with the target grid (sigma 1 px at 2K). When the source is
        # enlarged, its pixel stair-steps are `upscale` px wide, so the blur also scales to
        # half a source pixel: that removes jaggies while a 1-source-px line still reaches
        # ~0.32 luminance at its centre, i.e. it stays below the 0.5 threshold.
        sigma = max(0.5, float(max(H, W)) / 2048.0, 0.5 * upscale)
        L_blur = cv2.GaussianBlur(L, (0, 0), sigmaX=sigma, sigmaY=sigma)
        # Sigmoid sharpening around 0.5 with k=18 preserves smooth anti-aliasing.
        z = np.clip(-np.float32(HD_SIGMOID_K) * (L_blur - np.float32(0.5)), np.float32(-40.0), np.float32(40.0))
        L_clean = (np.float32(1.0) / (np.float32(1.0) + np.exp(z))).astype(np.float32, copy=False)
    else:
        L_clean = np.clip(L, 0.0, 1.0).astype(np.float32, copy=False)

    resolved_polarity = detect_polarity(L_clean) if polarity == POLARITY_AUTO else polarity

    if bg_removal:
        # Closing radius only needs to bridge the artist's contour strokes at
        # T-junctions (~0.35% of min dim, 12px at 4K). A wider kernel jumps
        # across narrow background gaps between separate limbs / toes. It scales with
        # resolution (the old 14 px cap halved the bridging at 8K).
        rad = max(3, int(round(0.0035 * float(min(H, W)))))
        if resolved_polarity == POLARITY_BLACK_ON_WHITE:
            L_clean = _knock_out_paper(L_clean, rad)
        M_white = (L_clean > np.float32(0.5)).astype(np.uint8)
        ksize = 2 * rad + 1
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        M_closed = cv2.morphologyEx(M_white, cv2.MORPH_CLOSE, kern)

        dark_closed = (1 - M_closed).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_closed, connectivity=8)

        min_hole_area = 0.005 * float(H * W)
        bg_labels = np.zeros(num_labels, dtype=bool)
        for lbl in range(1, num_labels):
            x = int(stats[lbl, cv2.CC_STAT_LEFT])
            y = int(stats[lbl, cv2.CC_STAT_TOP])
            bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
            bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
            area = float(stats[lbl, cv2.CC_STAT_AREA])
            touches_border = (x == 0) or (y == 0) or ((x + bw) >= W) or ((y + bh) >= H)
            if touches_border or area >= min_hole_area:
                bg_labels[lbl] = True

        seed_bg = bg_labels[labels]

        # Reclaim exterior concave background pockets closed over by M_closed
        # without leaking through thin interior contour lines.
        M_dark = (L_clean <= np.float32(0.5)).astype(np.uint8)
        d_dark = cv2.distanceTransform(M_dark, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        line_thresh = max(2.5, 0.0035 * float(min(H, W)))
        wide_dark = (d_dark > line_thresh).astype(np.uint8)
        reclaim_rad = int(np.ceil(line_thresh)) + 1
        reclaim_ksize = 2 * reclaim_rad + 1
        reclaim_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (reclaim_ksize, reclaim_ksize))
        reclaimable_dark = cv2.bitwise_and(cv2.dilate(wide_dark, reclaim_kern), M_dark)

        candidate_bg = ((reclaimable_dark > 0) | seed_bg).astype(np.uint8)
        n_cc, cc_labels = cv2.connectedComponents(candidate_bg, connectivity=8)
        bg_cc = np.unique(cc_labels[seed_bg])
        bg_cc = bg_cc[bg_cc > 0]
        keep = np.zeros(n_cc, dtype=bool)
        keep[bg_cc] = True
        M_bg_bool = keep[cc_labels]

        # Any dark pixel farther from white paper than a contour stroke can be
        # (rad * 1.15) cannot be an interior line: force it to background so
        # webbing between limbs / toes never survives inside M_sil.
        d_to_white = cv2.distanceTransform(
            (1 - M_white).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        M_bg_bool |= (M_white == 0) & (d_to_white > np.float32(rad * 1.15))

        M_bg = M_bg_bool.astype(np.float32)
        M_sil = (np.float32(1.0) - M_bg).astype(np.float32)
    else:
        M_bg = np.zeros((H, W), dtype=np.float32)
        M_sil = np.ones((H, W), dtype=np.float32)

    M_lines = ((np.float32(1.0) - L_clean) * M_sil).astype(np.float32)
    M_regions = (L_clean * M_sil).astype(np.float32)

    return {
        "L_clean": L_clean,
        "M_bg": M_bg,
        "M_sil": M_sil,
        "M_lines": M_lines,
        "M_regions": M_regions,
        "original_shape": original_shape,
        "shape": (H, W),
        "polarity": resolved_polarity,
    }
