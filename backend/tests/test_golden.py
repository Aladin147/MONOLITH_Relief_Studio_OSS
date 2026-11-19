"""Output fingerprint of the benchmark relief.

A silent library upgrade that changes the heightmap would become a physical
difference in a carved sculpture. This pins summary statistics of the benchmark
build. Regenerate deliberately with ``MONOLITH_UPDATE_GOLDEN=1``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from backend import db
from backend import main as api_main

GOLDEN_PATH = Path(__file__).resolve().parent / "golden" / "benchmark_1024.json"
AI_MODES = ("off", "guided_overlap")
FLOAT_KEYS = ("mean", "std", "p01", "p50", "p99")
TOLERANCE = 1e-5


def _fingerprint(ai_mode: str, image=None, style: str = "classic") -> dict:
    full_layers, h_current = api_main._build_layers_from_image(
        image or api_main.BENCHMARK_IMAGE_PATH,
        target_res=1024,
        bg_removal=True,
        hd_restore=True,
        ai_mode=ai_mode,
        params={**db.default_relief_params(), "style": style},
    )
    sil = np.asarray(full_layers["M_sil"]) > 0.5
    inside = np.asarray(h_current, dtype=np.float64)[sil]
    p01, p50, p99 = np.percentile(inside, [1, 50, 99])
    return {
        "sil_pixels": int(sil.sum()),
        "mean": round(float(inside.mean()), 8),
        "std": round(float(inside.std()), 8),
        "p01": round(float(p01), 8),
        "p50": round(float(p50), 8),
        "p99": round(float(p99), 8),
    }


def test_benchmark_output_fingerprint():
    current = {mode: _fingerprint(mode) for mode in AI_MODES}
    current["sculpt"] = _fingerprint("off", style="sculpt")

    if os.environ.get("MONOLITH_UPDATE_GOLDEN") == "1":
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(
            json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return

    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    for mode in current:
        assert current[mode]["sil_pixels"] == golden[mode]["sil_pixels"], mode
        for key in FLOAT_KEYS:
            assert abs(current[mode][key] - golden[mode][key]) <= TOLERANCE, (
                f"{mode}.{key}: {current[mode][key]} vs golden {golden[mode][key]}"
            )
