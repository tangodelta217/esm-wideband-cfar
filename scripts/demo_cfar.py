#!/usr/bin/env python
"""Command-line demo that runs CA-CFAR on synthetic data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from src.cfar import ca_cfar_1d
except ImportError:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.cfar import ca_cfar_1d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CA-CFAR on synthetic wideband data.")
    parser.add_argument("--snr-db", type=float, default=10.0, help="Signal-to-noise ratio in dB.")
    parser.add_argument("--samples", type=int, default=2048, help="Number of complex samples.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    noise = (rng.standard_normal(args.samples) + 1j * rng.standard_normal(args.samples)) / np.sqrt(2)
    signal = np.zeros_like(noise)
    signal_index = args.samples // 2
    snr_linear = 10 ** (args.snr_db / 10.0)
    signal_power = snr_linear
    signal[signal_index] = np.sqrt(signal_power)

    iq = noise + signal
    power = np.abs(iq) ** 2
    detections, thresholds = ca_cfar_1d(power)
    hits = np.nonzero(detections)[0]

    print(f"Injected target @ {signal_index}, detected bins: {hits.tolist()}")
    if hits.size:
        for idx in hits:
            print(f"  bin {idx}: power={power[idx]:.3f}, threshold={thresholds[idx]:.3f}")
    else:
        print("No detections; consider increasing --snr-db.")


if __name__ == "__main__":
    main()
