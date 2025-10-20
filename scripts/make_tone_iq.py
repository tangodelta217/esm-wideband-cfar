#!/usr/bin/env python
"""Generate a complex float32 IQ recording with a tone immersed in noise."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def main() -> None:
    sample_rate = 2_000_000  # 2 MS/s
    duration_s = 5.0
    tone_freq = 75_000.0  # 75 kHz offset
    noise_power = 0.05
    amplitude = 0.8
    seed = 2024

    output_dir = Path("data/iq_examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    iq_path = output_dir / "tone_2msps_cf32.iq"

    num_samples = int(sample_rate * duration_s)
    time = np.arange(num_samples, dtype=np.float64) / sample_rate
    tone = amplitude * np.exp(1j * 2.0 * np.pi * tone_freq * time)

    rng = np.random.default_rng(seed)
    noise = (
        rng.standard_normal(num_samples) + 1j * rng.standard_normal(num_samples)
    ) * np.sqrt(noise_power / 2.0)

    iq = np.asarray(tone + noise, dtype=np.complex64)
    iq.view(np.float32).tofile(iq_path)

    print(f"Wrote {iq_path} with {num_samples} samples at {sample_rate} Hz")


if __name__ == "__main__":
    main()

