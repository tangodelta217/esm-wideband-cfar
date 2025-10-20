"""Monte Carlo metrics for CFAR detector evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from src.cfar import ca_cfar_1d


@dataclass
class MonteCarloResult:
    """Summary of CFAR performance at a given SNR."""

    snr_db: float
    pd: float
    pfa_empirical: float
    trials: int


def _synthesize_iq(
    rng: np.random.Generator,
    snr_db: float,
    num_samples: int,
    target_bin: int,
) -> np.ndarray:
    noise = (rng.standard_normal(num_samples) + 1j * rng.standard_normal(num_samples)) / np.sqrt(2.0)
    signal = np.zeros(num_samples, dtype=complex)
    snr_linear = 10 ** (snr_db / 10.0)
    signal[target_bin] = np.sqrt(snr_linear)
    return noise + signal


def monte_carlo_pd_pfa(
    snr_values_db: Sequence[float],
    trials: int,
    *,
    num_samples: int = 1024,
    num_train: int = 16,
    num_guard: int = 4,
    pfa: float = 1e-3,
    rng_seed: int | None = None,
) -> List[MonteCarloResult]:
    """Estimate Pd and Pfa empirically for a set of SNR values using CA-CFAR."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if trials <= 0:
        raise ValueError("trials must be positive")

    window = num_guard + num_train
    if num_samples <= 2 * window:
        raise ValueError("num_samples too small for the requested CFAR configuration")

    target_bin = num_samples // 2
    if target_bin <= window or target_bin >= num_samples - window:
        target_bin = window

    rng = np.random.default_rng(rng_seed)
    results: List[MonteCarloResult] = []

    for snr_db in snr_values_db:
        detections = 0
        false_alarms = 0
        background_cells = 0

        for _ in range(trials):
            iq = _synthesize_iq(rng, snr_db, num_samples, target_bin)
            power = np.abs(iq) ** 2
            det, thr = ca_cfar_1d(power, num_train=num_train, num_guard=num_guard, pfa=pfa)
            valid_mask = ~np.isnan(thr)

            if not valid_mask[target_bin]:
                raise RuntimeError("Target bin falls outside CFAR valid region; adjust parameters.")

            detections += int(det[target_bin])

            non_target_mask = valid_mask.copy()
            non_target_mask[target_bin] = False
            background_cells += int(np.count_nonzero(non_target_mask))
            false_alarms += int(np.count_nonzero(det & non_target_mask))

        pd = detections / trials
        pfa_empirical = false_alarms / background_cells if background_cells else 0.0
        results.append(MonteCarloResult(snr_db=float(snr_db), pd=pd, pfa_empirical=pfa_empirical, trials=trials))

    return results
