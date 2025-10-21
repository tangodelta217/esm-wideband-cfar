#!/usr/bin/env python
"""Herramienta de línea de comando para evaluar el detector CA-CFAR."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

try:
    from eval.metrics import monte_carlo_pd_pfa
    from src.cfar import ca_cfar_1d
except ImportError:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from eval.metrics import monte_carlo_pd_pfa
    from src.cfar import ca_cfar_1d


def evaluate_pfa(
    *,
    num_samples: int = 2048,
    num_guard: int = 4,
    trials: int = 200_000,
    train_cells: tuple[int, ...] = (8, 12, 16),
    pfa_targets: tuple[float, ...] = (1e-2, 1e-3, 1e-4),
) -> None:
    """Compara Pfa teórica vs. empírica para diferentes configuraciones."""
    rng = np.random.default_rng(42)
    print("### Evaluación de Probabilidad de Falsa Alarma (Pfa)")
    print(f"Iteraciones por configuración: {trials:,}, NFFT: {num_samples}")
    print("| N Train | Pfa teórica | Pfa empírica | Límite 95% (±) | Resultado |")
    print("|:-------:|:-----------:|:------------:|:--------------:|:---------:|")

    for train_per_side in train_cells:
        border = 2 * (train_per_side + num_guard)
        valid_cells = num_samples - border
        if valid_cells <= 0:
            print(f"| {2*train_per_side:<7} | - | - | - | N/A |")
            continue

        for pfa in pfa_targets:
            false_alarms = 0
            remaining = trials
            batch_size = min(trials, 10_000)
            while remaining > 0:
                current = min(batch_size, remaining)
                noise = (
                    rng.standard_normal((current, num_samples))
                    + 1j * rng.standard_normal((current, num_samples))
                ) / math.sqrt(2.0)
                power = np.abs(noise) ** 2
                for row in power:
                    det, _ = ca_cfar_1d(
                        row,
                        num_train=train_per_side,
                        num_guard=num_guard,
                        pfa=pfa,
                    )
                    false_alarms += int(np.count_nonzero(det))
                remaining -= current

            total_cells = trials * valid_cells
            empirical = false_alarms / total_cells
            margin = 1.96 * math.sqrt((pfa * (1 - pfa)) / total_cells)
            status = "✅" if abs(empirical - pfa) <= margin else "❌"
            print(
                f"| {2*train_per_side:<7} | {pfa:<11.0e} | "
                f"{empirical:<12.4e} | {margin:<12.2e} | {status:^9} |"
            )


def evaluate_pd(
    *,
    snr_start: float = -10.0,
    snr_stop: float = 20.0,
    snr_step: float = 2.5,
    trials: int = 2_000,
    num_samples: int = 2048,
    num_train: int = 16,
    num_guard: int = 4,
    pfa: float = 1e-4,
) -> None:
    """Calcula Pd vs. SNR usando simulaciones Monte Carlo."""
    snrs = np.arange(snr_start, snr_stop + snr_step, snr_step)
    results = monte_carlo_pd_pfa(
        snrs,
        trials,
        num_samples=num_samples,
        num_train=num_train,
        num_guard=num_guard,
        pfa=pfa,
    )
    print("\n### Probabilidad de Detección (Pd) vs. SNR")
    print(f"Configuración: NFFT={num_samples}, Train={num_train}, Guard={num_guard}, Pfa={pfa:.1e}")
    print("| SNR (dB) | Pd (%) | Pfa empírica | Trials |")
    print("|:--------:|:------:|:------------:|:------:|")
    for row in results:
        print(
            f"| {row.snr_db:>8.1f} | {row.pd*100:6.2f} | "
            f"{row.pfa_empirical:>12.4e} | {row.trials:>6} |"
        )


def measure_latency(
    *,
    fft_sizes: tuple[int, ...] = (1024, 2048, 4096, 8192, 16_384),
    iterations: int = 1_000,
) -> None:
    """Mide latencia media de ca_cfar_1d para distintos tamaños de FFT."""
    rng = np.random.default_rng(77)
    print("\n### Medición de Latencia Algorítmica")
    print(f"Iteraciones por tamaño: {iterations:,}")
    print("| NFFT  | Tiempo medio (µs) |")
    print("|:------|:-----------------:|")
    for nfft in fft_sizes:
        power = np.abs((rng.standard_normal(nfft) + 1j * rng.standard_normal(nfft)) / math.sqrt(2.0)) ** 2
        ca_cfar_1d(power)  # warm-up
        start = time.perf_counter()
        for _ in range(iterations):
            ca_cfar_1d(power)
        elapsed = time.perf_counter() - start
        avg_us = (elapsed / iterations) * 1e6
        print(f"| {nfft:<5} | {avg_us:>17.2f} |")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluación empírica del detector CA-CFAR.")
    parser.add_argument(
        "experiment",
        choices=["pfa", "pd", "latency", "all"],
        nargs="?",
        default="all",
        help="Experimento a ejecutar (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.experiment in ("pfa", "all"):
        evaluate_pfa()
    if args.experiment in ("pd", "all"):
        evaluate_pd()
    if args.experiment in ("latency", "all"):
        measure_latency()


if __name__ == "__main__":
    main()
