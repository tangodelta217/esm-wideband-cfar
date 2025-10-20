"""Script for empirical evaluation of CFAR performance and latency."""

import argparse
import sys
import time
from math import sqrt
from pathlib import Path

# Add repo root to path to allow importing `src`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from tqdm import tqdm

from src.cfar import ca_cfar_1d


def evaluate_pfa(
    nfft: int = 2048,
    num_guard: int = 4,
    num_iter: int = 200_000,
):
    """Run Monte Carlo simulation to compare empirical vs. theoretical Pfa."""
    pfa_targets = [1e-2, 1e-3, 1e-4]
    num_train_targets = [8, 12, 16]  # Corresponds to N = {16, 24, 32}

    print("### Evaluación de Probabilidad de Falsa Alarma (Pfa)")
    print(f"Iteraciones por configuración: {num_iter:,}, NFFT: {nfft}")
    print("| N Train | Pfa Teórica | Pfa Empírica | Límite Conf. 95% (±) | Pasó |")
    print("|:-------:|:-----------:|:--------------:|:----------------------:|:----:|")

    rng = np.random.default_rng(42)

    for num_train in num_train_targets:
        for pfa in pfa_targets:
            # Edge effect calculation
            border_cells = 2 * (num_train + num_guard)
            valid_cells = nfft - border_cells
            if valid_cells <= 0:
                print(f"| {2*num_train:<7} | {pfa:<11} | N/A (NFFT muy pequeño) | | |")
                continue

            total_valid_cells = num_iter * valid_cells
            false_alarms = 0

            # Batch processing for efficiency
            batch_size = min(num_iter, 10_000)
            remaining = num_iter
            while remaining > 0:
                current = min(batch_size, remaining)
                noise = (rng.standard_normal((current, nfft)) + 1j * rng.standard_normal((current, nfft))) / sqrt(2.0)
                power = np.abs(noise) ** 2
                for i in range(current):
                    det, _ = ca_cfar_1d(power[i, :], num_train=num_train, num_guard=num_guard, pfa=pfa)
                    false_alarms += np.sum(det)
                remaining -= current

            empirical_pfa = false_alarms / total_valid_cells
            
            # Confidence interval calculation
            conf_interval = 1.96 * sqrt((pfa * (1 - pfa)) / total_valid_cells)
            passed = "✅" if abs(empirical_pfa - pfa) < conf_interval else "❌"

            print(
                f"| {2*num_train:<7} | {pfa:<11.0e} | {empirical_pfa:<14.4e} | {conf_interval:<22.2e} | {passed} |")


def evaluate_pd(
    nfft: int = 2048,
    num_train: int = 16,
    num_guard: int = 4,
    pfa: float = 1e-4,
    num_iter: int = 2_000,
):
    """Run Monte Carlo simulation to evaluate Probability of Detection vs. SNR."""
    snr_dbs = np.arange(-10, 21, 2.5)
    
    print("\n### Evaluación de Probabilidad de Detección (Pd) vs. SNR")
    print(f"Iteraciones por SNR: {num_iter:,}, N: {2*num_train}, Pfa: {pfa:.0e}")
    print("| SNR (dB) | Detecciones | Pd (%) |")
    print("|:--------:|:-----------:|:------:|")

    rng = np.random.default_rng(123)
    tone_bin = nfft // 2

    for snr in snr_dbs:
        detections = 0
        signal_power = 10 ** (snr / 10.0)
        
        for _ in tqdm(range(num_iter), desc=f"SNR={snr}dB", leave=False):
            noise = (rng.standard_normal(nfft) + 1j * rng.standard_normal(nfft)) / sqrt(2.0)
            power_noise = np.abs(noise) ** 2
            
            power_with_signal = np.copy(power_noise)
            power_with_signal[tone_bin] += signal_power

            det, _ = ca_cfar_1d(power_with_signal, num_train=num_train, num_guard=num_guard, pfa=pfa)
            if det[tone_bin]:
                detections += 1
        
        pd_percent = (detections / num_iter) * 100.0
        print(f"| {snr:<8.1f} | {detections:<11,}/{num_iter} | {pd_percent:>5.1f}% |")


def measure_latency(num_iter: int = 1_000):
    """Measure algorithmic latency for different FFT sizes."""
    nfft_sizes = [1024, 2048, 4096, 8192, 16384]
    print("\n### Medición de Latencia Algorítmica")
    print(f"Iteraciones por tamaño: {num_iter:,}")
    print("| NFFT  | Tiempo Promedio (µs) |")
    print("|:------|:--------------------:|")

    rng = np.random.default_rng(77)

    for nfft in nfft_sizes:
        power = np.abs((rng.standard_normal(nfft) + 1j * rng.standard_normal(nfft)) / sqrt(2.0)) ** 2
        
        # Warm-up run
        ca_cfar_1d(power)

        start_time = time.perf_counter()
        for _ in range(num_iter):
            ca_cfar_1d(power)
        end_time = time.perf_counter()

        avg_time_us = ((end_time - start_time) / num_iter) * 1e6
        print(f"| {nfft:<5} | {avg_time_us:>20.2f} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluación empírica del rendimiento del detector CFAR."
    )
    parser.add_argument(
        "experiment",
        choices=["pfa", "pd", "latency", "all"],
        nargs="?",
        default="all",
        help="El experimento a ejecutar (por defecto: all)."
    )
    args = parser.parse_args()

    if args.experiment == "pfa" or args.experiment == "all":
        evaluate_pfa()
    if args.experiment == "pd" or args.experiment == "all":
        evaluate_pd()
    if args.experiment == "latency" or args.experiment == "all":
        measure_latency()
