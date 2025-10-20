"""Tests for the 1-D CA-CFAR implementation."""

import numpy as np
import pytest

from src.cfar import ca_cfar_1d


@pytest.mark.parametrize(
    "num_train, pfa, tolerance",
    [
        (16, 1e-2, 1e-2),
        (32, 1e-3, 5e-4),
        (64, 1e-4, 5e-5),
    ],
)
def test_cfar1d_empirical_pfa_matches_design(num_train, pfa, tolerance):
    """Check that the empirical Pfa matches the design Pfa for various parameters."""
    rng = np.random.default_rng(42)
    # Use enough samples to get statistically significant results
    samples = int(500 / pfa)
    noise = (rng.standard_normal(samples) + 1j * rng.standard_normal(samples)) / np.sqrt(2.0)
    power = np.abs(noise) ** 2

    det, thr = ca_cfar_1d(power, num_train=num_train, num_guard=4, pfa=pfa)
    valid = ~np.isnan(thr)
    
    # Avoid division by zero if there are no valid cells to test
    if np.sum(valid) == 0:
        assert samples < (2 * (num_train + 4) + 1), "No valid cells with enough samples"
        return

    empirical_pfa = det[valid].mean()
    assert abs(empirical_pfa - pfa) < tolerance


def test_cfar1d_detects_strong_tone():
    rng = np.random.default_rng(123)
    samples = 2048
    noise = (rng.standard_normal(samples) + 1j * rng.standard_normal(samples)) / np.sqrt(2.0)
    power_noise = np.abs(noise) ** 2

    tone_bin = samples // 2
    tone_power = 100.0
    power_noise[tone_bin] += tone_power

    det, thr = ca_cfar_1d(power_noise, num_train=16, num_guard=4, pfa=1e-4)
    assert det[tone_bin], "Strong tone should be detected"
    assert not np.isnan(thr[tone_bin])


def test_cfar1d_robust_to_non_centered_spectrum():
    """Ensure CFAR logic is independent of spectrum centering (fftshift)."""
    rng = np.random.default_rng(456)
    samples = 4096
    noise = (rng.standard_normal(samples) + 1j * rng.standard_normal(samples)) / np.sqrt(2.0)
    power_noise = np.abs(noise) ** 2

    # Place tone in a non-centered position (e.g., first quarter)
    tone_bin = samples // 4
    tone_power = 100.0
    power_noise[tone_bin] += tone_power

    det, thr = ca_cfar_1d(power_noise, num_train=32, num_guard=8, pfa=1e-5)
    assert det[tone_bin], "Should detect tone in non-centered spectrum"
    assert not np.isnan(thr[tone_bin])
    # Check that no other detections were triggered
    assert np.sum(det) == 1