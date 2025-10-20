"""Tests for the 2-D CA-CFAR implementation."""

import numpy as np
from scipy.ndimage import gaussian_filter

from src.cfar import ca_cfar_2d


def test_cfar2d_detects_hotspot():
    rng = np.random.default_rng(99)
    shape = (64, 64)
    noise = rng.standard_normal(shape)
    spec = np.abs(noise) ** 2

    hotspot = (32, 33)
    spec[hotspot] += 50.0

    det, thr = ca_cfar_2d(spec, train=(4, 4), guard=(2, 2), pfa=1e-4)

    assert det[hotspot], "Hotspot should trigger detection"
    assert not np.isnan(thr[hotspot])
    assert np.isnan(thr[0, 0]), "Borders must remain undefined"


def test_cfar2d_pfa_in_correlated_noise():
    """Characterize CFAR performance in correlated noise.
    
    The empirical Pfa is expected to be higher than the design Pfa because the
    I.I.D. noise assumption is violated.
    """
    rng = np.random.default_rng(111)
    shape = (256, 256)
    # Generate correlated noise by applying a Gaussian filter
    noise = rng.standard_normal(shape)
    correlated_noise = gaussian_filter(noise, sigma=2.0)
    spec = np.abs(correlated_noise) ** 2

    pfa = 1e-3
    train = (8, 8)
    guard = (3, 3)
    det, thr = ca_cfar_2d(spec, train=train, guard=guard, pfa=pfa)
    
    valid_mask = ~np.isnan(thr)
    assert np.any(valid_mask), "No valid cells found for CFAR calculation"

    empirical_pfa = np.sum(det[valid_mask]) / np.sum(valid_mask)

    # We expect the Pfa to be significantly higher than the design Pfa
    assert empirical_pfa > pfa * 2
    print(f"Design Pfa: {pfa}, Empirical Pfa in correlated noise: {empirical_pfa:.4f}")
