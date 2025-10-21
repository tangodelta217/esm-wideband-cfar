import numpy as np
from src.cfar.cfar1d import ca_cfar_1d

def make_spectrum(nfft, snr_db=None, k_unshifted=None, rng=None):
    """
    Devuelve potencia por bin (periodograma) con fftshift aplicado.
    SNR objetivo se interpreta COMO SNR EN EL BIN DEL DETECTOR.
    """
    rng = rng or np.random.default_rng()
    x = (rng.standard_normal(nfft) + 1j*rng.standard_normal(nfft)) / np.sqrt(2.0)
    if snr_db is not None and k_unshifted is not None:
        snr_lin = 10.0**(snr_db/10.0)      # SNR deseado en el bin
        A = (snr_lin / nfft) ** 0.5        # hace que A^2 * NFFT = snr_lin
        ph = rng.uniform(0, 2*np.pi)
        n = np.arange(nfft)
        tone = A * np.exp(1j*(2*np.pi * k_unshifted * n / nfft + ph))
        x = x + tone
    X = np.fft.fft(x)
    P = (np.abs(X)**2) / nfft
    return np.fft.fftshift(P)

def sweep_snr(snr_list_db, n_trials=1000, pfa=1e-3, num_train=16, num_guard=4, nfft=4096):
    rng = np.random.default_rng(0)
    k_unshifted = 200
    k_shifted = (k_unshifted + nfft//2) % nfft

    # Pfa empírica (ruido)
    fa_trials = 3000
    valid_bins = nfft - (num_train + num_guard) * 2 - 1
    fa_count = 0
    for _ in range(fa_trials):
        Pn = make_spectrum(nfft, None, None, rng)
        det, _ = ca_cfar_1d(Pn, num_train, num_guard, pfa)
        fa_count += det.sum()
    pfa_emp = fa_count / (fa_trials * valid_bins)

    # Pd vs SNR
    results = []
    for snr in snr_list_db:
        hits = 0
        for _ in range(n_trials):
            P = make_spectrum(nfft, snr, k_unshifted, rng)
            det, _ = ca_cfar_1d(P, num_train, num_guard, pfa)
            hits += bool(det[k_shifted])
        results.append((snr, hits/n_trials, pfa_emp))
    return results
