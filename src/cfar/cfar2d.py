"""Two-dimensional Cell-Averaging CFAR detector."""

from __future__ import annotations

import numpy as np
from scipy.signal import convolve2d


def _build_training_kernel(train: tuple[int, int], guard: tuple[int, int]) -> np.ndarray:
    height = 2 * (train[0] + guard[0]) + 1
    width = 2 * (train[1] + guard[1]) + 1
    kernel = np.ones((height, width), dtype=float)

    guard_top = train[0]
    guard_bottom = guard_top + 2 * guard[0] + 1
    guard_left = train[1]
    guard_right = guard_left + 2 * guard[1] + 1
    kernel[guard_top:guard_bottom, guard_left:guard_right] = 0.0

    return kernel


def ca_cfar_2d(
    spec: np.ndarray,
    train: tuple[int, int] = (6, 6),
    guard: tuple[int, int] = (2, 2),
    pfa: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply 2-D CA-CFAR to a spectrogram or heatmap.

    This implementation uses `scipy.signal.convolve2d` in 'valid' mode for
    high performance, avoiding Python loops and redundant convolutions.

    Contratos (Precondiciones):
    ---------------------------
    - `spec` debe ser un array NumPy 2D de valores no negativos.
    - `train` debe ser una tupla de 2 enteros positivos (rows, cols).
    - `guard` debe ser una tupla de 2 enteros no negativos (rows, cols).
    - `pfa` debe ser un float en el rango (0, 1).

    Manejo de Bordes:
    ------------------
    El umbral no se calcula para los píxeles de borde donde la ventana de
    entrenamiento es incompleta. El tamaño del borde depende de `train` y `guard`.
    - En el array `det`, estos píxeles siempre serán `False`.
    - En el array `thr`, estos píxeles contendrán `np.nan`.

    Parameters
    ----------
    spec:
        2-D array with non-negative power values.
    train:
        Training cells in (rows, columns) on each side of the Cell Under Test.
    guard:
        Guard cells in (rows, columns) excluded from averaging.
    pfa:
        Desired probability of false alarm.

    Returns
    -------
    det:
        Boolean matrix of detections.
    thr:
        Threshold matrix with `np.nan` where the CFAR statistic is undefined.
    """
    # --- Contratos (Aserciones) ---
    assert spec.ndim == 2, "spec debe ser un array 2D"
    assert np.all(spec >= 0), "spec debe contener valores no negativos"
    assert isinstance(train, tuple) and len(train) == 2 and all(isinstance(t, int) and t > 0 for t in train), "train debe ser una tupla de 2 enteros positivos"
    assert isinstance(guard, tuple) and len(guard) == 2 and all(isinstance(g, int) and g >= 0 for g in guard), "guard debe ser una tupla de 2 enteros no negativos"
    assert 0 < pfa < 1, "pfa debe estar en el rango (0, 1)"

    kernel = _build_training_kernel(train, guard)
    training_cells = int(np.sum(kernel))
    assert training_cells > 0, "El kernel de entrenamiento no puede estar vacío"

    # The convolution in 'valid' mode gives the sum of noise power for each
    # CUT that has a full set of training cells.
    sum_train = convolve2d(spec, kernel, mode="valid")
    mean_power = sum_train / training_cells

    alpha = training_cells * (pfa ** (-1.0 / training_cells) - 1.0)
    valid_thr = alpha * mean_power

    # Extract the corresponding central part of the spectrogram for comparison.
    h_ker, w_ker = kernel.shape
    h_spec, w_spec = spec.shape
    cut_spec = spec[
        h_ker // 2 : h_spec - h_ker // 2,
        w_ker // 2 : w_spec - w_ker // 2,
    ]
    valid_det = cut_spec > valid_thr

    # Pad the valid results to match the original spectrogram shape.
    pad_h = (h_ker // 2, h_spec - valid_det.shape[0] - h_ker // 2)
    pad_w = (w_ker // 2, w_spec - valid_det.shape[1] - w_ker // 2)

    det = np.pad(valid_det, (pad_h, pad_w), mode="constant", constant_values=False)
    thr = np.pad(valid_thr, (pad_h, pad_w), mode="constant", constant_values=np.nan)

    return det, thr
