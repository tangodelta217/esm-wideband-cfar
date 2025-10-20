"""One-dimensional Cell-Averaging CFAR detector."""

from __future__ import annotations

import numpy as np


def ca_cfar_1d(
    power_lin: np.ndarray,
    num_train: int = 16,
    num_guard: int = 4,
    pfa: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Realiza detección CFAR 1D (Cell-Averaging) sobre un vector de potencia.

    Esta implementación es vectorizada y no utiliza bucles de Python para el
    cálculo del umbral, asegurando un alto rendimiento.

    Contratos (Precondiciones):
    ---------------------------
    - `power_lin` debe ser un array NumPy 1D de valores no negativos.
    - `num_train` debe ser un entero positivo.
    - `num_guard` debe ser un entero no negativo.
    - `pfa` debe ser un float en el rango (0, 1).

    Manejo de Bordes:
    ------------------
    El detector no puede calcular un umbral para los bins en los bordes del
    array, donde la ventana de entrenamiento estaría incompleta.
    - El número de bins sin evaluar en cada borde es `num_train + num_guard`.
    - En el array `det`, estos bins siempre serán `False`.
    - En el array `thr`, estos bins contendrán `np.nan`.

    Parameters
    ----------
    power_lin:
        Array 1D con muestras de potencia lineal no negativa (|IQ|^2).
    num_train:
        Número de celdas de entrenamiento a cada lado de la celda bajo prueba (CUT).
    num_guard:
        Número de celdas de guarda a cada lado de la CUT.
    pfa:
        Probabilidad de falsa alarma deseada.

    Returns
    -------
    det:
        Array booleano marcando una detección (`True`) para cada posición.
    thr:
        Array de punto flotante con el umbral calculado para cada posición.
    """
    # --- Contratos (Aserciones) ---
    assert power_lin.ndim == 1, "power_lin debe ser un array 1D"
    assert np.all(power_lin >= 0), "power_lin debe contener valores no negativos"
    assert isinstance(num_train, int) and num_train > 0, "num_train debe ser un entero positivo"
    assert isinstance(num_guard, int) and num_guard >= 0, "num_guard debe ser un entero no negativo"
    assert 0 < pfa < 1, "pfa debe estar en el rango (0, 1)"

    n_cells = power_lin.size
    det = np.zeros(n_cells, dtype=bool)
    thr = np.full(n_cells, np.nan, dtype=float)

    half_window = num_train + num_guard
    if n_cells < 2 * half_window + 1:
        return det, thr

    train_cells = 2 * num_train
    alpha = train_cells * (pfa ** (-1.0 / train_cells) - 1.0)

    # Vectorized implementation using prefix sums
    cumsum = np.concatenate(([0.0], np.cumsum(power_lin, dtype=float)))

    # Sum of all cells in the full window (2 * half_window + 1)
    full_sum = cumsum[2 * half_window + 1 :] - cumsum[: - (2 * half_window + 1)]

    # Sum of all cells in the guard window plus the CUT (2 * num_guard + 1)
    guard_sum = (
        cumsum[half_window + num_guard + 1 : - (half_window - num_guard)]
        - cumsum[half_window - num_guard : - (half_window + num_guard + 1)]
    )

    # The training cell sum is the full window sum minus the guard window sum
    noise_power = (full_sum - guard_sum) / train_cells

    # Calculate threshold only for the valid central part of the array
    valid_thr = alpha * noise_power
    thr[half_window : n_cells - half_window] = valid_thr

    # Compare power of CUTs with their corresponding threshold
    cut_power = power_lin[half_window : n_cells - half_window]
    valid_det = cut_power > valid_thr
    det[half_window : n_cells - half_window] = valid_det

    return det, thr
