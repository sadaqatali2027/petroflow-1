import numpy as np
from numba import njit


@njit(nogil=True)
def split(logs, length, step):
    res = np.empty(((logs.shape[1] - length) // step + 1, logs.shape[0], length), dtype=logs.dtype)
    for i in range(res.shape[0]):
        res[i, :, :] = logs[:, i * step : i * step + length]
    return res


@njit(nogil=True)
def random_split(logs, length, n_segments, positions):
    res = np.empty((n_segments, logs.shape[0], length), dtype=logs.dtype)
    for i in range(res.shape[0]):
        res[i, :, :] = logs[:, positions[i] : positions[i] + length]
    return res
