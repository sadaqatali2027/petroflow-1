import numpy as np
from numba import njit


@njit(nogil=True)
def split(logs, length, split_positions):
    shape = (len(split_positions), *logs.shape[:-1], length)
    res = np.empty(shape, dtype=logs.dtype)
    for i in range(res.shape[0]):
        res[i] = logs[..., split_positions[i] : split_positions[i] + length]
    return res
