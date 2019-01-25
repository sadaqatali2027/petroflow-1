import numpy as np
from numba import njit


@njit(nogil=True)
def split(logs, length, split_positions):
    shape = (len(split_positions), *logs.shape[:-1], length)
    res = np.empty(shape, dtype=logs.dtype)
    for i in range(res.shape[0]):
        res[i] = logs[..., split_positions[i] : split_positions[i] + length]
    return res


def aggregate(comp, step, agg_fn):
    length = comp.shape[-1]
    n_crops = comp.shape[0]
    padded_length = (n_crops - 1) * step + length
    tmp_comp = np.full((*comp.shape[:-1], padded_length), np.nan, dtype=comp.dtype)
    for ix in range(n_crops):
        tmp_comp[ix, ..., ix * step : ix * step + length] = comp[ix]
    tmp_comp = agg_fn(tmp_comp, axis=0)
    return tmp_comp
