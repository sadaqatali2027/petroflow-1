"""Auxiliary functions for well logs processing."""

import numpy as np
from numba import njit


@njit(nogil=True)
def crop(logs, length, positions):
    """Crop segments with given ``length`` and start ``positions`` from
    ``logs`` along the last axis.

    Parameters
    ----------
    logs : n-D ndarray
        Logs to crop segments from.
    length : positive int
        Length of each crop along the last axis.
    positions : array-like of positive ints
        Starting indices of cropped segments in the initial logs.

    Returns
    -------
    logs : (n+1)-D ndarray
        Cropped segments, stacked along the new axis with index 0.
    """
    shape = (len(positions), *logs.shape[:-1], length)
    res = np.empty(shape, dtype=logs.dtype)
    for i in range(res.shape[0]):  # pylint: disable=unsubscriptable-object
        res[i] = logs[..., positions[i] : positions[i] + length]
    return res


def aggregate(logs, step, agg_fn):
    """Undo the application of ``WellLogsBatch.crop`` method by aggregating
    the resulting crops using ``agg_fn``.

    Parameters
    ----------
    logs : n-D ndarray
        Segments, cropped from the initial logs and stacked along the first
        axis.
    step : positive int
        ``WellLogsBatch.crop`` step argument.
    agg_fn : callable
        An aggregation function to combine values in case of crop overlay.

    Returns
    -------
    logs : (n-1)-D ndarray
        Aggregated logs.
    """
    length = logs.shape[-1]
    n_crops = logs.shape[0]
    padded_length = (n_crops - 1) * step + length
    res = np.full((*logs.shape[:-1], padded_length), np.nan, dtype=logs.dtype)
    for ix in range(n_crops):
        res[ix, ..., ix * step : ix * step + length] = logs[ix]
    res = agg_fn(res, axis=0)
    return res
