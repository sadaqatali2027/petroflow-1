"""Miscellaneous utility functions."""

import functools
import inspect

import numpy as np

from ..batchflow import FilesIndex

def for_each_component(method):
    """Independently call a wrapped method for each component in the
    ``components`` argument, which can be a string or an array-like object.
    """
    @functools.wraps(method)
    def wrapped_method(self, *args, **kwargs):
        if "components" in kwargs:
            components = kwargs.pop("components")
        else:
            components = inspect.signature(method).parameters["components"].default
        if components is inspect.Parameter.empty:
            method(self, *args, **kwargs)
        else:
            components = np.unique(np.asarray(components).ravel())
            for comp in components:
                method(self, *args, components=comp, **kwargs)
        return self
    return wrapped_method


def to_list(obj):
    """Cast an object to a list. Almost identical to `list(obj)` for 1-D
    objects, except for `str`, which won't be split into separate letters but
    transformed into a list of a single element.
    """
    return np.array(obj).ravel().tolist()

def leq_notclose(x1, x2):
    """Return element-wise truth value of
    (x1 <= x2) AND (x1 is NOT close to x2).
    """
    return np.less_equal(x1, x2) & ~np.isclose(x1, x2)

def leq_close(x1, x2):
    """Return element-wise truth value of
    (x1 <= x2) OR (x1 is close to x2).
    """
    return np.less_equal(x1, x2) | np.isclose(x1, x2)

def geq_close(x1, x2):
    """Return element-wise truth value of
    (x1 >= x2) OR (x1 is close to x2).
    """
    return np.greater_equal(x1, x2) | np.isclose(x1, x2)

def get_path(batch, index, src):
    """Get path corresponding to index."""
    if src is not None:
        path = src[index]
    elif isinstance(batch.index, FilesIndex):
        path = batch.index.get_fullpath(index)
    else:
        raise ValueError("Source path is not specified")
    return path

def fill_nans_around(arr, period):
    """Fill nan array values with closest not nan values on fixed period.

    Usage example for period=1:

    Input array:  __1___23_4___
    Output array: _111_223344__

    Parameters
    ----------
    arr : np.array
        Input array.
    period : int
        Number of values to fill.

    Returns
    -------
    arr : np.array
        Array with nans filled.
    """
    arr = arr.copy()
    nan_mask = np.isnan(arr)
    lims = [False] + np.logical_xor(nan_mask[1:], nan_mask[:-1]).tolist()
    lims = np.where(lims)[0]
    # If initial array starts or ends with nan value, limits of nan intervals
    # are padded with required indices, since xor trick above doesn't handle it
    left_lim = [0] if nan_mask[0] else []
    right_lim = [arr.size] if nan_mask[-1] else []
    lims = np.concatenate([left_lim, lims, right_lim]).astype(int)
    # As the original `period` value might be excessive for some intervals with
    # smaller lengths, limits of intervals to fill are checked for overlapping
    lefts = lims.copy()
    rights = lims.copy()
    lengths = lims[1::2] - lims[::2]
    lefts[::2] = lefts[1::2] - np.minimum(period, lengths // 2)
    rights[1::2] = rights[::2] + np.minimum(period, lengths // 2 + lengths % 2)
    # Zip limits of nan intervals in pairs `(from, to)`
    lefts = list(zip(lefts[::2], lefts[1::2]))
    rights = list(zip(rights[::2], rights[1::2]))
    # Iterate all nan intervals and fill them in calculated limits
    for i in range(lengths.size):
        left_slice = slice(*lefts[i])
        if left_slice.stop != arr.size:
            arr[left_slice] = arr[left_slice.stop]
        right_slice = slice(*rights[i])
        if right_slice.start != 0:
            arr[right_slice] = arr[right_slice.start - 1]
    return arr
