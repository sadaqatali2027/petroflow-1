"""Miscellaneous utility functions."""

import functools

import numpy as np


def to_list(obj):
    """Cast an object to a list. Almost identical to `list(obj)` for 1-D
    objects, except for `str`, which won't be split into separate letters but
    transformed into a list of a single element.
    """
    return np.array(obj).ravel().tolist()


def process_columns(method):
    """Decorate a `method` so that it is applied to `src` columns of an `attr`
    well attribute and stores the result in `dst` columns of the same
    attribute.

    Adds the following additional arguments to the decorated method:
    ----------------------------------------------------------------
    attr : str, optional
        `WellSegment` attribute to get the data from. Defaults to "logs".
    src : str or list of str or None, optional
        `attr` columns to pass to the method. Defaults to all columns.
    except_src : str or list of str or None, optional
        All `attr` columns, except these, will be passed to the method. Can't
        be specified together with `src`. By default, all columns are passed.
    dst : str or list of str or None, optional
        `attr` columns to save the result into. Defaults to `src`.
    drop_src : bool, optional
        Specifies whether to drop `src` columns from `attr` after the method
        call. Defaults to `False`.
    """
    @functools.wraps(method)
    def wrapper(self, *args, attr="logs", src=None, except_src=None, dst=None, drop_src=False, **kwargs):
        df = getattr(self, attr)
        if (src is not None) and (except_src is not None):
            raise ValueError("src and except_src can't be specified together")
        if src is not None:
            src = to_list(src)
        elif except_src is not None:
            # Calculate the difference between df.columns and except_src, preserving the order of columns in df
            except_src = np.unique(except_src)
            src = np.setdiff1d(df.columns, except_src, assume_unique=True)
        else:
            src = df.columns
        dst = src if dst is None else to_list(dst)
        df[dst] = method(self, df[src], *args, **kwargs)
        if drop_src:
            df.drop(set(src) - set(dst), axis=1, inplace=True)
        return self
    return wrapper


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

    Parameters
    ----------
    arr : 1-d np.array
        Input array.
    period : int
        Number of array positions to fill (counting from last not nan value).

    Returns
    -------
    arr : 1-d np.array
        Array with nans filled.

    Examples
    -----
    Usage for period=1:

        Input array:  __1___23_4___
        Output array: _111_223344__
    """
    arr = arr.copy()
    nan_mask = np.isnan(arr)
    bounds = [False] + np.logical_xor(nan_mask[1:], nan_mask[:-1]).tolist()
    bounds = np.where(bounds)[0]
    # If initial array starts or ends with nan value, bounds of nan intervals
    # are padded with array limits, since xor trick above doesn't handle it
    left_lim = [0] if nan_mask[0] else []
    right_lim = [arr.size] if nan_mask[-1] else []
    bounds = np.concatenate([left_lim, bounds, right_lim]).astype(int)
    # Create two `bounds` subarrays — `lefts` for intervals to be filled with
    # their most right value, and `rights` — with their most left value
    lefts = bounds.copy()
    rights = bounds.copy()
    # As the original `period` value might be excessive for some intervals with
    # smaller lengths, bounds of intervals to fill are checked for overlapping
    lengths = bounds[1::2] - bounds[::2]
    lefts[::2] = lefts[1::2] - np.minimum(period, lengths // 2)
    rights[1::2] = rights[::2] + np.minimum(period, lengths // 2 + lengths % 2)
    # Zip bounds of nan intervals in pairs `(from, to)`
    lefts = list(zip(lefts[::2], lefts[1::2]))
    rights = list(zip(rights[::2], rights[1::2]))
    # Iterate all nan intervals and fill them in calculated bounds
    for i in range(lengths.size):
        left_slice = slice(*lefts[i])
        if left_slice.stop != arr.size:
            arr[left_slice] = arr[left_slice.stop]
        right_slice = slice(*rights[i])
        if right_slice.start != 0:
            arr[right_slice] = arr[right_slice.start - 1]
    return arr
