"""Miscellaneous utility functions."""

import functools
import inspect

import numpy as np

from ..batchflow import FilesIndex


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


def leq_notclose(x1, x2):
    """Return the truth value of (x1 <= x2) AND (x1 is NOT close to x2) element-wise."""
    return np.less_equal(x1, x2) & ~np.isclose(x1, x2)


def leq_close(x1, x2):
    """Return the truth value of (x1 <= x2) OR (x1 is close to x2) element-wise."""
    return np.less_equal(x1, x2) | np.isclose(x1, x2)


def geq_close(x1, x2):
    """Return the truth value of (x1 >= x2) OR (x1 is close to x2) element-wise."""
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
