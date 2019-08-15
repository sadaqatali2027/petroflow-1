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
