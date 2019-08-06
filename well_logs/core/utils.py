"""Miscellaneous utility functions."""

import functools
import inspect

import numpy as np


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
    return np.array(obj).ravel().tolist()
