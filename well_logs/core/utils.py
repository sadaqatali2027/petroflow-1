import functools

import numpy as np


def for_each_component(method):
    @functools.wraps(method)
    def wrapped_method(self, *args, components, **kwargs):
        components = np.asarray(components).ravel()
        for comp in components:
            method(self, *args, components=comp, **kwargs)
        return self
    return wrapped_method
