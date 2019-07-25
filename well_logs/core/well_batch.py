from abc import ABCMeta
from functools import wraps
from collections import Counter

import numpy as np

from ..batchflow import FilesIndex, Batch, Dataset, action, inbatch_parallel
from .well import Well

from .abstract_well import AbstractWell
from .well_segment import WellSegment

class WellDelegatingMeta(ABCMeta):
    def __new__(mcls, name, bases, namespace):
        abstract_methods = frozenset().union(*(
            base.__abstractmethods__ for base in bases if hasattr(base, '__abstractmethods__')
        ))
        for name in abstract_methods:
            if name not in namespace:
                namespace[name] = mcls._make_parallel_action(name)
        return super().__new__(mcls, name, bases, namespace)

    @staticmethod
    def _make_parallel_action(name):
        @wraps(getattr(Well, name))
        def batch_delegator(self, index, *args, **kwargs):
            pos = self.get_pos(None, "wells", index)
            func = getattr(Well, name)
            res = func(self.wells[pos], *args, **kwargs)
            return res
        return action()(inbatch_parallel(init="indices", target="threads")(batch_delegator))

class WellBatch(Batch, AbstractWell, metaclass=WellDelegatingMeta):
    components = "wells",

    def __init__(self, index, preloaded=None, **kwargs):
        super().__init__(index, preloaded, **kwargs)
        self.wells = [None] * len(self.index)
    
    @action
    @inbatch_parallel(init="indices", target="threads")
    def load(self, index, src=None, *args, **kwargs):
        if src is not None:
            path = src[index]
        elif isinstance(self.index, FilesIndex):
            path = self.index.get_fullpath(index)
        else:
            raise ValueError("Source path is not specified")
        
        well = Well(path, *args, **kwargs)
        i = self.get_pos(None, "wells", index)     
        self.wells[i] = well

    @action
    def get_crops(self, src, dst):
        if not isinstance(src, (tuple, list)):
            src = [src]
        if not isinstance(dst, (tuple, list)):
            dst = [dst]
        for attr_from, attr_to in zip(src, dst):
            crops = [getattr(segment, attr_from) for well in self.wells for segment in well.iter_level()]
            setattr(self, attr_to, crops)
        return self
        
        
