from abc import ABCMeta
from functools import wraps
from copy import copy
from collections import Counter

import numpy as np

from .abstract_well import AbstractWell
from .well_segment import WellSegment
from ..batchflow import timeit

class SegmentDelegatingMeta(ABCMeta):
    def __new__(mcls, name, bases, namespace):
        abstract_methods = frozenset().union(*(base.__abstractmethods__ for base in bases))
        for name in abstract_methods:
            if name not in namespace:
                namespace[name] = mcls._make_delegator(name)
        return super().__new__(mcls, name, bases, namespace)

    @staticmethod
    def _make_delegator(name):
        @wraps(getattr(WellSegment, name))
        def delegator(self, *args, **kwargs):
            results = []
            for segment in self.segments:
                res = getattr(segment, name)(*args, **kwargs)
                if not isinstance(res, list):
                    res = [res]
                results.append(res)
            res_well = self.copy()
            res_well.segments = sum(results, [])
            return res_well
        return delegator


class Well(AbstractWell, metaclass=SegmentDelegatingMeta):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.segments = [WellSegment(*args, **kwargs)]

    def copy(self):
        return copy(self)

    def split_segments(self, connected=False):
        self.segments = [item for segment in self.segments for item in segment.split_segments(connected)]
    
#    def drop_segments(self, indices):
#        self.segments = [segment for i, segment in self.segments if not i in indices]
    
    def random_crop(self, height, n_crops=1, divide_by=None):
        if divide_by is None:
            p = np.array([item.depth_to - item.depth_from for item in self.segments])
            random_segments = Counter(np.random.choice(self.segments, n_crops, p=p/sum(p)))
            self.segments = [s for segment, n_crops in random_segments.items() for s in segment.random_crop(height, n_crops)]
        else:
            positive = [segment for segment in self.segments if divide_by(segment) == 1]
            negative = [segment for segment in self.segments if divide_by(segment) == 0]
            self.segments = []
            for segments in positive, negative:
                p = np.array([item.depth_to - item.depth_from for item in segments])
                random_segments = Counter(np.random.choice(segments, n_crops, p=p/sum(p)))
                self.segments.extend(
                    [s for segment, n_crops in random_segments.items() for s in segment.random_crop(height, n_crops)]
                )
            self.segments = np.random.choice(self.segments, size=2*n_crops, replace=False)