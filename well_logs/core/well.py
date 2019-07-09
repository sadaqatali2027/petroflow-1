from abc import ABCMeta
from functools import wraps
from copy import copy
from collections import Counter

import numpy as np

from .abstract_well import AbstractWell
from .well_segment import WellSegment

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
    def __init__(self, segments=None, *args, **kwargs):
        super().__init__()
        if segments is None:
            self.segments = [WellSegment(*args, **kwargs)]
        else:
            self.segments = segments

    def copy(self):
        return copy(self)

    def split_segments(self, connected=False):
        self.segments = [Well(segments=segment.split_segments(connected)) for segment in self.segments]
        return self.segments
    
    @property
    def depth(self):
        return sum([segment.depth for segment in self.segments])
    
#    def drop_segments(self, indices):
#        self.segments = [segment for i, segment in self.segments if not i in indices]
    
    def random_crop(self, height, n_crops=1, divide_by=None):
        p = np.array([item.depth for item in self.segments])
        random_segments = Counter(np.random.choice(self.segments, n_crops, p=p/sum(p)))
        self.segments = [Well(segments=segment.random_crop(height, n_crops)) for segment, n_crops in random_segments.items()]
        print('self.segments:', self.segments)
        return self.segments

    def crop(self, height, step, drop_last=True):
        self.segments = [Well(segments=segment.crop(height, step, drop_last)) for segment in self.segments]

    # def assemble_crops(self, crops, name):
    #     i = 0
    #     for segment in self.segments:
    #         for subsegment in segment:
    #             setattr(subsegment, name, crops[i])
    #             i += 1

    # def aggregate(self, name, func):
    #     for i in range(len(self.segments)):
    #         self.segments[i] = [self.segments[i], func([getattr(subsegment, name) for subsegment in self.segments[i]])]

    def dump(self, path):
        self.aggregate().segments[0].dump(path)
        return self
