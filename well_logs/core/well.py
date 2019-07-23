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
    def __init__(self, *args, segments=None, **kwargs):
        super().__init__()
        if segments is None:
            self.segments = [WellSegment(*args, **kwargs)]
        else:
            self.segments = segments
        
        self.tree_depth = 2 if self._has_segments() else self.segments[0].tree_depth + 1

    def _has_segments(self):
        return all([isinstance(item, WellSegment) for item in self.segments])

    def iter_level(self, level=-1):
        level = level if level >= 0 else self.tree_depth + level
        if level > self.tree_depth:
            raise ValueError("Level ({}) can't exceed depth ({})".format(level, self.tree_depth))
        if level == 0:
            return [self]
        elif level == 1:
            return self.segments
        else:
            return [item for well in self.segments for item in well.iter_level(level-1)]

    def copy(self):
        return copy(self)

    def create_segments(self, src, connected=True):
        wells = self.iter_level(-2)
        for well in wells:
            well.segments = [
                Well(segments=segment.create_segments(src, connected)) for segment in well.segments
            ]
        self.tree_depth += 1

    @property
    def length(self):
        return sum([segment.length for segment in self.segments])
    
    @property
    def depth_from(self):
        return self.segments[0].depth_from
    
    @property
    def depth_to(self):
        return self.segments[-1].depth_to

    def random_crop(self, height, n_crops=1):
        wells = self.iter_level(-2)
        p = np.array([item.length for item in wells])
        if len(wells) > 0:
            random_wells = Counter(np.random.choice(wells, n_crops, p=p/sum(p)))
            for well, n_well_crops in random_wells.items():
                if len(well.segments) > 0:
                    p = np.array([item.length for item in well.segments])
                    random_segments = Counter(np.random.choice(well.segments, n_well_crops, p=p/sum(p)))
                    well.segments = [
                        Well(segments=segment.random_crop(height, n_segment_crops))
                        for segment, n_segment_crops in random_segments.items()
                    ]
                else:
                    well.segments = []
            self.tree_depth += 1

    def crop(self, height, step, drop_last=True):
        wells = self.iter_level(-2)
        for well in wells:
            well.segments = [
                Well(segments=segment.crop(height, step, drop_last))
                for segment in well.segments
            ]
        self.tree_depth += 1


    # def assemble_crops(self, crops, name):
    #     i = 0
    #     for segment in self.segments:
    #         for subsegment in segment:
    #             setattr(subsegment, name, crops[i])
    #             i += 1

    # def aggregate(self, name, func):
    #     for i in range(len(self.segments)):
    #         self.segments[i] = [self.segments[i], func([getattr(subsegment, name) for subsegment in self.segments[i]])]

    # def dump(self, path):
    #     self.aggregate().segments[0].dump(path)
    #     return self
