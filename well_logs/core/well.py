"""Implements Well class."""

from abc import ABCMeta
from copy import copy
from functools import wraps
from collections import Counter

import numpy as np

from .abstract_classes import AbstractWell
from .well_segment import WellSegment


class SegmentDelegatingMeta(ABCMeta):
    """ Metaclass to delegate abstract methods from `Well` to its children
    (instances of `Well` or `WellSegment`). """
    def __new__(mcls, name, bases, namespace):
        abstract_methods = [
            base.__abstractmethods__ for base in bases if hasattr(base, "__abstractmethods__")
        ]
        abstract_methods = frozenset().union(*abstract_methods)
        for name in abstract_methods:
            if name not in namespace:
                namespace[name] = mcls._make_delegator(name)
        return super().__new__(mcls, name, bases, namespace)

    @staticmethod
    def _make_delegator(name):
        @wraps(getattr(WellSegment, name))
        def delegator(self, *args, **kwargs):
            results = []
            for segment in self:
                res = getattr(segment, name)(*args, **kwargs)
                if not isinstance(res, list):
                    res = [res]
                results.extend(res)
            res_well = self.copy()
            res_well.segments = results
            return res_well
        return delegator


class Well(AbstractWell, metaclass=SegmentDelegatingMeta):
    def __init__(self, *args, segments=None, **kwargs):
        super().__init__()
        if segments is None:
            self.segments = [WellSegment(*args, **kwargs)]
        else:
            self.segments = segments

    @property
    def tree_depth(self):
        if self._has_segments():
            return 2
        return self.segments[0].tree_depth + 1

    @property
    def length(self):
        return self.depth_to - self.depth_from

    @property
    def depth_from(self):
        return min([well.depth_from for well in self])

    @property
    def depth_to(self):
        return max([well.depth_to for well in self])

    @property
    def n_segments(self):
        return len(self.iter_level())

    def _has_segments(self):
        return all(isinstance(item, WellSegment) for item in self)

    def __iter__(self):
        for segment in self.segments:
            yield segment

    def iter_level(self, level=-1):
        level = level if level >= 0 else self.tree_depth + level
        if (level < 0) or (level > self.tree_depth):
            raise ValueError("Level ({}) can't be negative or exceed tree depth ({})".format(level, self.tree_depth))
        if level == 0:
            return [self]
        if level == 1:
            return self.segments
        return [item for well in self for item in well.iter_level(level - 1)]

    def prune(self):
        # TODO: raise EmptyWellException if no segments left
        self.segments = [well for well in self if isinstance(well, WellSegment) or well.n_segments > 0]
        for well in self:
            if isinstance(well, Well):
                _ = well.prune()
        return self

    def copy(self):
        return copy(self)

    def dump(self, path):
        # TODO: aggregate before dumping
        self.segments[0].dump(path)
        return self

    def create_segments(self, src, connected=True):
        wells = self.iter_level(-2)
        for well in wells:
            well.segments = [
                Well(segments=segment.create_segments(src, connected)) for segment in well
            ]
        return self

    def crop(self, height, step, drop_last=True):
        wells = self.iter_level(-2)
        for well in wells:
            well.segments = [
                Well(segments=segment.crop(height, step, drop_last))
                for segment in well
            ]
        return self

    def random_crop(self, height, n_crops=1):
        # TODO: current implementation uses well.length to choose cropping probability, but lenght is estimated
        # incorrectly in case of overlapping segments
        wells = self.iter_level(-2)
        p = np.array([sum([segment.length for segment in item]) for item in wells])
        random_wells = Counter(np.random.choice(wells, n_crops, p=p/sum(p)))
        for well in wells:
            if well in random_wells:
                n_well_crops = random_wells[well]
                p = np.array([item.length for item in well])
                random_segments = Counter(np.random.choice(well.segments, n_well_crops, p=p/sum(p)))
                well.segments = [
                    Well(segments=segment.random_crop(height, n_segment_crops))
                    for segment, n_segment_crops in random_segments.items()
                ]
            else:
                well.segments = []
        return self.prune()

    def drop_nans(self, components_to_drop_nans):
        wells = self.iter_level(-2)
        for well in wells:
            well.segments = [
                Well(segments=segment.drop_nans(components_to_drop_nans)) for segment in well
            ]
        return self.prune()

    def drop_short_segments(self, min_length):
        wells = self.iter_level(-2)
        for well in wells:
            well.segments = [segment for segment in well if segment.length > min_length]
        return self.prune()

    # def assemble_crops(self, crops, name):
    #     i = 0
    #     for segment in self.segments:
    #         for subsegment in segment:
    #             setattr(subsegment, name, crops[i])
    #             i += 1

    # def aggregate(self, name, func):
    #     for i in range(len(self.segments)):
    #         self.segments[i] = [self.segments[i], func([getattr(subsegment, name) for subsegment in self.segments[i]])]
