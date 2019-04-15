from abc import ABCMeta
from functools import wraps
from copy import copy

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
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.segments = [WellSegment(*args, **kwargs)]

    def copy(self):
        return copy(self)
