"""Implements delegators - metaclasses that create absent abstract methods of
`WellBatch` and `Well` classes."""

from abc import ABCMeta
from functools import wraps

from ..batchflow import action, inbatch_parallel
from .well import Well
from .well_segment import WellSegment


class BaseDelegator(ABCMeta):
    """Base metaclass that searches for absent abstract methods and creates
    them."""

    def __new__(mcls, name, bases, namespace):
        abstract_methods = [base.__abstractmethods__ for base in bases if hasattr(base, "__abstractmethods__")]
        abstract_methods = frozenset().union(*abstract_methods)
        for method in abstract_methods:
            if method not in namespace:
                mcls._create_method(method, namespace)
        return super().__new__(mcls, name, bases, namespace)

    @classmethod
    def _create_method(mcls, method, namespace):
        """Create a method, absent in the `namespace`. Must be overridden in
        child classes."""
        raise NotImplementedError


class WellDelegatingMeta(BaseDelegator):
    """A metaclass to delegate calls to absent abstract methods of a
    `WellBatch` to `Well` objects in `wells` component."""

    @classmethod
    def _create_method(mcls, method, namespace):
        target = namespace["targets"][method]
        namespace[method] = mcls._make_parallel_action(method, target)

    @staticmethod
    def _make_parallel_action(name, target):
        @wraps(getattr(Well, name))
        def delegator(self, well, *args, **kwargs):
            _ = self
            return getattr(well, name)(*args, **kwargs)
        return action(inbatch_parallel(init="wells", post="_filter_assemble", target=target)(delegator))


class SegmentDelegatingMeta(BaseDelegator):
    """A metaclass to delegate calls to absent abstract methods of a `Well` to
    its children (instances of `Well` or `WellSegment`)"""

    @classmethod
    def _create_method(mcls, method, namespace):
        delegate_fn = getattr(mcls, namespace["delegators"][method])
        namespace[method] = delegate_fn(method)

    @staticmethod
    def segment_delegator(name):
        """Delegate the call to each segment of a `Well`. Acts as the default
        delegator."""
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

    @staticmethod
    def aggregating_delegator(name):
        """If `aggregate` is `False`, delegate the call to each segment of a
        `Well`. Otherwise, aggregate the `Well` and call the method."""
        @wraps(getattr(WellSegment, name))
        def delegator(self, *args, aggregate=True, **kwargs):
            segments = [self.aggregated_segment] if aggregate else self.iter_level()
            for segment in segments:
                getattr(segment, name)(*args, **kwargs)
            return self
        return delegator

    @staticmethod
    def well_delegator(name):
        """Delegate the call to each segment of a `Well` and create new
        `Well`s from the corresponding results. Increases the depth of the
        segment tree."""
        @wraps(getattr(WellSegment, name))
        def delegator(self, *args, **kwargs):
            wells = self.iter_level(-2)
            for well in wells:
                well.segments = [Well(segments=getattr(segment, name)(*args, **kwargs)) for segment in well]
            return self.prune()
        return delegator
