"""Implements WellBatch class."""

from abc import ABCMeta
from functools import wraps

from ..batchflow import FilesIndex, Batch, action, inbatch_parallel
from .well import Well
from .abstract_classes import AbstractWell
from .utils import to_list


class WellDelegatingMeta(ABCMeta):
    def __new__(mcls, name, bases, namespace):
        abstract_methods = [base.__abstractmethods__ for base in bases if hasattr(base, "__abstractmethods__")]
        abstract_methods = frozenset().union(*abstract_methods)
        for name in abstract_methods:
            if name not in namespace:
                namespace[name] = mcls._make_parallel_action(name)
        return super().__new__(mcls, name, bases, namespace)

    @staticmethod
    def _make_parallel_action(name):
        @wraps(getattr(Well, name))
        def delegator(self, index, *args, **kwargs):
            pos = self.get_pos(None, "wells", index)
            func = getattr(Well, name)
            self.wells[pos] = func(self.wells[pos], *args, **kwargs)
        # TODO: choose inbatch_parallel target depending on action name
        return action()(inbatch_parallel(init="indices", target="threads")(delegator))


class WellBatch(Batch, AbstractWell, metaclass=WellDelegatingMeta):
    components = ("wells",)

    def __init__(self, index, preloaded=None, **kwargs):
        super().__init__(index, preloaded, **kwargs)
        self.wells = [None] * len(self.index)
        self._init_wells(**kwargs)

    @inbatch_parallel(init="indices", target="threads")
    def _init_wells(self, index, src=None, **kwargs):
        if src is not None:
            path = src[index]
        elif isinstance(self.index, FilesIndex):
            path = self.index.get_fullpath(index)
        else:
            raise ValueError("Source path is not specified")

        well = Well(path, **kwargs)
        i = self.get_pos(None, "wells", index)
        self.wells[i] = well

    @action
    def get_crops(self, src, dst):
        src = to_list(src)
        dst = to_list(dst)
        for attr_from, attr_to in zip(src, dst):
            crops = [getattr(segment, attr_from) for well in self.wells for segment in well.iter_level()]
            setattr(self, attr_to, crops)
        return self
