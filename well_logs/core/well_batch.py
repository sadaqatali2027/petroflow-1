"""Implements WellBatch class."""

from abc import ABCMeta
from functools import wraps

import numpy as np

from ..batchflow import FilesIndex, Batch, action, inbatch_parallel
from .well import Well
from .abstract_classes import AbstractWell
from .utils import to_list

TARGET = dict() # inbatch_parallel target depending on action name

class WellDelegatingMeta(ABCMeta):
    """ Metaclass to delegate abstract methods from `WellBatch` to `Well` objects
    in `wells` component. """
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
        target = TARGET.get(name, 'threads')
        return action()(inbatch_parallel(init="indices", target=target)(delegator))


class WellBatch(Batch, AbstractWell, metaclass=WellDelegatingMeta):
    """A batch class for well data storing and processing.  Batch class inherits
    all abstract methods from `Well` class and realize some extra functionality.
    To execute method for each well in a batch you should add that method into pipeline.

    Parameters
    ----------
    index : DatasetIndex
        Unique identifiers of wells in the batch.
    preloaded : tuple, optional
        Data to put in the batch if given. Defaults to ``None``.

    Attributes
    ----------
    index : DatasetIndex
        Unique identifiers of wells in the batch.
    wells : 1-D ndarray
        An array of `Well` instances.

    Note
    ----
    Some batch methods take ``index`` as their first argument after ``self``.
    You should not specify it in your code since it will be implicitly passed
    by ``inbatch_parallel`` decorator.
    """

    components = ("wells",)

    def __init__(self, index, preloaded=None, **kwargs):
        super().__init__(index, preloaded, **kwargs)
        self.wells = np.array([None] * len(self.index))
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
        """ Get some attributes from well and put them into batch variables.

        Parameters
        ----------
        src : str or iterable
            attributes of wells to load into batch
        dst : str or iterable
            batch variables to save well attributes. Must be of the same length as 'src'.

        Returns
        -------
        batch : WellLogsBatch
            Batch with loaded components. Changes batch data inplace.
        """
        src = to_list(src)
        dst = to_list(dst)
        if len(src) != len(dst):
            raise ValueError(
                "'src' and 'dst' must be of the same length but {} and {} were given".format(len(src), len(dst))
            )
        for attr_from, attr_to in zip(src, dst):
            crops = [[getattr(segment, attr_from) for segment in well.iter_level()] for well in self.wells]
            setattr(self, attr_to, np.array(crops))
        return self
