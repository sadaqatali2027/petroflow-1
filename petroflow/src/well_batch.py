"""Implements WellBatch class."""
# pylint: disable=abstract-method

import traceback
from abc import ABCMeta
from functools import wraps

import numpy as np

from ..batchflow import Batch, SkipBatchException, action, inbatch_parallel, any_action_failed
from .well import Well
from .abstract_classes import AbstractWell
from .exceptions import SkipWellException
from .utils import get_path


class WellDelegatingMeta(ABCMeta):
    """A metaclass to delegate abstract methods from `WellBatch` to `Well`
    objects in `wells` component."""

    def __new__(mcls, name, bases, namespace):
        abstract_methods = [base.__abstractmethods__ for base in bases if hasattr(base, "__abstractmethods__")]
        abstract_methods = frozenset().union(*abstract_methods)
        for method_name in abstract_methods:
            if method_name not in namespace:
                target = namespace['targets'].get(name, 'threads')
                namespace[method_name] = mcls._make_parallel_action(method_name, target)
        return super().__new__(mcls, name, bases, namespace)

    @staticmethod
    def _make_parallel_action(name, target):
        @wraps(getattr(Well, name))
        def delegator(self, index, *args, **kwargs):
            pos = self.get_pos(None, "wells", index)
            return getattr(Well, name)(self.wells[pos], *args, **kwargs)
        return action()(inbatch_parallel(init="indices", post="_filter_assemble", target=target)(delegator))


class WellBatch(Batch, AbstractWell, metaclass=WellDelegatingMeta):
    """A batch class for well data storing and processing.

    Batch class inherits all abstract methods from `Well` class and implements
    some extra functionality. To execute a method for each well in a batch you
    should add that method into a pipeline.

    Parameters
    ----------
    index : DatasetIndex
        Unique identifiers of wells in the batch.
    preloaded : tuple, optional
        Data to put in the batch if given. Defaults to `None`.
    kwargs : misc
        Any additional named arguments to `Well.__init__`.

    Attributes
    ----------
    index : DatasetIndex
        Unique identifiers of wells in the batch.
    wells : 1-D ndarray
        An array of `Well` instances.

    Note
    ----
    Some batch methods take `index` as their first argument after `self`. You
    should not specify it in your code since it will be implicitly passed by
    `inbatch_parallel` decorator.
    """

    components = ("wells",)
    targets = dict()  # inbatch_parallel target depending on action name

    def __init__(self, index, *args, preloaded=None, **kwargs):
        super().__init__(index, *args, preloaded=preloaded, **kwargs)
        if preloaded is None:
            self.wells = np.array([None] * len(self.index))
            self._init_wells(**kwargs)

    @inbatch_parallel(init="indices", target="threads")
    def _init_wells(self, index, src=None, **kwargs):
        path = get_path(self, index, src)

        well = Well(path, **kwargs)
        i = self.get_pos(None, "wells", index)
        self.wells[i] = well

    def _filter_assemble(self, results, *args, **kwargs):
        skip_mask = np.array([isinstance(res, SkipWellException) for res in results])
        if sum(skip_mask) == len(self):
            raise SkipBatchException
        results = np.array(results)[~skip_mask]  # pylint: disable=invalid-unary-operand-type
        if any_action_failed(results):
            errors = self.get_errors(results)
            print(errors)
            traceback.print_tb(errors[0].__traceback__)
            raise RuntimeError("Could not assemble the batch")
        self.index = self.index.create_subset(self.indices[~skip_mask])  # pylint: disable=invalid-unary-operand-type, attribute-defined-outside-init, line-too-long
        self.wells = results
        return self
