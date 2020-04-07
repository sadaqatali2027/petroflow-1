"""Named expression for wells."""
import pandas as pd

from ..batchflow import NamedExpression
from ..batchflow.batchflow.named_expr import _DummyBatch  # pylint: disable=import-error
from .utils import to_list

class NestedList:
    """Wrapper for nested lists."""
    def __init__(self, nested_list):
        self._nested_list = nested_list

    def __getattr__(self, key):
        return NestedList([[getattr(item, key) for item in inner_list] for inner_list in self._nested_list])

    def __setattr__(self, key, value):
        if key == '_nested_list':
            self.__dict__['_nested_list'] = value
        else:
            for item, val in zip(self.ravel(), value):
                setattr(item, key, val)

    def __getitem__(self, key):
        return NestedList([[item[key] for item in inner_list] for inner_list in self._nested_list])

    def __setitem__(self, key, value):
        key = to_list(key)
        for item, val in zip(self.ravel(), value):
            val = pd.DataFrame(val, columns=key, index=item.index)
            item[key] = val

    def __repr__(self):
        return repr(self._nested_list)

    def __copy__(self):
        return NestedList([[item.copy() for item in inner_list] for inner_list in self._nested_list])

    def to_list(self):
        """Return wrapped list."""
        return self._nested_list

    def ravel(self):
        """Flatten a nested list into a list."""
        return sum(self._nested_list, [])

class WS(NamedExpression):
    """Component or attribute of each well segment.

    Notes
    -----
    `WS()` returns list of wells.

    To avoid unexpected data changes the copy of the segments data may be
    returned, if `copy=True`.

    Examples
    --------
    ::

        WS('samples')
        WS('core_dl')
        WS(copy=True)
    """
    def __init__(self, name=None, mode='w', copy=False):
        super().__init__(name, mode)
        self.copy = copy

    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value of a well component """
        if self.params:
            batch, pipeline, model = self.params
        name = super()._get_name(batch=batch, pipeline=pipeline, model=model)
        if isinstance(batch, _DummyBatch):
            raise ValueError("WS expressions are not allowed in static models: WS('%s')" % name)
        nested_list = NestedList([[segment for segment in well.iter_level()] for well in batch.wells])  # pylint: disable=unnecessary-comprehension, line-too-long
        if name is None:
            return nested_list.copy() if self.copy else nested_list
        return getattr(nested_list, name)

    def assign(self, value, batch=None, pipeline=None, model=None):
        """ Assign a value to a well component """
        if self.params:
            batch, pipeline, model = self.params
        name = super()._get_name(batch=batch, pipeline=pipeline, model=model)
        nested_list = NestedList([[segment for segment in well.iter_level()] for well in batch.wells])  # pylint: disable=unnecessary-comprehension, line-too-long
        if name is not None:
            setattr(nested_list, name, value)
