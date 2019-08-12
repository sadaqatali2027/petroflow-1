"""Named expression for wells."""

import copy

from ..batchflow import NamedExpression
from ..batchflow.batchflow.named_expr import _DummyBatch

class NestedList:
    """Wrapper for nested lists."""
    def __init__(self, nested_list):
        self.nested_list = nested_list

    def __getattr__(self, key):
        return NestedList([[getattr(item, key) for item in inner_list] for inner_list in self.nested_list])

    def __getitem__(self, key):
        return NestedList([[item[key] for item in inner_list] for inner_list in self.nested_list])

    def __repr__(self):
        return repr(self.nested_list)

    def __copy__(self):
        return NestedList([[copy.copy(item) for item in inner_list] for inner_list in self.nested_list])

    def to_list(self):
        """Return wrapped list."""
        return self.nested_list

class WS(NamedExpression):
    """Component or attribute name of well segments.

    Notes
    -----
    ``W()`` return list of wells.

    To avoid unexpected data changes the copy of the segments data may be returned, if ``copy=True``.

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
        nested_list = NestedList([[segment for segment in well.iter_level()] for well in batch.wells])
        if name is None:
            return nested_list.copy() if self.copy else nested_list
        return getattr(nested_list, name)

    def assign(self, *args, **kwargs):
        raise TypeError("WS expressions can't be overriden.")
