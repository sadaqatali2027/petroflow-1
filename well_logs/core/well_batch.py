from collections import Counter

import numpy as np

from ..batchflow import FilesIndex, Batch, action, inbatch_parallel
from .well import Well

class WellBatch(Batch):
    components = "wells",

    def __init__(self, index, preloaded=None, **kwargs):
        super().__init__(index, preloaded, **kwargs)
        self.wells = [None] * len(self.index)
    
    @action
    @inbatch_parallel(init="indices", target="threads")
    def load(self, index, src=None, *args, **kwargs):
        if src is not None:
            path = src[index]
        elif isinstance(self.index, FilesIndex):
            path = self.index.get_fullpath(index)
        else:
            raise ValueError("Source path is not specified")
        
        well = Well(path, *args, **kwargs)
        i = self.get_pos(None, "wells", index)
        
        self.wells[i] = well

    @action
    @inbatch_parallel(init="indices", target="threads")
    def split_segments(self, index, *args, **kwargs):
        i = self.get_pos(None, "wells", index)
        self.wells[i].split_segments(*args, **kwargs)

    @action
    @inbatch_parallel(init="indices", target="threads")
    def random_crop(self, index, n_crops, height, divide_by=None, *args, **kwargs):
        pos = self.get_pos(None, "wells", index)
        self.wells[pos].random_crop(height, n_crops, divide_by)
    
    @action
    @inbatch_parallel(init="indices", target="threads")
    def crop(self, index, height, n_crops, *args, **kwargs):
        pos = self.get_pos(None, "wells", index)
        self.wells[pos].crop(height, n_crops)
