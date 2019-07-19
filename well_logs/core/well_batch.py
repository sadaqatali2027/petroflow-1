from collections import Counter

import numpy as np

from ..batchflow import Dataset, FilesIndex, Batch, action, inbatch_parallel
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
    def crop(self, index, height, step, *args, **kwargs):
        pos = self.get_pos(None, "wells", index)
        self.wells[pos].crop(height, step)
    
    @action
    def assemble_crops(self, crops, name):
        pos = 0
        res = []
        for well in self.wells:
            length = sum([len(segment) for segment in well.segments])
            well.assemble_crops(crops[pos:pos+length], name)
        return self
    
    @action
    @inbatch_parallel(init="indices", target="threads")
    def aggregate(self, index, name, func):
        pos = self.get_pos(None, "wells", index)
        self.wells[pos].aggregate(name, func)

class WellDataset(Dataset):
    def __init__(self, index=None, batch_class=WellBatch, preloaded=None, index_class=FilesIndex, *args, **kwargs):
        if index is None:
            index = index_class(*args, **kwargs)
        super().__init__(index, batch_class, preloaded)
