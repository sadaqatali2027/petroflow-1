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
    def split_by_core(self, index):
        i = self.get_pos(None, "wells", index)
        self.wells[i] = self.wells[i].split_by_core()
