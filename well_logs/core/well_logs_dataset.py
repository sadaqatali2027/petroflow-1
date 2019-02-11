from .. import batchflow as bf
from .well_logs_batch import WellLogsBatch


class WellLogsDataset(bf.Dataset):
    def __init__(self, index=None, batch_class=WellLogsBatch, preloaded=None, index_class=bf.FilesIndex,
                 *args, **kwargs):
        if index is None:
            index = index_class(*args, **kwargs)
        super().__init__(index, batch_class, preloaded)
