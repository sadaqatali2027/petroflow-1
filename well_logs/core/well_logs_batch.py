from copy import deepcopy
from math import ceil

import numpy as np
import matplotlib.pyplot as plt

from .. import batchflow as bf
from . import well_logs_batch_tools as bt


class WellLogsBatch(bf.Batch):
    components = "dept", "logs", "meta", "mask", "predictions"

    def __init__(self, index, preloaded=None):
        super().__init__(index, preloaded)
        self.dept = self.array_of_nones
        self.logs = self.array_of_nones
        self.meta = self.array_of_dicts
        self.mask = self.array_of_nones
        self.predictions = self.array_of_nones

    @property
    def array_of_nones(self):
        return np.array([None] * len(self.index))

    @property
    def array_of_dicts(self):
        return np.array([{} for _ in range(len(self.index))])

    def _reraise_exceptions(self, results):
        if bf.any_action_failed(results):
            all_errors = self.get_errors(results)
            raise RuntimeError("Cannot assemble the batch", all_errors)

    @staticmethod
    def _to_array(arr):
        return np.array(list(arr) + [None])[:-1]

    @bf.action
    def load(self, src=None, fmt=None, components=None, *args, **kwargs):
        if components is None:
            components = self.components
        components = np.asarray(components).ravel()
        if fmt == "npz":
            return self._load_npz(src=src, fmt=fmt, components=components, *args, **kwargs)
        return super().load(src, fmt, components, *args, **kwargs)

    @bf.inbatch_parallel(init="indices", target="threads")
    def _load_npz(self, index, src=None, fmt=None, components=None, *args, **kwargs):
        if src is not None:
            path = src[index]
        elif isinstance(self.index, bf.FilesIndex):
            path = self.index.get_fullpath(index)
        else:
            raise ValueError("Source path is not specified")

        i = self.get_pos(None, "logs", index)
        well_data = np.load(path)

        missing_components = set(components) - set(well_data.keys())
        if missing_components:
            err_msg = "File {} does not contain components {}".format(path, ", ".join(missing_components))
            raise ValueError(err_msg)
        for comp in components:
            getattr(self, comp)[i] = well_data[comp]

        extra_components = set(well_data.keys()) - set(components)
        for comp in extra_components:
            self.meta[i][comp] = well_data[comp]

    def show_logs(self, index=None, start=None, end=None, plot_mask=False, subplot_size=(15, 2)):
        i = 0 if index is None else self.get_pos(None, "signal", index)
        dept, logs, mnemonics = self.dept[i], self.logs[i], self.meta[i]["mnemonics"]
        if plot_mask:
            mask = self.mask[i]
            logs = np.concatenate([logs, mask[np.newaxis, ...]], axis=0)
            mnemonics = np.append(mnemonics, "Collector mask")
        num_channels = logs.shape[0]

        start = dept[0] if start is None else start
        start_ix = (np.abs(dept - start)).argmin()
        end = dept[-1] if end is None else end
        end_ix = (np.abs(dept - end)).argmin()

        figsize = (subplot_size[0], subplot_size[1] * num_channels)
        _, axes = plt.subplots(num_channels, 1, squeeze=False, figsize=figsize)
        for channel, (ax,) in enumerate(axes):
            ax.plot(dept[start_ix:end_ix], logs[channel, start_ix:end_ix])
            ax.set_xlim(start, end)
            ax.set_title(mnemonics[channel])
            ax.set_xlabel("Depth (m)")
            ax.grid(True, which="major")

        plt.tight_layout()
        plt.show()

    @bf.action
    @bf.inbatch_parallel(init="indices", post="_assemble_drop_nans", target="threads")
    def drop_nans(self, index):
        i = self.get_pos(None, "logs", index)
        dept, logs, meta, mask = self.dept[i], self.logs[i], self.meta[i], self.mask[i]
        not_nan_mask = np.all(~np.isnan(logs), axis=0)
        not_nan_indices = np.where(not_nan_mask)[0]
        borders = np.where((not_nan_indices[1:] - not_nan_indices[:-1]) != 1)[0] + 1
        splits = []
        for i, indices in enumerate(np.split(not_nan_indices, borders)):
            if len(indices) == 0:
                continue
            splits.append([str(index) + "_" + str(i), deepcopy(meta),
                           dept[indices], logs[:, indices], mask[indices]])
        return splits

    def _assemble_drop_nans(self, results, *args, **kwargs):
        _ = args, kwargs
        self._reraise_exceptions(results)
        results = sum(results, [])
        if len(results) == 0:
            raise bf.SkipBatchException("All batch data was dropped")
        indices, meta, dept, logs, mask = zip(*results)
        batch = self.__class__(bf.DatasetIndex(indices))
        batch.meta = self._to_array(meta)
        batch.dept = self._to_array(dept)
        batch.logs = self._to_array(logs)
        batch.mask = self._to_array(mask)
        return batch

    @bf.action
    @bf.inbatch_parallel(init="indices", target="threads")
    def fill_nans(self, index, fill_value=0):
        logs = self.logs[self.get_pos(None, "logs", index)]
        logs[np.isnan(logs)] = fill_value

    def _filter_batch(self, keep_mask):
        indices = self.indices[keep_mask]
        if len(indices) == 0:
            raise bf.SkipBatchException("All batch data was dropped")
        batch = self.__class__(bf.DatasetIndex(indices))
        for component in self.components:
            setattr(batch, component, getattr(self, component)[keep_mask])
        return batch

    @bf.action
    def drop_short_logs(self, min_length, axis=-1):
        keep_mask = np.array([log.shape[axis] >= min_length for log in self.logs])
        return self._filter_batch(keep_mask)

    @staticmethod
    def _check_positive_int(val, val_name):
        if (val <= 0) or not isinstance(val, int):
            raise ValueError("{} must be positive integer".format(val_name))

    @staticmethod
    def _pad(logs, new_length, pad_value):
        pad_len = new_length - logs.shape[1]
        if pad_len == 0:
            return logs
        return np.pad(logs, ((0, 0), (pad_len, 0)), "constant", constant_values=pad_value)

    @bf.action
    @bf.inbatch_parallel(init="indices", target="threads")
    def split_logs(self, index, length, step, pad_value=0, split_mask=False):
        self._check_positive_int(length, "Segment length")
        self._check_positive_int(step, "Step size")
        i = self.get_pos(None, "logs", index)

        log_length = self.logs[i].shape[1]
        n_segments = ceil(max(log_length - length, 0) / step) + 1
        new_length = (n_segments - 1) * step + length
        pad_length = new_length - log_length
        self.meta[i].update({"n_segments": n_segments})

        split_positions = np.arange(n_segments) * step

        # TODO: split dept component
        padded_logs = self._pad(self.logs[i], new_length, pad_value)
        self.logs[i] = bt.split(padded_logs, length, split_positions)

        if split_mask:
            padded_mask = self._pad(self.mask[i][np.newaxis, ...], new_length, pad_value)
            self.mask[i] = bt.split(padded_mask, length, split_positions)

    @bf.action
    @bf.inbatch_parallel(init="indices", target="threads")
    def random_split_logs(self, index, length, n_segments, pad_value=0, split_mask=False):
        self._check_positive_int(length, "Segment length")
        self._check_positive_int(n_segments, "The number of segments")
        i = self.get_pos(None, "logs", index)
        if self.logs[i].shape[1] < length:
            tmp_logs = self._pad(self.logs[i], length, pad_value)
            self.logs[i] = np.tile(tmp_logs, (n_segments, 1, 1))
            if split_mask:
                tmp_mask = self._pad(self.mask[i][np.newaxis, ...], length, pad_value)
                self.mask[i] = np.tile(tmp_mask, (n_segments, 1, 1))
        else:
            split_positions = np.random.randint(0, self.logs[i].shape[1] - length + 1, n_segments)
            self.logs[i] = bt.split(self.logs[i], length, split_positions)
            if split_mask:
                self.mask[i] = bt.split(self.mask[i][np.newaxis, ...], length, split_positions)

    @bf.action
    def split_by_well(self, components):
        components = np.asarray(components).ravel()
        split_indices = [meta.get("n_segments") for meta in self.meta]
        if any(ix is None for ix in split_indices):
            raise ValueError("The number of log segments for a well is unknown")
        split_indices = np.cumsum(split_indices)[:-1]
        for comp in components:
            setattr(self, comp, self._to_array(np.split(getattr(self, comp), split_indices)))
        return self

    @bf.action
    @bf.inbatch_parallel(init="indices", post="_assemble_predictions", target="threads")
    def average_prediction(self, index, length, step, shapes):
        predictions = self.predictions[self.get_pos(None, "logs", index)]
        log_length = (len(predictions) - 1) * step + length
        average_prediction = np.zeros((len(predictions), log_length, 1))
        denom = np.zeros(log_length)
        for i, item in enumerate(predictions):
            # print(item.shape, predictions.shape, average_prediction[i, i * step: i * step+length].shape)
            average_prediction[i, i * step: i * step+length] = item
            denom[i * step: i * step+length] += 1
        average_prediction = np.sum(average_prediction, axis=0).swapaxes(0, 1) / denom
        return average_prediction

    def _assemble_predictions(self, list_of_res, *args, **kwargs):
        if not bf.any_action_failed(list_of_res):
            setattr(self, 'predictions', self._to_array(list_of_res))
        else:
            raise Exception(list_of_res)
        return self

    @bf.action
    @bf.inbatch_parallel(init="indices", target="threads")
    def standardize(self, index, axis=-1, eps=1e-10):
        i = self.get_pos(None, "logs", index)
        logs = self.logs[i]
        self.logs[i] = ((logs - np.mean(logs, axis=axis, keepdims=True)) /
                         np.std(logs, axis=axis, keepdims=True) + eps)
