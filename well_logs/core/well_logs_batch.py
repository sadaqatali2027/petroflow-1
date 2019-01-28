from copy import deepcopy
from math import ceil

import numpy as np
import matplotlib.pyplot as plt

from .. import batchflow as bf
from . import well_logs_batch_tools as bt
from .utils import for_each_component


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
    def _pad(comp, new_length, pad_value):
        pad_len = new_length - comp.shape[-1]
        if pad_len == 0:
            return comp
        pad_width = [(0, 0)] * (comp.ndim - 1) + [(pad_len, 0)]
        return np.pad(comp, pad_width, "constant", constant_values=pad_value)

    @staticmethod
    def _pad_dept(dept, new_length):
        pad_len = new_length - dept.shape[-1]
        if pad_len == 0:
            return dept
        pad_width = [(0, 0)] * (dept.ndim - 1) + [(pad_len, 0)]
        end_values = (dept[0] - (dept[1] - dept[0]) * pad_len, 0)
        return np.pad(dept, pad_width, "linear_ramp", end_values=end_values)

    @bf.action
    @bf.inbatch_parallel(init="indices", target="threads")
    def split_logs(self, index, length, step, pad_value=0, *, components):
        self._check_positive_int(length, "Segment length")
        self._check_positive_int(step, "Step size")
        i = self.get_pos(None, "logs", index)

        log_length = self.logs[i].shape[-1]
        n_segments = ceil(max(log_length - length, 0) / step) + 1
        new_length = (n_segments - 1) * step + length
        pad_length = new_length - log_length
        additional_meta = {
            "n_segments": n_segments,
            "pad_length": pad_length,
            "split_step": step,
        }
        self.meta[i].update(additional_meta)

        components = set(np.unique(np.asarray(components).ravel()))
        split_positions = np.arange(n_segments) * step
        if "dept" in components:
            padded_dept = self._pad_dept(self.dept[i], new_length)
            self.dept[i] = bt.split(padded_dept, length, split_positions)
            components.remove("dept")
        for comp in components:
            padded_comp = self._pad(getattr(self, comp)[i], new_length, pad_value)
            getattr(self, comp)[i] = bt.split(padded_comp, length, split_positions)

    @bf.action
    @bf.inbatch_parallel(init="indices", target="threads")
    def random_split_logs(self, index, length, n_segments, pad_value=0, *, components):
        self._check_positive_int(length, "Segment length")
        self._check_positive_int(n_segments, "The number of segments")
        i = self.get_pos(None, "logs", index)

        if self.logs[i].shape[-1] < length:
            if "dept" in components:
                padded_dept = self._pad_dept(self.dept[i], length)
                self.dept[i] = np.tile(padded_dept, (n_segments,) + (1,) * padded_dept.ndim)
                components.remove("dept")
            for comp in components:
                padded_comp = self._pad(getattr(self, comp)[i], length, pad_value)
                getattr(self, comp)[i] = np.tile(padded_comp, (n_segments,) + (1,) * padded_comp.ndim)
        else:
            split_positions = np.random.randint(0, self.logs[i].shape[-1] - length + 1, n_segments)
            for comp in components:
                getattr(self, comp)[i] = bt.split(getattr(self, comp)[i], length, split_positions)

    @for_each_component
    def _split_by_well(self, split_indices, *, components):
        comp = self._to_array(np.split(getattr(self, components), split_indices))
        setattr(self, components, comp)

    @bf.action
    def split_by_well(self, *, components):
        split_indices = [meta.get("n_segments") for meta in self.meta]
        if any(ix is None for ix in split_indices):
            raise ValueError("The number of log segments for a well is unknown")
        split_indices = np.cumsum(split_indices)[:-1]
        return self._split_by_well(split_indices, components=components)

    @for_each_component
    @bf.inbatch_parallel(init="indices", target="for")
    def _aggregate(self, index, agg_fn, *, components):
        i = self.get_pos(None, components, index)
        comp = getattr(self, components)[i]
        meta = self.meta[i]
        tmp_comp = bt.aggregate(comp, meta["split_step"], agg_fn)
        getattr(self, components)[i] = tmp_comp[..., meta["pad_length"]:]

    @bf.action
    def aggregate(self, agg_fn="mean", *, components):
        if isinstance(agg_fn, str):
            if not agg_fn.startswith("nan"):
                agg_fn = "nan" + agg_fn
            try:
                agg_fn = getattr(np, agg_fn)
            except AttributeError:
                raise ValueError("agg_fn must be a valid numpy aggregation function name")
        elif not callable(agg_fn):
            raise ValueError("agg_fn must be a callable or a valid numpy aggregation function name")
        return self._aggregate(agg_fn, components=components)

    @bf.action
    @for_each_component
    @bf.inbatch_parallel(init="indices", target="for")
    def standardize(self, index, axis=-1, eps=1e-10, *, components):
        i = self.get_pos(None, components, index)
        comp = getattr(self, components)[i]
        comp = ((comp - np.nanmean(comp, axis=axis, keepdims=True)) /
                np.nanstd(comp, axis=axis, keepdims=True) + eps)
        getattr(self, components)[i] = comp
