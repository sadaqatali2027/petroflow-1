from copy import deepcopy
from math import ceil

import numpy as np
import matplotlib.pyplot as plt

from .. import batchflow as bf
from . import well_logs_batch_tools as bt
from .utils import for_each_component


class WellLogsBatch(bf.Batch):
    components = "dept", "logs", "meta"

    def __init__(self, index, preloaded=None):
        super().__init__(index, preloaded)
        self.dept = self.array_of_nones
        self.logs = self.array_of_nones
        self.meta = self.array_of_dicts

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
    
    # Input/output methods

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

    # Channels processing

    @staticmethod
    def _get_mnemonics_key(component):
        return component + "_mnemonics"

    @bf.action
    @bf.inbatch_parallel(init="indices", target="threads")
    def convert_names(self, index):
        i = self.get_pos(None, 'logs', index)
        self.meta[i]['logs_mnemonics'] = self.meta[i].pop('mnemonics')

    def _generate_mask(self, index, mnemonics=None, indices=None, invert_mask=False, *, components):
        i = self.get_pos(None, components, index)
        component = getattr(self, components)[i]
        n_channels = component.shape[0]
        component_mnemonics = self.meta[i].get(self._get_mnemonics_key(components))
        mask = np.zeros(n_channels, dtype=np.bool)
        if indices is not None:
            mask |= np.in1d(np.arange(n_channels), indices)
        if mnemonics is not None:
            if component_mnemonics is None:
                raise ValueError("Mnemonics for {} component are not defined in meta".format(components))
            mask |= np.in1d(component_mnemonics, mnemonics)
        if invert_mask:
            mask = ~mask
        return mask

    @bf.inbatch_parallel(init="indices", target="threads")
    def _filter_channels(self, index, mnemonics=None, indices=None, invert_mask=False, *, components):
        mask = self._generate_mask(index, mnemonics, indices, invert_mask, components=components)
        if np.sum(mask) == 0:
            raise ValueError("All channels cannot be dropped")
        i = self.get_pos(None, components, index)
        getattr(self, components)[i] = getattr(self, components)[i][mask]
        mnemonics_key = self._get_mnemonics_key(components)
        if mnemonics_key in self.meta[i]:
            self.meta[i][mnemonics_key] = np.asarray(self.meta[i][mnemonics_key])[mask]

    @bf.action
    @for_each_component
    def drop_channels(self, mnemonics=None, indices=None, *, components="logs"):
        if mnemonics is None and indices is None:
            raise ValueError("Both mnemonics and indices cannot be empty")
        return self._filter_channels(mnemonics, indices, invert_mask=True, components=components)

    @bf.action
    @for_each_component
    def keep_channels(self, mnemonics=None, indices=None, *, components="logs"):
        if mnemonics is None and indices is None:
            raise ValueError("Both mnemonics and indices cannot be empty")
        return self._filter_channels(mnemonics, indices, invert_mask=False, components=components)

    @bf.action
    @for_each_component
    @bf.inbatch_parallel(init="indices", target="threads")
    def rename_channels(self, index, rename_dict, *, components="logs"):
        i = self.get_pos(None, components, index)
        mnemonics_key = self._get_mnemonics_key(components)
        old_mnemonics = self.meta[i].get(mnemonics_key)
        if old_mnemonics is not None:
            new_mnemonics = np.array([rename_dict.get(name, name) for name in old_mnemonics], dtype=object)
            self.meta[i][mnemonics_key] = new_mnemonics

    @bf.action
    @for_each_component
    @bf.inbatch_parallel(init="indices", target="threads")
    def reorder_channels(self, index, new_order, *, components="logs"):
        i = self.get_pos(None, components, index)
        mnemonics_key = self._get_mnemonics_key(components)
        old_order = self.meta[i].get(mnemonics_key)
        if old_order is None:
            raise ValueError("Mnemonics for {} component are not defined in meta".format(components))
        diff = np.setdiff1d(new_order, old_order)
        if diff.size > 0:
            raise ValueError("Unknown mnemonics: {}".format(", ".join(diff)))
        if len(new_order) == 0:
            raise ValueError("All channels cannot be dropped")
        transform_dict = {k: v for v, k in enumerate(old_order)}
        indices = [transform_dict[k] for k in new_order]
        getattr(self, components)[i] = getattr(self, components)[i][indices]
        self.meta[i][mnemonics_key] = old_order[indices]

    @bf.action
    @bf.inbatch_parallel(init="indices", target="threads")
    def split_by_mnemonic(self, index, mnemonics, component_from, component_to, all_mnemonics=None):
        if callable(mnemonics):
            mnemonics = mnemonics(all_mnemonics)
        mask = self._generate_mask(index, mnemonics, components=component_from)
        i = self.get_pos(None, component_from, index)
        mnemonics_key_to = self._get_mnemonics_key(component_to)
        mnemonics_key_from = self._get_mnemonics_key(component_from)

        getattr(self, component_to)[i] = getattr(self, component_from)[i][mask]
        self.meta[i][mnemonics_key_to] = self.meta[i][mnemonics_key_from][mask]

        getattr(self, component_from)[i] = getattr(self, component_from)[i][~mask]
        self.meta[i][mnemonics_key_from] = self.meta[i][mnemonics_key_from][~mask]

    @bf.action
    @bf.inbatch_parallel(init="indices", target="threads")
    def create_input_target(self, index, component_input, component_target, component_mask, all_mnemonics):
        i = self.get_pos(None, component_input, index)
        is_crops = (len(getattr(self, component_input)[i].shape) == 3)
        
        components = [component_input, component_target]
        
        for component in components:
            mnemonics = self.meta[i][self._get_mnemonics_key(component)]        
            data = getattr(self, component)[i]
            #new_data = np.zeros((len(all_mnemonics), *data.shape[1: ]))
            new_data = np.zeros((*data.shape[:-2], len(all_mnemonics), data.shape[-1]))
            mask = np.isin(all_mnemonics, mnemonics)
            new_data[..., np.where(mask)[0], :] = data
            getattr(self, component)[i] = new_data
        
        mask = mask[np.newaxis, ...]
        if is_crops:
            mask = np.tile(mask, (getattr(self, component_input)[i].shape[0], 1))
        getattr(self, component_mask)[i] = mask.astype('float32')

    # Logs processing

    @bf.inbatch_parallel(init="indices", post="_assemble_drop_nans", target="threads")
    def _drop_nans(self, index, components_to_split, components_to_copy):
        components = self[index]
        not_nan_mask = np.all(~np.isnan(components.logs), axis=0)
        not_nan_indices = np.where(not_nan_mask)[0]
        borders = np.where((not_nan_indices[1:] - not_nan_indices[:-1]) != 1)[0] + 1
        splits = []
        for i, indices in enumerate(np.split(not_nan_indices, borders)):
            if len(indices) == 0:
                continue
            split_components = [getattr(components, comp)[..., indices] for comp in components_to_split]
            copy_components = [deepcopy(getattr(components, comp)) for comp in components_to_copy]
            splits.append([str(index) + "_" + str(i)] + split_components + copy_components)
        return splits

    def _assemble_drop_nans(self, results, components_to_split, components_to_copy, *args, **kwargs):
        _ = args, kwargs
        self._reraise_exceptions(results)
        results = sum(results, [])
        if len(results) == 0:
            raise bf.SkipBatchException("All batch data was dropped")
        indices, *components = zip(*results)
        batch = self.__class__(bf.DatasetIndex(indices))
        for comp_name, comp_data in zip(components_to_split + components_to_copy, components):
            setattr(batch, comp_name, self._to_array(comp_data))
        return batch

    @bf.action
    def drop_nans(self, *, components_to_split=None, components_to_copy=None):
        if components_to_split is None:
            components_to_split = set()
        else:
            components_to_split = set(np.unique(np.asarray(components_to_split).ravel()))
        components_to_split = sorted(components_to_split | {"dept", "logs"})

        if components_to_copy is None:
            components_to_copy = set()
        else:
            components_to_copy = set(np.unique(np.asarray(components_to_copy).ravel()))
        components_to_copy = sorted(components_to_copy | {"meta"})

        return self._drop_nans(components_to_split, components_to_copy)

    @bf.action
    @for_each_component
    @bf.inbatch_parallel(init="indices", target="threads")
    def fill_nans(self, index, fill_value=0, *, components):
        comp = getattr(self, components)[self.get_pos(None, components, index)]
        comp[np.isnan(comp)] = fill_value

    def _filter_batch(self, keep_mask):
        indices = self.indices[keep_mask]
        if len(indices) == 0:
            raise bf.SkipBatchException("All batch data was dropped")
        batch = self.__class__(bf.DatasetIndex(indices))
        for comp in self.components:
            setattr(batch, comp, getattr(self, comp)[keep_mask])
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
    def crop(self, index, length, step, pad_value=0, *, components):
        self._check_positive_int(length, "Segment length")
        self._check_positive_int(step, "Step size")
        i = self.get_pos(None, "logs", index)

        log_length = self.logs[i].shape[-1]
        n_crops = ceil(max(log_length - length, 0) / step) + 1
        new_length = (n_crops - 1) * step + length
        pad_length = new_length - log_length
        additional_meta = {
            "n_crops": n_crops,
            "pad_length": pad_length,
            "split_step": step,
        }
        self.meta[i].update(additional_meta)

        components = set(np.unique(np.asarray(components).ravel()))
        split_positions = np.arange(n_crops) * step
        if "dept" in components:
            padded_dept = self._pad_dept(self.dept[i], new_length)
            self.dept[i] = bt.split(padded_dept, length, split_positions)
            components.remove("dept")
        for comp in components:
            padded_comp = self._pad(getattr(self, comp)[i], new_length, pad_value)
            getattr(self, comp)[i] = bt.split(padded_comp, length, split_positions)

    @bf.action
    @bf.inbatch_parallel(init="indices", target="threads")
    def random_crop(self, index, length, n_crops, pad_value=0, *, components):
        self._check_positive_int(length, "Segment length")
        self._check_positive_int(n_crops, "The number of segments")
        i = self.get_pos(None, "logs", index)

        if self.logs[i].shape[-1] < length:
            if "dept" in components:
                padded_dept = self._pad_dept(self.dept[i], length)
                self.dept[i] = np.tile(padded_dept, (n_crops,) + (1,) * padded_dept.ndim)
                components.remove("dept")
            for comp in components:
                padded_comp = self._pad(getattr(self, comp)[i], length, pad_value)
                getattr(self, comp)[i] = np.tile(padded_comp, (n_crops,) + (1,) * padded_comp.ndim)
        else:
            split_positions = np.random.randint(0, self.logs[i].shape[-1] - length + 1, n_crops)
            for comp in components:
                getattr(self, comp)[i] = bt.split(getattr(self, comp)[i], length, split_positions)

    @for_each_component
    def _split_by_well(self, split_indices, *, components):
        comp = self._to_array(np.split(getattr(self, components), split_indices))
        setattr(self, components, comp)

    @bf.action
    def split_by_well(self, *, components):
        split_indices = [meta.get("n_crops") for meta in self.meta]
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
