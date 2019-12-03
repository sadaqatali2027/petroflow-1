"""Implements WellLogsBatch class."""

from copy import deepcopy
from math import ceil

import numpy as np
import matplotlib.pyplot as plt

from . import well_logs_batch_tools as bt
from .utils import for_each_component, get_path
from ..batchflow import Batch, DatasetIndex, SkipBatchException
from ..batchflow import action, inbatch_parallel, any_action_failed


class WellLogsBatch(Batch):
    """A batch class for well logs storing and processing.

    Parameters
    ----------
    index : DatasetIndex
        Unique identifiers of well logs in the batch.
    preloaded : tuple, optional
        Data to put in the batch if given. Defaults to ``None``.

    Attributes
    ----------
    index : DatasetIndex
        Unique identifiers of well logs in the batch.
    dept : 1-D ndarray
        An array of 1-D ndarrays, containing information about depth for each
        sample.
    logs : 1-D ndarray
        An array of 2-D ndarrays with well logs data in channels first format.
    meta : 1-D ndarray
        An array of dicts with additional metadata about logs, such as
        mnemonics for each channel.

    Note
    ----
    Some batch methods take ``index`` as their first argument after ``self``.
    You should not specify it in your code since it will be implicitly passed
    by ``inbatch_parallel`` decorator.
    """

    components = "dept", "logs", "meta"

    def __init__(self, index, preloaded=None, **kwargs):
        super().__init__(index, preloaded, **kwargs)
        self.dept = self.array_of_nones
        self.logs = self.array_of_nones
        self.meta = self.array_of_dicts

    @property
    def array_of_nones(self):
        """1-D ndarray: ``NumPy`` array with ``None`` values."""
        return np.array([None] * len(self.index))

    @property
    def array_of_dicts(self):
        """1-D ndarray: ``NumPy`` array with empty ``dict`` values."""
        return np.array([{} for _ in range(len(self.index))])

    def _reraise_exceptions(self, results):
        if any_action_failed(results):
            all_errors = self.get_errors(results)
            raise RuntimeError("Cannot assemble the batch", all_errors)

    @staticmethod
    def _to_array(arr):
        return np.array(list(arr) + [None])[:-1]

    @staticmethod
    def _preprocess_components(components):
        return np.unique(np.asarray(components).ravel()).tolist()

    # Input/output methods

    @action
    def load(self, src=None, fmt=None, components=None, *args, **kwargs):
        """Load given batch components from source.

        This method supports loading of well logs in npz format.

        Parameters
        ----------
        src : misc, optional
            Source to load components from. Must be a collection, that can be
            indexed by indices of a batch. If ``None`` and ``self.index`` has
            ``FilesIndex`` type, paths from ``self.index`` are used.
        fmt : str, optional
            Source files extension. Can be one of:
            npz:
                Arrays from the source file are loaded in the components with
                the corresponding names. Arrays, that are not listed in the
                ``components`` argument, are stored in ``meta`` component
                under the same keys.
        components : str or array-like, optional
            Components to load. If ``None``, all batch components are loaded.

        Returns
        -------
        batch : WellLogsBatch
            Batch with loaded components. Changes batch data inplace.
        """
        if components is None:
            components = self.components
        components = np.asarray(components).ravel()
        if fmt == "npz":
            return self._load_npz(src=src, fmt=fmt, components=components, *args, **kwargs)
        return super().load(src, fmt, components, *args, **kwargs)

    @inbatch_parallel(init="indices", target="threads")
    def _load_npz(self, index, src=None, fmt=None, components=None, *args, **kwargs):
        _ = fmt
        path = get_path(self, index, src)

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

    def _show_logs(self, index=None, start=None, end=None, plot_mask=False, subplot_size=(15, 2)):
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

    # Channels processing methods

    @staticmethod
    def _get_mnemonics_key(component):
        return component + "_mnemonics"

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

    @inbatch_parallel(init="indices", target="threads")
    def _filter_channels(self, index, mnemonics=None, indices=None, invert_mask=False, *, components):
        mask = self._generate_mask(index, mnemonics, indices, invert_mask, components=components)
        if np.sum(mask) == 0:
            raise ValueError("All channels cannot be dropped")
        i = self.get_pos(None, components, index)
        getattr(self, components)[i] = getattr(self, components)[i][mask]
        mnemonics_key = self._get_mnemonics_key(components)
        if mnemonics_key in self.meta[i]:
            self.meta[i][mnemonics_key] = np.asarray(self.meta[i][mnemonics_key])[mask]

    @action
    @for_each_component
    def drop_channels(self, mnemonics=None, indices=None, *, components="logs"):
        """Drop channels from ``components`` whose names are in ``mnemonics``
        or whose indices are in ``indices``.

        Parameters
        ----------
        mnemonics : str or list or tuple, optional
            Mnemonics of channels to be dropped from a batch.
        indices : int or list or tuple, optional
            Indices of channels to be dropped from a batch.
        components : str or array-like, optional
            Components to be processed. Defaults to ``logs``.

        Returns
        -------
        batch : WellLogsBatch
            Batch with dropped channels. Changes its ``components`` and
            ``meta`` inplace.

        Raises
        ------
        ValueError
            If mnemonics for any of the ``components`` are not defined in
            ``self.meta``.
        ValueError
            If both ``mnemonics`` and ``indices`` are empty.
        ValueError
            If all channels should be dropped.
        """
        if mnemonics is None and indices is None:
            raise ValueError("Both mnemonics and indices cannot be empty")
        return self._filter_channels(mnemonics, indices, invert_mask=True, components=components)

    @action
    @for_each_component
    def keep_channels(self, mnemonics=None, indices=None, *, components="logs"):
        """Drop channels from ``components`` whose names are not in
        ``mnemonics`` and whose indices are not in ``indices``.

        Parameters
        ----------
        mnemonics : str or list or tuple, optional
            Mnemonics of channels to be kept in a batch.
        indices : int or list or tuple, optional
            Indices of channels to be kept in a batch.
        components : str or array-like, optional
            Components to be processed. Defaults to ``logs``.

        Returns
        -------
        batch : WellLogsBatch
            Batch with dropped channels. Changes its ``components`` and
            ``meta`` inplace.

        Raises
        ------
        ValueError
            If mnemonics for any of the ``components`` are not defined in
            ``self.meta``.
        ValueError
            If both ``mnemonics`` and ``indices`` are empty.
        ValueError
            If all channels should be dropped.
        """
        if mnemonics is None and indices is None:
            raise ValueError("Both mnemonics and indices cannot be empty")
        return self._filter_channels(mnemonics, indices, invert_mask=False, components=components)

    @action
    @for_each_component
    @inbatch_parallel(init="indices", target="threads")
    def rename_channels(self, index, rename_dict, *, components="logs"):
        """Rename channels of ``components`` with corresponding values from
        ``rename_dict``.

        Parameters
        ----------
        rename_dict : dict
            Dictionary containing ``(old mnemonic : new mnemonic)`` pairs.
            Mnemonics, that are not specified in ``rename_dict`` keys, remain
            unchanged.
        components : str or array-like, optional
            Components to be processed. Defaults to ``logs``.

        Returns
        -------
        batch : WellLogsBatch
            Batch with renamed channels. Changes ``self.meta`` inplace.
        """
        i = self.get_pos(None, components, index)
        mnemonics_key = self._get_mnemonics_key(components)
        old_mnemonics = self.meta[i].get(mnemonics_key)
        if old_mnemonics is not None:
            new_mnemonics = np.array([rename_dict.get(name, name) for name in old_mnemonics], dtype=object)
            self.meta[i][mnemonics_key] = new_mnemonics

    @action
    @for_each_component
    @inbatch_parallel(init="indices", target="threads")
    def reorder_channels(self, index, new_order, *, components="logs"):
        """Change the order of channels of specified ``components`` according
        to the ``new_order``.

        Parameters
        ----------
        new_order : array_like
            A list of mnemonics, specifying the order of channels in the
            transformed ``components``.
        components : str or array-like, optional
            Components to be processed. Defaults to ``logs``.

        Returns
        -------
        batch : WellLogsBatch
            Batch with reordered channels. Changes its ``components`` and
            ``meta`` inplace.

        Raises
        ------
        ValueError
            If mnemonics for any of the ``components`` are not defined in
            ``self.meta``.
        ValueError
            If unknown mnemonics are specified.
        ValueError
            If all channels should be dropped.
        """
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

    @action
    @inbatch_parallel(init="indices", target="threads")
    def split_by_mnemonics(self, index, mnemonics, component_from, component_to):
        """Move channels with given ``mnemonics`` from ``component_from`` to
        ``component_to``.

        Parameters
        ----------
        mnemonics : str or list or tuple
            Mnemonics of channels to be moved.
        component_from : str
            A component to move channels from.
        component_to : str
            A component to move channels to.

        Returns
        -------
        batch : WellLogsBatch
            Batch with moved channels. Changes its ``component_from``,
            ``component_to`` and ``meta`` inplace.

        Raises
        ------
        ValueError
            If mnemonics for ``component_from`` are not defined in
            ``self.meta``.
        """
        mask = self._generate_mask(index, mnemonics, components=component_from)
        i = self.get_pos(None, component_from, index)
        mnemonics_key_to = self._get_mnemonics_key(component_to)
        mnemonics_key_from = self._get_mnemonics_key(component_from)

        getattr(self, component_to)[i] = getattr(self, component_from)[i][mask]
        self.meta[i][mnemonics_key_to] = self.meta[i][mnemonics_key_from][mask]

        getattr(self, component_from)[i] = getattr(self, component_from)[i][~mask]
        self.meta[i][mnemonics_key_from] = self.meta[i][mnemonics_key_from][~mask]

    @action
    def copy_components(self, components_from, components_to):
        """Create copies of components.

        Parameters
        ----------
        components_from : str or list or tuple
            Components to be copied.
        components_to : str or list or tuple
            Components to save the copies.

        Returns
        -------
        batch : WellLogsBatch
            Batch with copied components.

        Raises
        ------
        ValueError
            If `components_from` and `components_to` have different lengths.
        """
        components_from = self._preprocess_components(components_from)
        components_to = self._preprocess_components(components_to)
        if len(components_from) != len(components_to):
            err_msg = ("components_from and components_to must be convertible to lists of the same length, " +
                       "but lists of lengths {} and {} were given").format(len(components_from), len(components_to))
            raise ValueError(err_msg)
        for component_from, component_to in zip(components_from, components_to):
            setattr(self, component_to, deepcopy(getattr(self, component_from)))
        return self

    @action
    @inbatch_parallel(init="indices", target="threads")
    def fill_channels(self, index, channels, channels_mask=None, value=0, axis=0, *, components):
        """Fill `channels` with `value`.

        Parameters
        ----------
        channels : list or tuple or callable
            Indices of channels to be filled with `value`. If `callable`, it
            should accept component length along specified `axis` and return
            indices of channels to be filled.
        channels_mask : str
            Component to save a binary mask with ones for padded channels. If
            `None`, the mask will not be saved.
        value : float
            Value to be used for filling.
        axis : int
            Channels axis.
        components : str or list or tuple
            Components to be processed.

        Returns
        -------
        batch : WellLogsBatch
            Batch with channels, filled with `value`. Changes its `components`
            inplace.
        """
        components = self._preprocess_components(components)
        i = self.get_pos(None, components[0], index)
        data = getattr(self, components[0])[i]
        n_channels = data.shape[axis]

        if callable(channels):
            channels = channels(n_channels)

        indices = [slice(None)] * len(data.shape)
        indices[axis] = channels

        for component in components:
            getattr(self, component)[i][indices] = value

        if channels_mask:
            getattr(self, channels_mask)[i] = np.isin(np.arange(n_channels), channels).astype('float32')

    # Logs processing methods

    @inbatch_parallel(init="indices", post="_assemble_drop_nans", target="threads")
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
            raise SkipBatchException("All batch data was dropped")
        indices, *components = zip(*results)
        batch = self.__class__(DatasetIndex(indices))
        for comp_name, comp_data in zip(components_to_split + components_to_copy, components):
            setattr(batch, comp_name, self._to_array(comp_data))
        return batch

    @action
    def drop_nans(self, *, components_to_split=None, components_to_copy=None):
        """Select connected areas of well logs without ``nan`` values and
        store them in a new batch instance under modified indices.

        Parameters
        ----------
        components_to_split : str or array-like, optional
            Components, whose elements are indexed by positions of selected
            connected areas.
        components_to_copy : str or array-like, optional
            Components, whose elements are copied for each selected connected
            area.

        Returns
        -------
        batch : WellLogsBatch
            Batch with dropped ``nan`` values. Creates a new ``WellLogsBatch``
            instance.

        Raises
        ------
        SkipBatchException
            If all batch data was dropped.
        """
        def _init_components(components):
            return set() if components is None else set(self._preprocess_components(components))

        components_to_split = sorted(_init_components(components_to_split) | {"dept", "logs"})
        components_to_copy = sorted(_init_components(components_to_copy) | {"meta"})
        return self._drop_nans(components_to_split, components_to_copy)

    @action
    @for_each_component
    @inbatch_parallel(init="indices", target="threads")
    def fill_nans(self, index, fill_value=0, *, components):
        """Replace ``nan`` values in specified ``components`` with given
        ``fill_value``.

        Parameters
        ----------
        fill_value : int or float
            A value to replace ``nan`` with.
        components : str or array-like, optional
            Components to be processed.

        Returns
        -------
        batch : WellLogsBatch
            Batch with replaced ``nan`` values. Changes its ``components``
            inplace.
        """
        comp = getattr(self, components)[self.get_pos(None, components, index)]
        comp[np.isnan(comp)] = fill_value

    def _filter_batch(self, keep_mask):
        indices = self.indices[keep_mask]
        if len(indices) == 0:
            raise SkipBatchException("All batch data was dropped")
        batch = self.__class__(DatasetIndex(indices))
        for comp in self.components:
            setattr(batch, comp, getattr(self, comp)[keep_mask])
        return batch

    @action
    def drop_short_logs(self, min_length, axis=-1):
        """Drop short logs from a batch.

        Parameters
        ----------
        min_length : positive int
            Minimal log length.
        axis : int, optional
            Axis along which length is calculated. Defaults to the last axis.

        Returns
        -------
        batch : WellLogsBatch
            Filtered batch. Creates a new ``WellLogsBatch`` instance.

        Raises
        ------
        SkipBatchException
            If all batch data was dropped.
        """
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

    @action
    @inbatch_parallel(init="indices", target="threads")
    def crop(self, index, length, step, pad_value=0, *, components):
        """Crop segments from ``components`` along the last axis with given
        ``length`` and ``step``.

        If the length of a component along the last axis is less than
        ``length``, it is padded to the left with ``pad_value``.

        Notice, that each element of the resulting components will have an
        additional axis with index 0, along which crops were stacked.

        Parameters
        ----------
        length : positive int
            Length of each segment along the last axis.
        step : positive int
            The number of array elements between starting indices of cropped
            segments.
        pad_value : float, optional
            Padding value. Defaults to 0.
        components : str or array-like
            Components to crop segments from.

        Returns
        -------
        batch : WellLogsBatch
            A batch of cropped components. Changes its ``components`` inplace.

        Raises
        ------
        ValueError
            If ``length`` or ``step`` are negative or non-integer.
        """
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

        components = self._preprocess_components(components)
        crop_positions = np.arange(n_crops) * step
        if "dept" in components:
            padded_dept = self._pad_dept(self.dept[i], new_length)
            self.dept[i] = bt.crop(padded_dept, length, crop_positions)
            components.remove("dept")
        for comp in components:
            padded_comp = self._pad(getattr(self, comp)[i], new_length, pad_value)
            getattr(self, comp)[i] = bt.crop(padded_comp, length, crop_positions)

    @action
    @inbatch_parallel(init="indices", target="threads")
    def random_crop(self, index, length, n_crops, pad_value=0, *, components):
        """Crop ``n_crops`` segments from ``components`` along the last axis
        with random start positions and given ``length``.

        If the length of a component along the last axis is less than
        ``length``, it is padded to the left with ``pad_value``.

        Notice, that each element of the resulting components will have an
        additional axis with index 0, along which crops were stacked.

        Parameters
        ----------
        length : positive int
            Length of each segment along the last axis.
        n_crops : positive int
            The number of segments to be cropped.
        pad_value : float, optional
            Padding value. Defaults to 0.
        components : str or array-like
            Components to crop segments from.

        Returns
        -------
        batch : WellLogsBatch
            A batch of cropped components. Changes its ``components`` inplace.

        Raises
        ------
        ValueError
            If ``length`` or ``n_crops`` are negative or non-integer.
        """
        self._check_positive_int(length, "Segment length")
        self._check_positive_int(n_crops, "The number of segments")
        i = self.get_pos(None, "logs", index)

        components = self._preprocess_components(components)
        if self.logs[i].shape[-1] < length:
            if "dept" in components:
                padded_dept = self._pad_dept(self.dept[i], length)
                self.dept[i] = np.tile(padded_dept, (n_crops,) + (1,) * padded_dept.ndim)
                components.remove("dept")
            for comp in components:
                padded_comp = self._pad(getattr(self, comp)[i], length, pad_value)
                getattr(self, comp)[i] = np.tile(padded_comp, (n_crops,) + (1,) * padded_comp.ndim)
        else:
            crop_positions = np.random.randint(0, self.logs[i].shape[-1] - length + 1, n_crops)
            for comp in components:
                getattr(self, comp)[i] = bt.crop(getattr(self, comp)[i], length, crop_positions)

    @for_each_component
    def _split_by_well(self, split_indices, *, components):
        comp = self._to_array(np.split(getattr(self, components), split_indices))
        setattr(self, components, comp)

    @action
    def split_by_well(self, *, components):
        """Split an array, stored in each of the specified ``components``,
        along axis 0 into an array of arrays so that the shape of the
        resulting arrays along axis 0 equals to ``n_crops`` value, stored in
        the corresponding ``meta`` dict by the ``WellLogsBatch.crop`` method.

        This method is useful for splitting model predictions for the
        concatenated batch of crops into separate predictions for crops from
        each individual log in the original batch.

        Returns
        -------
        batch : WellLogsBatch
            A batch with split components. Changes its ``components`` inplace.

        Raises
        ------
        ValueError
            If ``WellLogsBatch.crop`` method was not called beforehand.
        """
        split_indices = [meta.get("n_crops") for meta in self.meta]
        if any(ix is None for ix in split_indices):
            raise ValueError("The number of log segments for a well is unknown")
        split_indices = np.cumsum(split_indices)[:-1]
        return self._split_by_well(split_indices, components=components)

    @for_each_component
    @inbatch_parallel(init="indices", target="for")
    def _aggregate(self, index, agg_fn, *, components):
        i = self.get_pos(None, components, index)
        comp = getattr(self, components)[i]
        meta = self.meta[i]
        tmp_comp = bt.aggregate(comp, meta["split_step"], agg_fn)
        getattr(self, components)[i] = tmp_comp[..., meta["pad_length"]:]

    @action
    def aggregate(self, agg_fn="mean", *, components):
        """Undo the application of ``WellLogsBatch.crop`` method by
        aggregating the resulting crops using ``agg_fn``.

        Parameters
        ----------
        agg_fn : str or callable
            An aggregation function to combine values in case of crop overlay.
            If ``str``, it must be a valid numpy NaN-aggregation function
            name.
        components : str or array-like
            Components to be aggregated.

        Returns
        -------
        batch : WellLogsBatch
            Batch with aggregated components. Changes its ``components``
            inplace.

        Raises
        ------
        ValueError
            If ``agg_fn`` is not a callable or a valid numpy aggregation
            function name.
        """
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

    @staticmethod
    def _insert_axes(*arrays, axes):
        axes = np.array(axes).ravel()
        res = []
        for arr in arrays:
            ndim = arr.ndim + len(axes)
            axes = np.where(axes < 0, axes + ndim, axes)
            shape = np.ones(ndim, dtype=np.int32)
            shape[np.setdiff1d(np.arange(ndim), axes)] = arr.shape
            res.append(arr.reshape(shape))
        return tuple(res)

    @inbatch_parallel(init="indices", target="for")
    def _norm_mean_std(self, index, axis, mean, std, eps, *, component):
        i = self.get_pos(None, component, index)
        comp = getattr(self, component)[i]
        if mean is None:
            mean = np.nanmean(comp, axis=axis)
        if std is None:
            std = np.nanstd(comp, axis=axis)

        mean, std = self._insert_axes(mean, std, axes=axis)  # pylint: disable=unbalanced-tuple-unpacking
        getattr(self, component)[i] = (comp - mean) / (std + eps)

    @action
    def norm_mean_std(self, axis=-1, mean=None, std=None, eps=1e-10, *, components):
        """Standardize components along specified axes by subtracting the mean
        and scaling to unit variance.

        Parameters
        ----------
        axis : ``None`` or int or tuple of ints, optional
            Axis or axes along which standardization is performed. Defaults to
            the last axis.
        mean : None or ndarray or tuple or list of ndarrays
            Mean to be subtracted. If ``None``, it is calculated independently
            for each element of the batch along specified ``axis``.
        std : None or ndarray or tuple or list of ndarrays
            Standard deviation to be divided by. If ``None``, it is calculated
            independently for each element of the batch along specified
            ``axis``.
        eps: float, optional
            A small float to be added to the denominator to avoid division by
            zero.
        components : str or array-like
            Components to be standardized. If multiple components are given,
            ``mean`` and ``std`` must be either single ``ndarrays``, that will
            be used for each component, or tuples or lists of the same length,
            representing mean values and standard deviations of the
            corresponding components.

        Returns
        -------
        batch : WellLogsBatch
            Batch with standardized components. Changes its ``components``
            inplace.
        """
        components = np.asarray(components).ravel()
        if not isinstance(mean, (tuple, list)):
            mean = [mean] * len(components)
        if not isinstance(std, (tuple, list)):
            std = [std] * len(components)

        if not (len(mean) == len(std) == len(components)):
            raise ValueError("Mean, std and components lists must be of the same length")

        for comp_mean, comp_std, comp in zip(mean, std, components):
            self._norm_mean_std(axis, comp_mean, comp_std, eps, component=comp)
        return self

    @inbatch_parallel(init="indices", target="for")
    def _norm_min_max(self, index, axis, min, max, *, component):  # pylint: disable=redefined-builtin
        i = self.get_pos(None, component, index)
        comp = getattr(self, component)[i]
        if min is None:
            min = np.nanmin(comp, axis=axis)
        if max is None:
            max = np.nanmax(comp, axis=axis)

        min, max = self._insert_axes(min, max, axes=axis)  # pylint: disable=unbalanced-tuple-unpacking
        getattr(self, component)[i] = (comp - min) / (max - min)

    @action
    def norm_min_max(self, axis=-1, min=None, max=None, *, components):  # pylint: disable=redefined-builtin
        """Linearly scale components to a [0, 1] range along specified axes.

        Parameters
        ----------
        axis : ``None`` or int or tuple of ints, optional
            Axis or axes along which scaling is performed. Defaults to the
            last axis.
        min : None or ndarray or tuple or list of ndarrays
            Minimum values of the components. If ``None``, it is calculated
            independently for each element of the batch along specified
            ``axis``.
        max : None or ndarray or tuple or list of ndarrays
            Maximum values of the components. If ``None``, it is calculated
            independently for each element of the batch along specified
            ``axis``.
        components : str or array-like
            Components to be scaled. If multiple components are given, ``min``
            and ``max`` must be either single ``ndarrays``, that will be used
            for each component, or tuples or lists of the same length,
            representing minimum and maximum values of the corresponding
            components.

        Returns
        -------
        batch : WellLogsBatch
            Batch with scaled components. Changes its ``components`` inplace.
        """
        components = np.asarray(components).ravel()
        if not isinstance(min, (tuple, list)):
            min = [min] * len(components)
        if not isinstance(max, (tuple, list)):
            max = [max] * len(components)

        if not (len(min) == len(max) == len(components)):
            raise ValueError("Min, max and components lists must be of the same length")

        for comp_min, comp_max, comp in zip(min, max, components):
            self._norm_min_max(axis, comp_min, comp_max, component=comp)
        return self
