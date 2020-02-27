"""Implements WellSegment - a class, representing a contiguous part of a well.
"""
# pylint: disable=no-member,protected-access

import os
import re
import json
import base64
import shutil
from copy import copy, deepcopy
from glob import glob
from functools import reduce
from itertools import chain, repeat
from math import ceil

import numpy as np
import pandas as pd
import lasio
import PIL
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
import cv2
from plotly import graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, plot, iplot

from .abstract_classes import AbstractWellSegment
from .matching import select_contigious_intervals, match_boring_sequence, find_best_shifts, create_zero_shift
from .joins import cross_join, between_join, fdtd_join
from .utils import to_list, process_columns, leq_notclose, leq_close, geq_close
from .exceptions import SkipWellException, DataRegularityError


def add_attr_properties(cls):
    """Add missing properties for lazy loading of `WellSegment` table-based
    attributes."""
    for attr in cls.attrs_depth_index + cls.attrs_fdtd_index + cls.attrs_no_index:
        if hasattr(cls, attr):
            continue
        def prop(self, attr=attr):
            if getattr(self, "_" + attr) is None:
                getattr(self, "load_" + attr)()
            return getattr(self, "_" + attr)
        setattr(cls, attr, property(prop))
    return cls


def add_attr_loaders(cls):
    """Add missing loaders of `WellSegment` table-based attributes."""
    attr_iter = chain(
        zip(cls.attrs_depth_index, repeat(cls._load_depth_df)),
        zip(cls.attrs_fdtd_index, repeat(cls._load_fdtd_df)),
        zip(cls.attrs_no_index, repeat(cls._load_df))
    )
    for attr, loader in attr_iter:
        if hasattr(cls, "load_" + attr):
            continue
        def load_factory(attr, loader):
            def load(self, *args, **kwargs):
                data = loader(self, self._get_full_name(self.path, attr), *args, **kwargs)
                setattr(self, "_" + attr, data)
                return self
            return load
        setattr(cls, "load_" + attr, load_factory(attr, loader))
    return cls


@add_attr_properties
@add_attr_loaders
class WellSegment(AbstractWellSegment):
    """A class, representing a contiguous part of a well.

    Initially an instance of `Well` class consists of a single segment,
    representing the whole well. Subsequently, several `Well` methods, such as
    `crop`, `random_crop` or `drop_nans`, increase the number of segments,
    storing them in a tree-based structure.

    Unlike `Well`, which nearly always redirects method calls to its segments,
    `WellSegment` actually stores well data and implements its processing
    logic.

    Each table-based attribute of `WellSegment`, described in `Attributes`
    section, can be loaded in two different ways:
    1. via corresponding load method, e.g. `load_logs` or `load_layers`. All
       specified arguments will be passed to a loader, responsible for file's
       extension.
    2. via lazy loading mechanism, when an attribute is loaded at the time of
       the first access. In this case, default loading parameters are used.
    `core_dl` and `core_uv` attributes are loaded either by accessing them for
    the first time, or by calling `load_core` method.

    Parameters
    ----------
    path : str
        A path to a directory with well data, containing:
        - `meta.json` - a json dict with the following keys:
            - `name` - well name
            - `field` - field name
            - `depth_from` - minimum depth entry in the well logs
            - `depth_to` - maximum depth entry in the well logs
          These values will be stored as instance attributes.
        - `samples_dl` and `samples_uv` (optional) - directories, containing
          daylight and ultraviolet images of core samples respectively. Images
          of the same sample must have the same name in both dirs.
        - Optional `.csv`, `.las` or `.feather` file for certain class
          attributes (see more details in the `Attributes` section).
    core_width : positive float, optional
        The width of core samples in cm. Defaults to 10 cm.
    pixels_per_cm : positive int, optional
        The number of pixels in cm used to determine the loaded width of core
        sample images. Image height is calculated so as to keep the aspect
        ratio. Defaults to 5 pixels.

    Attributes
    ----------
    name : str
        Well name, loaded from `meta.json`.
    field : str
        Field name, loaded from `meta.json`.
    depth_from : float
        Minimum depth entry in the well logs, loaded from `meta.json`.
    depth_to : float
        Maximum depth entry in the well logs, loaded from `meta.json`.
    actual_depth_to : float
        Actual maximum segment depth. It is used when the segment is padded
        by the `crop` method and then used in `Well.aggregate` to drop padded
        part of the segment.
    logs : pandas.DataFrame
        Well logs, indexed by depth. Depth log in a source file must have
        `DEPTH` mnemonic. Mnemonics of the same log type in `logs` and
        `core_logs` should match. Loaded from the file with the same name from
        the well directory.
    inclination : pandas.DataFrame
        Well inclination. Loaded from the file with the same name from the
        well directory.
    layers : pandas.DataFrame
        Stratum layers names, indexed by depth range. Loaded from the file
        with the same name, having the following structure: DEPTH_FROM -
        DEPTH_TO - LAYER.
    boring_intervals : pandas.DataFrame
        Depths of boring intervals with core recovery in meters, indexed by
        depth range. Loaded from the file with the same name, having the
        following structure: DEPTH_FROM - DEPTH_TO - CORE_RECOVERY.
    boring_sequences : pandas.DataFrame
        Depth ranges of contiguous boring intervals, extracted one after
        another. If the file with the same name exists in the well directory,
        then `boring_sequences` is loaded from the file. Otherwise, it is
        calculated from `boring_intervals`. If core-to-log matching is
        performed, then extra `MODE` and `R2` columns with matching parameters
        and results are created.
    core_properties : pandas.DataFrame
        Properties of core plugs, indexed by depth. Depth column in source
        file must be called `DEPTH`. Loaded from the file with the same name
        from the well directory.
    core_lithology : pandas.DataFrame
        Lithological description of core samples, indexed by depth range.
        Loaded from the file with the same name, having the following
        structure: DEPTH_FROM - DEPTH_TO - FORMATION - COLOR - GRAINSIZE -
        GRAINCONTENT.
    core_logs : pandas.DataFrame
        Core logs, indexed by depth. Depth log in a source file must have
        `DEPTH` mnemonic. Mnemonics of the same log type in `logs` and
        `core_logs` should match. Loaded from the file with the same name from
        the well directory.
    samples : pandas.DataFrame
        Names of core sample images with their depth ranges. Loaded from the
        file with the same name, having the following structure: DEPTH_FROM -
        DEPTH_TO - SAMPLE.
    core_dl : numpy.ndarray
        Concatenated daylight image of all core samples in the segment. If
        core samples are absent for several depth ranges, corresponding
        `core_dl` values are equal to `numpy.nan`. Loaded from images in
        `samples_dl` directory, requires `samples` file to exist in the well
        directory.
    core_uv : numpy.ndarray
        Concatenated ultraviolet image of all core samples in the segment. If
        core samples are absent for several depth ranges, corresponding
        `core_uv` values are equal to `numpy.nan`. Loaded from images in
        `samples_uv` directory, requires `samples` file to exist in the well
        directory.
    """

    attrs_depth_index = ("logs", "core_properties", "core_logs")
    attrs_fdtd_index = ("layers", "boring_sequences", "boring_intervals", "core_lithology", "samples")
    attrs_no_index = ("inclination",)
    attrs_image = ("core_uv", "core_dl")

    def __init__(self, path, *args, core_width=10, pixels_per_cm=5, **kwargs):
        super().__init__()
        _ = args, kwargs
        self.path = path
        self.core_width = core_width
        self.pixels_per_cm = pixels_per_cm

        with open(os.path.join(self.path, "meta.json")) as meta_file:
            meta = json.load(meta_file)
        self.name = meta["name"]
        self.field = meta["field"]
        self.depth_from = float(meta["depth_from"])
        self.depth_to = float(meta["depth_to"])
        self.actual_depth_to = None

        self.has_samples = self._has_file("samples")

        self._logs = None
        self._inclination = None
        self._layers = None
        self._boring_intervals = None
        self._boring_sequences = None
        self._core_properties = None
        self._core_lithology = None
        self._core_logs = None
        self._samples = None
        self._core_dl = None
        self._core_uv = None
        self._boring_intervals_deltas = None
        self._core_lithology_deltas = None
        self._tolerance = 1e-3  # A tolerance to compare float-valued depths for equality.

        # In order to unify aggregate behavior in case of loaded and calculated `boring_sequences`,
        # they should be computed explicitly during the creation of a segment.
        if self._has_file("boring_intervals") and not self._has_file("boring_sequences"):
            _ = self.boring_sequences

    @property
    def length(self):
        """float: Length of the segment in meters."""
        return self.depth_to - self.depth_from

    @property
    def logs_step(self):
        """float: Step between measurements in `logs` in meters."""
        return round((self.logs.index[1:] - self.logs.index[:-1]).min(), 2)

    @property
    def boring_sequences(self):
        """pandas.DataFrame: Depth ranges of contiguous boring intervals,
        extracted one after another."""
        if self._boring_sequences is None:
            if self._has_file("boring_sequences"):
                self.load_boring_sequences()
            else:
                self._calc_boring_sequences()
        return self._boring_sequences

    def _calc_boring_sequences(self):
        """Calculate boring sequences given boring intervals."""
        data = []
        for segment in select_contigious_intervals(self.boring_intervals.reset_index()):
            data.append([segment["DEPTH_FROM"].min(), segment["DEPTH_TO"].max()])
        self._boring_sequences = pd.DataFrame(data, columns=["DEPTH_FROM", "DEPTH_TO"])
        self._boring_sequences.set_index(["DEPTH_FROM", "DEPTH_TO"], inplace=True)

    @staticmethod
    def _get_extension(path):
        """Get file extension from its name."""
        return os.path.splitext(path)[1][1:]

    @staticmethod
    def _load_las(path, *args, **kwargs):
        """Load a `.las` file into a `DataFrame`."""
        return lasio.read(path, *args, **kwargs).df().reset_index()

    @staticmethod
    def _load_csv(path, *args, **kwargs):
        """Load a `.csv` file into a `DataFrame`."""
        return pd.read_csv(path, *args, **kwargs)

    @staticmethod
    def _load_feather(path, *args, **kwargs):
        """Load a `.feather` file into a `DataFrame`."""
        return pd.read_feather(path, *args, **kwargs)

    def _load_df(self, path, *args, **kwargs):
        """Load a `DataFrame` from a table format (`.las`, `.csv` or
        `.feather`) depending on its extension."""
        ext = self._get_extension(path)
        if not hasattr(self, "_load_" + ext):
            raise ValueError("A loader for data in {} format is not implemented".format(ext))
        return getattr(self, "_load_" + ext)(path, *args, **kwargs)

    def _filter_depth_df(self, df):
        """Keep only depths between `self.depth_from` and `self.depth_to` in a
        `DataFrame`, indexed by depth."""
        df = df[self.depth_from:self.depth_to]
        if len(df) == 0:
            return df
        both_bounds_in_df = np.allclose([self.depth_from, self.depth_to],
                                        [df.index[0], df.index[-1]], rtol=0, atol=self._tolerance)
        if both_bounds_in_df:  # See __getitem__ docstring for an explanation.
            df.drop(df.index[-1], inplace=True)
        return df

    def _load_depth_df(self, path, *args, **kwargs):
        """Load a `DataFrame`, indexed by depth, from a table format and keep
        only depths between `self.depth_from` and `self.depth_to`."""
        df = self._load_df(path, *args, **kwargs).set_index("DEPTH")
        df = self._filter_depth_df(df)
        return df

    def _filter_fdtd_df(self, df):
        """Keep only depths between `self.depth_from` and `self.depth_to` in a
        `DataFrame`, indexed by depth range."""
        if len(df) == 0:
            return df
        depth_from, depth_to = zip(*df.index.values)
        mask = (np.array(depth_from) < self.depth_to) & (self.depth_from < np.array(depth_to))
        return df[mask]

    def _load_fdtd_df(self, path, *args, **kwargs):
        """Load a `DataFrame`, indexed by depth range, from a table format and
        keep only depths between `self.depth_from` and `self.depth_to`."""
        df = self._load_df(path, *args, **kwargs).set_index(["DEPTH_FROM", "DEPTH_TO"])
        df = self._filter_fdtd_df(df)
        return df

    def _has_file(self, name):
        """Check that exactly one file with a given name and any extension
        exists in a well directory."""
        files = glob(os.path.join(self.path, name + ".*"))
        if len(files) == 1:
            return True
        return False

    @classmethod
    def _get_full_name(cls, path, name):
        """Get full name of a file with a given `name` in a dir, specified in
        `path`.

        Parameters
        ----------
        path : str
            A path to a directiry to search for a file.
        name : str
            File name with or without extension.

        Returns
        -------
        full_name : str
            Full name of a file.

        Raises
        ------
        FileNotFoundError
            If a file does not exist in a dir.
        OSError
            If extension is not specified and several files with given name
            exist.
        """
        ext = cls._get_extension(name)
        if ext != "":
            full_name = os.path.join(path, name)
            if os.path.exists(full_name):
                return full_name
            raise FileNotFoundError("A file {} does not exist in {}".format(name, path))

        files = glob(os.path.join(path, name + ".*"))
        if len(files) == 0:
            raise FileNotFoundError("A file {} does not exist in {}".format(name, path))
        if len(files) > 1:
            raise OSError("Several files called {} are found in {}".format(name, path))
        return files[0]

    @property
    def core_dl(self):
        """numpy.ndarray: Concatenated daylight image of all core samples in
        the segment."""
        if self._core_dl is None:
            self.load_core()
        return self._core_dl

    @property
    def core_uv(self):
        """numpy.ndarray: Concatenated ultraviolet image of all core samples
        in the segment."""
        if self._core_uv is None:
            self.load_core()
        return self._core_uv

    @staticmethod
    def _load_image(path):
        """Open an image in `PIL` format."""
        return PIL.Image.open(path) if os.path.isfile(path) else None

    @staticmethod
    def _match_samples(dl_img, uv_img, height, width):
        """Match core samples in daylight and ultraviolet by resizing them to
        the given shape."""
        if (dl_img is not None) and (dl_img is not None):
            # TODO: contour matching instead of resizing
            dl_img = np.array(dl_img.resize((width, height), resample=PIL.Image.LANCZOS))
            uv_img = np.array(uv_img.resize((width, height), resample=PIL.Image.LANCZOS))
        elif dl_img is not None:
            dl_img = np.array(dl_img.resize((width, height), resample=PIL.Image.LANCZOS))
            uv_img = np.full((height, width, 3), np.nan, dtype=np.float32)
        else:
            dl_img = np.full((height, width, 3), np.nan, dtype=np.float32)
            uv_img = np.array(uv_img.resize((width, height), resample=PIL.Image.LANCZOS))
        return dl_img, uv_img

    def _meters_to_pixels(self, meters):
        """Convert meters to pixels given conversion ratio in
        `self.pixels_per_cm`."""
        return int(round(meters * 100)) * self.pixels_per_cm

    def load_core(self, core_width=None, pixels_per_cm=None):
        """Load core images in daylight and ultraviolet.

        If any of method arguments are not specified, those, passed to
        `__init__`, will be used. Otherwise, they will be overridden in
        `self`.

        Parameters
        ----------
        core_width : positive float, optional
            The width of core samples in centimeters.
        pixels_per_cm : positive int, optional
            The number of pixels in centimeters used to determine the loaded
            width of core sample images. Image height is calculated so as to
            keep the aspect ratio.

        Returns
        -------
        self : AbstractWellSegment or a child class
            Self with core images loaded for each segment.
        """
        self.core_width = core_width if core_width is not None else self.core_width
        self.pixels_per_cm = pixels_per_cm if pixels_per_cm is not None else self.pixels_per_cm

        height = self._meters_to_pixels(self.depth_to - self.depth_from)
        width = self.core_width * self.pixels_per_cm
        core_dl = np.full((height, width, 3), np.nan, dtype=np.float32)
        core_uv = np.full((height, width, 3), np.nan, dtype=np.float32)

        for (sample_depth_from, sample_depth_to), sample_name in self.samples["SAMPLE"].iteritems():
            sample_height = self._meters_to_pixels(sample_depth_to - sample_depth_from)
            sample_name = str(sample_name)

            dl_path = self._get_full_name(os.path.join(self.path, "samples_dl"), sample_name)
            dl_img = self._load_image(dl_path)
            uv_path = self._get_full_name(os.path.join(self.path, "samples_uv"), sample_name)
            uv_img = self._load_image(uv_path)
            dl_img, uv_img = self._match_samples(dl_img, uv_img, sample_height, width)

            top_crop = max(0, self._meters_to_pixels(self.depth_from - sample_depth_from))
            bottom_crop = sample_height - max(0, self._meters_to_pixels(sample_depth_to - self.depth_to))
            dl_img = dl_img[top_crop:bottom_crop]
            uv_img = uv_img[top_crop:bottom_crop]

            insert_pos = max(0, self._meters_to_pixels(sample_depth_from - self.depth_from))
            core_dl[insert_pos:insert_pos+dl_img.shape[0]] = dl_img
            core_uv[insert_pos:insert_pos+uv_img.shape[0]] = uv_img

        self._core_dl = core_dl
        self._core_uv = core_uv
        return self

    def dump(self, path):
        """Dump well segment data.

        Segment attributes are saved in the following manner:
        - `name`, `field`, `depth_from` and `depth_to` attributes are saved in
          `meta.json` file.
        - `core_dl` and `core_uv` are not saved. Instead, `samples_dl` and
          `samples_uv` directories are copied if exist.
        - All other attributes are dumped in feather format.

        Parameters
        ----------
        path : str
            A path to a directory, where well dir with dump will be created.

        Returns
        -------
        self : AbstractWellSegment or a child class
            Self unchanged.
        """
        path = os.path.join(path, self.name)
        if not os.path.exists(path):
            os.makedirs(path)

        meta = {
            "name": self.name,
            "field": self.field,
            "depth_from": self.depth_from,
            "depth_to": self.depth_to,
        }
        with open(os.path.join(path, "meta.json"), "w") as meta_file:
            json.dump(meta, meta_file)

        for attr in self.attrs_depth_index + self.attrs_fdtd_index + self.attrs_no_index:
            attr_val = getattr(self, "_" + attr)
            if attr_val is None:
                try:
                    shutil.copy2(self._get_full_name(self.path, attr), path)
                except Exception: # pylint: disable=broad-except
                    pass
            else:
                if attr not in self.attrs_no_index:
                    attr_val = attr_val.reset_index()
                attr_val.to_feather(os.path.join(path, attr + ".feather"))

        samples_dl_path = os.path.join(self.path, "samples_dl")
        if os.path.exists(samples_dl_path):
            shutil.copytree(samples_dl_path, os.path.join(path, "samples_dl"), copy_function=os.link)

        samples_uv_path = os.path.join(self.path, "samples_uv")
        if os.path.exists(samples_uv_path):
            shutil.copytree(samples_uv_path, os.path.join(path, "samples_uv"), copy_function=os.link)

        return self

    @staticmethod
    def _encode(img_path):
        """Encode an image into a `plotly` representation."""
        with open(img_path, "rb") as img:
            encoded_img = base64.b64encode(img.read()).decode()
        encoded_img = "data:image/png;base64," + encoded_img
        return encoded_img

    def plot(self, plot_core=True, interactive=True, subplot_height=700, subplot_width=200):
        """Plot well logs and core images.

        All well logs and core images in daylight and ultraviolet are plotted
        on separate subplots.

        Parameters
        ----------
        plot_core : bool
            Specifies whether to plot core images or not. Defaults to `True`.
        interactive : bool
            Specifies whether to draw a plot directly inside a Jupyter
            notebook. Defaults to `True`.
        subplot_height : positive int
            Height of each subplot with well log or core samples images in
            pixels. Defaults to 700.
        subplot_width : positive int
            Width of each subplot with well log or core samples images in
            pixels. Defaults to 200.

        Returns
        -------
        self : AbstractWellSegment or a child class
            Self unchanged.
        """
        init_notebook_mode(connected=True)

        # Approximate size of the left margin of a plot, that will be added to the calculated
        # figsize in order to prevent figure shrinkage in case of small number of subplots
        margin = 120

        n_cols = len(self.logs.columns)
        subplot_titles = list(self.logs.columns)
        if plot_core and self.has_samples:
            n_cols += 2
            subplot_titles += ["CORE DL", "CORE UV"]
            dl_col = n_cols - 1
            uv_col = n_cols

        fig = make_subplots(rows=1, cols=n_cols, subplot_titles=subplot_titles, shared_yaxes=True)
        for i, mnemonic in enumerate(self.logs.columns, 1):
            trace = go.Scatter(x=self.logs[mnemonic], y=self.logs.index, mode="lines", name=mnemonic)
            fig.append_trace(trace, 1, i)

        images = []
        if plot_core and self.has_samples:
            trace = go.Scatter(x=[0, 1], y=[self.depth_from, self.depth_to], opacity=0, name="CORE DL")
            fig.append_trace(trace, 1, dl_col)
            trace = go.Scatter(x=[0, 1], y=[self.depth_from, self.depth_to], opacity=0, name="CORE UV")
            fig.append_trace(trace, 1, uv_col)

            samples = self.samples.reset_index()[["DEPTH_FROM", "DEPTH_TO", "SAMPLE"]]
            for _, (depth_from, depth_to, sample_name) in samples.iterrows():
                sample_name = str(sample_name)

                dl_path = self._get_full_name(os.path.join(self.path, "samples_dl"), sample_name)
                sample_dl = self._encode(dl_path)
                sample_dl = go.layout.Image(source=sample_dl, xref="x"+str(dl_col), yref="y", x=0, y=depth_from,
                                            sizex=1, sizey=depth_to-depth_from, sizing="stretch", layer="below")
                images.append(sample_dl)

                uv_path = self._get_full_name(os.path.join(self.path, "samples_uv"), sample_name)
                sample_uv = self._encode(uv_path)
                sample_uv = go.layout.Image(source=sample_uv, xref="x"+str(uv_col), yref="y", x=0, y=depth_from,
                                            sizex=1, sizey=depth_to-depth_from, sizing="stretch", layer="below")
                images.append(sample_uv)

        layout = fig.layout
        fig_layout = go.Layout(title="{} {}".format(self.field.capitalize(), self.name), showlegend=False,
                               width=n_cols*subplot_width + margin, height=subplot_height,
                               yaxis=dict(range=[self.depth_to, self.depth_from]), images=images)
        layout.update(fig_layout)

        for key in layout:
            if key.startswith("xaxis"):
                layout[key]["fixedrange"] = True

        if plot_core and self.has_samples:
            layout["xaxis" + str(dl_col)]["showticklabels"] = False
            layout["xaxis" + str(dl_col)]["showgrid"] = False

            layout["xaxis" + str(uv_col)]["showticklabels"] = False
            layout["xaxis" + str(uv_col)]["showgrid"] = False

        for ann in layout["annotations"]:
            ann["font"]["size"] = 12

        plot_fn = iplot if interactive else plot
        plot_fn(fig)
        return self

    def __getitem__(self, key):
        """Select well logs by mnemonics or slice the well along the wellbore.

        Parameters
        ----------
        key : str, list of str or slice
            - If `key` is `str` or `list` of `str`, preserve only those logs
              in `self.logs`, that are in `key`.
            - If `key` is `slice` - perform well slicing along the wellbore.
              If both `start` and `stop` are in `self.logs.index`, then only
              `start` is kept in the resulting segment to ensure, that such
              methods as `crop` always return the same number of samples
              regardless of cropping position if crop size is given in meters.
              If only one of the ends of the slice present in the index, it is
              kept in the result contrary to usual python slices.

        Returns
        -------
        well : AbstractWellSegment
            A segment with filtered logs or depths.
        """
        if not isinstance(key, slice):
            return self.keep_logs(key)
        res = self.copy()
        if key.step is not None:
            raise ValueError("A well does not support slicing with a specified step")
        start = float(key.start) if key.start is not None else res.depth_from
        overlap_start = max(res.depth_from, start)
        stop = float(key.stop) if key.stop is not None else res.depth_to
        overlap_stop = min(res.depth_to, stop)
        if overlap_start > overlap_stop:
            raise SkipWellException("Slicing interval is out of segment bounds")
        res.depth_from = overlap_start
        res.depth_to = overlap_stop

        attr_iter = chain(
            zip(res.attrs_depth_index, repeat(res._filter_depth_df)),
            zip(res.attrs_fdtd_index, repeat(res._filter_fdtd_df))
        )
        for attr, filt in attr_iter:
            attr_val = getattr(res, "_" + attr)
            if attr_val is not None:
                setattr(res, "_" + attr, filt(attr_val))

        if (res._core_dl is not None) and (res._core_uv is not None):
            start_pos = self._meters_to_pixels(res.depth_from - self.depth_from)
            stop_pos = self._meters_to_pixels(res.depth_to - self.depth_from)
            res._core_dl = res._core_dl[start_pos:stop_pos]
            res._core_uv = res._core_uv[start_pos:stop_pos]
        return res

    def copy(self):
        """Perform a shallow copy of an object.

        Returns
        -------
        self : AbstractWellSegment
            Shallow copy.
        """
        return copy(self)

    def deepcopy(self):
        """Perform a deep copy of an object.

        Returns
        -------
        self : AbstractWellSegment
            Deep copy.
        """
        return deepcopy(self)

    def check_regularity(self):
        """Checks intervals data regularity.

        Following checks applied for boring_intervals dataframe:
        1. If any values of CORE_RECOVERY column are nan.
        2. If any values of CORE_RECOVERY column are greater
           than calculated CORE_INTERVAL values.
        3. If any values of DEPTH_FROM and DEPTH_TO columns of adjacent rows
           form intervals that overlap each other.
        4. If any values of DEPTH_FROM column are not increasing.
        5. If any values of DEPTH_FROM column are greater than
           values of DEPTH_TO column of the same row.

        Following checks applied for core_lithology dataframe:
        1. If any values of DEPTH_FROM and DEPTH_TO columns of adjacent rows
           form intervals that overlap each other.
        2. If any values of DEPTH_FROM column are not increasing.
        3. If any values of DEPTH_FROM column are greater than
           values of DEPTH_TO column of the same row.
        4. If any intervals formed by values of DEPTH_FROM and DEPTH_TO columns
           of the same row are not included in corresponding intervals from
           boring_intervals dataframe.
        5. If any values of CORE_TOTAL column (calculated as a sum of intervals
           included in the same interval of boring_intervals dataframe) are greater
           than values of CORE_RECOVERY column of boring_intervals dataframe.

        Raises
        ------
        DataRegularityError
            If any of checks above are not passed.
        """
        if not self._has_file("boring_intervals"):
            raise SkipWellException("boring_intervals file is reqiured to perform checks")

        boring_intervals = self.boring_intervals.copy()
        boring_intervals['DEPTH_FROM'] = boring_intervals.index.get_level_values('DEPTH_FROM')
        boring_intervals['DEPTH_TO'] = boring_intervals.index.get_level_values('DEPTH_TO')
        boring_intervals['CORE_INTERVAL'] = boring_intervals['DEPTH_TO'] - boring_intervals['DEPTH_FROM']

        # Check if any CORE_RECOVERY values are nan.
        nans_mask = boring_intervals['CORE_RECOVERY'].isna()
        nans = boring_intervals[nans_mask]
        if not nans.empty:
            raise DataRegularityError("boring_nans", nans[['CORE_RECOVERY']])

        # Check if any CORE_RECOVERY values are greater than CORE_INTERVAL ones.
        unfits_mask = leq_notclose(boring_intervals['CORE_INTERVAL'], boring_intervals['CORE_RECOVERY'])
        unfits = boring_intervals[unfits_mask]
        if not unfits.empty:
            raise DataRegularityError("boring_unfits", unfits[['CORE_RECOVERY', 'CORE_INTERVAL']])

        # Check if any adjacent boring intervals are overlapping.
        preceding = boring_intervals['DEPTH_FROM'].shift(-1) < boring_intervals['DEPTH_TO']
        following = boring_intervals['DEPTH_TO'].shift(-1) > boring_intervals['DEPTH_FROM']
        overlaps_mask = preceding & following
        overlaps = boring_intervals[overlaps_mask | overlaps_mask.shift(1)]
        if not overlaps.empty:
            raise DataRegularityError("boring_overlaps", overlaps[['CORE_RECOVERY']])

        # Check if boring intervals DEPTH_FROM values are not increasing.
        if not boring_intervals.index.is_monotonic_increasing:
            nonincreasing_mask = boring_intervals['DEPTH_FROM'].shift(-1) < boring_intervals['DEPTH_FROM']
            nonincreasing = boring_intervals[nonincreasing_mask | nonincreasing_mask.shift(1)]
            raise DataRegularityError("boring_nonincreasing", nonincreasing[['CORE_RECOVERY']])

        # Check if boring intervals DEPTH_FROM values are bigger than DEPTH_TO ones.
        disordered_mask = boring_intervals['DEPTH_FROM'] > boring_intervals['DEPTH_TO']
        disordered = boring_intervals[disordered_mask]
        if not disordered.empty:
            raise DataRegularityError('boring_disordered', disordered[['CORE_RECOVERY']])

        if not self._has_file("core_lithology"):
            return self

        lithology_intervals = self.core_lithology.copy()
        lithology_intervals['DEPTH_FROM'] = lithology_intervals.index.get_level_values('DEPTH_FROM')
        lithology_intervals['DEPTH_TO'] = lithology_intervals.index.get_level_values('DEPTH_TO')

        # Check if any adjacent lithology intervals are overlapping.
        preceding = lithology_intervals['DEPTH_FROM'].shift(-1) < lithology_intervals['DEPTH_TO']
        following = lithology_intervals['DEPTH_TO'].shift(-1) > lithology_intervals['DEPTH_FROM']
        overlaps_mask = preceding & following
        overlaps = lithology_intervals[overlaps_mask | overlaps_mask.shift(1)]
        if not overlaps.empty:
            raise DataRegularityError("lithology_overlaps", overlaps[['FORMATION']])

        # Check if lithology intervals DEPTH_FROM values are not increasing.
        if not lithology_intervals.index.is_monotonic_increasing:
            nonincreasing_mask = lithology_intervals['DEPTH_FROM'].shift(-1) < lithology_intervals['DEPTH_FROM']
            nonincreasing = lithology_intervals[nonincreasing_mask | nonincreasing_mask.shift(1)]
            raise DataRegularityError("lithology_nonincreasing", nonincreasing[['FORMATION']])

        # Check if lithology intervals DEPTH_FROM values are bigger than DEPTH_TO ones.
        disordered_mask = lithology_intervals['DEPTH_FROM'] > lithology_intervals['DEPTH_TO']
        disordered = lithology_intervals[disordered_mask]
        if not disordered.empty:
            raise DataRegularityError('lithology_disordered', disordered[['FORMATION']])

        # Check if any lithology intervals are not included in boring intervals.
        inclusions_mask = lithology_intervals.apply(lambda interval:
                                                    leq_close(boring_intervals['DEPTH_FROM'],
                                                              interval['DEPTH_FROM']).any() &
                                                    geq_close(boring_intervals['DEPTH_TO'],
                                                              interval['DEPTH_TO']).any(),
                                                    axis=1)
        exclusions = lithology_intervals[~inclusions_mask]
        if not exclusions.empty:
            raise DataRegularityError("lithology_exclusions", exclusions[['FORMATION']])

        # Check any CORE_TOTAL values are greater than the corresponding CORE_RECOVERY ones.
        lithology_intervals = lithology_intervals[['DEPTH_FROM', 'DEPTH_TO']].add_prefix('CORE_')
        combined = cross_join(boring_intervals, lithology_intervals)
        relevant_mask = (leq_close(combined['DEPTH_FROM'], combined['CORE_DEPTH_FROM']) &
                         geq_close(combined['DEPTH_TO'], combined['CORE_DEPTH_TO']))
        combined = combined[relevant_mask]
        combined['CORE_TOTAL'] = combined['CORE_DEPTH_TO'] - combined['CORE_DEPTH_FROM']
        combined = combined.groupby(['CORE_RECOVERY', 'DEPTH_FROM', 'DEPTH_TO'])['CORE_TOTAL'].sum()
        combined = pd.DataFrame(combined).reset_index('CORE_RECOVERY')
        unfits_mask = leq_notclose(combined.CORE_RECOVERY, combined.CORE_TOTAL)
        unfits = combined[unfits_mask]
        if not unfits.empty:
            raise DataRegularityError("lithology_unfits", unfits)

        return self

    def check_samples(self):
        """Checks samples filenames conformity.

        1. If duplicate filenames exist in samples.feather dataframe.
        2. If filenames have different extension lengths in samples.feather dataframe.
        3. If files in samples folder have same names but different extensions.
        4. If files from samples folder are not present in samples.feather dataframe.
        5. If files from samples.feather dataframe are not present in any of samples folders.

        Raises
        ------
        DataRegularityError
        """
        names = [str(name) for name in self.samples['SAMPLE'].values]
        if len(names) > len(set(names)):
            raise DataRegularityError("duplicate_files", "samples.feather")

        ext_lens = [len(os.path.splitext(name)[1]) for name in names]
        if ext_lens[1:] != ext_lens[:-1]:
            raise DataRegularityError("different_extensions", "samples.feather")

        desired_folders = set(["samples_dl", "samples_uv"])
        existing_folders = set(os.listdir(self.path))
        samples_folders = desired_folders.intersection(existing_folders)

        for folder in samples_folders:
            path = "{}/{}".format(self.path, folder)
            samples = os.listdir(path)
            if ext_lens[0] == 0:
                samples = [os.path.splitext(sample)[0] for sample in samples]
                if len(samples) != len(set(samples)):
                    raise DataRegularityError("duplicate_files", folder)

            samples_only = set(samples).difference(set(names))
            if len(samples_only) != 0:
                raise DataRegularityError("missing_files", folder, "samples.feather", samples_only)

            names_only = set(names).difference(set(samples))
            if len(names_only) != 0:
                raise DataRegularityError("missing_files", "samples.feather", folder, names_only)

    def _apply_matching(self):
        """Update depths in all core-related attributes given calculated
        deltas."""
        # Update DataFrames with depth index
        attrs_depth_index = [attr for attr in self.attrs_depth_index if attr.startswith("core_")]
        for attr in attrs_depth_index:
            if not self._has_file(attr):
                continue
            attr_df = getattr(self, attr).reset_index()
            columns = attr_df.columns
            merged_df = between_join(attr_df, self._core_lithology_deltas)
            merged_df["DEPTH"] += merged_df["DELTA"]
            setattr(self, "_" + attr, merged_df[columns].set_index("DEPTH").sort_index())

        # TODO: carfully update samples and reload core images if needed

        if self._has_file("core_lithology"):
            core_lithology = cross_join(self.core_lithology.reset_index(),
                                        self._core_lithology_deltas[["DEPTH_FROM", "DEPTH_TO", "DELTA"]],
                                        suffixes=("", "_deltas"))
            mask = ((core_lithology["DEPTH_FROM"] >= core_lithology["DEPTH_FROM_deltas"]) &
                    (core_lithology["DEPTH_TO"] <= core_lithology["DEPTH_TO_deltas"]))
            core_lithology = core_lithology[mask]
            core_lithology["DEPTH_FROM"] += core_lithology["DELTA"]
            core_lithology["DEPTH_TO"] += core_lithology["DELTA"]
            core_lithology = core_lithology[self.core_lithology.reset_index().columns]
            self._core_lithology = core_lithology.set_index(["DEPTH_FROM", "DEPTH_TO"]).sort_index()

        boring_intervals = pd.merge(self._boring_intervals.reset_index(),
                                    self._boring_intervals_deltas[["DEPTH_FROM", "DEPTH_TO", "DELTA"]],
                                    on=["DEPTH_FROM", "DEPTH_TO"])
        boring_intervals["DEPTH_FROM"] += boring_intervals["DELTA"]
        boring_intervals["DEPTH_TO"] += boring_intervals["DELTA"]
        boring_intervals = boring_intervals.drop("DELTA", axis=1)
        self._boring_intervals = boring_intervals.set_index(["DEPTH_FROM", "DEPTH_TO"]).sort_index()

    def _save_matching_report(self):
        """Save matching report in a well directory, specified in `self.path`.

        The report consists of two `.csv` files, containing depths of boring
        and lithology intervals respectively before and after matching.
        """
        boring_sequences = self.boring_sequences.reset_index()[["DEPTH_FROM", "DEPTH_TO", "MODE"]]
        not_none_mask = boring_sequences["MODE"].map(lambda x: x is not None)
        boring_sequences = boring_sequences[not_none_mask]
        if len(boring_sequences) == 0:
            return

        core_logs_list = []
        for _, (depth_from, depth_to, mode) in boring_sequences.iterrows():
            _, core_mnemonic, core_attr, _ = self._parse_matching_mode(mode)
            core_log_segment = getattr(self, core_attr)[core_mnemonic].dropna()[depth_from:depth_to]
            core_logs_list.append(core_log_segment.to_frame(name=mode))
        core_logs = pd.concat(core_logs_list)
        core_logs.index.rename("Увязанная глубина образца", inplace=True)
        core_logs.reset_index(inplace=True)

        boring_intervals = self._boring_intervals_deltas.reset_index()
        boring_intervals["DEPTH_FROM_DELTA"] = boring_intervals["DEPTH_FROM"] + boring_intervals["DELTA"]
        boring_intervals["DEPTH_TO_DELTA"] = boring_intervals["DEPTH_TO"] + boring_intervals["DELTA"]
        boring_intervals = boring_intervals[["DEPTH_FROM", "DEPTH_TO", "DEPTH_FROM_DELTA", "DEPTH_TO_DELTA"]]
        boring_intervals.columns = [
            "Кровля интервала долбления",
            "Подошва интервала долбления",
            "Увязанная кровля интервала долбления",
            "Увязанная подошва интервала долбления"
        ]
        boring_intervals = between_join(core_logs, boring_intervals, left_on="Увязанная глубина образца",
                                        right_on=("Увязанная кровля интервала долбления",
                                                  "Увязанная подошва интервала долбления"))
        boring_intervals.to_csv(os.path.join(self.path, self.name + "_boring_intervals_matching.csv"),
                                index=False)

        lithology_intervals = self._core_lithology_deltas.reset_index()
        lithology_intervals["DEPTH_FROM_DELTA"] = lithology_intervals["DEPTH_FROM"] + lithology_intervals["DELTA"]
        lithology_intervals["DEPTH_TO_DELTA"] = lithology_intervals["DEPTH_TO"] + lithology_intervals["DELTA"]
        lithology_intervals = lithology_intervals[["DEPTH_FROM", "DEPTH_TO", "DEPTH_FROM_DELTA", "DEPTH_TO_DELTA"]]
        lithology_intervals.columns = [
            "Кровля интервала литописания",
            "Подошва интервала литописания",
            "Увязанная кровля интервала литописания",
            "Увязанная подошва интервала литописания"
        ]
        lithology_intervals.to_csv(os.path.join(self.path, self.name + "_lithology_intervals_matching.csv"),
                                   index=False)

    @staticmethod
    def _parse_matching_mode(mode):
        """Split matching mode string into well log mnemonic, core log or
        property mnemonic, class attribute to get core data from and a sign of
        the theoretical correlation between well and core data."""
        mode = mode.replace(" ", "")
        split_mode = mode.split("~")
        if len(split_mode) != 2:
            raise ValueError("Incorrect mode format")
        log_mnemonic = split_mode[0]
        split_core_mode = split_mode[1].split(".")
        if len(split_core_mode) != 2:
            raise ValueError("Incorrect mode format")
        core_attr, core_mnemonic = split_core_mode

        sign = "+"
        if core_attr[0] in {"+", "-"}:
            sign = core_attr[0]
            core_attr = core_attr[1:]
        sign = 1 if sign == "+" else -1
        return log_mnemonic, core_mnemonic, core_attr, sign

    def _select_matching_mode(self, segment, mode_list, min_points, min_points_per_meter):
        """Select appropriate matching mode based on data, availible for given
        segment."""
        segment_depth_from = segment["DEPTH_FROM"].min()
        segment_depth_to = segment["DEPTH_TO"].max()
        core_len = segment["CORE_RECOVERY"].sum()
        for mode in mode_list:
            log_mnemonic, core_mnemonic, core_attr, _ = self._parse_matching_mode(mode)
            if log_mnemonic in self.logs and self._has_file(core_attr) and core_mnemonic in getattr(self, core_attr):
                well_log = self.logs[log_mnemonic].dropna()
                core_log = getattr(self, core_attr)[core_mnemonic].dropna()
                well_log_len = len(well_log[segment_depth_from:segment_depth_to])
                core_log_len = len(core_log[segment_depth_from:segment_depth_to])
                if min(well_log_len, core_log_len) >= max(min_points_per_meter * core_len, min_points):
                    return mode
        return None

    @classmethod
    def _unify_matching_mode(cls, mode):
        """Delete all spaces and add `+` sign if omitted to a matching mode
        string."""
        log_mnemonic, core_mnemonic, core_attr, sign = cls._parse_matching_mode(mode)
        mode = "{}~{}{}.{}".format(log_mnemonic, "+" if sign == 1 else "-", core_attr, core_mnemonic)
        return mode

    @classmethod
    def _unify_matching_modes(cls, modes):
        """Delete all spaces and add `+` sign if omitted to each matching mode
        string in a `modes` list."""
        return [cls._unify_matching_mode(mode) for mode in to_list(modes)]

    def _blur_log(self, log, win_size):
        """Blur a log with a Gaussian filter of size `win_size`."""
        if win_size is None:
            return log
        old_index = log.index
        new_index = np.arange(old_index.min(), old_index.max(), 0.01)
        log = log.reindex(index=new_index, method="nearest", tolerance=self._tolerance)
        log = log.interpolate(limit_direction="both")
        std = win_size / 6  # three-sigma rule
        log = log.rolling(window=win_size, min_periods=1, win_type="gaussian", center=True).mean(std=std)
        log = log.reindex(index=old_index, method="nearest")
        return log

    def match_core_logs(self, mode="GK ~ core_logs.GK", split_lithology_intervals=True, gaussian_win_size=None,
                        min_points=3, min_points_per_meter=1, min_gap=0.5, max_shift=10, delta_from=-8, delta_to=8,
                        delta_step=0.1, max_iter=50, max_iter_time=0.25, save_report=False):
        """Perform core-to-log matching by shifting core samples in order to
        maximize correlation between well and core logs.

        Parameters
        ----------
        mode : str or list of str, optional
            Matching mode precedence from highest to lowest. The mode is
            independently selected for each boring sequence. Each mode has the
            following structure: <well_log> ~ <sign><core_attr>.<core_log>,
            where:
            - well_log - mnemonic of a well log to use
            - sign - a sign of the theoretical correlation between well and
              core data, defaults to `+` if not given
            - core_attr - an attribute of `self` to get core data from
            - core_log - mnemonic of a core log or property to use
            Defaults to gamma ray matching.
        split_lithology_intervals : bool, optional
            Specifies whether to independently shift lithology intervals
            inside a boring interval. Defaults to `True`.
        gaussian_win_size : int, optional
            A Gaussian filter size in samples to perform log blurring before
            matching. No blurring is performed by default.
        min_points : int, optional
            A minimum number of samples in logs to perform matching. Defaults
            to 3.
        min_points_per_meter : int, optional
            A minimum number of samples per meter in logs to perform matching.
            Defaults to 1.
        min_gap : float, optional
            A minimum gap between two boring intervals in meters to set the
            bottom depth of the upper interval equal to the top depth of the
            lower one due to inaccuracies in depth measurements. Defaults to
            0.5.
        max_shift : positive float, optional
            Maximum shift of a boring sequence in meters. Defaults to 10.
        delta_from : float, optional
            Start of the grid of initial shifts in meters. Defaults to -8.
        delta_to : float, optional
            End of the grid of initial shifts in meters. Defaults to 8.
        delta_step : float, optional
            Step of the grid of initial shifts in meters. Defaults to 0.1.
        max_iter : positive int, optional
            Maximum number of `SLSQP` iterations. Defaults to 50.
        max_iter_time, optional
            Maximum time for an optimization iteration in seconds. Defaults to
            0.25.
        save_report : bool, optional
            Specifies whether to save matching report in a well directory.
            Defaults to `False`.

        Returns
        -------
        well : AbstractWellSegment
            Matched well segment with updated core depths. Changes all
            core-related depths inplace.
        """
        if max_shift <= 0:
            raise ValueError("max_shift must be positive")
        if delta_from > delta_to:
            raise ValueError("delta_to must be greater than delta_from")
        if max(np.abs(delta_from), np.abs(delta_to)) > max_shift:
            raise ValueError("delta_from and delta_to must not exceed max_shift in absolute value")

        # TODO: extra checks for min_points
        min_points = max(min_points, 2)

        mode_list = self._unify_matching_modes(mode)

        # If a gap between any two boring intervals is less than `min_gap`, treat it as inaccuracies in depth
        # measurements and set the bottom depth of the upper interval equal to the top depth of the lower one.
        boring_intervals = self.boring_intervals.reset_index()
        bi_depth_from = boring_intervals["DEPTH_FROM"]
        bi_depth_to = boring_intervals["DEPTH_TO"]
        boring_intervals["DEPTH_TO"] = np.where((bi_depth_from.shift(-1) - bi_depth_to) < min_gap,
                                                bi_depth_from.shift(-1), bi_depth_to)
        self._boring_intervals = boring_intervals.set_index(["DEPTH_FROM", "DEPTH_TO"])
        self._calc_boring_sequences()

        split_lithology_intervals = split_lithology_intervals and self._has_file("core_lithology")
        lithology_df = self.core_lithology if split_lithology_intervals else self.boring_intervals
        lithology_intervals = lithology_df.reset_index()[["DEPTH_FROM", "DEPTH_TO"]]

        # `boring_sequences` is a list of DataFrames, containing contiguous boring intervals, extracted one after
        # another. They are considered together since they must be shifted by the same delta.

        # `boring_groups` is a list of DataFrames, containing a number of boring sequences, extracted close to each
        # other. They are considered together since they can possibly overlap during optimization.
        boring_groups = select_contigious_intervals(self.boring_intervals.reset_index(), max_gap=2*max_shift)

        matched_boring_sequences = []
        matched_lithology_intervals = []
        sequences_modes = []
        sequences_r2 = []

        for group in boring_groups:
            boring_sequences = select_contigious_intervals(group)
            sequences_shifts = []

            # Independently optimize R^2 for each boring sequence
            for sequence in boring_sequences:
                sequence_depth_from = sequence["DEPTH_FROM"].min()
                sequence_depth_to = sequence["DEPTH_TO"].max()

                mode = self._select_matching_mode(sequence, mode_list, min_points, min_points_per_meter)
                sequences_modes.append(mode)
                if mode is None:
                    # Don't shift a sequence if there's no data to perform matching
                    sequences_shifts.append([create_zero_shift(sequence_depth_from, sequence_depth_to)])
                    continue

                log_mnemonic, core_mnemonic, core_attr, sign = self._parse_matching_mode(mode)
                well_log = self.logs[log_mnemonic].dropna()
                well_log = well_log[sequence_depth_from - max_shift : sequence_depth_to + max_shift]
                well_log = self._blur_log(well_log, gaussian_win_size)
                core_log = sign * getattr(self, core_attr)[core_mnemonic].dropna()
                core_log = core_log[sequence_depth_from:sequence_depth_to]
                core_log = self._blur_log(core_log, gaussian_win_size)

                shifts = match_boring_sequence(sequence, lithology_intervals, well_log, core_log,
                                               max_shift, delta_from, delta_to, delta_step,
                                               max_iter, timeout=max_iter*max_iter_time)
                sequences_shifts.append(shifts)

            best_shifts = find_best_shifts(sequences_shifts, self.name, self.field)

            # Store shift deltas, mode and R^2
            for sequence, shift in zip(boring_sequences, best_shifts):
                mask = ((lithology_intervals["DEPTH_FROM"] >= sequence["DEPTH_FROM"].min()) &
                        (lithology_intervals["DEPTH_TO"] <= sequence["DEPTH_TO"].max()))
                sequence_lithology_intervals = lithology_intervals[mask].copy()  # to avoid SettingWithCopyWarning
                sequence_lithology_intervals["DELTA"] = shift.interval_deltas
                sequence["DELTA"] = shift.sequence_delta

                sequences_r2.append(shift.loss**2)
                matched_boring_sequences.append(sequence)
                matched_lithology_intervals.append(sequence_lithology_intervals)

        if all(mode is None for mode in sequences_modes):
            raise SkipWellException("None of the boring sequences can be matched")

        self._boring_intervals_deltas = pd.concat(matched_boring_sequences)
        self._core_lithology_deltas = pd.concat(matched_lithology_intervals)
        self._apply_matching()

        self._calc_boring_sequences()
        self._boring_sequences["MODE"] = sequences_modes
        self._boring_sequences["R2"] = sequences_r2

        if save_report:
            self._save_matching_report()
        return self

    @staticmethod
    def _calc_matching_r2(well_log, core_log, eps=1e-8):
        """Calculate squared correlation coefficient between well and core
        logs.

        If well and core logs are defined on different sets of depths, then at
        first well log values are estimated at core log depths by linear
        interpolation and then `R^2` is calculated for the resulting arrays.
        """
        well_log = well_log.dropna()
        interpolator = interp1d(well_log.index, well_log, kind="linear", fill_value="extrapolate")
        well_log = interpolator(core_log.index)
        cov = np.mean(well_log * core_log) - well_log.mean() * core_log.mean()
        cor = np.clip(cov / ((well_log.std() + eps) * (core_log.std() + eps)), -1, 1)
        return cor**2

    def plot_matching(self, mode=None, scale=False, interactive=True, subplot_height=700, subplot_width=200):
        """Plot well log and corresponding core log for each boring sequence.

        This method can be used to illustrate results of core-to-log matching.

        Parameters
        ----------
        mode : str or list of str
            Specify type of well log and core log or property to plot. If
            `str`, the same mode will be used for all boring sequences. If
            `list` of `str`, than each boring sequence will have its own mode.
            In this case length of the `list` should match the number of
            boring sequences.

            Each mode has the same structure as `mode` in `match_core_logs`.
            If `None` and core-to-log matching was performed beforehand,
            chosen matching modes are used.
            Defaults to `None`.
        scale : bool
            Specifies whether to lineary scale core log values to well log
            values. Defaults to `False`.
        interactive : bool
            Specifies whether to draw a plot directly inside a Jupyter
            notebook. Defaults to `True`.
        subplot_height : positive int
            Height of each subplot with well and core logs. Defaults to 700.
        subplot_width : positive int
            Width of each subplot with well and core logs. Defaults to 200.

        Returns
        -------
        self : AbstractWellSegment or a child class
            Self unchanged.
        """
        init_notebook_mode(connected=True)

        # Approximate size of the left margin of a plot, that will be added to the calculated
        # figsize in order to prevent figure shrinkage in case of small number of subplots
        margin = 120

        boring_sequences = self.boring_sequences.reset_index()
        if mode is None and "MODE" not in boring_sequences.columns:
            raise ValueError("Core-to-log matching has to be performed beforehand if mode is not specified")
        if mode is not None:
            mode_list = self._unify_matching_modes(mode)
            if len(mode_list) == 1:
                mode_list = mode_list * len(boring_sequences)
            if len(mode_list) != len(boring_sequences):
                raise ValueError("Mode length must match the number of matching intervals")
            boring_sequences["MODE"] = mode_list
            r2_list = []
            for _, (depth_from, depth_to, _mode) in boring_sequences[["DEPTH_FROM", "DEPTH_TO", "MODE"]].iterrows():
                log_mnemonic, core_mnemonic, core_attr, sign = self._parse_matching_mode(_mode)
                well_log = self.logs[log_mnemonic].dropna()
                core_log_segment = sign * getattr(self, core_attr)[core_mnemonic].dropna()[depth_from:depth_to]
                r2_list.append(self._calc_matching_r2(well_log, core_log_segment))
            boring_sequences["R2"] = r2_list
        boring_sequences = boring_sequences[["DEPTH_FROM", "DEPTH_TO", "MODE", "R2"]]
        not_none_mask = boring_sequences["MODE"].map(lambda x: x is not None)
        boring_sequences = boring_sequences[not_none_mask]

        depth_from_list = boring_sequences["DEPTH_FROM"]
        depth_to_list = boring_sequences["DEPTH_TO"]
        mode_list = boring_sequences["MODE"]
        r2_list = boring_sequences["R2"]

        n_cols = len(boring_sequences)
        subplot_titles = ["{}<br>R^2 = {:.3f}".format(mode, r2) for mode, r2 in zip(mode_list, r2_list)]
        fig = make_subplots(rows=1, cols=n_cols, subplot_titles=subplot_titles)

        for i, (depth_from, depth_to, _mode) in enumerate(zip(depth_from_list, depth_to_list, mode_list), 1):
            log_mnemonic, core_mnemonic, core_attr, sign = self._parse_matching_mode(_mode)
            well_log_segment = self.logs[log_mnemonic].dropna()[depth_from - 3 : depth_to + 3]
            core_log_segment = sign * getattr(self, core_attr)[core_mnemonic].dropna()[depth_from:depth_to]

            if scale and min(len(well_log_segment), len(core_log_segment)) > 1:
                log_interpolator = interp1d(well_log_segment.index, well_log_segment, kind="linear",
                                            fill_value="extrapolate")
                X = np.array(core_log_segment).reshape(-1, 1)
                y = log_interpolator(core_log_segment.index)
                reg = LinearRegression().fit(X, y)
                core_log_segment = pd.Series(reg.predict(X), index=core_log_segment.index)

            well_log_trace = go.Scatter(x=well_log_segment, y=well_log_segment.index, name="Well " + log_mnemonic,
                                        line=dict(color="rgb(255, 127, 14)"), showlegend=False)
            drawing_mode = "markers" if core_attr == "core_properties" else None
            core_log_trace = go.Scatter(x=core_log_segment, y=core_log_segment.index, name="Core " + core_mnemonic,
                                        line=dict(color="rgb(31, 119, 180)"), mode=drawing_mode, showlegend=False)
            fig.append_trace(well_log_trace, 1, i)
            fig.append_trace(core_log_trace, 1, i)

        layout = fig.layout
        fig_layout = go.Layout(title="{} {}".format(self.field.capitalize(), self.name),
                               legend=dict(orientation="h"), width=n_cols*subplot_width + margin,
                               height=subplot_height)
        layout.update(fig_layout)

        for key in layout:
            if key.startswith("xaxis"):
                layout[key]["fixedrange"] = True

        for ann in layout["annotations"]:
            ann["font"]["size"] = 14

        for ix in range(n_cols):
            axis_ix = str(ix + 1) if ix > 0 else ""
            axis_name = "yaxis" + axis_ix
            layout[axis_name]["autorange"] = "reversed"

        plot_fn = iplot if interactive else plot
        plot_fn(fig)
        return self

    def add_depth_log(self):
        """Copy the index of `self.logs` to its column `DEPTH`.

        Returns
        -------
        well : AbstractWellSegment or a child class
            A segment with created depth log.
        """
        self.logs["DEPTH"] = self.logs.index
        return self

    def drop_logs(self, mnemonics):
        """Drop well logs whose mnemonics are in `mnemonics`.

        Parameters
        ----------
        mnemonics : str or list of str
            Mnemonics of well logs to be dropped.

        Returns
        -------
        well : AbstractWellSegment or a child class
            A segment with filtered logs.
        """
        res = self.copy()
        res._logs.drop(to_list(mnemonics), axis=1, inplace=True)
        return res

    def keep_logs(self, mnemonics):
        """Drop well logs whose mnemonics are not in `mnemonics`.

        Parameters
        ----------
        mnemonics : str or list of str
            Mnemonics of well logs to be kept.

        Returns
        -------
        well : AbstractWellSegment or a child class
            A segment with filtered logs.
        """
        missing_mnemonics = np.setdiff1d(mnemonics, self.logs.columns)
        if len(missing_mnemonics) > 0:
            err_msg = "The following logs were not recorded for the well: {}".format(", ".join(missing_mnemonics))
            raise SkipWellException(err_msg)
        res = self.copy()
        res._logs = res.logs[to_list(mnemonics)]
        return res

    def rename_logs(self, rename_dict):
        """Rename well logs with corresponding mnemonics in `rename_dict`.

        Parameters
        ----------
        rename_dict : dict
            Dictionary containing `(old mnemonic : new mnemonic)` pairs.

        Returns
        -------
        well : AbstractWellSegment or a child class
            A segment with renamed logs. Changes `self.logs` inplace.
        """
        self.logs.rename(columns=rename_dict, inplace=True)
        return self

    def _filter_layers(self, layers, connected, invert_mask):
        """Keep or drop layers whose names match any pattern from `layers`
        depending on an `invert_mask` flag.

        Parameters
        ----------
        layers : str or list of str
            Regular expressions, specifying layer names to keep or drop.
        connected : bool
            Specifies whether to join segments with kept layers, that go one
            after another.
        invert_mask : bool
            If `False`, drop layers, whose names don't match any pattern from
            `layers`. If `True`, drop layers, whose names match any of them.

        Returns
        -------
        well : list of WellSegment
            Segments, representing kept layers.
        """
        if not self._has_file("layers"):
            raise SkipWellException("Layers are not defined for a well {}".format(self.name))
        reg_list = [re.compile(layer, re.IGNORECASE) for layer in to_list(layers)]
        mask = np.array([any(regex.fullmatch(layer) for regex in reg_list) for layer in self.layers["LAYER"]])
        if invert_mask:
            mask = ~mask
        # TODO: unify with _create_segments_by_fdtd
        depth_df = self.layers.reset_index()[mask]
        if connected:
            depth_df = self._core_chunks(depth_df)
        segments = [self[top:bottom] for _, (top, bottom) in depth_df[['DEPTH_FROM', 'DEPTH_TO']].iterrows()]
        return segments

    def drop_layers(self, layers, connected=True):
        """Drop layers, whose names match any pattern from `layers`.

        Parameters
        ----------
        layers : str or list of str
            Regular expressions, specifying layer names to drop.
        connected : bool, optional
            Specifies whether to join segments with kept layers, that go one
            after another. Defaults to `True`.

        Returns
        -------
        well : list of WellSegment
            Segments, representing kept layers.
        """
        return self._filter_layers(layers, connected, invert_mask=True)

    def keep_layers(self, layers, connected=True):
        """Drop layers, whose names don't match any pattern from `layers`.

        Parameters
        ----------
        layers : str or list of str
            Regular expressions, specifying layer names to keep.
        connected : bool, optional
            Specifies whether to join segments with kept layers, that go one
            after another. Defaults to `True`.

        Returns
        -------
        well : list of WellSegment
            Segments, representing kept layers.
        """
        return self._filter_layers(layers, connected, invert_mask=False)

    def keep_matched_sequences(self, mode=None, threshold=0.6):
        """Keep boring sequences, matched using given `mode` with `R^2`
        greater than `threshold`.

        Parameters
        ----------
        mode : str or list of str
            Chosen matching mode to keep a sequence. It has the same structure
            as `mode` in `match_core_logs`.
        threshold : float
            Minimum value of `R^2` to keep a sequence.

        Returns
        -------
        segments : list of WellSegment
            Kept boring sequences.
        """
        mask = self.boring_sequences["R2"] > threshold
        if mode is not None:
            mode_list = self._unify_matching_modes(mode)
            mask &= self.boring_sequences["MODE"].isin(mode_list)
        sequences = self.boring_sequences[mask].reset_index()[["DEPTH_FROM", "DEPTH_TO"]]
        res_segments = []
        for _, (depth_from, depth_to) in sequences.iterrows():
            res_segments.append(self[depth_from:depth_to])
        return res_segments

    def create_segments(self, src, connected=True):
        """Split `self` into segments with depth ranges, specified in
        attributes in `src`.

        Parameters
        ----------
        src : str or iterable
            Names of attributes to get depths ranges for splitting. If `src`
            consists of attributes in fdtd format then each row will represent
            a new segment. Otherwise, an exception will be raised.
        connected : bool, optional
            Join segments which are one after another. Defaults to `True`.

        Returns
        -------
        segments : list of WellSegment
            Split segments.
        """
        if not isinstance(src, list):
            src = [src]
        if all([item in self.attrs_fdtd_index for item in src]):
            res = self._create_segments_by_fdtd(src, connected)
        else:
            # TODO: create_segments from depth_index
            raise ValueError(
                'All `src` must be in fdtd format:',
                [item for item in src if item not in self.attrs_fdtd_index]
            )
        return res

    def _create_segments_by_fdtd(self, src, connected):
        """Get segments from depth ranges, specified in fdtd attributes."""
        tables = [getattr(self, item).reset_index() for item in src]
        df = tables[0] if len(tables) == 1 else reduce(fdtd_join, tables)
        if connected:
            df = self._core_chunks(df)
        segments = [self[top:bottom] for _, (top, bottom) in df[['DEPTH_FROM', 'DEPTH_TO']].iterrows()]
        return segments

    @staticmethod
    def _core_chunks(df):
        if len(df) > 0:
            chunks = [(item.DEPTH_FROM.min(), item.DEPTH_TO.max()) for item in select_contigious_intervals(df)]
            chunks = pd.DataFrame(chunks, columns=["DEPTH_FROM", "DEPTH_TO"])
            return chunks
        return pd.DataFrame(columns=["DEPTH_FROM", "DEPTH_TO"])

    def random_crop(self, length, n_crops=1):
        """Create random crops from the segment. All cropped segments have the same
        length, their positions are sampled uniformly from the segment.

        Parameters
        ----------
        length : positive float
            Length of each crop in meters.
        n_crops : positive int, optional
            The number of crops from the segment. Defaults to 1.

        Returns
        -------
        segments : list of WellSegment
            Cropped segments.
        """
        bounds = self.depth_from, max(self.depth_from, self.depth_to-length)
        positions = np.sort(np.random.uniform(*bounds, size=n_crops))
        return [self[pos:pos+length] for pos in positions]

    def crop(self, length, step, drop_last=True, fill_value=0):
        """Create crops from the segment. All cropped segments have the same
        length and are cropped with some fixed step.

        Parameters
        ----------
        length : positive float
            Length of each crop in meters.
        step : positive float
            Step of cropping in meters.
        drop_last : bool, optional
            If `True`, only crops that lie within the segment will be kept.
            If `False`, the first crop which comes out of segment bounds will
            also be kept to cover the whole segment with crops. Its `logs`
            will be padded by `fill_value` at the end to have given `length`.
            Defaults to `True`.
        fill_value : float, optional
            Value to fill padded part of `logs`. Defaults to 0.

        Returns
        -------
        segments : list of WellSegment
            Cropped segments.
        """
        n_crops_in = ceil((self.depth_to - self.depth_from - length) / step)
        crops_in = np.arange(n_crops_in) * step + self.depth_from
        segments_in = [self[pos:pos+length] for pos in crops_in]
        if drop_last or np.allclose(crops_in[-1] + length, self.depth_to, rtol=0, atol=self._tolerance):
            return segments_in

        crop_out = crops_in[-1] + step
        self.actual_depth_to = self.depth_to
        self.depth_to = crop_out + length
        n_pads = ceil((self.depth_to - self.depth_from) / self.logs_step) + 1
        pad_index = np.arange(n_pads) * self.logs_step + self.depth_from
        pad_logs = self.logs.reindex(index=pad_index, method="nearest",
                                     tolerance=self._tolerance, fill_value=fill_value)
        setattr(self, '_logs', pad_logs)
        return segments_in + [self[crop_out:crop_out+length]]

    def create_mask(self, src, column, mapping=None, mode='logs', default=np.nan, dst='mask'):
        """Transform a column from some `WellSegment` attribute into a mask
        correponding to a well log or to a core image.

        Parameters
        ----------
        src : str
            Attribute to get the column from.
        column : str
            Name of the column to transform.
        mapping : dict or None
            Mapping for column values. If `None`, original attribure values
            will be kept in the mask. Defaults to `None`.
        mode : 'logs' or 'core'
            A mode, specifying the length of computed mask. If 'logs', mask
            lenght will match the length of well logs. If 'core', mask length
            will match the heigth of `core_dl` in pixels. Defaults to
            `'logs'`.
        default : float, optional
            Default value for mask if `src` doesn't contain information for
            corresponding depth. Defaults to `numpy.nan`.
        dst : str, optional
            `WellSegment` attribute to save the mask to. Defaults to `mask`.

        Returns
        -------
        self : AbstractWellSegment or a child class
            Self with mask.
        """
        if src in self.attrs_fdtd_index:
            self._create_mask_fdtd(src, column, mapping, mode, default, dst)
        elif src in self.attrs_depth_index:
            self._create_mask_depth_index(src, column, mapping, mode, default, dst)
        else:
            ValueError('Unknown src: ', src)
        return self

    def _create_mask_fdtd(self, src, column, mapping, mode, default, dst):
        """Create mask from fdtd data."""
        if mode == 'core':
            mask = np.ones(len(self.core_dl)) * default
        elif mode == 'logs':
            mask = np.ones(len(self.logs)) * default
        else:
            raise ValueError('Unknown mode: ', mode)

        table = getattr(self, src)
        factor = len(mask) / self.length
        for row in table.iterrows():
            depth_from, depth_to = row[1].name
            start = np.floor((max(depth_from, self.depth_from) - self.depth_from) * factor)
            end = np.ceil((min(depth_to, self.depth_to) - self.depth_from) * factor)
            mask[int(start):int(end)] = row[1][column] if mapping is None else mapping[row[1][column]]
        setattr(self, dst, mask)

    def _create_mask_depth_index(self, src, column, mapping, mode, default, dst):
        """Create mask from depth-indexed data."""
        # TODO: fix interpolation
        if mode == 'core':
            mask = np.ones(len(self.core_dl)) * default
        elif mode == 'logs':
            mask = np.ones(len(self.logs)) * default
        else:
            raise ValueError('Unknown mode: ', mode)

        table = getattr(self, src)
        factor = len(mask) / self.length
        for row in table.iterrows():
            depth = row[1].name
            pos = np.floor((depth - self.depth_from) * factor)
            mask[int(pos)] = row[1][column] if mapping is None else mapping[row[1][column]]
        setattr(self, dst, mask)

    @process_columns
    def apply(self, df, fn, *args, axis=None, **kwargs):
        """Apply a function to each row of a segment attribute.

        Parameters
        ----------
        fn : callable
            A function to be applied. See Notes section for caveats.
        axis : {0, 1, None}, optional
            An axis, along which the function is applied:
            * 0: apply the function to each column
            * 1: apply the function to each row
            * `None`: apply the function to the whole `DataFrame`
            Defaults to `None`.
        args : misc
            Any additional positional arguments to pass to `fn` after data
            from `attr`.
        kwargs : misc
            Any additional keyword arguments to pass as keyword arguments to
            `fn`.

        Notes
        -----
        Currently, callables from `numpy` without pure Python implementation
        can't be passed as `fn` if they get more than one non-keyword
        argument and `axis` is specified.

        E.g. `apply(np.divide, 1000, axis=1, src='DEPTH')` will fail.
        Use `apply(lambda x: x / 1000, axis=1, src='DEPTH')` instead.

        Returns
        -------
        well : AbstractWellSegment
            The segment with applied function.
        """
        _ = self
        if axis is None:
            res = fn(df, *args, **kwargs)
        else:
            res = df.apply(fn, axis=axis, raw=True, result_type="expand", args=args, **kwargs)
        if isinstance(res, pd.Series):
            res = res.to_frame()
        return res

    def _filter_depth_attrs(self, attrs=None):
        """Return intersection of `attrs` and `self.attrs_depth_index`."""
        attrs = self.attrs_depth_index if attrs is None else attrs
        return np.intersect1d(attrs, self.attrs_depth_index)

    def reindex(self, step, attrs=None):
        """Conform depth-indexed `attrs` of the segment to a new index,
        starting from `self.depth_from` to `self.depth_to` with a step `step`,
        placing `nan` values in locations having no value in the previous
        index.

        Parameters
        ----------
        step : positive float
            Distance between any two adjacent values in the new index in
            meters.
        attrs : str or list of str
            Depth-indexed attributes of the segment to be reindexed.

        Returns
        -------
        well : AbstractWellSegment or a child class
            The segment with reindexed `attrs`.
        """
        new_index = np.arange(self.depth_from, self.depth_to, step)
        for attr in self._filter_depth_attrs(attrs):
            res = getattr(self, attr).reindex(index=new_index, method="nearest", tolerance=self._tolerance)
            setattr(self, "_" + attr, res)
        return self

    def interpolate(self, *args, attrs=None, **kwargs):
        """Interpolate `nan` values in `attrs`.

        Parameters
        ----------
        attrs : str or list of str
            Depth-indexed attributes of the segment to be interpolated.
        args : misc
            Any additional positional arguments to `pandas.interpolate`.
        kwargs : misc
            Any additional named arguments to `pandas.interpolate`.

        Returns
        -------
        well : AbstractWellSegment or a child class
            The segment with interpolated values in `attrs`.
        """
        for attr in self._filter_depth_attrs(attrs):
            res = getattr(self, attr).interpolate(*args, **kwargs)
            setattr(self, "_" + attr, res)
        return self

    def gaussian_blur(self, win_size, std=None, attrs=None):
        """Blur columns of `attrs` with a Gaussian filter.

        Parameters
        ----------
        win_size : int
            Size of the kernel.
        std : float, optional
            The standard deviation of the normal distribution. Equals
            `win_size / 6' by default.
        attrs : str or list of str
            Depth-indexed attributes of the segment to be blurred.

        Returns
        -------
        well : AbstractWellSegment
            The segment with blurred logs in `attrs`.
        """
        if std is None:
            std = win_size / 6  # three-sigma rule
        for attr in self._filter_depth_attrs(attrs):
            val = getattr(self, attr)
            nan_mask = val.isna()
            val = val.rolling(window=win_size, min_periods=1, win_type="gaussian", center=True).mean(std=std)
            val[nan_mask] = np.nan
            setattr(self, "_" + attr, val)
        return self

    def drop_nans(self, logs=None):
        """Create segments that does not contain `nan` values in logs,
        indicated in `logs`.

        Parameters
        ----------
        logs : None or int or list of str
            - If `None`, create segments without `nan` values in all logs.
            - If `int`, create segments so that each row of logs has at least
              `logs` not-nan values.
            - If `list`, create segments without `nan` values in logs with
              mnemonics in `logs`.
            Defaults to `None`.

        Returns
        -------
        segments : list of WellSegment
            Segments with dropped `nan` values.
        """
        logs = self.logs.columns if logs is None else logs
        if isinstance(logs, int):
            not_nan_mask = (~np.isnan(self.logs)).sum(axis=1) >= logs
        else:
            not_nan_mask = np.all(~np.isnan(self.logs[logs]), axis=1)

        if not not_nan_mask.any():
            return []

        borders = not_nan_mask[not_nan_mask ^ not_nan_mask.shift(1)].index.tolist()
        # If last mask element is True (segment ends with not nan value in logs)
        # then the xor trick above misses the last slicing border and therefore
        # None should be added so the last slice could be done as `self[a:None]`
        if not_nan_mask.iloc[-1]:
            borders.append(None)
        borders = zip(borders[0::2], borders[1::2])
        return [self[a:b] for a, b in borders]

    @process_columns
    def norm_mean_std(self, df, mean=None, std=None, eps=1e-10):
        """Standardize well logs by subtracting the mean and scaling to unit
        variance.

        Parameters
        ----------
        mean : None or ndarray, optional
            Mean to be subtracted. If `None`, it is calculated independently
            for each log in the segment.
        std : None or ndarray, optional
            Standard deviation to be divided by. If `None`, it is calculated
            independently for each log in the segment.
        eps: float, optional
            A small float to be added to the denominator to avoid division by
            zero.

        Returns
        -------
        well : AbstractWellSegment or a child class
            The segment with standardized logs.
        """
        _ = self
        if mean is None:
            mean = df.mean()
        if std is None:
            std = df.std()
        return (df - mean) / (std + eps)

    @process_columns
    def norm_min_max(self, df, min=None, max=None, q_min=None, q_max=None, clip=True):  # pylint: disable=redefined-builtin
        """Linearly scale well logs to a [0, 1] range.

        Parameters
        ----------
        min : None or ndarray, optional
            Minimum values of the logs. If `None`, it is calculated
            independently for each log in the segment.
        max : None or ndarray or tuple or list of ndarrays
            Maximum values of the logs. If `None`, it is calculated
            independently for each log in the segment.
        q_min : float
            A quantile of logs to use as `min` if `min` is not given.
        q_max : float
            A quantile of logs to use as `max` if `max` is not given.
        clip : bool
            Specify, whether to clip scaled logs to a [0, 1] range.

        Returns
        -------
        well : AbstractWellSegment or a child class
            The segment with normalized logs.
        """
        _ = self

        if min is None and q_min is None:
            min = df.min()
        elif q_min is not None:
            min = df.quantile(q_min)

        if max is None and q_max is None:
            max = df.max()
        elif q_max is not None:
            max = df.quantile(q_max)

        df = (df - min) / (max - min)
        if clip:
            df.clip(0, 1, inplace=True)
        return df

    def equalize_histogram(self, src=None, dst=None, channels='last'):
        """Normalize core images by histogram equalization.

        Parameters
        ----------
        src : None, str or iterable
            Attributes to normalize. If `None`, `src` will be
            `('core_dl', 'core_uv')`. Defaults to `None`.
        dst : None, str or iterable
            Attributes to save normalized images to. If `None`, images will be
            saved into `src`. Defaults to `None`.
        channels : 'last' or 'first'
            Channels axis in images. Defaults to 'last'.

        Returns
        -------
        self : AbstractWellSegment or a child class
            Self with normalized images.
        """
        src = to_list(src)
        for item in src:
            _ = getattr(self, item)
        src = ['_' + item if item in ['core_dl', 'core_uv'] else item for item in src]
        if dst is None:
            dst = src
        else:
            dst = to_list(dst)
        for _src, _dst in zip(src, dst):
            img = getattr(self, _src)
            if img.ndim == 3:
                img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2YCrCb)
                _slice = [slice(None)] * 3
                axis = -1 if channels == 'last' else 0
                _slice[axis] = 0
                img[_slice] = cv2.equalizeHist(img[_slice])
                img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
            else:
                img = cv2.equalizeHist(img)
            setattr(self, _dst, img)
        return self

    def shift_logs(self, max_period, mnemonics=None):
        """Shift every `logs` column from `mnemonics` by a random step sampled
        from discrete uniform distribution in [-`max_period`, `max_period`].
        All new resulting empty positions are filled with first/last
        column value depending on shift direction.

        Parameters
        ----------
        max_period : float
            Max possible shift period in meters.
        mnemonics : None or str or list of str
            - If `None`, shift all logs columns.
            - If `str`, shift single column from logs with `mnemonics` name.
            - If `list`, shift all logs columnns with names in `mnemonics`.
            Defaults to `None`.

        Returns
        -------
        self : AbstractWellSegment or a child class
            Self with shifted logs columns.
        """
        mnemonics = self.logs.columns if mnemonics is None else to_list(mnemonics)
        max_period = int(np.floor(max_period * (len(self.logs) / self.length)))
        periods = np.random.randint(-max_period, max_period + 1, len(mnemonics))
        for mnemonic, period in zip(mnemonics, periods):
            fill_index = 0 if period > 0 else -1
            fill_value = self.logs[mnemonic].iloc[fill_index]
            self.logs[mnemonic] = self.logs[mnemonic].shift(periods=period, fill_value=fill_value)
        return self
