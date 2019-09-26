"""Implements Well class."""
# pylint: disable=abstract-method

from abc import ABCMeta
from copy import copy
from functools import wraps
from collections import Counter
import re

import numpy as np
import pandas as pd

from .abstract_classes import AbstractWell
from .well_segment import WellSegment
from .exceptions import SkipWellException


class SegmentDelegatingMeta(ABCMeta):
    """A metaclass to delegate abstract methods from `Well` to its children
    (instances of `Well` or `WellSegment`)."""
    def __new__(mcls, name, bases, namespace):
        abstract_methods = [
            base.__abstractmethods__ for base in bases if hasattr(base, "__abstractmethods__")
        ]
        abstract_methods = frozenset().union(*abstract_methods)
        for method_name in abstract_methods:
            if method_name not in namespace:
                namespace[method_name] = mcls._make_delegator(method_name)
        return super().__new__(mcls, name, bases, namespace)

    @staticmethod
    def _make_delegator(name):
        @wraps(getattr(WellSegment, name))
        def delegator(self, *args, **kwargs):
            results = []
            for segment in self:
                res = getattr(segment, name)(*args, **kwargs)
                if not isinstance(res, list):
                    res = [res]
                results.extend(res)
            res_well = self.copy()
            res_well.segments = results
            return res_well
        return delegator


class Well(AbstractWell, metaclass=SegmentDelegatingMeta):
    """A class, representing a well.

    A well consists of segments - instances of `WellSegment` class,
    representing a contiguous part of a well. Initially an instance of `Well`
    class has a single segment, containing information about the whole well.
    Subsequently, several `Well` methods, such as `crop`, `random_crop` or
    `drop_nans`, increase the number of segments, storing them in a tree-based
    structure.

    All methods, declared in `AbstractWell` and not overridden in `Well` will
    be created by `SegmentDelegatingMeta` metaclass. These methods will
    delegate the call to each segment of the well.

    Parameters
    ----------
    path : str or None
        If `None`, `Well` will be created from `segments`. If `str`, a path to
        a directory with well data, containing:
        - `meta.json` - a json dict with the following keys:
            - `name` - well name
            - `field` - field name
            - `depth_from` - minimum depth entry in the well logs
            - `depth_to` - maximum depth entry in the well logs
          These values will be stored as segment attributes.
        - `samples_dl` and `samples_uv` (optional) - directories, containing
          daylight and ultraviolet images of core samples respectively. Images
          of the same sample must have the same name in both dirs.
        - Optional `.csv`, `.las` or `.feather` file for certain segment
          attributes (see more details in the `WellSegment.Attributes`
          section).
    core_width : positive float, optional
        The width of core samples in cm. Defaults to 10 cm.
    pixels_per_cm : positive int, optional
        The number of pixels in cm used to determine the loaded width of core
        sample images. Image height is calculated so as to keep the aspect
        ratio. Defaults to 5 pixels.
    segments : list of `WellSegment` or `Well` instances or None, optional
        Segments to put into `segments` attribute. Usually is used by methods
        which increase the tree depth. If `None`, `path` must be defined.

    Attributes
    ----------
    segments : list of `WellSegment` or `Well` instances or None
        A segment tree of the well. All leave nodes are instances of
        `WellSegment` class, representing a contiguous part of a well. All
        other tree nodes are instances of `Well` class.
    """
    def __init__(self, *args, segments=None, **kwargs):
        super().__init__()
        if segments is None:
            self.segments = [WellSegment(*args, **kwargs)]
        else:
            self.segments = segments

    @property
    def tree_depth(self):
        """positive int: Depth of the tree consisting of `Well` and
        `WellSegment` instances. Initial depth of a created `Well` is 2: a
        root and a single segment.
        """
        if self._has_segments():
            return 2
        return self.segments[0].tree_depth + 1

    @property
    def length(self):
        """float: Length of the well in meters."""
        return self.depth_to - self.depth_from

    @property
    def depth_from(self):
        """float: Top of the well in meters."""
        return min([well.depth_from for well in self])

    @property
    def depth_to(self):
        """float: Bottom of the well in meters."""
        return max([well.depth_to for well in self])

    @property
    def n_segments(self):
        """int: Total number of `WellSegment` instances at the last level of
        the segment tree."""
        return len(self.iter_level())

    def _has_segments(self):
        return all(isinstance(item, WellSegment) for item in self)

    def __iter__(self):
        """Iterate over segments."""
        for segment in self.segments:
            yield segment

    def iter_level(self, level=-1):
        """Iterate over segments at some fixed level of the segment tree.

        Parameters
        ----------
        level : int
            Level of the tree to iterate over.

        Returns
        -------
        segments : list of WellSegment or Well
            Segments from given level.
        """
        level = level if level >= 0 else self.tree_depth + level
        if (level < 0) or (level > self.tree_depth):
            raise ValueError("Level ({}) can't be negative or exceed tree depth ({})".format(level, self.tree_depth))
        if level == 0:
            return [self]
        if level == 1:
            return self.segments
        return [item for well in self for item in well.iter_level(level - 1)]

    def _prune(self):
        """Recursively prune segment tree."""
        self.segments = [well for well in self if isinstance(well, WellSegment) or well.n_segments > 0]
        for well in self:
            if isinstance(well, Well):
                well._prune() # pylint: disable=protected-access

    def prune(self):
        """Remove subtrees without `WellSegment` instances at the last level
        of the tree.

        Returns
        -------
        self : AbstractWell or a child class
            Self with prunned tree.
        """
        self._prune()
        if not self.segments:
            raise SkipWellException("Empty well after prunning")
        return self

    def copy(self):
        """Perform shallow copy of an object.

        Returns
        -------
        self : AbstractWell or a child class
            Shallow copy.
        """
        return copy(self)

    def dump(self, path):
        """Dump well data. The well will be aggregated and the resulting
        segment will be dumped. Segment attributes are saved in the following
        manner:
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
        self : AbstractWell or a child class
            Self unchanged.
        """
        # TODO: aggregate before dumping
        self.segments[0].dump(path)
        return self

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
        self : AbstractWell or a child class
            The well with kept matched segments.
        """
        for well in self.iter_level(-2):
            well.segments = [
                Well(segments=segment.keep_matched_sequences(mode, threshold)) for segment in well
            ]
        return self.prune()

    def create_segments(self, src, connected=True):
        """Split segments at the last level of the segment tree into parts
        with depth ranges, specified in attributes in `src`.

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
        self : AbstractWell or a child class
            The well with split segments.
        """
        wells = self.iter_level(-2)
        for well in wells:
            well.segments = [
                Well(segments=segment.create_segments(src, connected)) for segment in well
            ]
        return self

    def crop(self, length, step, drop_last=True):
        """Create crops from segments at the last level. All cropped segments
        have the same length and are cropped with some fixed step. The tree depth
        will be increased.

        Parameters
        ----------
        length : positive float
            Length of each crop in meters.
        step : positive float
            Step of cropping in meters.
        drop_last : bool, optional
            If `True`, all segment that are out of segment bounds will be
            dropped. If `False`, the whole segment will be covered by crops.
            The first crop which comes out of segment bounds will be kept, the
            following crops will be dropped. Defaults to `True`.

        Returns
        -------
        self : AbstractWell or a child class
            The well with cropped segments.
        """
        wells = self.iter_level(-2)
        for well in wells:
            well.segments = [
                Well(segments=segment.crop(length, step, drop_last))
                for segment in well
            ]
        return self

    def random_crop(self, length, n_crops=1):
        """Create random crops from segments at the last level. All cropped
        segments have the same length, their positions are sampled uniformly
        from the segment. The tree depth will be increased. Branches of the
        tree without segments at last level will be dropped.

        Parameters
        ----------
        length : positive float
            Length of each crop in meters.
        n_crops : positive int, optional
            The number of crops from the segment. Defaults to 1.

        Returns
        -------
        self : AbstractWell or a child class
            The well with cropped segments.
        """
        wells = self.iter_level(-2)
        p = np.array([sum([segment.length for segment in item]) for item in wells])
        random_wells = Counter(np.random.choice(wells, n_crops, p=p/sum(p)))
        for well in wells:
            if well in random_wells:
                n_well_crops = random_wells[well]
                p = np.array([item.length for item in well])
                random_segments = Counter(np.random.choice(well.segments, n_well_crops, p=p/sum(p)))
                well.segments = [
                    Well(segments=segment.random_crop(length, n_segment_crops))
                    for segment, n_segment_crops in random_segments.items()
                ]
            else:
                well.segments = []
        return self.prune()

    def drop_nans(self, logs=None):
        """Split a well into contiguous segments, that does not contain `nan`
        values in logs, indicated in `logs`. The tree depth will be increased.

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
        self : AbstractWell
            The well with dropped `nan` values from segments.
        """
        wells = self.iter_level(-2)
        for well in wells:
            well.segments = [Well(segments=segment.drop_nans(logs=logs)) for segment in well]
        return self.prune()

    def drop_short_segments(self, min_length):
        """Drop segments at the last level with length smaller than
        `min_length`.

        Parameters
        ----------
        min_length : positive float
            Segments shorter than `min_length` are dropped.

        Returns
        -------
        self : AbstractWell
            The well with dropped short segments.
        """
        wells = self.iter_level(-2)
        for well in wells:
            well.segments = [segment for segment in well if segment.length > min_length]
        return self.prune()

    def _aggregate_array(self, func, attr, aggregate):
        pixels_per_m = self.segments[0].pixels_per_cm * 100

        agg_array_hight_pix = round((self.depth_to-self.depth_from)*pixels_per_m)
        attr_val_shape = getattr(self.segments[0], attr).shape

        total = np.zeros((agg_array_hight_pix, *attr_val_shape[1:]), dtype=int)
        background = np.full_like(total, np.nan, dtype=np.double)
        for segment in self.segments:

            attr_val = getattr(segment, attr)
            segment_place = slice(round((segment.depth_from-self.depth_from)*pixels_per_m),
                                  round((segment.depth_to-self.depth_from)*pixels_per_m))

            if aggregate is False:
                background[segment_place] = attr_val
                continue
            if func == 'max':
                background[segment_place] = np.fmax(background[segment_place], attr_val)
                continue
            if func == 'mean':
                background[segment_place] = np.nansum([background[segment_place], attr_val], axis=0)
                total[segment_place] += 1

        if func == 'max' or aggregate is False:
            return background
        if func == 'mean':
            total = np.where(total == 0, 1, total)
            return background / total
        raise ValueError("Aggregation function can't be ({})".format(func))

    def _aggregate_logs(self, func, step, aggregate):
        logs_from = min(self.iter_level(), key=lambda segment: segment.logs.index[0]).logs.index[0]
        logs_to = max(self.iter_level(), key=lambda segment: segment.logs.index[-1]).logs.index[-1]
        mnemonics = self.iter_level()[0].logs.columns

        indices = pd.Index(np.arange(logs_from, logs_to + step, step), name='DEPTH')
        total = np.zeros((indices.shape[0], len(mnemonics)), dtype=int)
        background = np.full_like(total, np.nan, dtype=np.double)

        for segment in self.segments:
            logs = segment.logs
            logs_place = slice(round(float(logs.index[0] - logs_from) / step),
                               round(float(logs.index[-1] - logs_from) / step) + 1)

            if aggregate is False:
                background[logs_place] = logs.values
                continue
            if func == 'max':
                background[logs_place] = np.fmax(background[logs_place], logs.values)
                continue
            if func == 'mean':
                mask = np.isnan(background[logs_place]) & np.isnan(logs.values)
                background[logs_place] = np.nansum([background[logs_place], logs.values], axis=0)
                background[logs_place] = np.where(mask, np.nan, background[logs_place])
                total[logs_place] += 1

        if func == 'max' or aggregate is False:
            return pd.DataFrame(background, index=indices, columns=mnemonics)
        if func == 'mean':
            total = np.where(total == 0, 1, total)
            return pd.DataFrame(background / total, index=indices, columns=mnemonics)
        raise ValueError("Aggregation function can't be ({})".format(func))

    def aggregate(self, func, attrs_to_aggregate=None, level=None): # pylint: disable=too-many-branches
        if level in (-1, -2):
            raise ValueError("Level can't be ({})".format(level))

        level = -self.tree_depth if level is None else level
        attrs_to_aggregate = ['logs', 'core_uv', 'core_dl'] if attrs_to_aggregate is None else attrs_to_aggregate

        aggregate_attrs = list(set(WellSegment.attrs_depth_index[1:]) & set(attrs_to_aggregate))
        check_to_concat = WellSegment.attrs_depth_index[1:]+WellSegment.attrs_fdtd_index
        concat_attrs = [attr for attr in check_to_concat if attr not in aggregate_attrs]

        wells = self.iter_level(level)
        for well in wells:
            well.segments = well.iter_level()
            seg_0 = well.segments[0]

            for attr in WellSegment.attrs_pixel_index:
                setattr(seg_0, '_'+attr, well._aggregate_array(func, attr, attr in attrs_to_aggregate)) # pylint: disable=protected-access

            step = (seg_0.logs.index[1:] - seg_0.logs.index[:-1]).min()
            setattr(seg_0, '_logs', well._aggregate_logs(func, step, 'logs' in attrs_to_aggregate)) # pylint: disable=protected-access
            # Reset index of all segments and merge it
            for attr in aggregate_attrs+concat_attrs:
                attr_val_0 = getattr(seg_0, attr)
                if not attr_val_0 is None:
                    attr_val_0.reset_index(inplace=True)

                for i, segment in enumerate(well.segments[1:]):

                    attr_val = getattr(segment, attr)

                    if attr_val is None:
                        continue

                    attr_val.reset_index(inplace=True)

                    if attr_val.dropna(how='all').empty:
                        continue

                    if attr in aggregate_attrs:
                        attr_val_0 = pd.merge_ordered(attr_val_0, attr_val, on='DEPTH',
                                                      suffixes=('_'+str(i), ''))
                    else:
                        attr_val_0 = pd.concat([attr_val_0, attr_val])

                setattr(seg_0, '_'+attr, attr_val_0)


            for attr in concat_attrs:
                attr_val_0 = getattr(seg_0, '_'+attr)
                if not attr_val_0 is None and not attr_val_0.empty:
                    attr_val_0.drop_duplicates(inplace=True)
                    if 'DEPTH' in attr_val_0.columns:
                        attr_val_0 = attr_val_0.set_index('DEPTH')
                    else:
                        attr_val_0 = attr_val_0.set_index(['DEPTH_FROM', 'DEPTH_TO'])
                    attr_val_0.sort_index(inplace=True)
                    setattr(seg_0, '_'+attr, attr_val_0)

            for attr in aggregate_attrs:
                attr_val_0 = getattr(seg_0, '_'+attr)

                if not attr_val_0 is None and not attr_val_0.empty:
                    attr_val_0 = attr_val_0.set_index('DEPTH')
                else:
                    continue

                columns = [column for column in attr_val_0 if not re.match(r'.*_\d*$', column)] # List of origin columns
                duplicate_columns = [[] for i in range(len(columns))] #  List of lists duplicates of origin column

                # Fill duplicate_columns
                for column_copy in attr_val_0.columns:
                    for i, column in enumerate(columns):
                        pattern = re.compile(column+r'_\d*$')
                        if re.match(pattern, column_copy):
                            duplicate_columns[i].append(column_copy)

                # Aggregate and drop duplicate_columns
                for column, duplicate in zip(columns, duplicate_columns):
                    attr_val_0[column] = getattr(attr_val_0[duplicate+[column]], func)(axis=1)
                    attr_val_0.drop(duplicate, axis=1, inplace=True)

                setattr(seg_0, '_'+attr, attr_val_0)
            setattr(seg_0, 'depth_from', well.depth_from)
            setattr(seg_0, 'depth_to', well.depth_to)
            well.segments = [seg_0]
        return self.prune()
