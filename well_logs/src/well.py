"""Implements Well class."""
# pylint: disable=abstract-method

from abc import ABCMeta
from copy import copy
from functools import wraps
from collections import Counter

import numpy as np

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
