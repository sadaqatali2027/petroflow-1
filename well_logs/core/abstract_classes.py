"""Implements abstract classes for WellSegment, Well and WellBatch."""

from abc import ABCMeta, abstractmethod


class AbstractWellSegment(metaclass=ABCMeta):
    """Abstract class to check that all nesessary methods are
    implemented in `WellSegment` class."""
    @abstractmethod
    def __getitem__(self, key):
        pass

    @abstractmethod
    def dump(self):
        pass

    @abstractmethod
    def copy(self):
        pass

    @abstractmethod
    def match_core_logs(self):
        pass

    @abstractmethod
    def plot(self):
        pass

    @abstractmethod
    def plot_matching(self):
        pass

    @abstractmethod
    def drop_logs(self):
        pass

    @abstractmethod
    def keep_logs(self):
        pass

    @abstractmethod
    def rename_logs(self):
        pass

    @abstractmethod
    def fill_nans(self):
        pass

    @abstractmethod
    def norm_mean_std(self):
        pass

    @abstractmethod
    def norm_min_max(self):
        pass


class AbstractWell(AbstractWellSegment):
    """Abstract class to check that all nesessary methods are
    implemented in `Well` and `WellBatch` classes. """
    @abstractmethod
    def keep_matched_sequences(self):
        pass

    @abstractmethod
    def create_segments(self):
        pass

    @abstractmethod
    def drop_short_segments(self):
        pass

    @abstractmethod
    def crop(self):
        pass

    @abstractmethod
    def random_crop(self):
        pass

    @abstractmethod
    def drop_layers(self):
        pass

    @abstractmethod
    def keep_layers(self):
        pass

    @abstractmethod
    def drop_nans(self):
        pass

    @abstractmethod
    def create_mask(self):
        pass
