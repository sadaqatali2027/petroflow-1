from abc import ABCMeta, abstractmethod

class AbstractWell(metaclass=ABCMeta):
    @abstractmethod
    def __getitem__(self, key):
        pass

    @abstractmethod
    def dump(self, key):
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
    def drop_logs(self):
        pass

    @abstractmethod
    def keep_logs(self):
        pass

    @abstractmethod
    def rename_logs(self):
        pass

    @abstractmethod
    def keep_matched_intervals(self):
        pass

    @abstractmethod
    def split_segments(self):
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
    def fill_nans(self):
        pass

    @abstractmethod
    def norm_mean_std(self):
        pass

    @abstractmethod
    def norm_min_max(self):
        pass
