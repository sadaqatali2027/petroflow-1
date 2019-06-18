import os
import json
import base64
from copy import copy
from glob import glob
from itertools import chain, repeat

import numpy as np
import pandas as pd
import lasio
import PIL
from scipy.interpolate import interp1d
from plotly import tools
from plotly import graph_objs as go
from plotly.offline import init_notebook_mode, plot

from .abstract_well import AbstractWell
from .matching import select_contigious_intervals, join_samples, optimize_shift


def add_attr_properties(cls):
    for attr in cls.attrs_depth_index + cls.attrs_fdtd_index + cls.attrs_no_index:
        def prop(self, attr=attr):
            if getattr(self, "_" + attr) is None:
                getattr(self, "load_" + attr)()
            return getattr(self, "_" + attr)
        setattr(cls, attr, property(prop))
    return cls


def add_attr_loaders(cls):
    attr_iter = chain(
        zip(cls.attrs_depth_index, repeat(cls._load_depth_df)),
        zip(cls.attrs_fdtd_index, repeat(cls._load_fdtd_df)),
        zip(cls.attrs_no_index, repeat(cls._load_df))
    )
    for attr, loader in attr_iter:
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
class WellSegment(AbstractWell):
    attrs_depth_index = ("logs", "core_properties", "core_logs")
    attrs_fdtd_index = ("layers", "boring_intervals", "core_lithology", "samples")
    attrs_no_index = ("inclination",)

    def __init__(self, path, core_width=10, pixels_per_cm=5):
        super().__init__()
        self.path = path
        self.core_width = core_width
        self.pixels_per_cm = pixels_per_cm

        with open(os.path.join(self.path, "meta.json")) as meta_file:
            meta = json.load(meta_file)
        self.name = meta["name"]
        self.field = meta["field"]
        self.depth_from = meta["depth_from"]
        self.depth_to = meta["depth_to"]

        self._logs = None
        self._inclination = None
        self._layers = None
        self._boring_intervals = None
        self._core_properties = None
        self._core_lithology = None
        self._core_logs = None
        self._samples = None
        self._core_dl = None
        self._core_uv = None

    @staticmethod
    def _load_las(path, *args, **kwargs):
        return lasio.read(path, *args, **kwargs).df().reset_index()

    @staticmethod
    def _load_csv(path, *args, **kwargs):
        return pd.read_csv(path, *args, **kwargs)

    @staticmethod
    def _load_feather(path, *args, **kwargs):
        return pd.read_feather(path, *args, **kwargs)

    def _load_df(self, path, *args, **kwargs):
        ext = os.path.splitext(path)[1][1:]
        if not hasattr(self, "_load_" + ext):
            raise ValueError("A loader for data in {} format is not implemented".format(ext))
        return getattr(self, "_load_" + ext)(path, *args, **kwargs)

    def _filter_depth_df(self, df):
        return df[self.depth_from:self.depth_to]

    def _load_depth_df(self, path, *args, **kwargs):
        df = self._load_df(path, *args, **kwargs).set_index("DEPTH")
        df = self._filter_depth_df(df)
        return df

    def _filter_fdtd_df(self, df):
        depth_from, depth_to = zip(*df.index.values)
        mask = (np.array(depth_from) < self.depth_to) & (self.depth_from < np.array(depth_to))
        return df[mask]

    def _load_fdtd_df(self, path, *args, **kwargs):
        df = self._load_df(path, *args, **kwargs).set_index(["DEPTH_FROM", "DEPTH_TO"])
        df = self._filter_fdtd_df(df)
        return df

    @staticmethod
    def _get_full_name(path, name):
        files = glob(os.path.join(path, name + ".*"))
        if len(files) == 0:
            raise FileNotFoundError("A file {} does not exist in {}".format(name, path))
        if len(files) > 1:
            raise OSError("Several files called {} are found in {}".format(name, path))
        return files[0]

    @property
    def core_dl(self):
        if self._core_dl is None:
            self.load_core()
        return self._core_dl

    @property
    def core_uv(self):
        if self._core_uv is None:
            self.load_core()
        return self._core_uv

    @staticmethod
    def _load_image(path):
        return PIL.Image.open(path) if os.path.isfile(path) else None

    @staticmethod
    def _match_samples(dl_img, uv_img, height, width):
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

    def load_core(self, core_width=None, pixels_per_cm=None):
        self.core_width = core_width if core_width is not None else self.core_width
        self.pixels_per_cm = pixels_per_cm if pixels_per_cm is not None else self.pixels_per_cm

        depth_from = self.logs.index.min()
        depth_to = self.logs.index.max()
        height = int(round((depth_to - depth_from) * 100)) * self.pixels_per_cm
        width = self.core_width * self.pixels_per_cm
        core_dl = np.full((height, width, 3), np.nan, dtype=np.float32)
        core_uv = np.full((height, width, 3), np.nan, dtype=np.float32)

        sample_names = self.samples.index
        for sample in sample_names:
            sample_depth_from, sample_depth_to = self.samples.loc[sample, ["DEPTH_FROM", "DEPTH_TO"]]
            sample_height = int(round((sample_depth_to - sample_depth_from) * 100)) * self.pixels_per_cm

            dl_path = os.path.join(self.path, "samples_dl", str(sample) + ".png")
            uv_path = os.path.join(self.path, "samples_uv", str(sample) + ".png")

            dl_img = self._load_image(dl_path)
            uv_img = self._load_image(uv_path)

            dl_img, uv_img = self._match_samples(dl_img, uv_img, sample_height, width)

            top_crop = max(0, int(round((depth_from - sample_depth_from) * 100)) * self.pixels_per_cm)
            bottom_crop = sample_height - max(0, int(round((sample_depth_to - depth_to) * 100)) * self.pixels_per_cm)
            dl_img = dl_img[top_crop:bottom_crop]
            uv_img = uv_img[top_crop:bottom_crop]

            insert_pos = max(0, int(round((sample_depth_from - depth_from) * 100)) * self.pixels_per_cm)
            core_dl[insert_pos:insert_pos+dl_img.shape[0]] = dl_img
            core_uv[insert_pos:insert_pos+uv_img.shape[0]] = uv_img

        self._core_dl = core_dl / 255
        self._core_uv = core_uv / 255
        return self

    @staticmethod
    def _encode(img_path):
        with open(img_path, "rb") as img:
            encoded_img = base64.b64encode(img.read()).decode()
        encoded_img = "data:image/png;base64," + encoded_img
        return encoded_img

    def plot(self, plot_core=True, subplot_height=500, subplot_width=150):
        init_notebook_mode(connected=True)
        depth_from = self.logs.index.min()
        depth_to = self.logs.index.max()

        n_cols = len(self.logs.columns)
        subplot_titles = list(self.logs.columns)
        if plot_core and self.has_samples:
            n_cols += 2
            subplot_titles += ["CORE DL", "CORE UV"]
            dl_col = n_cols - 1
            uv_col = n_cols

        fig = tools.make_subplots(rows=1, cols=n_cols, subplot_titles=subplot_titles, shared_yaxes=True,
                                  print_grid=False)
        for i, mnemonic in enumerate(self.logs.columns, 1):
            trace = go.Scatter(x=self.logs[mnemonic], y=self.logs.index, mode="lines", name=mnemonic)
            fig.append_trace(trace, 1, i)

        images = []
        if plot_core and self.has_samples:
            trace = go.Scatter(x=[0, 1], y=[depth_from, depth_to], opacity=0, name="CORE DL")
            fig.append_trace(trace, 1, dl_col)
            trace = go.Scatter(x=[0, 1], y=[depth_from, depth_to], opacity=0, name="CORE UV")
            fig.append_trace(trace, 1, uv_col)

            samples = self.samples.index
            for sample in samples:
                depth_from, depth_to = self.samples.loc[sample, ["DEPTH_FROM", "DEPTH_TO"]]

                sample_dl = self._encode(os.path.join(self.path, "samples_dl", str(sample) + ".png"))
                sample_dl = go.layout.Image(source=sample_dl, xref="x"+str(dl_col), yref="y", x=0, y=depth_from,
                                            sizex=1, sizey=depth_to-depth_from, sizing="stretch", layer="below")
                images.append(sample_dl)

                sample_uv = self._encode(os.path.join(self.path, "samples_uv", str(sample) + ".png"))
                sample_uv = go.layout.Image(source=sample_uv, xref="x"+str(uv_col), yref="y", x=0, y=depth_from,
                                            sizex=1, sizey=depth_to-depth_from, sizing="stretch", layer="below")
                images.append(sample_uv)

        layout = fig.layout
        fig_layout = go.Layout(title="Скважина {}".format(self.name), showlegend=False, width=n_cols*subplot_width,
                               height=subplot_height, yaxis=dict(range=[depth_to, depth_from]), images=images)
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

        plot(fig)
        return self

    def __getitem__(self, key):
        if not isinstance(key, slice):
            return self.keep_logs(key)
        res = self.copy()
        res._logs = res.logs[key]

        start, stop = key.start, key.stop
        mask = (res.samples["DEPTH_FROM"] < stop) & (start < res.samples["DEPTH_TO"])
        res._samples = res.samples[mask]

        # TODO: slice all other dataframes

        if (res._core_dl is not None) and (res._core_uv is not None):
            depth_from = self.logs.index.min()
            start_pos = int(round((start - depth_from) * 100)) * self.pixels_per_cm
            stop_pos = int(round((stop - depth_from) * 100)) * self.pixels_per_cm
            res._core_dl = res._core_dl[start_pos:stop_pos]
            res._core_uv = res._core_uv[start_pos:stop_pos]
        return res

    def copy(self):
        return copy(self)

    def match_core_logs(self, mnemonic="GK", max_shift=5, delta_from=-1, delta_to=1, delta_step=0.1):
        log = self.logs[mnemonic]
        core_log = self.core_logs[mnemonic]
        log_interpolator = interp1d(log.index, log, kind="linear")
        contigious_samples_list = select_contigious_intervals(self.samples, max_shift)

        samples_df_list = []
        for samples_df in contigious_samples_list:
            joined_df = join_samples(samples_df, core_log)
            _, best_deltas = optimize_shift(samples_df, joined_df, log_interpolator, max_shift,
                                            delta_from, delta_to, delta_step)
            samples_df["DELTA"] = best_deltas
            samples_df_list.append(samples_df)

        self._samples = pd.concat(samples_df_list)
        return self

    def drop_logs(self, mnemonics=None):
        res = self.copy()
        res._logs.drop(mnemonics, axis=1, inplace=True)
        return res

    def keep_logs(self, mnemonics=None):
        res = self.copy()
        mnemonics = np.asarray(mnemonics).tolist()
        res._logs = res.logs[mnemonics]
        return res

    def rename_logs(self, rename_dict):
        self.logs.columns = [rename_dict.get(name, name) for name in self.logs.columns]
        return self
    
    def _core_chunks(self):
        samples = self.samples.copy()
        gaps = samples['DEPTH_FROM'][1:].values - samples['DEPTH_TO'][:-1].values

        if any(gaps < 0):
            raise ValueError('Core intersects the previous one: ', list(samples.index[1:][gaps < 0]))
        
        samples['TOP'] = True
        samples['TOP'][1:] = (gaps != 0)

        samples['BOTTOM'] = True
        samples['BOTTOM'][:-1] = (gaps != 0)

        chunks = pd.DataFrame({
            'TOP': samples[samples.TOP].DEPTH_FROM.values,
            'BOTTOM': samples[samples.BOTTOM].DEPTH_TO.values
        })

        return chunks
    
    def split_by_core(self):
        chunks = self._core_chunks()
        segments = []
        for _, (top, bottom) in self._core_chunks().iterrows():
            segments.append(self[top:bottom])
        return segments

    def drop_layers(self):
        pass

    def keep_layers(self):
        pass

    def drop_nans(self):
        pass

    def fill_nans(self):
        pass

    def norm_mean_std(self, axis=-1, mean=None, std=None, eps=1e-10, *, components):
        pass

    def norm_min_max(self, axis=-1, min=None, max=None, *, components):
        pass
