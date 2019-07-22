import os
import json
import base64
import shutil
from copy import copy
from glob import glob
from itertools import chain, repeat

import numpy as np
import pandas as pd
import lasio
import PIL
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from plotly import tools
from plotly import graph_objs as go
from plotly.offline import init_notebook_mode, plot

from .abstract_well import AbstractWell
from .matching import select_contigious_intervals, match_segment
from .joins import cross_join, between_join

def _min(x, y):
    if x is None:
        return y
    elif y is None:
        return x
    else:
        return min(x, y)

def _max(x, y):
    if x is None:
        return y
    elif y is None:
        return x
    else:
        return max(x, y)

def add_attr_properties(cls):
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
                # TODO: transform depths if core-to-log matching has already been performed
                return self
            return load
        setattr(cls, "load_" + attr, load_factory(attr, loader))
    return cls


@add_attr_properties
@add_attr_loaders
class WellSegment(AbstractWell):
    attrs_depth_index = ("logs", "core_properties", "core_logs")
    attrs_fdtd_index = ("layers", "matching_intervals", "boring_intervals", "core_lithology", "samples")
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

        self.has_samples = (len(glob(os.path.join(self.path, "samples.*"))) == 1)

        self._logs = None
        self._inclination = None
        self._layers = None
        self._matching_intervals = None
        self._boring_intervals = None
        self._core_properties = None
        self._core_lithology = None
        self._core_logs = None
        self._samples = None
        self._core_dl = None
        self._core_uv = None

    @property
    def length(self):
        return self.depth_to - self.depth_from

    @property
    def matching_intervals(self):
        if self._matching_intervals is None:
            if len(glob(os.path.join(self.path, "matching_intervals.*"))) == 1:
                self.load_matching_intervals()
            else:
                self._calc_matching_intervals()
        return self._matching_intervals

    def _calc_matching_intervals(self, mnemonic=None):
        data = []
        for segment in select_contigious_intervals(self.boring_intervals.reset_index()):
            data.append([segment["DEPTH_FROM"].min(), segment["DEPTH_TO"].max()])
        self._matching_intervals = pd.DataFrame(data, columns=["DEPTH_FROM", "DEPTH_TO"])

        if mnemonic is not None:
            well_log = self.logs[mnemonic]
            core_log = self.core_logs[mnemonic]
            r2_list = []
            for _, (depth_from, depth_to) in self._matching_intervals.iterrows():
                r2_list.append(self._calc_matching_r2(well_log, core_log[depth_from:depth_to]))
            self._matching_intervals["R2"] = r2_list
        self._matching_intervals.set_index(["DEPTH_FROM", "DEPTH_TO"], inplace=True)

    @staticmethod
    def _get_extension(path):
        return os.path.splitext(path)[1][1:]

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
        ext = self._get_extension(path)
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

    @classmethod
    def _get_full_name(cls, path, name):
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

    def _meters_to_pixels(self, meters):
        return int(round(meters * 100)) * self.pixels_per_cm

    def load_core(self, core_width=None, pixels_per_cm=None):
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

        self._core_dl = core_dl / 255
        self._core_uv = core_uv / 255
        return self

    def dump(self, path):
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
                except Exception:
                    pass
            else:
                if attr not in self.attrs_no_index:
                    attr_val = attr_val.reset_index()
                attr_val.to_feather(os.path.join(path, attr + ".feather"))

        # TODO: probably it makes sense to dump _boring_intervals_deltas and _core_lithology_deltas

        samples_dl_path = os.path.join(self.path, "samples_dl")
        if os.path.exists(samples_dl_path):
            shutil.copytree(samples_dl_path, os.path.join(path, "samples_dl"), copy_function=os.link)

        samples_uv_path = os.path.join(self.path, "samples_uv")
        if os.path.exists(samples_uv_path):
            shutil.copytree(samples_uv_path, os.path.join(path, "samples_uv"), copy_function=os.link)

        return self

    @staticmethod
    def _encode(img_path):
        with open(img_path, "rb") as img:
            encoded_img = base64.b64encode(img.read()).decode()
        encoded_img = "data:image/png;base64," + encoded_img
        return encoded_img

    def plot(self, plot_core=True, subplot_height=500, subplot_width=150):
        init_notebook_mode(connected=True)

        n_cols = len(self._logs.columns)
        subplot_titles = list(self._logs.columns)
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
        fig_layout = go.Layout(title="Скважина {}".format(self.name), showlegend=False, width=n_cols*subplot_width,
                               height=subplot_height, yaxis=dict(range=[self.depth_to, self.depth_from]),
                               images=images)
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
        res.depth_from, res.depth_to = key.start, key.stop

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
        return copy(self)

    @staticmethod
    def _calc_matching_r2(well_log, core_log):
        well_log = well_log.dropna()
        interpolator = interp1d(well_log.index, well_log, kind="linear", fill_value="extrapolate")
        well_log = interpolator(core_log.index).reshape(-1, 1)
        reg = LinearRegression().fit(well_log, core_log)
        return r2_score(core_log, reg.predict(well_log))

    def _apply_matching(self, mnemonic):
        core_lithology_deltas = self._core_lithology_deltas.reset_index()
        core_lithology_deltas["key"] = 1

        for attr in ["core_logs", "core_properties"]:
            # TODO: load conditionally
            attr_df = getattr(self, attr).reset_index()
            columns = attr_df.columns

            attr_df["key"] = 1
            merged_df = pd.merge(attr_df, core_lithology_deltas, on="key")
            merged_df = merged_df[((merged_df["DEPTH"] >= merged_df["DEPTH_FROM"]) &
                                   (merged_df["DEPTH"] < merged_df["DEPTH_TO"]))]
            merged_df["DEPTH"] += merged_df["DELTA"]
            setattr(self, "_" + attr, merged_df[columns].set_index("DEPTH").sort_index())

        # TODO: update fdtd dataframes
        # TODO: carfully update samples

        boring_intervals = pd.merge(self._boring_intervals.reset_index(),
                                    self._boring_intervals_deltas[["DEPTH_FROM", "DEPTH_TO", "DELTA"]],
                                    on=["DEPTH_FROM", "DEPTH_TO"])
        boring_intervals["DEPTH_FROM"] += boring_intervals["DELTA"]
        boring_intervals["DEPTH_TO"] += boring_intervals["DELTA"]
        self._boring_intervals = boring_intervals.drop("DELTA", axis=1).set_index(["DEPTH_FROM", "DEPTH_TO"]).sort_index()

        self._calc_matching_intervals(mnemonic=mnemonic)

    def _save_matching_report(self, mnemonic):
        boring_intervals = self._boring_intervals_deltas
        boring_intervals["DEPTH_FROM_DELTA"] = boring_intervals["DEPTH_FROM"] + boring_intervals["DELTA"]
        boring_intervals["DEPTH_TO_DELTA"] = boring_intervals["DEPTH_TO"] + boring_intervals["DELTA"]
        boring_intervals = boring_intervals[["DEPTH_FROM", "DEPTH_TO", "DEPTH_FROM_DELTA", "DEPTH_TO_DELTA"]]
        boring_intervals.columns = ["Кровля интервала долбления", "Подошва интервала долбления",
                                    "Увязанная кровля интервала долбления", "Увязанная подошва интервала долбления"]

        lithology_intervals = self._core_lithology_deltas
        lithology_intervals["DEPTH_FROM_DELTA"] = lithology_intervals["DEPTH_FROM"] + lithology_intervals["DELTA"]
        lithology_intervals["DEPTH_TO_DELTA"] = lithology_intervals["DEPTH_TO"] + lithology_intervals["DELTA"]
        lithology_intervals = lithology_intervals[["DEPTH_FROM", "DEPTH_TO", "DEPTH_FROM_DELTA", "DEPTH_TO_DELTA"]]
        lithology_intervals.columns = ["Кровля интервала литописания", "Подошва интервала литописания",
                                       "Увязанная кровля интервала литописания", "Увязанная подошва интервала литописания"]

        cross = cross_join(boring_intervals, lithology_intervals)
        mask = ((cross["Кровля интервала литописания"] >= cross["Кровля интервала долбления"]) &
                (cross["Подошва интервала литописания"] <= cross["Подошва интервала долбления"]))
        cross = cross[mask]

        core_log = self.core_logs[mnemonic].reset_index()
        core_log.columns = ["Глубина " + mnemonic, "Значение " + mnemonic]

        report = between_join(core_log, cross, left_on="Глубина "+mnemonic,
                              right_on=("Увязанная кровля интервала литописания",
                                        "Увязанная подошва интервала литописания"))
        report.to_csv(os.path.join(self.path, self.name + "_matching_report.csv"), index=False)

    def match_core_logs(self, mnemonic="GK", max_shift=5, delta_from=-4, delta_to=4, delta_step=0.1,
                        max_iter=50, max_iter_time=0.25, save_report=False):
        if max_shift <= 0:
            raise ValueError("max_shift must be positive")
        if delta_from > delta_to:
            raise ValueError("delta_to must be greater than delta_from")
        if max(np.abs(delta_from), np.abs(delta_to)) > max_shift:
            raise ValueError("delta_from and delta_to must not exceed max_shift in absolute value")

        well_log = self.logs[mnemonic]
        core_log = self.core_logs[mnemonic]

        lithology_intervals = self.core_lithology.reset_index()[["DEPTH_FROM", "DEPTH_TO"]]
        contigious_segments = select_contigious_intervals(self.boring_intervals.reset_index(), 2 * max_shift)

        segments = []
        lithology_segments = []

        for segment in contigious_segments:
            segment, lithology_segment = match_segment(segment, lithology_intervals, well_log, core_log,
                                                       max_shift, delta_from, delta_to, delta_step,
                                                       max_iter, timeout=max_iter*max_iter_time)
            segments.append(segment)
            lithology_segments.append(lithology_segment)
        self._boring_intervals_deltas = pd.concat(segments)
        self._core_lithology_deltas = pd.concat(lithology_segments)
        self._apply_matching(mnemonic=mnemonic)

        if save_report:
            self._save_matching_report(mnemonic=mnemonic)
        return self

    def plot_matching(self, mnemonic="GK", subplot_height=750, subplot_width=200):
        init_notebook_mode(connected=True)

        well_log = self.logs[mnemonic]
        core_log = self.core_logs[mnemonic]

        self._calc_matching_intervals(mnemonic=mnemonic)
        n_cols = len(self._matching_intervals)
        subplot_titles = ["R^2 = {:.3f}".format(r2) for _, r2 in self._matching_intervals["R2"].iteritems()]
        fig = tools.make_subplots(rows=1, cols=n_cols, print_grid=False, subplot_titles=subplot_titles)

        intervals = self._matching_intervals.reset_index()[["DEPTH_FROM", "DEPTH_TO"]]
        for i, (_, (depth_from, depth_to)) in enumerate(intervals.iterrows(), 1):
            well_log_segment = well_log[depth_from - 3 : depth_to + 3]
            core_log_segment = core_log[depth_from:depth_to]
            well_log_trace = go.Scatter(x=well_log_segment, y=well_log_segment.index, name="Well " + mnemonic,
                                        line=dict(color="rgb(255, 127, 14)"), showlegend=(i==1))
            core_log_trace = go.Scatter(x=core_log_segment, y=core_log_segment.index, name="Core " + mnemonic,
                                        line=dict(color="rgb(31, 119, 180)"), showlegend=(i==1))
            fig.append_trace(well_log_trace, 1, i)
            fig.append_trace(core_log_trace, 1, i)

        layout = fig.layout
        fig_layout = go.Layout(title="Скважина {}".format(self.name), legend=dict(orientation="h"),
                               width=n_cols*subplot_width, height=subplot_height)
        layout.update(fig_layout)

        for key in layout:
            if key.startswith("xaxis"):
                layout[key]["fixedrange"] = True

        for ann in layout["annotations"]:
            ann["font"]["size"] = 16

        for ix in range(n_cols):
            axis_ix = str(ix + 1) if ix > 0 else ""
            axis_name = "yaxis" + axis_ix
            layout[axis_name]["autorange"] = "reversed"

        plot(fig)
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

    def keep_matched_intervals(self, threshold=0.6):
        mask = self.matching_intervals["R2"] > threshold
        intervals = self.matching_intervals[mask].reset_index()[["DEPTH_FROM", "DEPTH_TO"]]
        res_segments = []
        for _, (depth_from, depth_to) in intervals.iterrows():
            res_segments.append(self[depth_from:depth_to])
        return res_segments

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

    def split_segments(self, connected=True):
        segments = []
        if connected:
            df = self._core_chunks()
        else:
            df = self.samples.reset_index()[['DEPTH_FROM', 'DEPTH_TO']]
        for _, (top, bottom) in df.iterrows():
            segments.append(self[top:bottom])
        return segments

    def random_crop(self, height, n_crops=1):
        positions = np.random.uniform(self.depth_from, self.depth_to-height, size=n_crops)
        return [self[pos:pos+height] for pos in positions]
    
    def crop(self, height, step, drop_last=True):
        positions = np.arange(self.depth_from, self.depth_to, step)
        if drop_last and positions[-1]+height > self.depth_to:
            positions = positions[:-1]
        else:
            height = min(height, self.depth_to-positions[-1])
        return [self[pos:pos+height] for pos in positions]

    def create_mask(self, src, column, labels, mode, default=-1, dst='mask'):
        if src in self.attrs_fdtd_index:
            self._create_mask_fdtf(src, column, labels, mode, default, dst)
        else:
            # TODO: create mask from depth_index
            pass

    def _create_mask_fdtf(self, src, column, labels, mode, default=-1, dst='mask'):
        if mode == 'core':
            mask = np.ones(len(self.core_dl)) * default
        elif mode == 'logs':
            mask = np.ones(len(self.logs)) * default
        else:
            raise ValueError('Unknown mode: ', mode)

        table = getattr(self, src)
        for row in table.iterrows():
            factor = len(mask) / self.length if mode == 'core' else len(mask)
            depth_from, depth_to = row[1].name
            start = np.floor((max(depth_from, self.depth_from) - self.depth_from) * factor)
            end = np.ceil((min(depth_to, self.depth_to) - self.depth_from) * factor)
            mask[int(start):int(end)] = labels[row[1][column]]
        setattr(self, dst, mask)
        

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
