import os
import json
import base64
import shutil
from copy import copy
from glob import glob
from functools import reduce
from itertools import chain, repeat, product

import numpy as np
import pandas as pd
import lasio
import PIL
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from plotly import graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, plot

from .abstract_well import AbstractWell
from .matching import select_contigious_intervals, match_boring_sequence, Shift
from .joins import cross_join, between_join, fdtd_join


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
    attrs_fdtd_index = ("layers", "boring_sequences", "boring_intervals", "core_lithology", "samples")
    attrs_no_index = ("inclination",)

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

        self.has_samples = self._has_file("samples")

        self._logs = None
        self._inclination = None
        self._layers = None
        self._boring_sequences = None
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
    def boring_sequences(self):
        if self._boring_sequences is None:
            if self._has_file("boring_sequences"):
                self.load_boring_sequences()
            else:
                self._calc_boring_sequences()
        return self._boring_sequences

    def _calc_boring_sequences(self):
        data = []
        for segment in select_contigious_intervals(self.boring_intervals.reset_index()):
            data.append([segment["DEPTH_FROM"].min(), segment["DEPTH_TO"].max()])
        self._boring_sequences = pd.DataFrame(data, columns=["DEPTH_FROM", "DEPTH_TO"])
        self._boring_sequences.set_index(["DEPTH_FROM", "DEPTH_TO"], inplace=True)

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

    def _has_file(self, name):
        files = glob(os.path.join(self.path, name + ".*"))
        if len(files) == 1:
            return True
        return False

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
        if key.start is not None:
            res.depth_from = float(max(res.depth_from, key.start))
        if key.stop is not None:
            res.depth_to = float(min(res.depth_to, key.stop))

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

    def _apply_matching(self):
        core_lithology_deltas = self._core_lithology_deltas.reset_index()

        # Update DataFrames with depth index
        attrs_depth_index = [attr for attr in self.attrs_depth_index if attr.startswith("core_")]
        for attr in attrs_depth_index:
            if not self._has_file(attr):
                continue
            attr_df = getattr(self, attr).reset_index()
            columns = attr_df.columns
            merged_df = between_join(attr_df, core_lithology_deltas)
            merged_df["DEPTH"] += merged_df["DELTA"]
            setattr(self, "_" + attr, merged_df[columns].set_index("DEPTH").sort_index())

        # TODO: carfully update samples and reload core images if needed

        core_lithology = pd.merge(self._core_lithology.reset_index(),
                                  self._core_lithology_deltas[["DEPTH_FROM", "DEPTH_TO", "DELTA"]],
                                  on=["DEPTH_FROM", "DEPTH_TO"])
        core_lithology["DEPTH_FROM"] += core_lithology["DELTA"]
        core_lithology["DEPTH_TO"] += core_lithology["DELTA"]
        core_lithology = core_lithology.drop("DELTA", axis=1)
        self._core_lithology = core_lithology.set_index(["DEPTH_FROM", "DEPTH_TO"]).sort_index()

        boring_intervals = pd.merge(self._boring_intervals.reset_index(),
                                    self._boring_intervals_deltas[["DEPTH_FROM", "DEPTH_TO", "DELTA"]],
                                    on=["DEPTH_FROM", "DEPTH_TO"])
        boring_intervals["DEPTH_FROM"] += boring_intervals["DELTA"]
        boring_intervals["DEPTH_TO"] += boring_intervals["DELTA"]
        boring_intervals = boring_intervals.drop("DELTA", axis=1)
        self._boring_intervals = boring_intervals.set_index(["DEPTH_FROM", "DEPTH_TO"]).sort_index()

    def _save_matching_report(self):
        boring_sequences = self.boring_sequences.reset_index()[["DEPTH_FROM", "DEPTH_TO", "MODE"]]
        not_none_mask = boring_sequences["MODE"].map(lambda x: x is not None)
        boring_sequences = boring_sequences[not_none_mask]
        if len(boring_sequences) == 0:
            return

        core_logs_list = []
        for _, (depth_from, depth_to, mode) in boring_sequences.iterrows():
            _, core_mnemonic, core_attr = self._parse_matching_mode(mode)
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
        split_mode = mode.split("~")
        if len(split_mode) != 2:
            raise ValueError("Incorrect mode format")
        log_mnemonic = split_mode[0]
        split_core_mode = split_mode[1].split(".")
        if len(split_core_mode) != 2:
            raise ValueError("Incorrect mode format")
        core_attr, core_mnemonic = split_core_mode
        return log_mnemonic, core_mnemonic, core_attr

    def _select_matching_mode(self, segment, mode_list):
        segment_depth_from = segment["DEPTH_FROM"].min()
        segment_depth_to = segment["DEPTH_TO"].max()
        core_len = segment["CORE_RECOVERY"].sum()
        for mode in mode_list:
            log_mnemonic, core_mnemonic, core_attr = self._parse_matching_mode(mode)
            if log_mnemonic in self.logs and self._has_file(core_attr) and core_mnemonic in getattr(self, core_attr):
                well_log = self.logs[log_mnemonic].dropna()
                core_log = getattr(self, core_attr)[core_mnemonic].dropna()
                well_log_len = len(well_log[segment_depth_from:segment_depth_to])
                core_log_len = len(core_log[segment_depth_from:segment_depth_to])
                if min(well_log_len, core_log_len) > max(core_len, 1):
                    return mode
        return None

    @staticmethod
    def _unify_matching_mode(mode):
        mode_list = [mode] if isinstance(mode, str) else mode
        mode_list = [mode.replace(" ", "") for mode in mode_list]
        return mode_list

    def match_core_logs(self, mode="GK ~ core_logs.GK", max_shift=5, delta_from=-4, delta_to=4, delta_step=0.1,
                        max_iter=50, max_iter_time=0.25, save_report=False):
        if max_shift <= 0:
            raise ValueError("max_shift must be positive")
        if delta_from > delta_to:
            raise ValueError("delta_to must be greater than delta_from")
        if max(np.abs(delta_from), np.abs(delta_to)) > max_shift:
            raise ValueError("delta_from and delta_to must not exceed max_shift in absolute value")

        mode_list = self._unify_matching_mode(mode)

        if not self._has_file("core_lithology"):
            core_lithology = self.boring_intervals.reset_index()[["DEPTH_FROM", "DEPTH_TO"]]
            self._core_lithology = core_lithology.set_index(["DEPTH_FROM", "DEPTH_TO"])
        lithology_intervals = self.core_lithology.reset_index()[["DEPTH_FROM", "DEPTH_TO"]]

        # `boring_sequences` is a list of DataFrames, containing contigious boring intervals, extracted one after
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
                mode = self._select_matching_mode(sequence, mode_list)
                sequences_modes.append(mode)
                if mode is None:
                    # Don't shift a sequence if there's no data to perform matching
                    segment_depth_from = sequence["DEPTH_FROM"].min()
                    segment_depth_to = sequence["DEPTH_TO"].max()
                    zero_shift = Shift(segment_depth_from, segment_depth_to, 0, 0, np.nan)
                    sequences_shifts.append([zero_shift])
                    continue

                log_mnemonic, core_mnemonic, core_attr = self._parse_matching_mode(mode)
                well_log = self.logs[log_mnemonic].dropna()
                core_log = getattr(self, core_attr)[core_mnemonic].dropna()

                shifts = match_boring_sequence(sequence, lithology_intervals, well_log, core_log,
                                               max_shift, delta_from, delta_to, delta_step,
                                               max_iter, timeout=max_iter*max_iter_time)
                sequences_shifts.append(shifts)

            # Choose best shift for each boring sequence so that they don't overlap and maximize matching R^2
            best_shifts = None
            best_loss = None
            for shifts in product(*sequences_shifts):
                sorted_shifts = sorted(shifts, key=lambda x: x.depth_from)
                do_overlap = False
                for int1, int2 in zip(sorted_shifts[:-1], sorted_shifts[1:]):
                    if int1.depth_to > int2.depth_from:
                        do_overlap = True
                        break
                loss = np.nanmean([interval.loss for interval in sorted_shifts])
                if (not do_overlap) and (best_shifts is None or loss < best_loss):
                    best_shifts = shifts
                    best_loss = loss

            # Store shift deltas, mode and R^2
            for sequence, shift in zip(boring_sequences, best_shifts):
                mask = ((lithology_intervals["DEPTH_FROM"] >= sequence["DEPTH_FROM"].min()) &
                        (lithology_intervals["DEPTH_TO"] <= sequence["DEPTH_TO"].max()))
                sequence_lithology_intervals = lithology_intervals[mask]
                sequence_lithology_intervals["DELTA"] = shift.interval_deltas
                sequence["DELTA"] = shift.sequence_delta

                sequences_r2.append(shift.loss**2)
                matched_boring_sequences.append(sequence)
                matched_lithology_intervals.append(sequence_lithology_intervals)

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
    def _calc_matching_r2(well_log, core_log):
        well_log = well_log.dropna()
        interpolator = interp1d(well_log.index, well_log, kind="linear", fill_value="extrapolate")
        well_log = interpolator(core_log.index)
        return np.corrcoef(core_log, well_log)[0, 1]**2

    def plot_matching(self, mode=None, scale=False, subplot_height=750, subplot_width=200):
        init_notebook_mode(connected=True)

        boring_sequences = self.boring_sequences.reset_index()
        if mode is None and "MODE" not in boring_sequences.columns:
            raise ValueError("Core-to-log matching has to be performed beforehand if mode is not specified")
        if mode is not None:
            mode_list = self._unify_matching_mode(mode)
            if len(mode_list) == 1:
                mode_list = mode_list * len(boring_sequences)
            if len(mode_list) != len(boring_sequences):
                raise ValueError("Mode length must match the number of matching intervals")
            boring_sequences["MODE"] = mode_list
            r2_list = []
            for _, (depth_from, depth_to, mode) in boring_sequences[["DEPTH_FROM", "DEPTH_TO", "MODE"]].iterrows():
                log_mnemonic, core_mnemonic, core_attr = self._parse_matching_mode(mode)
                well_log = self.logs[log_mnemonic].dropna()
                core_log_segment = getattr(self, core_attr)[core_mnemonic].dropna()[depth_from:depth_to]
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

        for i, (depth_from, depth_to, mode) in enumerate(zip(depth_from_list, depth_to_list, mode_list), 1):
            log_mnemonic, core_mnemonic, core_attr = self._parse_matching_mode(mode)
            well_log_segment = self.logs[log_mnemonic].dropna()[depth_from - 3 : depth_to + 3]
            core_log_segment = getattr(self, core_attr)[core_mnemonic].dropna()[depth_from:depth_to]

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
        fig_layout = go.Layout(title="Скважина {}".format(self.name), legend=dict(orientation="h"),
                               width=n_cols*subplot_width, height=subplot_height)
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

    def keep_matched_intervals(self, mode=None, threshold=0.6):
        mask = self.boring_sequences["R2"] > threshold
        if mode is not None:
            mode_list = self._unify_matching_mode(mode)
            mask &= self.boring_sequences["MODE"].isin(mode_list)
        intervals = self.boring_sequences[mask].reset_index()[["DEPTH_FROM", "DEPTH_TO"]]
        res_segments = []
        for _, (depth_from, depth_to) in intervals.iterrows():
            res_segments.append(self[depth_from:depth_to])
        return res_segments

    def create_segments(self, src, connected=True):
        if not isinstance(src, list):
            src = [src]
        if all([item in self.attrs_fdtd_index for item in src]):
            res = self._create_segments_by_fdtd(src, connected)
        else:
            # TODO: create_segments from depth_index
            pass
        return res

    def _create_segments_by_fdtd(self, src, connected):
        tables = [getattr(self, item).reset_index() for item in src]
        df = tables[0] if len(tables) == 1 else reduce(fdtd_join, tables)
        if connected:
            df = self._core_chunks(df)
        segments = [self[top:bottom] for _, (top, bottom) in df[['DEPTH_FROM', 'DEPTH_TO']].iterrows()]
        return segments

    def _core_chunks(self, df):
        if len(df) > 0:
            chunks = [(item.DEPTH_FROM.min(), item.DEPTH_TO.max()) for item in select_contigious_intervals(df)]
            chunks = pd.DataFrame(chunks, columns=["DEPTH_FROM", "DEPTH_TO"])
            return chunks
        else:
            return pd.DataFrame(columns=["DEPTH_FROM", "DEPTH_TO"])

    def random_crop(self, height, n_crops=1):
        positions = np.random.uniform(self.depth_from, max(self.depth_from, self.depth_to-height), size=n_crops)
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
            # TODO: create_mask from depth_index
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

    def drop_nans(self, components_to_drop_nans):
        # Drop rows with at least one NaN in places components_to_drop_nans
        if isinstance(components_to_drop_nans, (list, tuple)):
            not_nan_mask = ~np.isnan(self[components_to_drop_nans].logs)
            not_nan_indices = np.where(np.all(not_nan_mask, axis=1))[0]

        # Drop rows with greater than or equal to components_to_drop_nans NaN in row
        elif isinstance(components_to_drop_nans, int):
            not_nan_mask = (~np.isnan(self.logs)).sum(axis=1)
            not_nan_indices = np.where(not_nan_mask >= components_to_drop_nans)[0]
        else:
            raise ValueError('components_to_drop_nans must be list or tuple or int')

        not_nan_depth = not_nan_mask.index[not_nan_indices]
        borders = np.where((not_nan_indices[1:] - not_nan_indices[:-1]) != 1)[0] + 1
        splits = np.split(not_nan_depth, borders)
        return [self[split[0]:split[-1]] for split in splits]

    def fill_nans(self):
        pass

    def norm_mean_std(self, axis=-1, mean=None, std=None, eps=1e-10, *, components):
        pass

    def norm_min_max(self, axis=-1, min=None, max=None, *, components):
        pass

    def drop_short_segments(self):
        pass
