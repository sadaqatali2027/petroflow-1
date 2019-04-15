import os
import base64
from copy import copy

import numpy as np
import pandas as pd
import lasio
from plotly import tools
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go

from .abstract_well import AbstractWell


class WellSegment(AbstractWell):
    def __init__(self, well_dir, depth_mnemonic="DEPTH", field=None):
        super().__init__()
        self.field = field
        self.name = os.path.basename(os.path.dirname(well_dir))
        self.well_dir = well_dir

        logs_path = os.path.join(well_dir, "logs.las")
        self.logs = self._load_logs(logs_path, depth_mnemonic)

        self.inclination = pd.read_csv(os.path.join(well_dir, "inclination.csv"))
        self.layers = pd.read_csv(os.path.join(well_dir, "layers.csv"))
        self.core = pd.read_csv(os.path.join(well_dir, "core.csv"))
        self.samples = pd.read_csv(os.path.join(well_dir, "samples.csv")).set_index("SAMPLE")

    @staticmethod
    def _load_logs(logs_path, depth_mnemonic):
        las = lasio.read(logs_path)
        logs = las.df().reset_index().set_index(depth_mnemonic)
        return logs

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
        if plot_core:
            n_cols += 1
            subplot_titles += ["CORE"]

        fig = tools.make_subplots(rows=1, cols=n_cols, subplot_titles=subplot_titles, shared_yaxes=True,
                                  print_grid=False)
        for i, mnemonic in enumerate(self.logs.columns, 1):
            trace = go.Scatter(x=self.logs[mnemonic], y=self.logs.index, mode="lines", name=mnemonic)
            fig.append_trace(trace, 1, i)

        if plot_core:
            trace = go.Scatter(x=[0, 1], y=[depth_from, depth_to], opacity=0, name="CORE")
            fig.append_trace(trace, 1, n_cols)

        images = []
        samples_path = os.path.join(self.well_dir, "samples/")
        for sample_path in os.listdir(samples_path):
            sample_name = int(os.path.splitext(sample_path)[0])
            img = self._encode(os.path.join(samples_path, sample_path))
            depth_from, depth_to = self.samples.loc[sample_name, ["DEPTH_FROM", "DEPTH_TO"]]
            img = go.layout.Image(source=img, xref="x"+str(n_cols), yref="y", x=0, y=depth_from,
                                  sizex=1, sizey=depth_to-depth_from, sizing="stretch", layer="below")
            images.append(img)

        layout = fig.layout
        fig_layout = go.Layout(title="Скважина {}".format(self.name), showlegend=False, width=n_cols*subplot_width,
                               height=subplot_height, yaxis=dict(range=[depth_to, depth_from]), images=images)
        layout.update(fig_layout)

        for key in layout:
            if key.startswith("xaxis"):
                layout[key]["fixedrange"] = True

        if plot_core:
            layout["xaxis" + str(n_cols)]["showticklabels"] = False
            layout["xaxis" + str(n_cols)]["showgrid"] = False

        for ann in layout["annotations"]:
            ann["font"]["size"] = 12

        iplot(fig)
        return self

    def __getitem__(self, key):
        if not isinstance(key, slice):
            return self.keep_logs(key)
        res = self.copy()
        res.logs = res.logs[key]
        # TODO: slice all other dataframes
        return res

    def copy(self):
        return copy(self)

    def drop_logs(self, mnemonics=None):
        res = self.copy()
        res.logs.drop(mnemonics, axis=1, inplace=True)
        return res

    def keep_logs(self, mnemonics=None):
        res = self.copy()
        mnemonics = np.asarray(mnemonics).tolist()
        res.logs = res.logs[mnemonics]
        return res

    def rename_logs(self, rename_dict):
        self.logs.columns = [rename_dict.get(name, name) for name in self.logs.columns]
        return self

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
