import os
import dill
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

from petroflow import WellDataset, WS
from petroflow.batchflow import B, V, L


def add_lithology_position(well, segment=0):
    segment = well.iter_level()[segment]
    core_lithology = segment.core_lithology
    image = segment.core_dl
    factor = image.shape[0] / segment.length
    positions = []
    for (depth_from, depth_to), formation in core_lithology.iterrows():
        positions.append(
            (max(0, depth_from - segment.depth_from) * factor,
             min(segment.length, depth_to - segment.depth_from) * factor)
        )
    return positions

