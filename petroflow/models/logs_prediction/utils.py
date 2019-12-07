import os
import sys

import numpy as np

from collections import OrderedDict
from numpy.lib.stride_tricks import as_strided
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, "/notebooks/goryachev/petroflow")

from petroflow import WellDataset


def build_dataset(batch):
    preloaded = ({k: v for k, v in zip(batch.indices, batch.wells)},)
    ds = WellDataset(index=batch.index, preloaded=preloaded)
    return ds

def calc_metrics(true, pred, display=True):
    metrics = OrderedDict()
#     metrics['MAE'] = mean_absolute_error(true, pred)
    metrics['MSE'] = mean_squared_error(true, pred)
    metrics['R^2'] = r2_score(true, pred)
    
    if display:
        for name, value in metrics.items():
            print(name + ": {0:.3f}".format(value))

    return metrics

def batch_mse(true, pred):
    return np.mean(mean_squared_error(true.ravel(), pred.ravel()))

def moving_average_1d(x, n):
    padding = [None] * (n // 2)
    windows = as_strided(x, (n, len(x) - n + 1), (x.itemsize, x.itemsize))
    averaged = windows.mean(axis=0)
    return np.concatenate([padding, averaged, padding])

def save_results(model, loss, metrics, path='results/'):
    if not os.path.exists(path):
        os.makedirs(path)
    model.save(path + 'model')
    with open(path + 'loss', 'wb') as f:
        pickle.dump(loss, f)
    with open(path + 'metrics', 'wb') as f:
        pickle.dump(metrics, f)

def compare_results(metrics, path='results/'):
    with open(path + "metrics", 'rb') as f:
        saved_metrics = pickle.load(f)
    if np.allclose(list(saved_metrics.values()), list(metrics.values())):
        print('Reproduced model metrics match saved.')
    else:
        print('Reproduced model metrics DO NOT match saved:\n')
        print('Saved metrics:     ', ', '.join(["{0}: {1:0.3f}".format(k, v) for k, v in saved_metrics.items()]))
        print('Reproduced metrics:', ', '.join(["{0}: {1:0.3f}".format(k, v) for k, v in metrics.items()]))
