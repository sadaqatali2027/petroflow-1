""" Auxiliary functions for core images. """

import os
import glob

import cv2
import PIL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Assemble: # pylint: disable=too-few-public-methods
    """Namespace for pipeline to assemble predictions."""
    @classmethod
    def assemble(cls, predictions, images, mode='mean'):
        """Transform crops predictions into whole images predictions."""
        _mode = mode if isinstance(mode, list) else [mode]
        res = [[] for _ in range(len(_mode))]
        i = 0
        for item in images:
            for j, m in enumerate(_mode):
                res[j].append(getattr(np, m)(predictions[i:i+item.shape[0]], axis=0))
            i = i + item.shape[0]
        res = [np.array(item) for item in res]
        if len(res) == 1:
            res = res[0]
        return res

def plot_pair(well_path, filename, length=1000, figsize=(5, 15)):
    """Plot DL and UV images."""
    plt.figure(figsize=figsize)
    for pos, name in enumerate(['dl', 'uv']):
        image = PIL.Image.open(os.path.join(well_path, 'samples_'+name, filename))
        if name == 'uv':
            image = image.convert('L')
            image = np.array(image)[:length]
            image = cv2.equalizeHist(image)  # pylint: disable=no-member
        else:
            image = np.array(image)[:length]
        plt.subplot(1, 2, pos+1)
        plt.title(name)
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image, cmap='gray')
    plt.show()

def read_annotation(well_path, df_name='samples.feather'):
    """Read annotation for all wells in path/glob."""
    path = os.path.join(well_path, '*')
    annotation = []
    for filename in glob.glob(os.path.join(path, df_name)):
        _df = pd.read_feather(filename)
        _df['SAMPLE'] = os.path.split(os.path.split(filename)[0])[1] + '_' + _df['SAMPLE']
        annotation.append(_df)
    annotation = pd.concat(annotation)
    annotation['QC'] = 1 - annotation['QC']
    annotation = annotation.set_index('SAMPLE')
    return annotation

def plot_crops_predictions(batch, dl_attr='dl', uv_attr='uv'):
    """Plot crops and corresponding predictions."""
    for i in np.random.choice(getattr(batch, dl_attr).shape[0], 5):
        img1 = np.squeeze(getattr(batch, dl_attr)[i])
        img2 = np.squeeze(getattr(batch, uv_attr)[i])
        shape = np.minimum(img1.shape, img2.shape) # pylint: disable=assignment-from-no-return
        img1 = img1[:shape[0], :shape[1]]
        img2 = img2[:shape[0], :shape[1]]
        image = np.concatenate((img1, img2), axis=1)

        print('Label: ', str(batch.labels[i]) + '   Prediction:' + str(batch.proba[i][1]))
        plt.imshow(image, cmap='gray')
        plt.show()

def _split(arr):
    if len(arr.shape) != 1:
        arr = np.split(arr, len(arr)) + [None]
        arr = np.array(arr)[:-1]
    return arr

def plot_images_predictions(ppl, mode='fn', threshold=0.5, n_images=None,
                            load_labels=True, sort=False, proba_index=0):
    """ Plot examples of predictions.

    Parameters
    ----------
    ppl : Pipeline
        Pipeline with `stat` variable.
    mode : str
        Predictions to plot:
            'p': positive,
            'n': negative,
            'tp': true positive,
            'tn': true negative,
            'fp': false positive,
            'fn': false negative.
    threshold : float
        Prediction threshold.
    n_images : int
        Number of predictions to plot.
    load_labels : bool
        If True, labels will be loaded from 'stat' variable of pipeline.
    sort : bool
        Sort or not plotted predictions by predicted proba.
    proba_index : int
        Mode of the probability aggregation:
            0 : 'median',
            1 : 'mean',
            2 : 'min',
            3 : 'max'.
    """
    stat = ppl.get_variable('stat')
    proba = np.concatenate([item[2][proba_index] for item in stat])
    if sort:
        order = list(enumerate(proba[:, 1]))
        order = np.array(sorted(order, key=lambda x: x[1], reverse=True))[:, 0].astype('int')
        proba = proba[order]
    else:
        order = np.arange(len(proba))

    dl_images = np.concatenate([_split(item[0]) for item in stat])[order]
    uv_images = np.concatenate([_split(item[1]) for item in stat])[order]
    index = ppl.dataset.indices[order]

    if load_labels:
        labels = np.concatenate([item[3] for item in stat])[order]
        indices = dict(
            fn=[i for i in range(len(labels)) if (labels[i] == 1) & (proba[i][1] < threshold)],
            fp=[i for i in range(len(labels)) if (labels[i] == 0) & (proba[i][1] > threshold)],
            tn=[i for i in range(len(labels)) if (labels[i] == 0) & (proba[i][1] < threshold)],
            tp=[i for i in range(len(labels)) if (labels[i] == 1) & (proba[i][1] > threshold)],
            all=np.arange(len(labels))
        )
    else:
        indices = dict()

    indices.update(**dict(
        p=[i for i in range(len(proba)) if (proba[i][1] > threshold)],
        n=[i for i in range(len(proba)) if (proba[i][1] <= threshold)],
        all=np.arange(len(proba))
    ))

    n_images = len(indices[mode]) if n_images is None else n_images
    if not isinstance(mode, list):
        mode = [mode]
    _indices = [item for m in mode for item in indices[m]]

    for i in _indices[:n_images]:
        img1 = np.squeeze(dl_images[i])
        img2 = np.squeeze(uv_images[i])
        shape = np.max((img1.shape[0], img2.shape[0])), np.min((img1.shape[1], img2.shape[1]))
        figsize = np.array([10, 3]) * shape / np.array([1500, 300])
        plt.figure(figsize=figsize)
        plt.subplot(2, 1, 1)
        filename = '_'.join(index[i].split('_')[1:])
        if load_labels:
            title = (("{}" + " " * 8) * 2 + "{:.02f}").format(filename, labels[i], proba[i][1])
        else:
            title = (("{}" + " " * 8) * 2).format(filename, proba[i][1])
        plt.title(title)

        plt.imshow(img1.transpose((1, 0, 2)))
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2, 1, 2)
        plt.imshow(img2.transpose(), cmap='gray')
        plt.xticks([])
        plt.yticks([])

        plt.show()

def fix_annotation(ppl, annotation, threshold=0.5):
    """Fix annotation for images where model provides wrong predictions."""
    new_annotation = annotation.copy()
    stat = ppl.get_variable('stat')
    dl_images = np.concatenate([_split(item[0]) for item in stat])
    uv_images = np.concatenate([_split(item[1]) for item in stat])
    proba = np.concatenate([item[2] for item in stat])
    labels = np.concatenate([item[3] for item in stat])

    indices = dict(
        fn=[i for i in range(len(labels)) if (labels[i] == 1) & (proba[i][1] < threshold)],
        fp=[i for i in range(len(labels)) if (labels[i] == 0) & (proba[i][1] > threshold)],
        tn=[i for i in range(len(labels)) if (labels[i] == 0) & (proba[i][1] < threshold)],
        tp=[i for i in range(len(labels)) if (labels[i] == 1) & (proba[i][1] > threshold)]
    )

    index = ppl.dataset.indices

    for i in indices['fn'] + indices['fp']:
        img1 = np.squeeze(dl_images[i])
        img2 = np.squeeze(uv_images[i])
        img1[img1 > 1] = 1
        img2[img2 > 1] = 1
        shape = np.min((img1.shape[0], img2.shape[0])), np.min((img1.shape[1], img2.shape[1]))
        plt.figure(figsize=(15, 10))
        image = np.concatenate((img1[:shape[0], :shape[1]], img2[:shape[0], :shape[1]]), axis=1)
        plt.imshow(image.transpose(), cmap='gray')
        plt.title(index[i] + '      ' + str(labels[i]) + '     ' + str(proba[i][1]))
        plt.xticks(np.arange(0, image.shape[0], 100))
        plt.grid(True, color='red', markevery='100')
        plt.show()

        flag = True
        while flag:
            try:
                new_annotation['QC'][index[i]] = int(input())
            except ValueError:
                print('Please enter the number')
            else:
                flag = False

    return new_annotation
