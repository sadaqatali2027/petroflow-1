""" Auxiliary functions for core images. """

import os
import glob

import cv2
import PIL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Assemble:
    """ Namespace for pipeline to assmble predictions"""
    @classmethod
    def assemble(cls, predictions, images, mode='mean'):
        """ Transform crops predictions into whole images predictions. """
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

def plot_pair(path, name, length=1000):
    """ Plot DL and UV images. """
    dl_image = PIL.Image.open(os.path.join(path, 'samples_dl', name))
    dl_image = np.array(dl_image)[:length]

    uv_image = PIL.Image.open(os.path.join(path, 'samples_uv', name)).convert('L')
    uv_image = np.array(uv_image)[:length]

    plt.figure(figsize=(5, 15))
    plt.subplot(1, 2, 1)
    plt.title('dl')
    plt.imshow(dl_image)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.title('uv')
    plt.imshow(cv2.equalizeHist(uv_image), cmap='gray')  # pylint: disable=no-member
    plt.xticks([])
    plt.yticks([])
    plt.show()

def read_annotation(path, df_name='samples.feather'):
    """ Read annotation for all wells in path/glob. """
    path = os.path.join(path, '*')
    annotation = []
    for filename in glob.glob(os.path.join(path, df_name)):
        _df = pd.read_feather(filename)
        _df['SAMPLE'] = os.path.split(os.path.split(filename)[0])[1] + '_' + _df['SAMPLE']
        annotation.append(_df)
    annotation = pd.concat(annotation)
    annotation['QC'] = 1 - annotation['QC']
    annotation = annotation.set_index('SAMPLE')
    return annotation

def plot_crops_predictions(batch):
    """ Plot crops and corresponding predictions. """
    for i in np.random.choice(batch.dl.shape[0], 5):
        img1 = np.squeeze(batch.dl[i])
        img2 = np.squeeze(batch.uv[i])
        print('Label: ', str(batch.labels[i]) + '   Prediction:' + str(batch.proba[i][1]))
        shape = np.min((img1.shape[0], img2.shape[0])), np.min((img1.shape[1], img2.shape[1]))
        plt.imshow(np.concatenate((img1[:shape[0], :shape[1]], img2[:shape[0], :shape[1]]), axis=1), cmap='gray')
        plt.show()

def _split(arr):
    if len(arr.shape) != 1:
        arr = np.split(arr, len(arr)) + [None]
        arr = np.array(arr)[:-1]
    return arr


def _split(arr):
    if len(arr.shape) != 1:
        arr = np.split(arr, len(arr)) + [None]
        arr = np.array(arr)[:-1]
    return arr

def _split(arr):
    if len(arr.shape) != 1:
        arr = np.split(arr, len(arr)) + [None]
        arr = np.array(arr)[:-1]
    return arr

def plot_images_predictions(ppl, mode='fn', threshold=0.5, n_images=None,
                            load_labels=True, sort=False, proba_index=0):
    """ Plot examples of predictions. """
    stat = ppl.get_variable('stat')
    proba = np.concatenate([item[3][proba_index] for item in stat])
    if sort:
        order = list(enumerate(proba[:, 1]))
        order = np.array(sorted(order, key=lambda x: x[1], reverse=True))[:, 0].astype('int')
        proba = proba[order]
    else:
        order = np.arange(len(proba))
    
    dl_images = np.concatenate([_split(item[0]) for item in stat])[order]
    uv_images = np.concatenate([_split(item[1]) for item in stat])[order]
    uv_images_bin = np.concatenate([_split(item[2]) for item in stat])[order]
    index = ppl.dataset.indices[order]
    
    if load_labels:
        labels = np.concatenate([item[4] for item in stat])[order]
        indices = dict(
            fn=[i for i in range(len(labels)) if (labels[i] == 1) & (proba[i][1] < threshold)],
            fp=[i for i in range(len(labels)) if (labels[i] == 0) & (proba[i][1] > threshold)],
            tn=[i for i in range(len(labels)) if (labels[i] == 0) & (proba[i][1] < threshold)],
            tp=[i for i in range(len(labels)) if (labels[i] == 1) & (proba[i][1] > threshold)],
            all=np.arange(len(labels))
        )
    else:
        indices = dict(
            p=[i for i in range(len(proba)) if (proba[i][1] > threshold)],
            n=[i for i in range(len(proba)) if (proba[i][1] <= threshold)],
            all=np.arange(len(proba))
        )

    n_images = len(indices[mode]) if n_images is None else n_images
    if not isinstance(mode, list):
        mode = [mode]
    _indices = [item for m in mode for item in indices[m]]

    for i in _indices[:n_images]:
        img1 = np.squeeze(dl_images[i])
        img2 = np.squeeze(uv_images[i])
        img3 = np.squeeze(uv_images_bin[i])
        shape = np.min((img1.shape[0], img2.shape[0])), np.min((img1.shape[1], img2.shape[1]))
        image = np.concatenate(
            (
                img1[:shape[0], :shape[1]],
                img2[:shape[0], :shape[1]],
                img3[:shape[0], :shape[1]]
            ), axis=1)
        figsize = np.array([10, 7]) * image.shape / np.array([1500, 300])
        plt.figure(figsize=figsize)
        plt.imshow(image.transpose(), cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if load_labels:
            plt.title(index[i] + '      ' + str(labels[i]) + '     ' + str(proba[i][1]))
        else:
            plt.title(index[i] + '      ' + str(proba[i][1]))
        plt.show()

def fix_annotation(ppl, annotation, threshold=0.5):
    """ Fix annotation for images where model provides wrong predictions. """
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
