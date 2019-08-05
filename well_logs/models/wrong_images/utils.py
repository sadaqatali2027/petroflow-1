""" Auxiliary functions for core images. """

import os
import glob

import PIL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_pair(path, name, threshold=0.5, length=1000):
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
    uv_image = uv_image / np.max(uv_image)
    uv_image[uv_image > threshold] = threshold
    uv_image = uv_image / threshold
    plt.imshow(uv_image, cmap='gray')
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

def make_data(batch):
    """ Transform array of arrays into array. """
    crops1 = np.concatenate(batch.dl_crops)
    crops2 = np.concatenate(batch.uv_crops)
    if crops1.ndim == 3:
        images = np.stack((crops1, crops2), axis=1)
    else:
        images = np.concatenate((crops1, crops2), axis=1)
    labels = np.concatenate(batch.labels_crops)
    return np.array(images, dtype='float32'), np.array(labels, dtype='float32')

def assemble(batch):
    """ Transform crops predictions into whole images predictions. """
    predictions = batch.proba
    res = []
    i = 0
    for item in batch.labels_crops:
        res.append(np.median(predictions[i:i+item.shape[0]], axis=0))
        i = i + item.shape[0]
    return np.array(res)

def get_bounds(path, percentile=90):
    """ Get percentiles for values for each well. """
    well_stat = dict()
    if not isinstance(percentile, list):
        percentile = [percentile, percentile]
    for well_path in glob.glob(path):
        well_stat[well_path] = {'dl': [], 'uv': []}
        for image in glob.glob(os.path.join(well_path, 'samples_uv', '*')):
            img = np.array(PIL.Image.open(image))
            well_stat[well_path]['uv'].append(img.flatten())
        for image in glob.glob(os.path.join(well_path, 'samples_dl', '*')):
            img = np.array(PIL.Image.open(image))
            well_stat[well_path]['dl'].append(img.flatten())
        well_stat[well_path]['dl'] = np.concatenate(well_stat[well_path]['dl'])
        well_stat[well_path]['uv'] = np.concatenate(well_stat[well_path]['uv'])

    bounds = dict()
    for well in well_stat:
        bounds[well.split('/')[-1]] = {
            'dl': np.percentile(well_stat[well]['dl'], percentile[0]),
            'uv': np.percentile(well_stat[well]['uv'], percentile[1])
        }
    return bounds

def plot_crops_predictions(batch):
    """ Plot crops and corresponding predictions. """
    for i in np.random.choice(batch.dl.shape[0], 5):
        img1 = np.squeeze(batch.dl[i])
        img2 = np.squeeze(batch.uv[i])
        print(batch.indices[i])
        print('Label: ', str(batch.labels[i]) + '   Prediction:' + str(batch.proba[i][1]))
        shape = np.min((img1.shape[0], img2.shape[0])), np.min((img1.shape[1], img2.shape[1]))
        plt.imshow(np.concatenate((img1[:shape[0], :shape[1]], img2[:shape[0], :shape[1]]), axis=1), cmap='gray')
        plt.show()

def _split(arr):
    if len(arr.shape) != 1:
        arr = np.split(arr, len(arr)) + [None]
        arr = np.array(arr)[:-1]
    return arr

def plot_images_predictions(ppl, mode='fn', threshold=0.5, n_images=None):
    """ Plot examples of predictions. """
    stat = ppl.get_variable('stat')
    dl_images = np.concatenate([_split(item[0]) for item in stat])
    uv_images = np.concatenate([_split(item[1]) for item in stat])
    proba = np.concatenate([item[2] for item in stat])
    labels = np.concatenate([item[3] for item in stat])

    indices = dict(
        fn=[i for i in range(len(labels)) if (labels[i] == 1) & (proba[i][1] < threshold)],
        fp=[i for i in range(len(labels)) if (labels[i] == 0) & (proba[i][1] > threshold)],
        tn=[i for i in range(len(labels)) if (labels[i] == 0) & (proba[i][1] < threshold)],
        tp=[i for i in range(len(labels)) if (labels[i] == 1) & (proba[i][1] > threshold)],
        all=np.arange(len(labels))
    )

    index = ppl.dataset.indices

    n_images = len(indices[mode]) if n_images is None else n_images
    if not isinstance(mode, list):
        mode = [mode]
    _indices = [item for m in mode for item in indices[m]]

    for i in _indices[:n_images]:
        img1 = np.squeeze(dl_images[i])
        img2 = np.squeeze(uv_images[i])
        shape = np.min((img1.shape[0], img2.shape[0])), np.min((img1.shape[1], img2.shape[1]))
        plt.figure(figsize=(15, 10))
        image = np.concatenate((img1[:shape[0], :shape[1]], img2[:shape[0], :shape[1]]), axis=1)
        plt.imshow(image.transpose(), cmap='gray')
        plt.title(index[i] + '      ' + str(labels[i]) + '     ' + str(proba[i][1]))
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
