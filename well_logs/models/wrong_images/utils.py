""" Auxiliary functions for core images. """

import os
import glob

import PIL
import numpy as np
import matplotlib.pyplot as plt

def plot_pair(path, name, length=1000):
    """ Plot DL and UV images. """
    dl_image = PIL.Image.open(os.path.join(path, 'samples_dl', name))
    dl_image = np.array(dl_image)[:length]

    uv_image = PIL.Image.open(os.path.join(path, 'samples_uv', name))
    uv_image = np.array(uv_image)[:length]

    plt.figure(figsize=(5, 15))
    plt.subplot(1, 2, 1)
    plt.title('dl')
    plt.imshow(dl_image)
    plt.subplot(1, 2, 2)
    plt.title('uv')
    plt.imshow(uv_image / np.max(uv_image))
    plt.show()

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
            'dl': np.percentile(well_stat[well]['dl'], percentile),
            'uv': np.percentile(well_stat[well]['uv'], percentile)
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
        plt.imshow(np.concatenate((img1[:shape[0], :shape[1]], img2[:shape[0],:shape[1]]), axis=1), cmap='gray')
        plt.show()

def _split(arr):
    if len(arr.shape) != 1:
        arr = np.split(arr, len(arr)) + [None]
        arr = np.array(arr)[:-1]
    return arr

def plot_images_predictions(ppl, mode='fn', threshold=0.5, n_images=10):
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
        tp=[i for i in range(len(labels)) if (labels[i] == 1) & (proba[i][1] > threshold)]
    )

    index = ppl.dataset.indices

    for i in indices[mode][:n_images]:
        img1 = np.squeeze(dl_images[i])
        img2 = np.squeeze(uv_images[i])
        img1[img1 > 1] = 1
        img2[img2 > 1] = 1
        shape = np.min((img1.shape[0], img2.shape[0])), np.min((img1.shape[1], img2.shape[1]))
        plt.figure(figsize=(15,10))
        plt.imshow(np.concatenate((img1[:shape[0],:shape[1]], img2[:shape[0],:shape[1]]), axis=1).transpose(), cmap='gray')
        plt.title(index[i] + '      ' + str(labels[i]) + '     ' + str(proba[i][1]))
        plt.show()
