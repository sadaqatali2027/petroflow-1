""" Auxiliary functions for core images. """

import os

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
