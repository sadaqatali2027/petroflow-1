import os

import PIL
import numpy as np
import matplotlib.pyplot as plt

def plot_pair(path, name, length=1000):
    dl = PIL.Image.open(os.path.join(path, 'samples_dl', name))
    dl = np.array(dl)[:length]

    uv = PIL.Image.open(os.path.join(path, 'samples_uv', name))
    uv = np.array(uv)[:length]

    plt.figure(figsize=(5, 15))
    plt.subplot(1, 2, 1)
    plt.title('dl')
    plt.imshow(dl)
    plt.subplot(1, 2, 2)
    plt.title('uv')
    plt.imshow(uv / np.max(uv))
    plt.show()
    
def make_data(batch):
    crops1 = np.concatenate(batch.dl_crops)
    crops2 = np.concatenate(batch.uv_crops)
    if crops1.ndim == 3:
        images = np.stack((crops1, crops2), axis=1)
    else:
        images = np.concatenate((crops1, crops2), axis=1)
    labels = np.concatenate(batch.labels_crops)
    return np.array(images, dtype='float32'), np.array(labels, dtype='float32')

def show_images(path):
    ind = index.create_subset(np.array([path]))
    b = Dataset(ind, CoreBatch).p.load(df).normalize().next_batch(1)
    plt.figure(figsize=(15, 10))
    plt.imshow(np.concatenate((b.dl[0], b.uv[0]), axis=1).transpose(1, 0, 2))
    plt.show()
    
def assemble(batch):
    predictions = batch.proba
    res = []
    i = 0
    for item in batch.labels_crops:
        res.append(np.median(predictions[i:i+item.shape[0]], axis=0))
        i = i + item.shape[0]
    return np.array(res)

def assemble_stat(batch):
    predictions = batch.proba
    res = []
    i = 0
    for item in batch.labels_crops:
        res.append(predictions[i:i+item.shape[0], 1])
        i = i + item.shape[0]
    return np.array(res)
