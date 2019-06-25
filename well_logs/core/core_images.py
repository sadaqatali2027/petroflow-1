import os

import numpy as np
import pandas as pd
import PIL
import matplotlib.pyplot as plt

from well_logs.batchflow import Dataset, Pipeline, B, V, Batch, ImagesBatch, action, inbatch_parallel, FilesIndex

class CoreBatch(ImagesBatch):
    def _get_components(self, index, components=('dl', 'uv', 'labels')):
        pos = self.get_pos(None, components[0], index)
        return [getattr(self, component)[pos] for component in components]
    
    @action
    @inbatch_parallel(init='indices', post='_assemble', dst=('dl', 'uv', 'labels'))
    def load(self, index, df=None, bw=False):
        full_path_dl = self._get_file_name(index, src=None)
        _path = os.path.split(full_path_dl)
        filename = _path[1]
        dirname_uv = _path[0][:-2] + 'uv'
        full_path_uv = os.path.join(dirname_uv, filename)
        label = 0 if df is None else df.loc[filename]['QC']
        res = (PIL.Image.open(full_path_dl), PIL.Image.open(full_path_uv))
        if bw:
            res = [item.convert('L') for item in res]
        return res[0], res[1], label
    
    @action
    @inbatch_parallel(init='indices', post='_assemble', dst=('dl', 'uv', 'labels'))
    def flip(self, index, p=0.5):
        img1, img2, label = self._get_components(index)
        if np.random.rand() < p:
            img2 = PIL.ImageOps.flip(img2)
            label = 1.
        return img1, img2, label
    
    @action
    def shuffle(self, p=0.5):
        n_permutations = int(np.ceil(len(self.indices) * p / 2))
        shuffled_indices = np.random.choice(self.indices, n_permutations, replace=False)
        for i, j in zip(self.indices[:n_permutations], shuffled_indices):
            if i != j:
                _, uv1, _ = self._get_components(i)
                _, uv2, _ = self._get_components(j)
                self.uv[self.get_pos(None, 'uv', i)] = uv2
                self.uv[self.get_pos(None, 'uv', j)] = uv1
                self.labels[self.get_pos(None, 'uv', i)] = 1
                self.labels[self.get_pos(None, 'uv', j)] = 1
        return self

    @action
    @inbatch_parallel(init='indices', post='_assemble', dst='short', target="threads")
    def short_cores(self, index, shape):
        img1, img2, label = self._get_components(index)
        _x = min(img1.size[1], img2.size[1])
        _y = min(img1.size[0], img2.size[0])
        return _x <= shape[0] or _y <= shape[1]
    
    @action
    @inbatch_parallel(init='indices', post='_assemble', dst=('dl', 'uv'), target="threads")
    def normalize(self, index):
        res = []
        src = ('dl', 'uv')
        for i in range(len(src)):
            pos = self.get_pos(None, src[i], index)
            image = np.array(getattr(self, src[i])[pos])
            _min = np.min(image)
            _max = np.max(image)
            if _max - _min != 0:
                image = (image - _min) / (_max - _min)
                res.append(PIL.Image.fromarray(image))
            else:
                res.append(getattr(self, src[i])[pos])
        return res
    
    @action
    @inbatch_parallel(init='indices', post='_assemble', dst=('dl', 'uv', 'labels'), target="threads")
    def random_crop(self, index, shape, proba=0.5):
        res = []
        img1, img2, label = self._get_components(index)
        _x = min(img1.shape[1], img2.shape[1])
        _y = min(img1.shape[2], img2.shape[2])
        x = np.random.randint(0, _x - shape[0])
        y = np.random.randint(0, _y - shape[1])
        _slice = [slice(None)] * img1.ndim
        _slice[-2] = slice(x, x + shape[0])
        _slice[-1] = slice(y, y + shape[1])
        img1 =  img1[_slice]
        p = np.random.random()
        if p < proba:
            shift = np.random.randint(100, 200) * np.random.choice([-1, 1])
            if (x + shift >= 0) and (x + shift + shape[0] <= _x):
                x += shift
                label = 1.
            _slice[-2] = slice(x, x + shape[0])
            _slice[-1] = slice(y, y + shape[1]) 
        img2 = img2[_slice]
        return img1, img2, label
    
    @action
    @inbatch_parallel(init='indices', post='_assemble', dst=('dl_crops', 'uv_crops', 'labels_crops'), target="threads")
    def crop(self, index, shape, step):
        img1, img2, label = self._get_components(index)
        _x = min(img1.shape[1], img2.shape[1])
        _y = min(img1.shape[2], img2.shape[2])
        positions = np.arange(0, _x-shape[0], step)
        _slice = [slice(None)] * img1.ndim

        crops1 = []
        crops2 = []
        
        _slice[-1] = slice(0, shape[1])
        for pos in positions:
            _slice[-2] = slice(pos, pos + shape[0])
            crops1.append(img1[_slice])
            crops2.append(img2[_slice])
        return np.concatenate(crops1), np.concatenate(crops2), np.array([label] * len(positions))

    @action
    @inbatch_parallel(init='indices', post='_assemble', dst='images')
    def concatenate(self, index):
        img1, img2, _ = self._get_components(index)
        if img1.ndim == 2:
            res = np.stack((img1, img2), axis=0)
        else:
            res = np.concatenate((img1, img2), axis=0)  
        return np.array(res, "float32")

    @action
    @inbatch_parallel(init='indices', post='_assemble', dst=('check', 's1', 's2'), target="threads")
    def check_shapes(self, index):
        img1, img2, label = self._get_components(index)
        return img1.size[0] != img2.size[0] or img1.size[1] != img2.size[1], img1.size, img2.size