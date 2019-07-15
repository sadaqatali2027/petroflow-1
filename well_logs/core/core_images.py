""" Batch class for bad core images detecting """

import os

import numpy as np
import PIL

from well_logs.batchflow import ImagesBatch, action, inbatch_parallel

class CoreBatch(ImagesBatch):
    """ Batch class for bad core images detecting. Contains core images in daylight (DL)
        and ultraviolet light (UV) and labels for that pairs: 1 if the pair has defects
        and 0 otherwise. Path to images must have the following form:
        '*/well_name/samples_{uv, dl}/*'.

        Parameters
        ----------
        index : DatasetIndex
            Unique identifiers of core images in the batch.
        Attributes
        ----------
        index : DatasetIndex
            Unique identifiers of core images in the batch.
        dl : 1-D ndarray
            Array of 3-D ndarrays with DL images.
        uv : 1-D ndarray
            Array of 3-D ndarrays with UV images.
        labels : 1-D ndarray
            Labels for images.
    """
    def _get_components(self, index, components=('dl', 'uv', 'labels')):
        pos = self.get_pos(None, components[0], index)
        return [getattr(self, component)[pos] for component in components]

    @action
    @inbatch_parallel(init='indices', post='_assemble', dst=('dl', 'uv', 'labels'))
    def load(self, index, df=None, grayscale=False):
        """ Load data.

        Parameters
        ----------
        df : pd.DataFrame or None
            pd.DataFrame must have index which corresponds to index of batch. Also dataframe includes
            'QC' column with labels for images where 0 corresponds to good pairs of DL and UV images
            and 1 to pairs with defects.
        grayscale : bool
            if True, convert images to gray scale.
        """
        full_path_dl = self._get_file_name(index, src=None)
        _path = os.path.split(full_path_dl)
        filename = _path[1]
        dirname_uv = _path[0][:-2] + 'uv'
        full_path_uv = os.path.join(dirname_uv, filename)
        label = 0 if df is None else df.loc[filename]['QC']
        res = (PIL.Image.open(full_path_dl), PIL.Image.open(full_path_uv))
        if grayscale:
            res = [item.convert('L') for item in res]
        return res[0], res[1], label

    @action
    @inbatch_parallel(init='indices', post='_assemble', dst=('dl', 'uv', 'labels'))
    def flip(self, index, proba=0.5):
        """ Randomly flip UV images. Flipped images always will have label 1.

        Parameters
        ----------
        p : float
            probability of flip.
        """
        img1, img2, label = self._get_components(index)
        if np.random.rand() < proba:
            img2 = PIL.ImageOps.flip(img2)
            label = 1.
        return img1, img2, label

    @action
    def shuffle(self, proba=0.5):
        """ Shuffle DL and UV images. Shuffled images will have label 1.

        Parameters
        ----------
        p : float
            probability that pair in the batch will be changed.
        """
        n_permutations = int(np.ceil(len(self.indices) * proba / 2))
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
        """ Find images which shape is less than some fixed value.

        Parameters
        ----------
        shape : tuple
            minimal shape that images must have.
        dst : str
            name of the aatribute to save result
        """
        img1, img2, _ = self._get_components(index)
        _x = min(img1.size[1], img2.size[1])
        _y = min(img1.size[0], img2.size[0])
        return _x <= shape[0] or _y <= shape[1]

    @action
    @inbatch_parallel(init='indices', post='_assemble', dst=('dl', 'uv'), target="threads")
    def normalize(self, index, bounds, cut=False):
        """ Normalize images.

            Parameters
            ----------
            bounds : dict of dicts
                keys are well names, values are dicts of the form {'dl': dl_bound, 'uv': uv_bound}.
                Arrays with DL and UV images will divided by corresponding bounds.
            cut : bool
                if True, all values which are greater than 1 will be defined as 1.
        """
        res = []
        src = ('dl', 'uv')
        well = index.split('_')[0]
        for component in src:
            pos = self.get_pos(None, component, index)
            image = np.array(getattr(self, component)[pos])
            if cut:
                image[image > bounds[well][component]] = bounds[well][component]
            image = image / bounds[well][component]
            res.append(PIL.Image.fromarray(image))
        return res

    @action
    @inbatch_parallel(init='indices', post='_assemble', dst=('dl', 'uv', 'labels'), target="threads")
    def random_crop(self, index, shape, proba=0.5):
        """ Get random crops from images.

            Parameters
            ----------
            shape : tuple

            proba : float
                probability that cropping from DL and UV images will be performed
                from different positions.

        """
        img1, img2, label = self._get_components(index)
        _x = min(img1.shape[1], img2.shape[1])
        _y = min(img1.shape[2], img2.shape[2])
        x = np.random.randint(0, _x - shape[0])
        y = np.random.randint(0, _y - shape[1])
        _slice = [slice(None)] * img1.ndim
        _slice[-2] = slice(x, x + shape[0])
        _slice[-1] = slice(y, y + shape[1])
        img1 = img1[_slice]
        if np.random.random() < proba:
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
        """ Get crops from images.

            Parameters
            ----------
            shape : tuple

            step : float
        """
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
        """ Concatenate DL and UV images into array and save into images attribute. """
        img1, img2, _ = self._get_components(index)
        if img1.ndim == 2:
            res = np.stack((img1, img2), axis=0)
        else:
            res = np.concatenate((img1, img2), axis=0)
        return np.array(res, "float32")

    @action
    @inbatch_parallel(init='indices', post='_assemble', dst=('check', 's1', 's2'), target="threads")
    def check_shapes(self, index):
        """ Check that DL and UV images have the same shape. """
        img1, img2, _ = self._get_components(index)
        return img1.size[0] != img2.size[0] or img1.size[1] != img2.size[1], img1.size, img2.size
