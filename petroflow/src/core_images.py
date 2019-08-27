"""Batch class for core images processing"""

import os
from itertools import product

import numpy as np
import PIL
import cv2

from petroflow.batchflow import FilesIndex, ImagesBatch, action, inbatch_parallel

class CoreIndex(FilesIndex):
    """FilesIndex with a well name added to its indices as a prefix."""
    def __init__(self, index=None, path=None, *args, **kwargs):
        """Create index.

        Parameters
        ----------
        index : int, 1-d array-like or callable
            defines structure of FilesIndex
        path : str
            path to folder with wells.
        *args, **kwargs
            parameters of FilesIndex.
        """
        path = path if path is None else os.path.join(path, '*/samples_dl/*.*')
        super().__init__(index, path, *args, **kwargs)

    @staticmethod
    def build_key(fullpathname, no_ext=False):
        """Create index item from full path name. Well name will be added
        to index as prefix. """
        folder_name = fullpathname
        splitted_path = []
        for _ in range(3):
            folder_name, _key_name = os.path.split(folder_name)
            splitted_path.append(_key_name)
        key_name = splitted_path[2] + '_' + splitted_path[0]
        if no_ext:
            dot_position = key_name.rfind('.')
            dot_position = dot_position if dot_position > 0 else len(key_name)
            key_name = key_name[:dot_position]
        return key_name, fullpathname

class CoreBatch(ImagesBatch):
    """Batch class for core images processing. Contains core images in daylight (DL)
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

    components = 'dl', 'uv', 'labels'

    @staticmethod
    def _mirror_padding(image, shape):
        new_shape = (np.array(shape) - image.size) * (np.array(shape) - image.size > 0)
        padding_shape = ((new_shape[1], new_shape[1]), (new_shape[0], new_shape[0]), (0, 0))
        image = np.array(image)
        if image.ndim == 2:
            padding_shape = padding_shape[:-1]
        return PIL.Image.fromarray(np.pad(image, padding_shape, mode='reflect'))

    @staticmethod
    def _get_uv_path(path_dl):
        _path = os.path.split(path_dl)
        filename = _path[1]
        dirname_uv = _path[0][:-2] + 'uv'
        path_uv = os.path.join(dirname_uv, filename)
        return path_uv

    def _get_components(self, index, components=None):
        if components is None:
            components = self.components
        elif isinstance(components, str):
            components = [components]
        pos = self.get_pos(None, components[0], index)
        res = [getattr(self, component)[pos] for component in components]
        if len(res) == 1:
            res = res[0]
        return res

    def _assemble_images(self, all_results, *args, dst=None, **kwargs):
        dst = self.components[:2] if dst is None else dst
        return self._assemble(all_results, *args, dst=dst, **kwargs)

    def _assemble_uv(self, all_results, *args, dst=None, **kwargs):
        dst = self.components[1] if dst is None else dst
        return self._assemble(all_results, *args, dst=dst, **kwargs)

    def _assemble_labels(self, all_results, *args, dst=None, **kwargs):
        dst = self.components[2] if dst is None else dst
        return self._assemble(all_results, *args, dst=dst, **kwargs)

    def _assemble_uv_labels(self, all_results, *args, dst=None, **kwargs):
        dst = self.components[1:] if dst is None else dst
        return self._assemble(all_results, *args, dst=dst, **kwargs)

    @action
    @inbatch_parallel(init='indices', post='_assemble_images')
    def load(self, index, grayscale=False, **kwargs):
        """Load data.

        Parameters
        ----------
        grayscale : bool
            if True, convert images to gray scale.
        dst : tuple
            components to save resulting images. Default: ('dl', 'uv').
        """
        full_path_dl = self._get_file_name(index, src=None)
        full_path_uv = self._get_uv_path(full_path_dl)
        res = (PIL.Image.open(full_path_dl), PIL.Image.open(full_path_uv))
        if grayscale:
            res = [item.convert('L') for item in res]
        return res[0], res[1]

    @action
    @inbatch_parallel(init='indices', post='_assemble_images')
    def to_grayscale(self, index, src=None, **kwargs):
        """Load data.

        Parameters
        ----------
        src : tuple of str
            components to process. Default: ('dl', 'uv').
        dst : tuple of str
            components to save resulting images. Default: ('dl', 'uv').
        """
        src = self.components[:2] if src is None else src
        return [img.convert('L') for img in self._get_components(index, src)]

    @action
    @inbatch_parallel(init='indices', post='_assemble_labels')
    def create_labels(self, index, labels=None, **kwargs):
        """Create labels from pd.DataSeries/dict

        Parameters
        ----------
        labels : pd.DataSeries/dict
            index/keys should correspond to index of the Dataset
        dst : str
            components to save resulting images. Default: 'labels'.
        """
        _ = self, kwargs
        label = 0 if labels is None else labels[index]
        return label

    @action
    @inbatch_parallel(init='indices', post='_assemble_images')
    def mirror_padding(self, index, shape, src=None, **kwargs):
        """Add padding to images, whose size is less than `shape`.

        Parameters
        ----------
        shape : tuple
            if image shape is less than size, image will be padded by reflections.
        src : tuple of str
            components to process. Default: ('dl', 'uv').
        dst : tuple of str
            components to save resulting images. Default: ('dl', 'uv').
        """
        _ = kwargs
        src = self.components[:2] if src is None else src
        return [self._mirror_padding(img, shape) for img in self._get_components(index, src)]

    @action
    @inbatch_parallel(init='indices', post='_assemble_images')
    def fix_shape(self, index, src=None, **kwargs):
        """Transform shapes of DL and UV to the same values. Image with larger
        shape will be croppped.

        Parameters
        ----------
        src : tuple of str
            components to process. Default: ('dl', 'uv').
        dst : tuple of str
            components to save resulting images. Default: ('dl', 'uv').
        """
        _ = kwargs
        src = self.components[:2] if src is None else src
        images = self._get_components(index, src)
        shape = (min(*[img.size[0] for img in images]), min(*[img.size[1] for img in images]))
        images = [img.crop((0, 0, shape[0], shape[1])) for img in images]
        return images

    @action
    @inbatch_parallel(init='indices', post='_assemble_uv_labels')
    def flip_uv(self, index, proba=0.5, src=None, **kwargs):
        """Randomly flip UV images. Flipped images always will have label 1.

        Parameters
        ----------
        proba : float
            probability of flip.
        src : tuple of str
            components to process. Default: ('uv', 'labels').
        dst : tuple of str
            components to save resulting images and labels. Default: ('uv', 'labels').
        """
        _ = kwargs
        src = self.components[1:] if src is None else src
        img, label = self._get_components(index, src)
        if np.random.rand() < proba:
            img = PIL.ImageOps.flip(img)
            label = 1.
        return img, label

    @action
    @inbatch_parallel(init='indices', post='_assemble_uv_labels')
    def shift_uv(self, index, proba=0.5, bounds=(10, 100), src=None, **kwargs):
        """Randomly shift UV images. Flipped images always will have label 1.

        Parameters
        ----------
        proba : float
            probability of shift.
        bounds : int
            maximal absolute value of shift.
        src : tuple of str
            components to process. Default: ('dl', uv', 'labels').
        dst : tuple of str
            components to save resulting images and labels. Default: ('dl', uv', 'labels').
        """
        _ = kwargs
        src = self.components[1:] if src is None else src
        img, label = self._get_components(index, src)
        if np.random.rand() < proba:
            lower = bounds[0]
            upper = min(bounds[1], img.size[1])
            if lower < upper:
                shift = np.random.randint(lower, upper)
                img = img.crop((0, shift, img.size[0], img.size[1]))
                label = 1.
        return img, label

    @action
    def shuffle_images(self, proba=0.5, src=None, dst=None):
        """Shuffle DL and UV images. Shuffled images will have label 1.

        Parameters
        ----------
        proba : float
            probability that pair in the batch will be changed.
        src : tuple of str
            components to process. Default: ('dl', uv', 'labels').
        dst : tuple of str
            components to save resulting images and labels. Default: ('dl', 'uv', 'labels').
        """
        n_permutations = int(np.ceil(len(self.indices) * proba / 2))
        shuffled_indices = np.random.choice(self.indices, n_permutations, replace=False)
        src = self.components if src is None else src
        dst = self.components if dst is None else dst
        for i, component in enumerate(src):
            setattr(self, dst[i], getattr(self, component))
        for i, j in zip(self.indices[:n_permutations], shuffled_indices):
            if i != j:
                uv1 = self._get_components(i, src[1])
                uv2 = self._get_components(j, src[1])
                getattr(self, dst[1])[self.get_pos(None, src[0], i)] = uv2
                getattr(self, dst[1])[self.get_pos(None, src[0], j)] = uv1
                getattr(self, dst[2])[self.get_pos(None, src[0], i)] = 1
                getattr(self, dst[2])[self.get_pos(None, src[0], j)] = 1
        return self

    @action
    @inbatch_parallel(init='indices', post='_assemble_images')
    def normalize(self, index, src=None, **kwargs):
        """Normalize images histograms.

        Parameters
        ----------
        src : tuple of str
            components to process. Default: ('dl', uv').
        dst : tuple of str
            components to save resulting images. Default: ('dl', 'uv').
        """
        _ = kwargs
        res = []
        src = self.components[:2] if src is None else src
        for component in src:
            pos = self.get_pos(None, component, index)
            image = np.array(getattr(self, component)[pos])
            res.append(PIL.Image.fromarray(cv2.equalizeHist(image))) # pylint: disable=no-member
        return res

    @action
    @inbatch_parallel(init='indices', post='_assemble_uv')
    def binarize(self, index, threshold=127, src=None, **kwargs):
        """Binarize images.

        Parameters
        ----------
        threshold : int
            binarization threshold.
        src : tuple of str
            components to process. Default: 'uv'.
        dst : tuple of str
            components to save resulting images. Default: 'uv'.
        """
        _ = kwargs
        src = self.components[1] if src is None else src
        pos = self.get_pos(None, src, index)
        image = np.array(getattr(self, src)[pos])
        return PIL.Image.fromarray(((image > threshold) * 255).astype('uint8'))

    @action
    @inbatch_parallel(init='indices', post='_assemble_uv')
    def blur(self, index, kernel=10, src=None, **kwargs):
        """Blur the images.

        Parameters
        ----------
        kernel : int, tuple
            kernel size to blur image. If int, image will be blurred with kernel
            `(kernel, kernel)`.
        src : tuple of str
            components to process. Default: 'uv'.
        dst : tuple of str
            components to save resulting images. Default: 'uv'.
        """
        _ = kwargs
        src = self.components[1] if src is None else src
        kernel = (kernel, kernel) if isinstance(kernel, int) else kernel
        kernel = np.ones(kernel, np.float32) / (kernel[0] * kernel[1])
        pos = self.get_pos(None, src, index)
        image = np.array(getattr(self, src)[pos])
        return PIL.Image.fromarray(cv2.filter2D(image, -1, kernel)) # pylint: disable=no-member

    @action
    def make_random_crops(self, shape, n_crops=1, src=None, channels='first', **kwargs):
        """Get random crops from images.

        Parameters
        ----------
        shape : tuple
            shape of crop.
        n_crops : int
            number of crops from one image.
        src : tuple of str
            components to process. Default: ('dl', uv').
        dst : tuple of str
            components to save resulting images and labels. Default: ('dl', 'uv').
        channels : str, 'first' or 'last'
            channels axis.
        """
        def _positions(image_shape, shape):
            return np.array(
                [np.random.randint(0, image_shape[i] - shape[i] + 1, size=n_crops) for i in range(2)]
            ).transpose()
        return self.make_crops(shape, _positions, None, src, channels, **kwargs)

    @action
    @inbatch_parallel(init='indices', post='_assemble_images')
    def make_crops(self, index, shape, positions=None, step=None, src=None, channels='first', **kwargs):
        """Get crops from images.

        Parameters
        ----------
        shape : tuple
            shape of crop.
        positions : None or callable
            positions of crops. If None, image will be croped with `step`.
            If callable, get image shape and shape of the crop and return array of positions.
        step : None, int or tuple of int
            step for cropping. If None, will be equal to `shape`. If int, step for both axes.
        src : tuple of str
            components to process. Default: ('dl', uv').
        dst : tuple of str
            components to save resulting images and labels. Default: ('dl', 'uv').
        channels : str, 'first' or 'last'
            channels axis.
        """
        _ = kwargs
        src = self.components[:2] if src is None else src
        images = self._get_components(index, src)
        if channels == 'first':
            spatial_axis = (1, 2)
        else:
            spatial_axis = (0, 1)
        image_shape = (
            min([img.shape[spatial_axis[0]] for img in images]),
            min([img.shape[spatial_axis[1]] for img in images])
        )
        if callable(positions):
            pos = positions(image_shape, shape)
        else:
            if step is None:
                step = shape
            elif isinstance(step, int):
                step = (step, step)
            pos = np.array(
                list(product(*[np.arange(0, image_shape[i] - shape[i] + 1, step[i]) for i in range(2)]))
            )

        crops = [[] for _ in range(len(images))]
        for _pos in pos:
            _slice = [slice(None)] * images[0].ndim
            for i, axis in enumerate(spatial_axis):
                _slice[axis] = slice(_pos[i], _pos[i] + shape[i])
            for i, img in enumerate(images):
                crops[i].append(img[_slice])
        return np.array(crops)
