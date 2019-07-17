""" Batch class for bad core images detecting """

import os

import numpy as np
import PIL

from well_logs.batchflow import FilesIndex, ImagesBatch, action, inbatch_parallel

def _get_well_name(path):
    well = path
    for _ in range(2):
        well = os.path.split(well)[-2]
    return os.path.split(well)[-1]

def _mirror_padding(image, shape):
    new_shape = (np.array(shape) - image.size) * (np.array(shape) - image.size > 0)
    padding_shape = ((new_shape[1], new_shape[1]), (new_shape[0], new_shape[0]), (0, 0))
    image = np.array(image)
    if image.ndim == 2:
        padding_shape = padding_shape[:-1]
    return PIL.Image.fromarray(np.pad(image, padding_shape, mode='reflect'))

class CoreIndex(FilesIndex):
    """ FilesIndex that include well name to indices. """
    @staticmethod
    def build_key(fullpathname, no_ext=False):
        """ Create index item from full path name. """
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
    @inbatch_parallel(init='indices', post='_assemble', dst=('dl', 'uv'))
    def mirror_padding(self, index, shape, **kwargs):
        """ Randomly flip UV images. Flipped images always will have label 1.

        Parameters
        ----------
        shape : tuple
            if image shape is less than size, image will be padded by reflections
        dst : tuple of str
            attributes to save resulting DL and UV images.
        """
        _ = kwargs
        img1, img2, _ = self._get_components(index)

        return _mirror_padding(img1, shape), _mirror_padding(img2, shape)

    @action
    @inbatch_parallel(init='indices', post='_assemble', dst=('dl', 'uv'))
    def fix_shape(self, index, **kwargs):
        """ Transform shapes of DL and UV to the same values.

        Parameters
        ----------
        dst : tuple of str
            attributes to save resulting DL and UV images.
        """
        _ = kwargs
        img1, img2, _ = self._get_components(index)
        img1 = np.array(img1)
        img2 = np.array(img2)

        shape = (min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1]))

        return PIL.Image.fromarray(img1[:shape[0], :shape[1]]), PIL.Image.fromarray(img2[:shape[0], :shape[1]])

    @action
    @inbatch_parallel(init='indices', post='_assemble', dst=('dl', 'uv', 'labels'))
    def flip(self, index, proba=0.5, **kwargs):
        """ Randomly flip UV images. Flipped images always will have label 1.

        Parameters
        ----------
        proba : float
            probability of flip.
        dst : tuple of str
            attributes to save resulting DL, UV images and labels.
        """
        _ = kwargs
        img1, img2, label = self._get_components(index)
        if np.random.rand() < proba:
            img2 = PIL.ImageOps.flip(img2)
            label = 1.
        return img1, img2, label

    @action
    def shuffle(self, proba=0.5, dst=('dl', 'uv', 'labels')):
        """ Shuffle DL and UV images. Shuffled images will have label 1.

        Parameters
        ----------
        proba : float
            probability that pair in the batch will be changed.
        dst : tuple of str
            attributes to save resulting DL, UV images and labels.
        """
        n_permutations = int(np.ceil(len(self.indices) * proba / 2))
        shuffled_indices = np.random.choice(self.indices, n_permutations, replace=False)
        setattr(self, dst[0], self.dl)
        setattr(self, dst[1], self.uv)
        setattr(self, dst[2], self.labels)
        for i, j in zip(self.indices[:n_permutations], shuffled_indices):
            if i != j:
                _, uv1, _ = self._get_components(i)
                _, uv2, _ = self._get_components(j)
                getattr(self, dst[1])[self.get_pos(None, 'uv', i)] = uv2
                getattr(self, dst[1])[self.get_pos(None, 'uv', j)] = uv1
                getattr(self, dst[2])[self.get_pos(None, 'uv', i)] = 1
                getattr(self, dst[2])[self.get_pos(None, 'uv', j)] = 1
        return self

    @action
    @inbatch_parallel(init='indices', post='_assemble', target="threads")
    def find_short_cores(self, index, shape, dst):
        """ Find images which shape is less than some fixed value.

        Parameters
        ----------
        shape : tuple
            minimal shape that images must have.
        dst : str
            name of the attribute to save result of check
        """
        _ = dst
        img1, img2, _ = self._get_components(index)
        _x = min(img1.size[1], img2.size[1])
        _y = min(img1.size[0], img2.size[0])
        return _x <= shape[0] or _y <= shape[1]

    @action
    @inbatch_parallel(init='indices', post='_assemble', target="threads")
    def check_shapes(self, index, dst):
        """ Check that DL and UV images have the same shape.

        Parameters
        ----------
        dst : str
            attributes to save result of check.
        """

        _ = dst
        img1, img2, _ = self._get_components(index)
        return img1.size[0] != img2.size[0] or img1.size[1] != img2.size[1]

    @action
    @inbatch_parallel(init='indices', post='_assemble', target="threads", dst=('dl', 'uv'))
    def normalize(self, index, bounds, cut=False, **kwargs):
        """ Normalize images.

        Parameters
        ----------
        bounds : dict of dicts
            keys are well names, values are dicts of the form {'dl': dl_bound, 'uv': uv_bound}.
            Arrays with DL and UV images will divided by corresponding bounds.
        cut : bool
            if True, all values which are greater than 1 will be defined as 1.
        dst : tuple of str
            attributes to save DL, UV images.
        """
        _ = kwargs
        res = []
        src = ('dl', 'uv')
        well = _get_well_name(self._get_file_name(index, src=None))
        for component in src:
            pos = self.get_pos(None, component, index)
            image = np.array(getattr(self, component)[pos])
            if cut:
                image[image > bounds[well][component]] = bounds[well][component]
            image = image / bounds[well][component]
            res.append(PIL.Image.fromarray(image))
        return tuple(res)

    @action
    @inbatch_parallel(init='indices', post='_assemble', target="threads", dst=('dl', 'uv', 'labels'))
    def random_crop(self, index, shape, proba=0.5, **kwargs):
        """ Get random crops from images.

        Parameters
        ----------
        shape : tuple

        proba : float
            probability that cropping from DL and UV images will be performed
            from different positions.
        dst : tuple of str
            attributes to save DL, UV images and labels.
        """
        _ = kwargs
        img1, img2, label = self._get_components(index)
        _x = min(img1.shape[1], img2.shape[1])
        _y = min(img1.shape[2], img2.shape[2])
        x = np.random.randint(0, _x - shape[0] + 1)
        y = np.random.randint(0, _y - shape[1] + 1)
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
    @inbatch_parallel(init='indices', post='_assemble', target="threads", dst=('dl', 'uv', 'labels'))
    def crop(self, index, shape, step, **kwargs):
        """ Get crops from images.

        Parameters
        ----------
        shape : tuple

        step : float

        dst : tuple of str
            attributes to save DL, UV images and labels.
        """
        _ = kwargs
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
    def concatenate(self, index, **kwargs):
        """ Concatenate DL and UV images into array and save into images attribute.

        Parameters
        ----------
        dst : str
            attribute to save concatenated images.
        """
        _ = kwargs
        img1, img2, _ = self._get_components(index)
        if img1.ndim == 2:
            res = np.stack((img1, img2), axis=0)
        else:
            res = np.concatenate((img1, img2), axis=0)
        return np.array(res, "float32")
