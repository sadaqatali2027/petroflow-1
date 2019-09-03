import os
import sys
import cv2
import pandas as pd
import numpy as np
import collections
from collections import Counter
import matplotlib.pyplot as plt
from petroflow import CoreBatch, CoreIndex
from petroflow.batchflow import Dataset, Pipeline, B, V

def plot_examples(batch, test=False, mapping=None, reverse_mapping=None, examples=50):
    mapping = dict() if mapping is None else mapping
    reverse_mapping = dict() if reverse_mapping is None else reverse_mapping
    images = np.stack([segment.core_dl / 255 for well in batch.wells for segment in well.iter_level()])
    labels = np.array([mapping.get(
        segment.core_lithology.FORMATION.values[0],
        segment.core_lithology.FORMATION.values[0]
    ) for well in batch.wells for segment in well.iter_level()])
    if test:
        predictions = batch.proba.argmax(axis=1)
    plt.figure(figsize=(15, int(40 * examples / 50)))
    for i, index in enumerate(np.random.choice(len(images), examples, replace=False)):
        plt.subplot(np.ceil(examples / 5).astype(int), 5, i+1)
        plt.imshow(images[index])
        title = 'label: ' + str(labels[index])
        title = title + '\nprediction: ' + str(
            reverse_mapping.get(predictions[index], predictions[index])
        ) if test else title
        color = 'k'
        if test:
            if labels[index] != reverse_mapping.get(predictions[index], predictions[index]):
                color = 'r'
        plt.title(title, color=color)
        plt.xticks([])
        plt.yticks([])
    plt.show()

class LithologyUtils:
    @staticmethod
    def process_lithology(lithology):
        def _process_item(item):
            _lithology = item.split()
            if _lithology[0] == 'Переслаивание':
                return item
            elif len(_lithology) > 1 and _lithology[1] == 'порода':
                return item
            else:
                return _lithology[0]
        return [df.apply(_process_item) for df in lithology]
    
    @staticmethod
    def rename_lithology(lithology, types, default='другое'):
        res = []
        for df in lithology:
            start = time.time()
            df[~df.isin(types)] = default
            res.append(df)
            print((time.time()-start))
        return res

    @staticmethod
    def get_labels_mapping(array, threshold=None, leave=None, counter=None):
        if threshold is not None:
            name_mapping = {item: item if counter[item] > threshold else 'другое' for item in counter}
        else:
            name_mapping = {item: item for item in array}
        if leave is not None:
            name_mapping = {k: v if k in leave else 'другое' for k, v in name_mapping.items()}
        label_mapping = {item: i for i, item in enumerate(np.unique(list(name_mapping.values())))}
        return {item: label_mapping[name_mapping[item]] for item in name_mapping}, name_mapping
    
    @staticmethod
    def get_loss_weights(counter, labels_mapping):
        weights = np.zeros(len(set(labels_mapping.values())))
        for key, value in counter.items():
            weights[labels_mapping[key]] += value
        weights = 1 / weights
        return weights / sum(weights)
    
    @staticmethod
    def drop_lithology(lithology, types):
        def _filter_item(item):
            return item in types
        print([len(df) for df in lithology])
        res = [df[df.FORMATION.apply(_filter_item)] for df in lithology]  
        default = pd.DataFrame({'DEPTH_FROM': [], 'DEPTH_TO': [], 'FORMATION': []}).set_index(['DEPTH_FROM', 'DEPTH_TO'])
        res = [df if len(df) > 0 else default for df in res]
        return res

def get_input_data(path, lithology, ratio=0.8):
    index = CoreIndex(path=path)
    annotation = pd.read_feather(os.path.join(path, 'annotation.feather'))
    annotation['SAMPLE'] = annotation['WELL'] + '_' + annotation['SAMPLE']
    annotation = annotation.set_index('SAMPLE')    
    annotation = annotation[annotation['LITHOLOGY'].isin(lithology)]

    np.random.seed(42)
    train_wells = np.random.choice(annotation.WELL.unique(),
                                   int(len(annotation.WELL.unique()) * ratio),
                                   replace=False)
    test_wells = np.setdiff1d(annotation.WELL.unique(), train_wells)

    index = index.create_subset(annotation.index.values)
    train_index = index.create_subset(annotation[annotation.WELL.isin(train_wells)].index.values)
    test_index = index.create_subset(annotation[annotation.WELL.isin(test_wells)].index.values)

    ds = Dataset(index, CoreBatch)
    ds_train = Dataset(train_index, CoreBatch)
    ds_test = Dataset(test_index, CoreBatch)
    
    return annotation, ds, ds_train, ds_test

def get_statistics(ds, annotation):
    counter_ppl = (
        Pipeline()
        .set_dataset(ds)
        .load(uv=False, dst=['dl'])
        .normalize(src='dl', dst='dl')
        .create_labels(annotation.LITHOLOGY.loc)
        .update(B('labels'), B('labels').tolist())
        .init_variable('lithology', default=[])
        .update(V('lithology', mode='e'), B('labels'))
    )

    (counter_ppl
     .after
     .add_namespace(collections)
     .init_variable('counter')
     .Counter(V('lithology'), save_to=V('counter'))
    )
    
    counter_ppl.run(16, n_epochs=1, drop_last=False)
    counter = counter_ppl.v('counter')

    labels_mapping = {i: k for k, i in enumerate(counter)}
    reverse_mapping = {v: k for k, v in labels_mapping.items()}
    weights = np.array([1 / counter[reverse_mapping[i]] for i in range(len(labels_mapping))])
    weights = weights / sum(weights)

    return labels_mapping, reverse_mapping, counter, weights

def encode(labels, mapping):
    return np.array([mapping[item] for item in labels])