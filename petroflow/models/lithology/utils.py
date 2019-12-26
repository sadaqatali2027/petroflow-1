import os
import dill
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

from petroflow import WellDataset, WS
from petroflow.batchflow import B, V, L

def build_dataset(batch):
    preloaded = ({k: v for k, v in zip(batch.indices, batch.wells)},)
    ds = WellDataset(index=batch.index, preloaded=preloaded, copy=False)
    return ds

def add_lithology_position(well, segment=0):
    segment = well.iter_level()[segment]
    core_lithology = segment.core_lithology
    image = segment.core_dl
    factor = image.shape[0] / segment.length
    positions = []
    for (depth_from, depth_to), formation in core_lithology.iterrows():
        positions.append(
            (max(0, depth_from - segment.depth_from) * factor,
             min(segment.length, depth_to - segment.depth_from) * factor)
        )
    return positions

def plot_wells(wells):
    for well in wells:
        for i, segment in enumerate(well.iter_level()):
            if len(segment.core_lithology) > 1:
                plt.figure(figsize=(4, 20))
                img = segment.core_dl / 255
                mask = segment.mask
                lithology = segment.core_lithology
                plt.imshow(img)
                for a, b in add_lithology_position(well, segment=i):
                    plt.hlines(a, 0, img.shape[1], colors='r')
                    plt.hlines(b, 0, img.shape[1], colors='r')
                plt.show()
                break

def plot_examples(batch, reverse_mapping):
    images = np.transpose(batch.core, axes=(0, 2, 3, 1))
    predictions = np.tile(batch.proba.argmax(1), (1, 250, 1)).transpose(0, 2, 1)
    targets = np.tile(batch.masks, (1, 250, 1)).transpose(0, 2, 1)

    cmap = colors.ListedColormap(
        ['green'] * 5 + ['blue', 'grey', 'yellow', 'w'] + ['orange'] * 8 + ['black']
    )

    bounds = np.arange(-0.5, len(reverse_mapping) + 0.5, 1)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    a = np.ones((20, 200))
    b = np.concatenate([i * a for i in range(len(reverse_mapping))], axis=0)
    plt.figure(figsize=(15, 10))
    plt.imshow(b, norm=norm, cmap=cmap)

    for i, value in reverse_mapping.items():
        plt.text(20, 20 * i + 11, value, color='black', fontsize=12, bbox=dict(facecolor='white'))
    plt.show()

    for i in np.random.choice(len(images), 10, replace=False):
        plt.figure(figsize=(15, 15))
        plt.subplot(131)
        plt.imshow(images[i] / 255)
        plt.subplot(132)
        plt.imshow(predictions[i], vmin=0, vmax=len(reverse_mapping), norm=norm, cmap=cmap)
        plt.subplot(133)
        plt.imshow(targets[i], vmin=0, vmax=len(reverse_mapping), norm=norm, cmap=cmap)
        plt.show()

def filter_dataset(ds):
    filter_ppl = (ds.p
                  .init_variable('wells', default=[])
                  .has_attr('core_lithology')
                  .update(V('wells', mode='e'), B().indices)
                  .run(10, n_epochs=1, drop_last=False, shuffle=False))

    filtered_index = ds.index.create_subset(filter_ppl.v('wells'))
    return WellDataset(index=filtered_index)

def concat(df):
    return df.FORMATION + ' ' + df.GRAIN
    
def get_classes(ds):
    classes_ppl = (ds.p
           .init_variable('classes', default=[])
           .update(V('classes', mode='a'), (
               WS('core_lithology')[['FORMATION', 'GRAIN']].apply(concat, axis=1).values.ravel()))
    )

    (classes_ppl.after
        .add_namespace(np)
        .concatenate(L(sum)(V('classes'), []), save_to=V('classes', mode='w'))
        .unique(V('classes'), save_to=V('classes'))
    )

    classes_ppl.run(32, n_epochs=1, drop_last=False)
    return classes_ppl.v('classes')

def dump_results(train_ppl, path):
    if not os.path.exists(path):
        os.makedirs(path)

    train_ppl.get_model_by_name('model').save(os.path.join(path, 'unet.torch'))

    with open(os.path.join(path, 'loss.pkl'), 'wb') as f:
        dill.dump(train_ppl.v('loss_history'), f)

def dump_metrics(test_ppl, path):
    metrics = test_ppl.v('metrics')
    with open(path, 'wb') as f:
        dill.dump(test_ppl.v('metrics'), f)
        
def get_last_model_path(path):
    return sorted(glob.glob(path))[-1]

from collections import OrderedDict

def get_classes_distribution(*datasets, columns=None):
    if columns is None:
        columns = range(len(datasets))
    distribution = []
    for ds, name in zip(datasets, columns):
        ppl = (ds.p
               .init_variable('df', default=[])
               .update(V('df', mode='e'), WS('core_lithology').ravel())
              )

        df = ppl.run(len(ds), n_epochs=1).v('df')

        df = (pd.concat(df)
              .reset_index(drop=False)
              .groupby(['FORMATION', 'GRAIN'])
              .apply(lambda x: (x.DEPTH_TO - x.DEPTH_FROM).sum()))

        df_index = df.index.to_frame().apply(concat, axis=1)
        stat = pd.concat([df_index, df], axis=1, sort=True)
        stat.columns = ['CLASS', name]
        distribution.append(stat.set_index('CLASS'))
    df = pd.concat(distribution, axis=1, sort=True).fillna(0)
    for name in columns:
        df[name+'_ratio'] = df[name] / df[name].sum()
    return df