[![License](https://img.shields.io/github/license/analysiscenter/batchflow.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python](https://img.shields.io/badge/python-3.5-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-1.12-orange.svg)](https://tensorflow.org)
[![PyTorch](https://img.shields.io/badge/torch-1.1-orange.svg)](https://pytorch.org)


# PetroFlow

`PetroFlow` is a library that allows to process well data (logs, core photo etc.) and conveniently train
machine learning models.

Main features:

* load and process well data:
    * well logs
    * core images in daylight (DL) and ultraviolet light (UV)
    * core logs
    * inclination
    * stratum layers
    * boring intervals
    * properties of core plugs
    * lithological description of core samples
* perform core-to-log matching
* predict porosity by logs
* detect mismatched DL and UV core images
* recover missed logs and DT log by other logs
* detect collectors by logs

## About PetroFlow

> `PetroFlow` is based on [BatchFlow](https://github.com/analysiscenter/batchflow). You might benefit from reading [its documentation](https://analysiscenter.github.io/batchflow).
However, it is not required, especially at the beginning.


`PetroFlow` has two main modules: [``src``](https://github.com/gazprom-neft/petroflow/tree/master/petroflow/src) and [``models``](https://github.com/gazprom-neft/petroflow/tree/master/petroflow/models).


``src`` module contains ``Well``, ``WellBatch``, ``CoreBatch`` and ``WellLogsBatch`` classes.
``Well`` is a class, representing a well and includes methods for well data processing. All these methods are inherited by ``WellBatch`` class and might be used to build multi-staged workflows that can also involve machine learning models. ``CoreBatch`` is a class for core images processing, especially for detection of mismatched pairs. ``WellLogsBatch`` allows working with well logs separately.

``models`` module provides several ready to use models for important problems:

* logs and core data matching
* predicting of some reservoir properties (e.g., porosity) by well logs
* detecting mismatched pairs of DL and UV core photos
* logs recovering (e.g., DT log) by other logs
* detecting of oil collectors by logs

## Basic usage

Here is an example of a pipeline that loads well data, makes preprocessing and trains
a model for porosity prediction for 3000 epochs:
```python
train_pipeline = (
  bf.Pipeline()
    .add_namespace(np)
    .init_variable("loss", init_on_each_run=list)
    .init_model("dynamic", UNet, "UNet", model_config)
    .keep_logs(LOG_MNEMONICS + ["DEPTH"])
    .interpolate(attrs="core_properties", limit=10, limit_area="inside")
    .norm_min_max(q1, q99)
    .random_crop(CROP_LENGTH_M, N_CROPS)
    .update(B("logs"), WS("logs").ravel())
    .stack(B("logs"), save_to=B("logs"))
    .swapaxes(B("logs"), 1, 2, save_to=B("logs"))
    .array(B("logs"), dtype=np.float32, save_to=B("logs"))
    .update(B("mask"), WS("core_properties")["POROSITY"].ravel())
    .stack(B("mask"), save_to=B("mask"))
    .expand_dims(B("mask"), 1, save_to=B("mask"))
    .divide(B("mask"), 100, save_to=B("mask"))
    .array(B("mask"), dtype=np.float32, save_to=B("mask"))
    .train_model("UNet", B("logs"), B("mask"), fetches="loss", save_to=V("loss", mode="a"))
    .run(batch_size=4, n_epochs=3000, shuffle=True, drop_last=True, bar=True, lazy=True)
)
```


## Installation

> `PetroFlow` module is in the beta stage. Your suggestions and improvements are very welcome.

> `PetroFlow` supports python 3.5 or higher.


### Installation as a python package

With [pipenv](https://docs.pipenv.org/):

    pipenv install git+https://github.com/gazprom-neft/petroflow.git#egg=petroflow

With [pip](https://pip.pypa.io/en/stable/):

    pip3 install git+https://github.com/gazprom-neft/petroflow.git

After that just import `petroflow`:
```python
import petroflow
```


### Installation as a project repository

When cloning repo from GitHub use flag ``--recursive`` to make sure that ``batchflow`` submodule is also cloned.

    git clone --recursive https://github.com/gazprom-neft/petroflow.git

## Citing PetroFlow

Please cite `PetroFlow` in your publications if it helps your research.

    Khudorozhkov R., Kuvaev A., Kozhevin A. PetroFlow library for data science research of well data. 2019.

```
@misc{
  author       = {R. Khudorozhkov and A. Kuvaev and A. Kozhevin},
  title        = {PetroFlow library for data science research of well data},
  year         = 2019
}
```
