import gc
import os
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.utils import Bunch

# noinspection PyUnresolvedReferences
import src.common.log
from src.common.audioviz_datastore import (
    AbstractAudiovizDataStore,
    AudiovizDataStoreFactory,
    Hdf5AudiovizDataStore,
)
from src.common.dataset_config import dataset_config_factory
from src.common.utils import DATA_INTERIM_DIR


class AudiovizDataset:
    """
    Represents a dataset. In essence a dictionary, which can also be accessed through attributes.
    Follows sckikit-learn conventions:

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : dataframe
            The data matrix.
        target: Series
            The classification target.
        target_names: list
            The names of target classes.
        frame: DataFrame
            DataFrame with `data` and `target`.
        DESCR: str
            The full description of the dataset.
        data_filename: str
            The path to the location of the data.
        metadata_filename: str
            The path to the location of the data.
        metadata: dict
            pd.Dataframe of any remaining metadata
    """

    def __init__(
        self,
        store,
        name: str,
        target_key: str = None,
        dataset_key: str = None,
        target_names: List[str] = None,
        metadata_keys: str = None,
        data_filename: str = None,
    ):

        self._store = store
        self.name = name
        self.target_key = target_key if target_key is not None else "class"
        self.dataset_key = dataset_key if dataset_key is not None else name
        self._target_names = target_names
        self.metadata_keys = metadata_keys
        self.data_filename = data_filename

    @property
    def data(self):
        return self._store[self.dataset_key]

    @property
    def target(self):
        return self._store[self.target_key]

    @property
    def target_names(self):
        self._target_names = (
            self._target_names if self._target_names is not None else set(self.target)
        )
        return self._target_names

    @property
    def shape(self):
        return self.data.shape

    def map_chunked(self, func, chunksize):
        with self._store:
            rows = self.shape[0]
            n = rows // chunksize
            for i in range(1, n + 1):
                chunk = self.data[i * chunksize - chunksize : i * chunksize]
                yield [func(row) for row in chunk]
            chunk = self.data[n * chunksize :]
            yield [func(row) for row in chunk]

    @staticmethod
    def load(name: str, store: AbstractAudiovizDataStore = None):
        if store is None:
            file_path = os.path.join(DATA_INTERIM_DIR, name + "_data.h5")
            store = Hdf5AudiovizDataStore(file_path)
        with store:
            target_names = store[name].attrs["classes"]
        return AudiovizDataset(
            store=store,
            name=name,
            target_key="instrument_id",
            target_names=target_names,
            data_filename=store.path,
        )


def save_medley_solos_db(
    name: str,
    dataset: Dict[str, Any],
    path: str,
    metadata_keys: List[str],
    store: AbstractAudiovizDataStore = None,
):
    store = (
        AudiovizDataStoreFactory.get_instance(path, "h5") if store is None else store
    )

    subsets_mapping = {"training": 0, "test": 1, "validation": 2}
    subsets_list = ["training", "test", "validation"]
    classes = [
        "clarinet",
        "distorted electric guitar",
        "female singer",
        "flute",
        "piano",
        "tenor saxophone",
        "trumpet",
        "violin",
    ]
    samplerate = next(iter(dataset.items()))[1].audio[1]
    dataset_metadata = {
        "subsets_list": subsets_list,
        "classes": classes,
        "samplerate": samplerate,
    }

    samples = []
    files_metadata = defaultdict(list)
    sample_shape = (
        next(iter(dataset.items()))[1].audio[0].shape
    )  # Gets the length of the sample

    shape = (len(dataset), *sample_shape)

    with store:
        store._file.create_dataset(name, shape=shape)
    i = 0
    chunksize = 200
    with store:
        for _, sample in dataset.items():
            samples.append(sample.audio[0])
            # Turn list of metadata dictionaries for each file into one dict with a list of vals
            for key in ["instrument_id", "subset", "song_id"]:
                value = getattr(sample, key)
                # Convert subset to nominal
                if key == "subset":
                    value = subsets_mapping[value]
                files_metadata[key].append(value)
            i += 1
            if i % 100 == 0:
                print(f"Processed {i} samples")
                if i % chunksize == 0:
                    ds = store[name]
                    ds[i - chunksize : i] = np.asarray(samples)
                print(f"Wrote chunk: [{i-chunksize}:{i}]")
            del samples
            gc.collect()
    samples = []
    ds = store[name]
    remainder = i % chunksize
    ds[i - remainder : i] = np.asarray(samples)
    for k, v in files_metadata.items():
        store._file.create_dataset(k, data=v)
        for k, v in dataset_metadata.items():
            ds.attrs[k] = v

    del samples

    store.close()


# class AudiovizDatasetFactory:
#     implemented = {"medley-solos-db": create_medley_solos_ds}
#
#     @classmethod
#     def get_instance(cls, dataset: str):
#         try:
#             return cls.implemented[dataset]
#         except KeyError as err:
#             msg = f"The dataset {dataset} has not been implemented"
#             logging.exception(msg)
#             raise NotImplementedError(msg) from err
