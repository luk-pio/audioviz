import gc
import os
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np

# noinspection PyUnresolvedReferences
import src.common.log
from src.common.audioviz_datastore import (
    AbstractAudiovizDataStore,
    AudiovizDataStoreFactory,
    Hdf5AudiovizDataStore,
)
from src.common.utils import DATA_INTERIM_DIR


class AudiovizDataset:
    def __init__(
        self,
        store,
        name: str,
        target_key: str = None,
        dataset_key: str = None,
        subset_key: str = None,
        target_names: List[str] = None,
        metadata_keys: str = None,
        data_filename: str = None,
    ):

        self.subset_key = subset_key if subset_key is not None else "subset"
        self._store = store
        self.name = name
        self.target_key = target_key if target_key is not None else "class"
        self.dataset_key = dataset_key if dataset_key is not None else name
        self.target_names = target_names
        self.metadata_keys = metadata_keys
        self.data_filename = data_filename
        self._color_map = None

    @property
    def data(self):
        return self._store[self.dataset_key]

    @property
    def target(self):
        return self._store[self.target_key]

    @property
    def shape(self):
        return self.data.shape

    @property
    def colors(self):
        return self.data.attrs["colors"]

    @property
    def subset(self):
        return self._store[self.subset_key]

    def get_subset_rows(self, subset: str):
        try:
            index = np.where(self.data.attrs["subsets_list"] == subset)[0][0] + 1
        except IndexError as e:
            raise KeyError(f"Could not find subset with key {subset} in Dataset") from e
        return np.where(self.subset[:] == index)

    def color_map(self, rows):
        if self._color_map is None:
            self._color_map = [self.colors[t] for t in self.target[rows]]
        return self._color_map

    def map_chunked(self, func, chunksize):
        # TODO Refactor this out to _store?
        with self._store:
            rows = self.shape[0]
            n = rows // chunksize
            for i in range(1, n + 1):
                chunk = self.data[i * chunksize - chunksize : i * chunksize]
                yield [func(row) for row in chunk]
            chunk = self.data[n * chunksize :]
            yield [func(row) for row in chunk]

    def get_samples_for_class(self, clazz: int, num_samples: int):
        ret = []
        for i, val in enumerate(self.target):
            if val == clazz:
                ret.append(i)
            if len(ret) < num_samples:
                return ret

    def get_samples_for_each_class(
        self, num=2,
    ):
        samples = [[] for _ in range(len(self.target_names))]
        for i, val in enumerate(self.target):
            if len(samples[val]) < num:
                samples[val].append(i)
        return samples

    @staticmethod
    def load(name: str, store: AbstractAudiovizDataStore = None):
        if store is None:
            file_path = os.path.join(DATA_INTERIM_DIR, name + "_data.h5")
            store = Hdf5AudiovizDataStore(file_path)
        with store:
            target_names = list(store[name].attrs["classes"])
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
    colors = [
        "#ff4000",
        "#ffbf00",
        "#0000ff",
        "#00ffbf",
        "#00bfff",
        "#8000ff",
        "#ff0088",
        "#aeff00",
    ]

    samplerate = next(iter(dataset.items()))[1].audio[1]
    dataset_metadata = {
        "subsets_list": subsets_list,
        "classes": classes,
        "samplerate": samplerate,
        "colors": colors,
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
