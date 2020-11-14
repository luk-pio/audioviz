import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict

import h5py

from src.common import AudiovizDataset
from src.common.utils import DATA_PROCESSED_DIR


class AbstractAudiovizDataStore(ABC):
    @abstractmethod
    def __getitem__(self, key):
        raise NotImplementedError(
            "Please provide a concrete implementation of this method in the subclass."
        )

    @abstractmethod
    def write(self, item: Any, metadata: Dict[str, Any]):
        raise NotImplementedError(
            "Please provide a concrete implementation of this method in the subclass."
        )


class Hdf5AudiovizDataStore(AbstractAudiovizDataStore):
    def __init__(self, path: str, dataset: AudiovizDataset):
        if not path or not os.path.isfile(path):
            path = os.path.join(DATA_PROCESSED_DIR, dataset.name + ".h5")
            logging.info(f"Creating new file at {path}")

        self._path = path
        self._dataset = dataset

    def open(self):
        self._file = h5py.File(self._path, "a")
        return self._file

    def __enter__(self):
        self.open()

    def __getitem__(self, key):
        # TODO test
        return self._file[key]

    def close(self):
        self._file.close()

    def __exit__(self, type, value, traceback):
        self.close()

    def __contains__(self, item):
        # TODO test
        return item in self._file

    def write(
        self,
        key: str,
        data: Any,
        metadata: Dict[str, Any] = None,
        dtype=None,
        shape=None,
    ):
        if not self._file:
            self.open()
        if shape is None:
            try:
                shape = data.shape
            except AttributeError:
                raise ValueError("Cannot determine shape of data")
        try:
            dset = self._file.create_dataset(key, data=data, dtype=dtype)
            if metadata:
                for k in metadata:
                    dset.attrs[k] = metadata[k]
        finally:
            self.close()


class AudiovizDataStoreFactory:
    implemented = {"h5": Hdf5AudiovizDataStore}

    @classmethod
    def get_instance(cls, path: str, storage_type: str, dataset):
        try:
            return cls.implemented[storage_type](path, dataset)
        except KeyError as err:
            msg = f"The I/O for {storage_type} has not been implemented."
            logging.exception(msg)
            raise NotImplementedError(msg) from err
