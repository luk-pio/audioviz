import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict

import h5py


class AbstractAudiovizDataStore(ABC):
    @abstractmethod
    def open(self):
        raise NotImplementedError(
            "Please provide a concrete implementation of this method in the subclass."
        )

    @abstractmethod
    def __enter__(self):
        raise NotImplementedError(
            "Please provide a concrete implementation of this method in the subclass."
        )

    @abstractmethod
    def close(self):
        raise NotImplementedError(
            "Please provide a concrete implementation of this method in the subclass."
        )

    @abstractmethod
    def __exit__(self, type, value, traceback):
        raise NotImplementedError(
            "Please provide a concrete implementation of this method in the subclass."
        )

    @abstractmethod
    def __contains__(self, item):
        raise NotImplementedError(
            "Please provide a concrete implementation of this method in the subclass."
        )

    @abstractmethod
    def __getitem__(self, key):
        raise NotImplementedError(
            "Please provide a concrete implementation of this method in the subclass."
        )

    @abstractmethod
    def write(self, key: str, data: Any, metadata: Dict[str, Any] = None, dtype=None):
        raise NotImplementedError(
            "Please provide a concrete implementation of this method in the subclass."
        )


# def open_if_closed(file_like, func):
#     def wrapper(*args, **kwargs):
#         was_open = bool(file_like)
#         if not was_open:
#             file_like.open()
#         try:
#             func(*args, **kwargs)
#         finally:
#             if not was_open:
#                 file_like.close()
#
#     return wrapper


class Hdf5AudiovizDataStore(AbstractAudiovizDataStore):
    def __init__(self, path: str):
        if not os.path.isdir(os.path.dirname(path)):
            raise OSError(f"{path} is not in a valid directory.")
        self.path = path
        self._file = None

    def open(self, mode="a"):
        self._file = h5py.File(self.path, mode)
        return self._file

    def __enter__(self, mode="a"):
        self.open(mode)

    def __delitem__(self, key):
        del self._file[key]

    def __getitem__(self, key):
        # TODO refactor to decorator
        was_open = bool(self._file)
        if not was_open:
            self.open()
        try:
            val = self._file[key]
        except ValueError:
            logging.exception(f"Invalid identifier for hdf5 dataset {key}")
            raise
        except KeyError:
            logging.exception(f"No key: {key} in h5 file")
            raise
        finally:
            if not was_open:
                self.close()
        return val

    def close(self):
        self._file.close()

    def __exit__(self, type, value, traceback):
        self.close()

    def __contains__(self, item):
        # TODO refactor opening and closing to a decorator
        was_open = bool(self._file)
        if not was_open:
            self.open()
        try:
            val = item in self._file
        finally:
            if not was_open:
                self.close()
        return val

    def write(
        self, key: str, data: Any, metadata: Dict[str, Any] = None, dtype=None,
    ):
        # TODO refactor to decorator
        was_open = bool(self._file)
        if not was_open:
            self.open()
        try:
            dset = self._file.create_dataset(key, data=data, dtype=dtype)
            if metadata:
                for k in metadata:
                    dset.attrs[k] = metadata[k]
        finally:
            if not was_open:
                self.close()


class AudiovizDataStoreFactory:
    implemented = {"h5": Hdf5AudiovizDataStore}

    @classmethod
    def get_instance(cls, path: str, storage_type: str) -> AbstractAudiovizDataStore:
        try:
            return cls.implemented[storage_type](path)
        except KeyError as err:
            msg = f"The I/O for {storage_type} has not been implemented."
            logging.exception(msg)
            raise NotImplementedError(msg) from err
