import logging
from typing import List, Set

import numpy as np
from tqdm import tqdm

from src.common.audioviz_dataset import AudiovizDataset
from src.common.audioviz_datastore import AbstractAudiovizDataStore
from src.common.fun_call import FunCall


class FuncallStore:
    """
    Represents a collection of features calculated for a dataset
    """

    def __init__(self, dataset: AudiovizDataset, store: AbstractAudiovizDataStore):
        self._dataset = dataset
        self._store = store
        self.__funcalls: List[FunCall] = []

    def update(self, funcalls: List[FunCall]):

        with self._store:
            for fa in funcalls:
                self.add(fa)

    @property
    def _funcalls(self):
        return [f.__repr__() for f in self.__funcalls]

    def __str__(self):
        return "".join([str(f) for f in self.__funcalls])

    def __repr__(self):
        return "".join([f.__repr__() for f in self.__funcalls])

    def __getitem__(self, key: FunCall):
        if key.__repr__() in self._funcalls:
            return self._store[key.__repr__()]
        else:
            raise KeyError(key)

    def __contains__(self, item: FunCall):
        return item.__repr__() in self._funcalls

    def __in_store(self, item: FunCall):
        return item.__repr__() in self._store

    def add(self, funcall: FunCall):
        # TODO refactor to decorator
        was_open = bool(self._store)
        if not was_open:
            self._store.open()
        try:
            if funcall in self:
                logging.info(
                    f"Funcall {funcall} is already in this {self.__class__.__name__}. Skipping ..."
                )
            else:
                if not self.__in_store(funcall):
                    self._process(funcall)
                self.__funcalls.append(funcall)
        finally:
            if not was_open:
                self._store.close()

    def get_as_matrix(self):
        # TODO implement iterator for FeatureCollection for convenience this then becomes "for fe in self"
        if not self._funcalls:
            return np.ndarray([])
        return np.concatenate([self[fe] for fe in self.__funcalls], axis=1)

    def _process(self, funcall: FunCall, chunksize=1000):
        logging.info(f"Calculating funcall {funcall.name} with chunksize {chunksize}")
        with self._store, self._dataset._store:
            len_samples = self._dataset.shape[0]
            pbar = tqdm(total=len_samples)
            pbar.set_description(funcall.name)
            ds = None
            for i, chunk in enumerate(
                self._dataset.map_chunked(funcall, chunksize), start=1,
            ):
                # We do this inside the loop because we need chunk[0].shape
                if ds is None:
                    logging.info(f"Starting to calculate feature {funcall.name}")
                    ds = self._store._file.create_dataset(
                        funcall.__repr__(), (len_samples, *chunk[0].shape),
                    )

                chunk_start = i * chunksize - chunksize
                chunk_end = i * chunksize
                ds[chunk_start:chunk_end] = chunk
                msg = f"Processed chunk [{chunk_start}:{chunk_end}]"
                logging.debug(msg)
                pbar.update(chunksize)
        self._funcalls.append(funcall)
