import logging
from typing import List, Set

import numpy as np
from tqdm import tqdm

from src.common.audioviz_dataset import AudiovizDataset
from src.common.audioviz_datastore import AbstractAudiovizDataStore
from src.common.fun_call import FunCall


class FeatureCollection:
    """
    Represents a collection fo features calculated for a particular dataset
    """

    def __init__(self, dataset: AudiovizDataset, store: AbstractAudiovizDataStore):
        self._dataset = dataset
        self._store = store
        self._features: Set[str] = set()

    def update(self, feature_extractors: List[FunCall]):
        """
        Applies a feature extraction function to the dataset if no already in the feature collection
        Parameters
        ----------
        feature_extractors : List of feature extraction functions to be added to the feature collection
        """

        with self._store:
            for fe in feature_extractors:
                self.add(fe)

    def __getitem__(self, key: str):
        if key in self._features:
            return self._store[key]
        else:
            raise KeyError(key)

    def __contains__(self, item: FunCall):
        return item.__repr__() in self._features

    def __in_store(self, item: FunCall):
        return item.__repr__() in self._store

    def add(self, feature_extractor: FunCall):
        # TODO refactor to decorator
        was_open = bool(self._store)
        if not was_open:
            self._store.open()
        try:
            if feature_extractor in self:
                logging.info(
                    f"Feature {feature_extractor} is already in this FeatureCollection. Skipping ..."
                )
            else:
                repr = feature_extractor.__repr__()
                if not self.__in_store(feature_extractor):
                    self._process_feature(feature_extractor)
                self._features.add(repr)
        finally:
            if not was_open:
                self._store.close()

    def get_as_dataset(self):
        # TODO implement iterator for FeatureCollection for convenience this then becomes "for fe in self"
        # TODO returning h5 objects after file closes results in error
        if not self._features:
            return np.ndarray([])
        return np.concatenate([self[fe] for fe in self._features], axis=1)

    def _process_feature(self, feature_extractor: FunCall):
        chunksize = 1000
        logging.info(
            f"Calculating feature {feature_extractor.name} with chunksize {chunksize}"
        )
        with self._store:
            with self._dataset._store:
                len_samples = self._dataset.shape[0]
                pbar = tqdm(total=len_samples)
                pbar.set_description(feature_extractor.name)
                ds = None
                for i, chunk in enumerate(
                    self._dataset.map_chunked(feature_extractor, chunksize), start=1,
                ):
                    if ds is None:
                        ds = self._store._file.create_dataset(
                            feature_extractor.__repr__(),
                            (len_samples, *chunk[0].shape),
                        )

                    chunk_start = i * chunksize - chunksize
                    chunk_end = i * chunksize
                    ds[chunk_start:chunk_end] = chunk
                    msg = f"Processed chunk [{chunk_start}:{chunk_end}]"
                    logging.debug(msg)
                    pbar.update(chunksize)
        self._features.add(feature_extractor.__repr__())
