import logging
from sys import path
from typing import List, Set

import numpy as np

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
                    extracted_feature = np.apply_along_axis(
                        feature_extractor, 1, self._dataset.data
                    )
                    self._store.write(repr, extracted_feature)
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
