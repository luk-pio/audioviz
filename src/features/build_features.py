import logging
from sys import path
from typing import List

import click
import numpy as np

# noinspection PyUnresolvedReferences
import src.common.log

from src.common.AudiovizDataset import AudiovizDataset, load_audioviz_dataset
from src.common.AudiovizDataStore import (
    AudiovizDataStoreFactory,
    AbstractAudiovizDataStore,
)
from src.features.feature_extractor import parse_feature_extractors, FeatureExtractor


class FeatureCollection:
    """
    Represents a collection fo features calculated for a particular dataset
    """

    def __init__(self, dataset: AudiovizDataset, store: AbstractAudiovizDataStore):
        self._dataset = dataset
        self._filepath = path
        self._store = store

    def extend(self, feature_extractors: List[FeatureExtractor]):
        """
        Applies a feature extraction function to the dataset if no already in the feature collection
        Parameters
        ----------
        feature_extractors : List of feature extraction functions to be added to the feature collection

        Returns
        -------

        """

        with self._store:
            for fe in feature_extractors:
                self.append(fe)

    def __getitem__(self, key):
        return self._store[key]

    def __contains__(self, item: FeatureExtractor):
        return item.__repr__() in self._store

    def append(self, feature_extractor: FeatureExtractor):
        was_open = bool(self._store)
        if not was_open:
            self._store.open()
        try:
            if feature_extractor in self:
                logging.info(
                    f"Feature {feature_extractor} is already in this FeatureCollection."
                )
            else:
                extracted_feature = np.apply_along_axis(
                    feature_extractor, 1, self._dataset.data
                )
                self._store.write(feature_extractor.__repr__(), extracted_feature)
        finally:
            if not was_open:
                self._store.close()


def build_features(dataset_name, features, output_path, storage_type, write=False):

    dataset = load_audioviz_dataset(dataset_name)
    feature_collection = FeatureCollection(
        dataset,
        AudiovizDataStoreFactory.get_instance(output_path, storage_type, dataset),
    )
    feature_extractors = parse_feature_extractors(features)
    feature_collection.extend(feature_extractors)
    if write:
        pass
    return feature_collection


@click.command()
@click.argument("dataset", default="medley-solos-db", type=str)
@click.argument("features", nargs=-1, type=str)  # features along with kwargs
@click.option("--score/--build", default=False)
@click.option("-scoring_alg")  # if -score
@click.option("-output_path", default=None, type=click.Path())
@click.option("-storage_type", default="h5", type=str)
def main(dataset, features, score, scoring_alg, output_path, storage_type):
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).

    1. does the feature file in output path exist? read or create as pd.Dataframe
    2. loop through all wanted features
      a. Is the feature in feature_file? if yes break
      b. calculate the feature applying given args
      c. append this feature, along with the pickled funcall as a column to identify this feature
    3. save df as file if modified
    4. -s? if not exit
    5. Make a new pd.Dataframe from selected feaures
    6. run a scoring function on this dataset
    7. output score to stdout
    """
    # TODO remove, this is for debugging purposes only since pycharm mangels string arguments in run targets
    features = ['{"name":"stft", "args":{}}']
    build_features(dataset, features, output_path, storage_type)

    # if score:
    #     score_features()


if __name__ == "__main__":
    main()
