import logging
import os
import pickle
from typing import List
import numpy as np

import click

from src.common.utils import PROJECT_DIR
import src.common.log


class FeatureCollection:
    def __init__(
        self, name: str, feature_array: np.ndarray = None, metadata: List[str] = None
    ):
        self._name = name
        self._feature_array = np.ndarray([]) if feature_array is None else feature_array
        self._metadata = [] if metadata is None else metadata

    def add_feature_set(self, name: str, features: np.ndarray, metadata: List[str]):
        self._name += name
        self._feature_array = np.concatenate((self._feature_array, features))
        self._metadata += metadata

    def write_data(self, path):
        np.save(os.join(path, self._name + "_features.npy"))

    def write_metadata(self, path):
        filename = os.join(path, self._name + "_features_metadata.pkl")
        with open(filename, "wb") as f:
            pickle.dump(self._metadata, f)
        logging.info(f"wrote feature metadata to file {filename}")


def extract_stft(input_file: str):
    pass


def extract_mfcc(input_file: str):
    pass


implemented_feature_sets_to_extractors_dict = {
    "stft": extract_stft,
    "mfcc": extract_mfcc,
}


def build_feature_collection(
    input_file: str, feature_names: List[str] = None, name_prefix: str = ""
) -> FeatureCollection:
    """
    Extracts all features given as input_feature_sets
    """
    implemented_feature_sets = list(implemented_feature_sets_to_extractors_dict.keys())
    if feature_names is None:
        feature_names = implemented_feature_sets[0]

    feature_sets_to_build = []
    for f in feature_names:
        if f not in implemented_feature_sets_to_extractors_dict.keys():
            logging.warning(
                f"Unknown feature set: {f}. This feature set will not be calculated."
            )
        else:
            feature_sets_to_build.append(f)

    feature_collection = FeatureCollection(name_prefix)
    for feature_set_name in feature_sets_to_build:
        logging.info(f"Extracting {feature_set_name} features.")
        feature_set = implemented_feature_sets_to_extractors_dict[feature_set_name](
            input_file
        )
        logging.info(f"{feature_set_name} features extracted!")
        feature_collection.add_feature_set(*feature_set)

    return feature_collection


@click.command()
@click.argument("-features", nargs=-1, case_sensitive=False)
@click.option("-input_file", default=None, type=click.Path(exists=True))
@click.option("-output_path", default=None, type=click.Path())
def main(input_file, output_path, features):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    input_file = (
        os.path.join(PROJECT_DIR, "data", "raw") if input_file is None else input_file
    )
    output_path = (
        os.path.join(PROJECT_DIR, "data", "processed")
        if output_path is None
        else output_path
    )

    feature_collection_name_prefix = input_file.rsplit(".npy", 1)[0].rsplit("_samples")[
        0
    ]
    feature_collection = build_feature_collection(
        input_file, features, name_prefix=feature_collection_name_prefix
    )
    feature_collection.write_data(output_path)
    feature_collection.write_metadata(output_path)
