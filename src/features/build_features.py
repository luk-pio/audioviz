import json
import os
from typing import Any, Dict, List

import click

from src.common.audioviz_dataset import AudiovizDataset
from src.common.audioviz_datastore import AudiovizDataStoreFactory
from src.common.fun_call import parse_funcall, parse_funcalls
from src.common.utils import DATA_FEATURES_DIR
from src.features.feature_collection import FuncallStore
from src.features.feature_extractors import FeatureExtractorFactory


def parse_feature_extractor(feature):
    return parse_funcall(feature, FeatureExtractorFactory)


def parse_feature_extractors(features):
    return parse_funcalls(features, FeatureExtractorFactory)


def get_features(
    dataset: AudiovizDataset, features: List[Dict[str, Any]], storage_type="h5"
):
    path = os.path.join(DATA_FEATURES_DIR, dataset.name + "." + storage_type)
    feature_collection = FuncallStore(
        dataset, AudiovizDataStoreFactory.get_instance(path, storage_type),
    )
    stringified_features = [json.dumps(feature) for feature in features]
    feature_extractors = parse_feature_extractors(stringified_features)
    feature_collection.update(feature_extractors)
    return feature_collection


@click.command()
@click.argument("dataset", default="medley_solos_db", type=str)
@click.argument("features", nargs=-1, type=str)  # features along with kwargs
@click.option("--score/--build", default=False)
@click.option("-scoring_alg")  # if -score
@click.option("-storage_type", default="h5", type=str)
def main(dataset, features, score, scoring_alg, storage_type):
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../features).

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
    # TODO remove, this is for debugging purposes only since pycharm mangles string arguments in run targets
    features = [{"name": "stft", "args": {}}]
    dataset = AudiovizDataset.load(dataset)
    get_features(dataset, features, storage_type)

    # if score:
    #     score_features()


if __name__ == "__main__":
    main()
