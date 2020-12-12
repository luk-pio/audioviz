import logging
import os

import click

from src.common.audioviz_dataset import AudiovizDataset
from src.common.audioviz_datastore import AudiovizDataStoreFactory
from src.common.fun_call import parse_funcall
from src.features.feature_extractors import FeatureExtractorFactory
from src.common.utils import DATA_PROCESSED_DIR
from src.features.feature_collection import FeatureCollection


def parse_feature_extractors(features):
    feature_extractors = []
    for feature in features:
        try:
            fe = parse_funcall(feature, FeatureExtractorFactory)
        except Exception:
            logging.exception(f"Skipping feature {feature} ...")
            continue
        feature_extractors.append(fe)
    return feature_extractors


def build_features(dataset, features, storage_type):
    path = os.path.join(DATA_PROCESSED_DIR, dataset + "." + storage_type)
    feature_collection = FeatureCollection(
        AudiovizDataset.load(dataset),
        AudiovizDataStoreFactory.get_instance(path, storage_type),
    )
    feature_extractors = parse_feature_extractors(features)
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
    # TODO remove, this is for debugging purposes only since pycharm mangles string arguments in run targets
    features = ['{"name":"stft", "args":{}}']
    build_features(dataset, features, storage_type)

    # if score:
    #     score_features()


if __name__ == "__main__":
    main()
