import copy
import json
import os
import time
from typing import List, Any, Dict

import click

from src.common.audioviz_dataset import AudiovizDataset
from src.common.audioviz_datastore import AudiovizDataStoreFactory
from src.common.evaluation import eval_metrics, normalize_scores
from src.common.fun_call import FunCall
from src.common.utils import DATA_DIMRED_DIR
from src.dimred.dimred_algorithms import parse_dimred_algorithms
from src.features.build_features import get_features

# noinspection PyUnresolvedReferences
import src.common.log
import logging


def apply_dimred(
    dataset: AudiovizDataset,
    dimred_algs: List[FunCall],
    features: List[FunCall],
    storage_type="h5",
    recalculate=False,
    score=True,
    path=None,
    rows=None,
):
    feature_collection = get_features(dataset, features)

    path = (
        os.path.join(DATA_DIMRED_DIR, f"{dataset.name}_dimred.{storage_type}")
        if path is None
        else path
    )
    dimred_store = AudiovizDataStoreFactory.get_instance(path, "h5")

    rows = slice(None, None) if not rows else rows
    with feature_collection._store, dimred_store, dataset._store:
        dimred_algs = copy.deepcopy(dimred_algs)
        for alg in dimred_algs:
            for f in feature_collection.funcalls:
                alg.add_suffix(f)
            if alg.__repr__() in dimred_store and recalculate:
                del dimred_store[alg.__repr__()]
            if alg.__repr__() not in dimred_store:
                nd = feature_collection.get_as_matrix(rows)
                logging.info(f"Starting to apply {str(alg)}")
                # start_time = time.perf_counter()
                reduced = alg(nd)
                # elapsed = time.perf_counter() - start_time
                scores = {
                    **(eval_metrics(reduced, dataset.target[rows]) if score else {}),
                }
                dimred_store.write(alg.__repr__(), reduced, scores)
    return dimred_store, dimred_algs, feature_collection


@click.command()
@click.argument("dataset", type=str)
@click.argument("dimred_algorithm", nargs=1, type=str)
@click.argument("features", nargs=-1, type=str)  # features along with kwargs
@click.option("-scoring_alg", type=str)  # if -score
@click.option("-output_path", default=None, type=click.Path())
@click.option("-storage_type", default="h5", type=str)
def main(dataset, dimred_algorithm, features, scoring_alg, output_path, storage_type):
    dataset = AudiovizDataset.load(dataset)
    apply_dimred(dataset, dimred_algorithm, features)


if __name__ == "__main__":
    main()
