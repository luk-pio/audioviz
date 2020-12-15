import json
import os
from typing import List, Any, Dict

import click

from src.common.audioviz_dataset import AudiovizDataset
from src.common.audioviz_datastore import AudiovizDataStoreFactory
from src.common.evaluation import eval_metrics, normalize_scores
from src.common.utils import DATA_DIMRED_DIR
from src.dimred.dimred_algorithms import parse_dimred_algorithms
from src.features.build_features import get_features

# noinspection PyUnresolvedReferences
import src.common.log
import logging


def apply_dimred(
    dataset: AudiovizDataset,
    dimred_algorithms: List[Dict[str, Any]],
    features: List[Dict[str, Any]],
    storage_type="h5",
):
    stringified_dimred_algs = [json.dumps(a) for a in dimred_algorithms]
    dimred_algs = parse_dimred_algorithms(stringified_dimred_algs)
    feature_collection = get_features(dataset, features)
    identifier = feature_collection.__repr__()

    path = os.path.join(DATA_DIMRED_DIR, f"{dataset.name}_dimred.{storage_type}")
    dimred_store = AudiovizDataStoreFactory.get_instance(path, "h5")

    with feature_collection._store, dimred_store, dataset._store:
        for alg in dimred_algs:
            alg.add_suffix(identifier)
            if alg.__repr__() not in dimred_store:
                nd = feature_collection.get_as_matrix()
                logging.info(f"Starting to apply {str(alg)}")
                reduced = alg(nd)
                dimred_store.write(
                    alg.__repr__(), reduced, eval_metrics(reduced, dataset.target)
                )
    return dimred_store, dimred_algs, feature_collection


@click.command()
@click.argument("dataset", type=str)
@click.argument("dimred_algorithm", nargs=1, type=str)
@click.argument("features", nargs=-1, type=str)  # features along with kwargs
@click.option("-scoring_alg", type=str)  # if -score
@click.option("-output_path", default=None, type=click.Path())
@click.option("-storage_type", default="h5", type=str)
def main(dataset, dimred_algorithm, features, scoring_alg, output_path, storage_type):
    # TODO remove these (debugging)
    features = ['{"name":"stft", "args":{}}']
    dimred_algorithm = ['{"name":"pca", "args":{}}']
    dataset = "medley_solos_db"

    dataset = AudiovizDataset.load(dataset)
    apply_dimred(dataset, dimred_algorithm, features)


def test():
    import json

    import matplotlib.pyplot as plt

    from src.common.audioviz_dataset import AudiovizDataset
    from src.common.evaluation import format_metrics, eval_metrics
    from src.dimred.dimred import apply_dimred
    from src.features.build_features import get_features
    from src.visualization.visualize import plot2D

    mfcc = {"name": "mfcc", "args": {}}
    features = [mfcc]

    # Grid search over umap parameters
    n_neighbors = [5, 10, 15, 30, 50]
    min_dists = [0.000, 0.001, 0.01, 0.1, 0.5]
    umap_algs = [
        {
            "name": "umap",
            "args": {"n_neighbors": n, "min_dist": d, "metric": "correlation"},
        }
        for n in n_neighbors
        for d in min_dists
    ]
    umap_alg_str = [json.dumps(umap_alg) for umap_alg in umap_algs]

    dataset = AudiovizDataset.load("medley_solos_db")
    dimred_collection, algs, feature_collection = apply_dimred(
        dataset, umap_alg_str, features
    )

    def normalize_scores(dimred_collection, labels, algs, do_rescore=False):
        def recalculate():
            scores = []
            for alg in algs:
                score = eval_metrics(dimred_collection[alg.__repr__()], labels)
                for k, v in score.items():
                    dimred_collection[alg.__repr__()].attrs[k] = v
                scores.append(score)
            return scores

        scores = (
            recalculate()
            if do_rescore
            else [dict(dimred_collection[alg.__repr__()].attrs) for alg in algs]
        )
        C_H = "Calinski_Harabash"
        S = "Silhouette"
        CV_H_O = "Convex_hull_overlap"

        c_h_weight = 1
        s_weight = 1
        cv_h_o_weight = 0.25

        # Normalize all the scores
        max_c_h = max([s[C_H] for s in scores])
        # Silhoutte is in range (-1,1) so we transfer to (0,1)
        max_s = (1 + max([s[S] for s in scores])) / 2
        max_cv_h_o = max([s[CV_H_O] for s in scores])

        for s in scores:
            del s["K-means V-measure"]
            del s["Time Total"]
            s[C_H] = s[C_H] / max_c_h
            s[S] = ((1 + s[S]) / 2) / max_s
            s[CV_H_O] /= max_cv_h_o

            # Apply weights
            s[C_H] *= c_h_weight
            s[S] *= s_weight
            s[CV_H_O] *= cv_h_o_weight

            s["Total"] = sum([s[C_H], s[S]]) - s[CV_H_O]

        return scores

    with feature_collection._store, dataset._store, dimred_collection:
        color_map = [dataset.colors[t] for t in dataset.target]
        scores = normalize_scores(dimred_collection, dataset.target, algs)

        fig, ax = plt.subplots(
            nrows=len(n_neighbors), ncols=len(min_dists), figsize=(30, 45)
        )

        for i, row in enumerate(ax):
            for j, col in enumerate(row):
                current_plot = i * len(min_dists) + j
                current_reduced = dimred_collection[algs[current_plot].__repr__()]
                col.scatter(
                    current_reduced[:, 0], current_reduced[:, 1], c=color_map, s=1
                )
                col.set_title(
                    f"neighbors: {n_neighbors[i]}, distances: {min_dists[i]}\n{format_metrics(scores[current_plot])}",
                    fontdict={"fontsize": 15},
                )
        plt.subplots_adjust(hspace=0.5)
        plt.show()
        total_scores = [s["Total"] for s in scores]
        max_score = max(total_scores)
        max_index = total_scores.index(max_score)
        plot2D(
            dimred_collection[algs[max_index].__repr__()],
            color_map,
            str(algs[max_index]),
            str(feature_collection),
        )
        print(format_metrics(scores[max_index]))


def test2():
    import matplotlib.pyplot as plt

    from src.common.audioviz_dataset import AudiovizDataset
    from src.common.evaluation import format_metrics
    from src.dimred.dimred import apply_dimred
    from src.visualization.visualize import plot2D

    stft = {"name": "stft", "args": {}}
    features = [stft]

    pca_alg = {"name": "pca", "args": {}}

    # Grid search over umap parameters
    n_neighbors = [5, 10, 15, 30, 50]
    min_dists = [0.000, 0.001, 0.01, 0.1, 0.5]
    umap_algs = [
        {
            "name": "umap",
            "args": {"n_neighbors": n, "min_dist": d, "metric": "correlation"},
        }
        for n in n_neighbors
        for d in min_dists
    ]
    dimred_algs = umap_algs + [pca_alg]

    dataset = AudiovizDataset.load("medley_solos_db")
    dimred_collection, algs, feature_collection = apply_dimred(
        dataset, dimred_algs, features
    )

    with feature_collection._store, dataset._store, dimred_collection:
        color_map = [dataset.colors[t] for t in dataset.target]
        scores = normalize_scores(dimred_collection, dataset.target, algs)

        fig, ax = plt.subplots(
            nrows=len(n_neighbors), ncols=len(min_dists), figsize=(30, 45)
        )

        for i, row in enumerate(ax):
            for j, col in enumerate(row):
                current_plot = i * len(min_dists) + j
                current_reduced = dimred_collection[algs[current_plot].__repr__()]
                col.scatter(
                    current_reduced[:, 0], current_reduced[:, 1], c=color_map, s=1
                )
                col.set_title(
                    f"neighbors: {n_neighbors[i]}, distances: {min_dists[i]}\n{format_metrics(scores[current_plot])}",
                    fontdict={"fontsize": 15},
                )
        plt.subplots_adjust(hspace=0.5)
        plt.show()
        total_scores = [s["Total"] for s in scores]
        max_score = max(total_scores)
        max_index = total_scores.index(max_score)
        plot2D(
            dimred_collection[algs[max_index].__repr__()],
            color_map,
            str(algs[max_index]),
            str(feature_collection),
        )
        print(format_metrics(scores[max_index]))


if __name__ == "__main__":
    # main()
    # test()
    test2()
