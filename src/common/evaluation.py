from collections import OrderedDict, defaultdict
from functools import reduce
import logging
from math import sqrt

from ripleyk import ripleyk
from scipy import spatial
from scipy.spatial import ConvexHull
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# noinspection PyUnresolvedReferences
import src.common.log


# construct a polygon from vertices of a convex hull
def convex_poly(points):
    hull = ConvexHull(points)
    return Polygon([points[v] for v in hull.vertices])


def overlap_ratio(data, labels):
    grouped = defaultdict(list)
    for i in range(len(data)):
        grouped[labels[i]].append(data[i])

    cluster_polygons = {name: convex_poly(group) for name, group in grouped.items()}
    polygons = list(cluster_polygons.values())

    # the area of the union of all cluster polygons over the sum of the areas of all polys
    return unary_union(polygons).area / sum([p.area for p in polygons])


def roundness(data, labels):
    grouped = defaultdict(list)
    for i in range(len(data)):
        grouped[labels[i]].append(data[i])

    cluster_polygons = {name: convex_poly(group) for name, group in grouped.items()}

    total_poly = unary_union(list(cluster_polygons.values()))
    return 4 * np.pi * total_poly.area / (total_poly.length ** 2)


def ripley(data, *args):
    radii = [0.05, 0.1, 0.25, 0.5]

    def make_tree(d1, d2):
        points = np.c_[d1.ravel(), d2.ravel()]
        return spatial.cKDTree(points)

    def calc_ripley_k(r, d1, d2):
        area = np.pi * r ** 2
        count = sum([len(tree.query_ball_point([x, y], r)) - 1 for x, y in zip(d1, d2)])
        return count / (area * len(d1))

    d1 = data[:, 0]
    d2 = data[:, 1]
    tree = make_tree(d1, d2)
    ripley_k = [calc_ripley_k(r, d1, d2) for r in radii]

    ripley_l = [sqrt(k / np.pi) for k in ripley_k]
    ripley_h = [l - r for r, l in zip(radii, ripley_l)]
    return sum(ripley_h) / len(radii)


SILHOUETTE = "Silhouette"
CONVEX_HULL_OVERLAP = "Convex_hull_overlap"
ROUNDNESS = "Roundness"
RIPLEY = "Ripley"

METRICS = OrderedDict(
    {
        SILHOUETTE: silhouette_score,
        ROUNDNESS: roundness,
        CONVEX_HULL_OVERLAP: overlap_ratio,
        RIPLEY: ripley,
    }
)


def eval_metrics(data, cluster_labels):
    metrics = {k: v(data, cluster_labels) for k, v in METRICS.items()}
    metrics["Total"] = sum(metrics.values())
    return metrics


def format_metrics(metrics):
    col_width = 25
    METRIC = "metric"
    SCORE = "score"
    header = METRIC + " " * (col_width - len(METRIC)) + SCORE
    sep = "_" * len(header)
    return "\n".join(
        [header, sep] + [f"{k:<{col_width}} {v:.5f}" for k, v in metrics.items()]
    )


def recalculate_scores(dimred_collection, dimred_algs, labels):
    scores = []
    for alg in dimred_algs:
        score = eval_metrics(dimred_collection[alg.__repr__()], labels)
        for k in dimred_collection[alg.__repr__()].attrs:
            del dimred_collection[alg.__repr__()].attrs[k]
        for k, v in score.items():
            dimred_collection[alg.__repr__()].attrs[k] = v
        scores.append(score)
        logging.info(
            f"Recalculated score for {str(alg)} [{len(scores)}/{len(dimred_algs)}]"
        )
    return scores


def get_scores(dimred_collection, dimred_algs):
    return [dict(dimred_collection[alg.__repr__()].attrs) for alg in dimred_algs]


def normalize_scores(dimred_collection, dimred_algs):
    scores = get_scores(dimred_collection, dimred_algs)

    weights = {
        SILHOUETTE: 2,
        ROUNDNESS: 1,
        CONVEX_HULL_OVERLAP: 1,
        RIPLEY: 2,
    }

    # Normalize all the scores with respect to each max value
    # Silhoutte is in range (-1,1) so we transfer to (0,1)
    roundness_max = max([s[ROUNDNESS] for s in scores])
    silhouette_max = (1 + max([s[SILHOUETTE] for s in scores])) / 2
    cv_h_o_max = max([s[CONVEX_HULL_OVERLAP] for s in scores])
    ripley_max = max([1 / abs(s[RIPLEY]) for s in scores])

    for s in scores:
        s[ROUNDNESS] /= roundness_max
        s[SILHOUETTE] = ((1 + s[SILHOUETTE]) / 2) / silhouette_max
        s[CONVEX_HULL_OVERLAP] /= cv_h_o_max
        s[RIPLEY] = 1 / abs(s[RIPLEY]) / ripley_max

        # Apply weights
        for k, v in weights.items():
            s[k] *= v

        s["Total"] = sum(s[metric] for metric in METRICS.keys())

    return scores
