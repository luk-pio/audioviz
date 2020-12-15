from collections import OrderedDict, defaultdict
from functools import reduce

from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from sklearn.metrics import silhouette_score, calinski_harabasz_score


# construct a polygon from vertices of a convex hull
def convex_poly(points):
    hull = ConvexHull(points)
    return Polygon([points[v] for v in hull.vertices])


def overlap_ratio(data, labels):
    grouped = defaultdict(list)
    for i in range(len(data)):
        grouped[labels[i]].append(data[i])

    cluster_polygons = {name: convex_poly(group) for name, group in grouped.items()}

    total_polygon = convex_poly(data)
    polygons = list(cluster_polygons.values()) + [total_polygon]

    # the area of the union of all cluster polygons over the area of the total plot
    return reduce(lambda a, b: a.union(b), polygons).area / sum(
        [p.area for p in polygons]
    )


def km_vmeasure(data, cluster_labels):
    # TODO
    return 0


def eval_metrics(data, cluster_labels):
    metrics = OrderedDict(
        {
            "Silhouette": silhouette_score(data, cluster_labels),
            "Convex_hull_overlap": overlap_ratio(data, cluster_labels),
            "Calinski_Harabash": calinski_harabasz_score(data, cluster_labels),
            "K-means V-measure": km_vmeasure(data, cluster_labels),
            # "Time Total": 0,
        }
    )
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
        s[C_H] = s[C_H] / max_c_h
        s[S] = ((1 + s[S]) / 2) / max_s
        s[CV_H_O] /= max_cv_h_o

        # Apply weights
        s[C_H] *= c_h_weight
        s[S] *= s_weight
        s[CV_H_O] *= cv_h_o_weight

        s["Total"] = sum([s[C_H], s[S]]) - s[CV_H_O]

    return scores
