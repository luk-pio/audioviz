from collections import defaultdict, OrderedDict
from functools import reduce

from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from sklearn.metrics import silhouette_score


# construct a polygon from vertices of a convex hull
def convex_poly(points):
    hull = ConvexHull(points)
    return Polygon([points[v] for v in hull.vertices])


def overlap_ratio(data, labels):
    # TODO better data structure?
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


def eval_metrics(data, cluster_labels):
    metrics = OrderedDict(
        {
            "Silhouette": silhouette_score(data, cluster_labels),
            "Convex hull overlap": overlap_ratio(data, cluster_labels),
        }
    )
    metrics["Total"] = sum(metrics.keys())
    return metrics


def print_metrics(metrics):
    print("metric\t\t score")
    print("_" * 35)
    for k, v in metrics.items():
        print(f"{k:<9}\t{v}")
