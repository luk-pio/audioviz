from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np

# The silhoutte coefficient
def computeSilCoeff(data, cluster_labels):
    silhouette_avg = silhouette_score(data, cluster_labels)
    return silhouette_avg

# Define the roundness & overlap metrics
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, MultiPolygon, Point

def clusterGeometryMetrics(drum_points, labels):
    grouped_drum_points = {}
    
    for i in range(len(drum_points)):
        if labels[i] not in grouped_drum_points:
            grouped_drum_points[labels[i]] = list()
        grouped_drum_points[labels[i]].append(drum_points[i])
    
    drum_polygons = {}
    
    for drum in grouped_drum_points:
        points = grouped_drum_points[drum]
        hull = ConvexHull(grouped_drum_points[drum])
        drum_polygons[drum] = Polygon([points[v] for v in hull.vertices])
    
    total_hull = ConvexHull(drum_points)
    total_polygon = Polygon([drum_points[v] for v in total_hull.vertices])
    p = [Point(c) for c in total_polygon.exterior.coords]
    
    calc_roundness = lambda poly : 4 * np.pi * poly.area / (poly.length**2)
    
    #measure of "roundness" of each polygon based on Polsby-Popper Test
    roundness = [calc_roundness(drum_polygons[d]) for d in drum_polygons]
    roundness_np = np.asarray(roundness)
    roundness_mean = np.mean(roundness_np)

   #the polygons themselves for further processing 
    polygons = drum_polygons.values() + [total_polygon]

    #the area of the union of all cluser polygons over the area of the total plot
    overlap_ratio = reduce(lambda a,b: a.union(b), polygons[1:], polygons[0]).area / sum([p.area for p in polygons])
    
    returnVal = {}
    returnVal['roundness'] = roundness
    returnVal['roundness_mean'] = roundness_mean
    returnVal['overlap_ratio'] = overlap_ratio
    return returnVal

# The combined scoring function
def evalMetrics(data, cluster_labels):
    silhouette_score = computeSilCoeff(data, cluster_labels)
    geometry_dict = clusterGeometryMetrics(data, cluster_labels)
    final_score = silhouette_score + geometry_dict['roundness_mean'] + geometry_dict['overlap_ratio']
    return (silhouette_score,geometry_dict['roundness_mean'],geometry_dict['overlap_ratio'],final_score)
