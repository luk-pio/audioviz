import umap
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from src.common.fun_call import FunCallFactory, parse_funcall, FunCall, parse_funcalls


def normalize(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    return scaler.transform(data)


def audioviz_pca(data, **kwargs):
    if len(data.shape) > 2:
        data = data.reshape(len(data), -1)
    reduced_data = PCA(**kwargs).fit_transform(data)
    return normalize(reduced_data)


def audioviz_umap(data, **kwargs):
    if len(data.shape) > 2:
        data = data.reshape(len(data), -1)
    reduced_data = umap.UMAP(**kwargs).fit_transform(data)
    return normalize(reduced_data)


def stat_shorten(data):
    """Takes mean, std dev and mean first-order difference for features in each row"""
    # If the feature has been saved as a (n,1,m, ...) shaped ndarray
    if len(data.shape) > 2 and data.shape[1] == 1:
        data = np.reshape(data, (data.shape[0], *data.shape[2:]))
    dims = len(data.shape)
    stddev_features = np.std(data, axis=dims - 1)
    mean_features = np.mean(data, axis=dims - 1)
    avg_diff = np.mean(np.diff(data), axis=dims - 1)
    concat_stat_features = np.stack(
        (stddev_features, mean_features, avg_diff), axis=dims - 1
    )
    shape = concat_stat_features.shape
    # MinMaxScaler doesn't work with dims > 2
    if len(shape) > 2:
        concat_stat_features = np.reshape(
            concat_stat_features, (shape[0] * shape[1], *shape[2:])
        )
    normalized = normalize(concat_stat_features)
    return np.reshape(normalized, shape)


def stat_shorten_vertical(data):
    """Works the same as stat_shorten but transposes feature vect for each row first"""
    if len(data.shape) > 2 and data.shape[1] == 1:
        data = np.reshape(data, (data.shape[0], *data.shape[2:]))
    if len(data.shape) != 3:
        raise ValueError(f"Shape must be 3 dimensional, was data.shape={data.shape}")
    return stat_shorten(np.transpose(data, (0, 2, 1)))


class DimredAlgorithmFactory(FunCallFactory):
    _implemented = {
        "pca": audioviz_pca,
        "umap": audioviz_umap,
        "stat_shorten": stat_shorten,
        "stat_shorten_vertical": stat_shorten_vertical,
    }


def parse_dimred_algorithm(algorithm: str) -> FunCall:
    return parse_funcall(algorithm, DimredAlgorithmFactory)


def parse_dimred_algorithms(dimred_algorithms):
    return parse_funcalls(dimred_algorithms, DimredAlgorithmFactory)
