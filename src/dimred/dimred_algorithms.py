import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from src.common.fun_call import FunCallFactory, parse_funcall, FunCall, parse_funcalls


def audioviz_pca(data, **kwargs):
    if len(data.shape) > 2:
        data = data.reshape(len(data), -1)
    reduced_data = PCA(**kwargs).fit_transform(data)
    scaler = MinMaxScaler()
    scaler.fit(reduced_data)
    return scaler.transform(reduced_data)


def audioviz_umap(data, **kwargs):
    if len(data.shape) > 2:
        data = data.reshape(len(data), -1)
    reduced_data = umap.UMAP(**kwargs).fit_transform(data)
    scaler = MinMaxScaler()
    scaler.fit(reduced_data)
    return scaler.transform(reduced_data)


class DimRedAlgorithmFactory(FunCallFactory):
    _implemented = {"pca": audioviz_pca, "umap": audioviz_umap}


def parse_dimred_algorithm(algorithm: str) -> FunCall:
    return parse_funcall(algorithm, DimRedAlgorithmFactory)


def parse_dimred_algorithms(dimred_algorithms):
    return parse_funcalls(dimred_algorithms, DimRedAlgorithmFactory)
