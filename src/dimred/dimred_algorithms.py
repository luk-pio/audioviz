from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from src.common.fun_call import FunCallFactory, parse_funcall, FunCall


def pca(data):
    if len(data.shape) > 2:
        data = data.reshape(len(data), -1)
    reduced_data = PCA(n_components=2).fit_transform(data)
    scaler = MinMaxScaler()
    scaler.fit(reduced_data)
    return scaler.transform(reduced_data)


class DimRedAlgorithmFactory(FunCallFactory):
    _implemented = {"pca": pca}


def parse_dimred_algorithm(algorithm: str) -> FunCall:
    return parse_funcall(algorithm, DimRedAlgorithmFactory)
