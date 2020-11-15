import click
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from src.common.fun_call import parse_funcall, FunCallFactory
from src.features.build_features import build_features


def pca(data):
    reduced_data = PCA(n_components=2).fit_transform(data)
    scaler = MinMaxScaler()
    scaler.fit(reduced_data)
    return scaler.transform(reduced_data)


class DimRedAlgorithmFactory(FunCallFactory):
    _implemented = {"pca": pca}


@click.command()
@click.argument("dataset", default="medley-solos-db", type=str)
@click.argument("dimred_algorithm", nargs=1, type=str)
@click.argument("features", nargs=-1, type=str)  # features along with kwargs
@click.option("-scoring_alg", type=str)  # if -score
@click.option("-output_path", default=None, type=click.Path())
@click.option("-storage_type", default="h5", type=str)
def main(dataset, dimred_algorithm, features, scoring_alg, output_path, storage_type):
    # TODO remove these (debugging)
    features = ['{"name":"stft", "args":{}}']
    dimred_algorithm = '{"name":"pca", "args":{}}'

    dimred_alg = parse_funcall(dimred_algorithm, DimRedAlgorithmFactory)
    feature_collection = build_features(dataset, features, storage_type)
    print(feature_collection._features)
    with feature_collection._store:
        nd = feature_collection.get_as_dataset()
        reduced = dimred_alg(nd)
    print(reduced)


if __name__ == "__main__":
    main()
