import click

from src.common.audioviz_dataset import load_audioviz_dataset
from src.common.evaluation import eval_metrics, print_metrics
from src.dimred.dimred_algorithms import parse_dimred_algorithm
from src.features.build_features import build_features


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

    dataset = load_audioviz_dataset(dataset)
    dimred_alg = parse_dimred_algorithm(dimred_algorithm)
    feature_collection = build_features(dataset, features, storage_type)
    with feature_collection._store:
        nd = feature_collection.get_as_dataset()
        reduced = dimred_alg(nd)
    metrics = eval_metrics(reduced, dataset.target)
    print_metrics(metrics)
    print(reduced)


if __name__ == "__main__":
    main()
