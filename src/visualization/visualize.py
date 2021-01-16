from matplotlib import pyplot as plt

from src.common.audioviz_datastore import AbstractAudiovizDataStore
from src.common.evaluation import normalize_scores, format_metrics, get_scores


def plot_samples(to_plot, name, color):
    _, ax = plt.subplots(1, len(to_plot), sharey=True, figsize=(50, 10))
    for i, axis in enumerate(ax):
        axis.plot(to_plot[i], c=color)
        if i == 0:
            axis.set_title(name + "\n", fontsize=50)


def plot2D(reduced_data, colorMap, dim_red_type, feature):
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=colorMap, s=2)
    plt.title(dim_red_type + " on " + feature)
    plt.show()


def plot_grid_search(
    feature_collection,
    dataset,
    dimred_collection: AbstractAudiovizDataStore,
    dimred_algs,
    plot_row_param,
    plot_col_param,
    rows,
    title="",
    scores=None,
):
    with feature_collection._store, dataset._store, dimred_collection:
        color_map = [dataset.colors[t] for t in dataset.target[rows]]
        scores = (
            get_scores(dimred_collection, dimred_algs) if scores is None else scores
        )
        fig, ax = plt.subplots(
            nrows=len(plot_row_param), ncols=len(plot_col_param), figsize=(30, 45)
        )
        fig.suptitle(title, fontsize=30)

        for i, row in enumerate(ax):
            for j, col in enumerate(row):
                current_plot = i * len(plot_col_param) + j
                current_reduced = dimred_collection[
                    dimred_algs[current_plot].__repr__()
                ]
                col.scatter(
                    current_reduced[:, 0], current_reduced[:, 1], c=color_map, s=1
                )
                col.set_title(
                    f"neighbors: {plot_row_param[i]}, distances: "
                    f"{plot_col_param[i]}\n{format_metrics(scores[current_plot])}",
                    fontdict={"fontsize": 15},
                )
        plt.subplots_adjust(hspace=0.5)
        plt.show()
        total_scores = [s["Total"] for s in scores]
        max_score = max(total_scores)
        max_index = total_scores.index(max_score)
        plot2D(
            dimred_collection[dimred_algs[max_index].__repr__()],
            color_map,
            str(dimred_algs[max_index]),
            str(feature_collection),
        )
        print(format_metrics(scores[max_index]))
