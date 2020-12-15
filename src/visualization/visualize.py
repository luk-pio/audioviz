from typing import Dict, List

from matplotlib import pyplot as plt

from src.common.audioviz_dataset import AudiovizDataset


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
