import os
import itertools
import pickle

import umap
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from utils import CLASSNAMES, CLASS_COLORS, SAMPLE_DIR, FEATURE_DIR
from scoring import evalMetrics

samples = {}
for d in CLASSNAMES:
    samples[d] = np.load(os.path.join(SAMPLE_DIR, d + '_samples.npy'))


def plot_samples(to_plot, name, color):
    _, ax = plt.subplots(1, len(to_plot), sharey=True, figsize=(50, 10))
    for i, axis in enumerate(ax):
        axis.plot(to_plot[i], c=color)
        if i == 0:
            axis.set_title(name + "\n", fontsize=50)


# Define some pretty printing
print('class_name\tdimensions')
print('_' * 30)
for classname in CLASSNAMES:
    print('%-9s\t%s' % (classname, samples[classname].shape))

# Visualize a few of the samples

# Plot the samples
# for classname, color in zip(CLASSNAMES, CLASS_COLORS):
#     plot_samples(samples[classname][:5], classname, color)


# Load all the precomputed features
def load_features(feature_name):
    features = []
    for classname in CLASSNAMES:
        file_path = os.path.join(
            FEATURE_DIR, '{}_{}.npy'.format(
                classname, feature_name))
        feature = np.load(file_path)
        features.extend(feature)
    features = np.asarray(features)
    print('{}\t{}'.format(feature_name, features.shape))
    features = features.reshape(len(features), -1)
    return features


def load_all_features(feature_dict):
    x_data_features = {}
    for feature_name in feature_dict:
        features = load_features(feature_name)
        x_data_features[feature_name] = features
    return x_data_features


# Define some pretty printing
print('feature_name\t original dimensions')
print('_' * 35)

# Load all the features
feature_list = ['stft', 'mfcc']
x_data_dict = load_all_features(feature_list)

print('feature_name\t flat dimensions')
print('_' * 35)
for feature_name in feature_list:
    print('{}\t{}'.format(feature_name, x_data_dict[feature_name].shape))

# Creating the ground truth labels

ground_labels = []
classnumbers = []
for i, classname in enumerate(CLASSNAMES):
    feature_file = np.load(os.path.join(FEATURE_DIR, classname + '_stft.npy'))
    no_samples = feature_file.shape[0]
    # Map how many of each sample in class
    # Classes are mapped classlabel = list index no
    ground_labels.extend([i] * no_samples)
    classnumbers.append(no_samples)
y_data_labels = np.asarray(ground_labels)

# print("y_data_labels.shape:", y_data_labels.shape)
# plt.figure()
# plt.plot(y_data_labels)

# Ground truth color mapping


def concatColors(segmentList, colorList):
    multiples = []
    for i in range(len(segmentList)):
        multiples.append([colorList[i]] * segmentList[i])
    return list(itertools.chain(*multiples))


colorMap = concatColors(classnumbers, CLASS_COLORS)

# Defining metrics for evaluation of the 2D viz

# Plotting the PCA


def returnPCA(data):
    reduced_data = PCA(n_components=2).fit_transform(data)
    scaler = MinMaxScaler()
    scaler.fit(reduced_data)
    return scaler.transform(reduced_data)


def plot2D(reduced_data, colorMap, dim_red_type, feature):
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=colorMap, s=2)
    plt.title(dim_red_type + " on " + feature)
    plt.show()


def printMetrics(scores):
    print('metric\t\t score')
    print('_' * 35)
    print('%-9s\t%s' % ('Silhouette', scores[0]))
    print('%-9s\t%s' % ('Roundess', scores[1]))
    print('%-9s\t%s' % ('Overlap', scores[2]))
    print('-' * 35)
    print('%-9s\t%s' % ('Final Score', scores[3]))


# Return all PCA reduced data for each feature and store in a dictionary
pca_data_dict = {}
feature_list = ['stft', 'mfcc']
for feature_name in feature_list:
    pca_data_dict[feature_name] = returnPCA(x_data_dict[feature_name])

# Return all metrics for each PCA+feature combination and store in a dictionary
pca_scores_dict = {}
for feature_name in feature_list:
    pca_scores_dict[feature_name] = evalMetrics(
        pca_data_dict[feature_name], y_data_labels)

plot2D(pca_data_dict['stft'], colorMap, 'PCA', 'STFT')
printMetrics(pca_scores_dict['stft'])

plot2D(pca_data_dict['mfcc'], colorMap, 'PCA', 'MFCC')
printMetrics(pca_scores_dict['mfcc'])


# UMAP 2D

def get_scaled_umap_embeddings(features, neighbour, distance):

    embedding = umap.UMAP(n_neighbors=neighbour,
                          min_dist=distance,
                          metric='correlation').fit_transform(features)
    scaler = MinMaxScaler()
    scaler.fit(embedding)
    return scaler.transform(embedding)


def gridSearchUMAP(x_data):
    umap_embeddings = []
    neighbours = [5, 10, 15, 30, 50]
    distances = [0.000, 0.001, 0.01, 0.1, 0.5]
    for i, neighbour in enumerate(neighbours):
        for j, distance in enumerate(distances):
            print("neighbour: ", neighbour, "distance", distance)
            umap_embedding = get_scaled_umap_embeddings(
                x_data, neighbour, distance)
            umap_embeddings.append(umap_embedding)
    return umap_embeddings


def gridSearchUMAPallFeatures(x_data_dict):
    umap_data_dict = {}
    feature_list = ['stft', 'mfcc']
    for feature_name in feature_list:
        umap_data_dict[feature_name] = gridSearchUMAP(
            x_data_dict[feature_name])
    return umap_data_dict

# umap_data_dict = gridSearchUMAPallFeatures(x_data_dict)
# with open('./artifacts/umap_map_all_features.pkl','wb+') as f:
#     pickle.dump(umap_data_dict, f)


umap_map_dict = pickle.load(
    open('./artifacts/umap_map_all_features.pkl', 'rb'))


def plotUMAPmap(embedding, labels):
    neighbours = [5, 10, 15, 30, 50]
    distances = [0.001, 0.01, 0.1, 0.5]
    fig, ax = plt.subplots(nrows=len(neighbours),
                           ncols=len(distances),
                           figsize=(30, 30))

    for i, row in enumerate(ax):
        for j, col in enumerate(row):
            current_plot = i * len(distances) + j
            col.scatter(embedding[current_plot].T[0],
                        embedding[current_plot].T[1],
                        c=colorMap,
                        s=1)
            (s_score, r_score, o_score, f_score) = evalMetrics(
                embedding[current_plot], labels)
            col.set_title("neighbours: " +
                          str(neighbours[i]) +
                          " distances: " +
                          str(distances[j]) +
                          '\n' +
                          "final score: " +
                          str(round(f_score, 5)), fontdict={'fontsize': 15})
    plt.show()


plotUMAPmap(umap_map_dict['stft'], y_data_labels)

plotUMAPmap(umap_map_dict['mfcc'], y_data_labels)
