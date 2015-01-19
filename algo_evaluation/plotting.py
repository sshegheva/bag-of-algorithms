import matplotlib.pyplot as plt
import numpy as np

n_classes = 2
plot_colors = "br"
plot_step = 0.02


def plot_training_points(decision_tree_classifier, feature_group, plot_number):
    plt.figure(figsize=(20, 20))
    nrows = len(decision_tree_classifier.feature_names)
    ncols = 1
    plt.subplot(nrows, ncols, plot_number)
    feature_1, feature_2 = feature_group
    group_1_ix = decision_tree_classifier.feature_names.index(feature_1)
    group_2_ix = decision_tree_classifier.feature_names.index(feature_2)
    X = decision_tree_classifier.features_train[:, [group_1_ix, group_2_ix]]
    Y = decision_tree_classifier.labels_train
    for i, color in zip(set(Y), plot_colors):
        idx = np.where(Y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=i, cmap=plt.cm.Paired)

    plt.xlabel(feature_1)
    plt.ylabel(feature_2)
    plt.axis("tight")
    plt.legend()
