import matplotlib.pyplot as plt
import numpy as np
from sklearn.learning_curve import learning_curve

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


def plot_learning_curve(estimator, title, X, y, axes, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.title(title, axes=axes)
    if ylim is not None:
        plt.ylim(*ylim, axes=axes)
    plt.xlabel("Training examples", axes=axes)
    plt.ylabel("Score", axes=axes)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r", axes=axes)
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g", axes=axes)
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score", axes=axes)
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score", axes=axes)

    plt.legend(loc="best", axes=axes)
    return plt

