from algo_evaluation.plotting import plot_learning_curve
from algo_evaluation import TEST_DATA_SPLIT
from sklearn import cross_validation
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

N_JOBS = 4


def plot_learning_curves(raw_data, limit_size=None):

    features, weights, labels = raw_data

    if limit_size is not None:
        features = features[:limit_size]
        weights = weights[:limit_size]
        labels = labels[:limit_size]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6), sharex=False, sharey=False)

    cv = cross_validation.ShuffleSplit(features.shape[0], n_iter=5,
                                       test_size=TEST_DATA_SPLIT, random_state=0)

    title = "Learning Curves (Decision Trees)"
    estimator = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=60)
    plot_learning_curve(estimator, title, features, labels, axes[0][0], ylim=(0.7, 1.01), cv=cv, n_jobs=N_JOBS)

    title = "Learning Curves (AdaBoost)"
    estimator = AdaBoostClassifier(n_estimators=100, learning_rate=1.0)
    plot_learning_curve(estimator, title, features, labels, axes[1][0], ylim=(0.7, 1.01), cv=cv, n_jobs=N_JOBS)

    title = "Learning Curves (K-Nearest Neighbour)"
    estimator = KNeighborsClassifier(n_neighbors=10, p=2)
    plot_learning_curve(estimator, title, features, labels, axes[0][1], ylim=(0.7, 1.01), cv=cv, n_jobs=N_JOBS)

    title = "Learning Curves (SVM)"
    estimator = svm.SVC(C=1.0, gamma=0.1)
    plot_learning_curve(estimator, title, features, labels, axes[1][1], ylim=(0.7, 1.01), cv=cv, n_jobs=N_JOBS)
