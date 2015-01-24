from algo_evaluation.plotting import plot_learning_curve
from sklearn import cross_validation
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm


def plot_learning_curves(raw_data):

    features, weights, labels = raw_data

    cv = cross_validation.ShuffleSplit(features.shape[0], n_iter=10,
                                       test_size=0.2, random_state=0)

    title = "Learning Curves (Decision Trees)"
    estimator = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=60)
    plot_learning_curve(estimator, title, features, labels, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

    title = "Learning Curves (AdaBoost)"
    estimator = AdaBoostClassifier(n_estimators=100, learning_rate=1.0)
    plot_learning_curve(estimator, title, features, labels, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

    title = "Learning Curves (K-Nearest Neighbour)"
    estimator = KNeighborsClassifier(n_neighbors=3, p=2)
    plot_learning_curve(estimator, title, features, labels, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

    title = "Learning Curves (SVM)"
    estimator = svm.SVC(C=1.0, gamma=0.1)
    plot_learning_curve(estimator, title, features, labels, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

    plt.show()
