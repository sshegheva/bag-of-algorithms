"""
http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#example-ensemble-plot-forest-importances-py
"""
from sklearn.ensemble import ExtraTreesClassifier


def test(data, n_estimators=250):
    features, weights, labels = data
    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=n_estimators,
                                  random_state=0)

    forest.fit(features, labels)
    importances = forest.feature_importances_
    return importances
