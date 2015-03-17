"""
http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#example-ensemble-plot-forest-importances-py
"""
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier


def rank_features(data, n_estimators=250):
    features, weights, labels = data
    feature_names = features.columns.tolist()
    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=n_estimators,
                                  random_state=0)

    forest.fit(features, labels)
    importances = forest.feature_importances_
    importances = zip(feature_names, importances)
    df = pd.DataFrame(importances, columns=['feature', 'rank']).set_index('feature').sort('rank', ascending=False)
    return df
