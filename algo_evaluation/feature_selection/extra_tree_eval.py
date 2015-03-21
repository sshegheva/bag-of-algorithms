"""
http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#example-ensemble-plot-forest-importances-py
"""
from time import time
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier


def rank_features(data, n_estimators=250, display=False):
    features, weights, labels = data
    feature_names = features.columns.tolist()
    # Build a forest and compute the feature importances
    start = time()
    forest = ExtraTreesClassifier(n_estimators=n_estimators,
                                  random_state=0)

    forest.fit(features, labels)
    elapsed = time() - start
    importances = forest.feature_importances_
    importances = zip(feature_names, importances)
    df = pd.DataFrame(importances, columns=['feature', 'rank']).set_index('feature').sort('rank', ascending=False)
    df['time'] = elapsed
    df['algo'] = 'extra_tree'
    if display:
        plot_rank(df)
    return df


def transform(data, n_components=3):
    features, weights, labels = data
    feature_names = features.columns.tolist()
    start = time()
    forest = ExtraTreesClassifier(random_state=0)
    forest.fit(features, labels)
    elapsed = time() - start
    importances = forest.feature_importances_
    importances = zip(feature_names, importances)
    importances = sorted(importances, key=lambda x: x[1], reverse=True)
    df = features[[f for f, _ in importances[:n_components]]]
    return df, elapsed

def plot_rank(df):
    df['rank'].plot(kind='bar', title='Higgs Feature Rank (largest information gain)', figsize=(10, 4))