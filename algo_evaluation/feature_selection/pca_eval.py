"""
http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
"""
from time import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cross_validation import cross_val_score


def rank_features(data, n_components, display=False):
    features, weights, labels = data
    feature_names = features.columns.tolist()
    start = time()
    pca = PCA(n_components=n_components)
    pca.fit(features)
    elapsed = time() - start
    variances = pca.explained_variance_ratio_
    variances = zip(feature_names, variances)
    df = pd.DataFrame(variances, columns=['feature', 'variance']).set_index('feature').sort('variance', ascending=False)
    df['time'] = elapsed
    df['algo'] = 'pca'
    if display:
        plot_rank(df)
    return df


def plot_rank(df):
    df['variance'].plot(kind='bar', title='Higgs Feature Rank (largest variance ratio)', figsize=(10, 4))


def transform(data, n_components=3):
    features, weights, labels = data
    feature_names = features.columns.tolist()
    start = time()
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(features)
    elapsed = time() - start
    variances = pca.explained_variance_ratio_
    variances = sorted(zip(feature_names, variances), key=lambda x: x[1])
    df = pd.DataFrame(transformed, columns=[f for f, _ in variances[:n_components]])
    return df, elapsed


def reconstruction_error(estimator, features):
    transformed = estimator.fit_transform(features)
    reconstruct = estimator.inverse_transform(transformed)
    residual = (reconstruct - features) * (reconstruct - features)
    return residual


def estimate_components(data):
    features, weights, labels = data
    n_components = features.shape[1]
    estimator = PCA()
    scores = []
    for n in range(n_components):
        estimator.n_components = n
        score = np.mean(cross_val_score(estimator, features))
        scores.append([n, score])
    df = pd.DataFrame.from_records(scores, columns=['components', 'score'])
    df['algo'] = 'pca'
    return df



