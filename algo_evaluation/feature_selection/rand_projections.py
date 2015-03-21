"""
http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.RandomizedPCA.html
"""
from time import time
import numpy as np
import pandas as pd
from sklearn import random_projection
from sklearn.cross_validation import cross_val_score


def rank_features(data, n_components, display=False):
    features, weights, labels = data
    feature_names = features.columns.tolist()
    start = time()
    rp = random_projection.SparseRandomProjection(n_components=n_components)
    rp.fit(features)
    elapsed = time() - start
    variances = rp.explained_variance_
    variances = zip(feature_names, variances)
    df = pd.DataFrame(variances, columns=['feature', 'variance']).set_index('feature').sort('variance', ascending=False)
    df['time'] = elapsed
    df['algo'] = 'pca_rand'
    if display:
        df.plot(kind='bar', title='Higgs Feature Variance', figsize=(10, 4))
    return df


def transform(data, n_components=3):
    features, weights, labels = data
    start = time()
    rp = random_projection.SparseRandomProjection(n_components=n_components)
    rp.fit(features)
    transformed = rp.transform(features)
    elapsed = time() - start
    df = pd.DataFrame(transformed)
    return df, elapsed


def estimate_components(data):
    features, weights, labels = data
    n_components = features.shape[1]
    estimator = random_projection.SparseRandomProjection()
    pca_scores = []
    for n in range(n_components):
        estimator.n_components = n
        score = np.mean(cross_val_score(estimator, features))
        pca_scores.append([n, score])
    df = pd.DataFrame.from_records(pca_scores, columns=['components', 'score'])
    df['algo'] = 'rand_projections'
    return df