"""
http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html
"""
import numpy as np
import pandas as pd
from time import time
from sklearn.decomposition import FastICA
from sklearn.cross_validation import cross_val_score


def transform(data, n_components=3):
    features, weights, labels = data
    start = time()
    ica = FastICA(n_components=n_components)
    transformed = ica.fit_transform(features)
    elapsed = time() - start
    df = pd.DataFrame(transformed)
    return df, elapsed


def estimate_components(data):
    features, weights, labels = data
    n_components = features.shape[1]
    estimator = FastICA()
    pca_scores = []
    for n in range(n_components):
        estimator.n_components = n
        score = np.mean(cross_val_score(estimator, features))
        pca_scores.append([n, score])
    df = pd.DataFrame.from_records(pca_scores, columns=['components', 'score'])
    df['algo'] = 'fast_ica'
    return df

