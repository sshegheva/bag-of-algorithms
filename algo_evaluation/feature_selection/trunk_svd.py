"""
http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html#sklearn.decomposition.FactorAnalysis
"""
import numpy as np
import pandas as pd
from time import time
from sklearn.decomposition import TruncatedSVD


def transform(data, n_components=3):
    features, weights, labels = data
    start = time()
    ica = TruncatedSVD(n_components=n_components)
    transformed = ica.fit_transform(features)
    elapsed = time() - start
    df = pd.DataFrame(transformed)
    return df, elapsed


def reconstruction_error(estimator, features):
    transformed = estimator.fit_transform(features)
    reconstruct = estimator.inverse_transform(transformed)
    residual = np.linalg.norm(reconstruct - features)
    return np.sqrt(residual)


def estimate_components(data):
    features, weights, labels = data
    n_components = features.shape[1]
    estimator = TruncatedSVD()
    scores = []
    for n in range(1, n_components):
        estimator.n_components = n
        score = reconstruction_error(estimator, features)
        scores.append([n, score])
    df = pd.DataFrame.from_records(scores, columns=['components', 'reconstruction_error'])
    df['algo'] = 'lsa'
    return df
