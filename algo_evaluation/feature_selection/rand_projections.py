from time import time
import numpy as np
import pandas as pd
from sklearn import random_projection


def project_features(data, n_components, display=False):
    features, weights, labels = data
    feature_names = features.columns.tolist()
    start = time()
    rp = random_projection.SparseRandomProjection(n_components=n_components)
    rp.fit(features)
    rp.transform(features)
    return rp


def transform(data, n_components=3):
    features, weights, labels = data
    start = time()
    rp = random_projection.SparseRandomProjection(n_components=n_components)
    rp.fit(features)
    transformed = rp.transform(features)
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
    estimator = random_projection.SparseRandomProjection()
    scores = []
    for n in range(1, n_components):
        estimator.n_components = n
        score = reconstruction_error(estimator, features)
        scores.append([n, score])
    df = pd.DataFrame.from_records(scores, columns=['components', 'reconstruction_error'])
    df['algo'] = 'rand_projections'
    return df