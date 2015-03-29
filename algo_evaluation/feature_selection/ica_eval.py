"""
http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from sklearn.decomposition import FastICA
import seaborn as sns


def transform(data, n_components=3):
    features, weights, labels = data
    start = time()
    ica = FastICA(n_components=n_components)
    transformed = ica.fit_transform(features)
    elapsed = time() - start
    df = pd.DataFrame(transformed)
    return df, elapsed


def mixing_matrix(data, n_components, display=True):
    features, weights, labels = data
    ica = FastICA(n_components=n_components)
    ica.fit_transform(features)
    mixing = ica.mixing_
    if display:
        f, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(mixing)
        plt.title('Signal Mixing Estimated Matrix')
    return mixing

def reconstruction_error(estimator, features):
    transformed = estimator.fit_transform(features)
    reconstruct = estimator.inverse_transform(transformed)
    residual = np.linalg.norm(reconstruct - features)
    return np.sqrt(residual)


def estimate_components(data):
    features, weights, labels = data
    n_components = features.shape[1]
    estimator = FastICA(max_iter=500)
    scores = []
    for n in range(1, n_components):
        estimator.n_components = n
        score = reconstruction_error(estimator, features)
        scores.append([n, score])
    df = pd.DataFrame.from_records(scores, columns=['components', 'reconstruction_error'])
    df['algo'] = 'fast_ica'
    return df

