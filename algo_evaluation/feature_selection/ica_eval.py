"""
http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from sklearn.decomposition import FastICA
import seaborn as sns
from scipy.stats import kurtosis


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
        f, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(mixing)
        plt.title('Signal Mixing Estimated Matrix')
    return mixing

def reconstruction_error(estimator, features):
    transformed = estimator.fit_transform(features)
    reconstruct = estimator.inverse_transform(transformed)
    residual = np.linalg.norm(reconstruct - features)
    return np.sqrt(residual)


def estimate_components(data, display=True):
    features, weights, labels = data
    feature_names = features.columns.tolist()
    data = [kurtosis(features[f]) for f in feature_names]
    df = pd.Series(data, index=feature_names).order(ascending=False)
    if display:
        df.plot(kind='bar', title='ICA Component Selection (kurtosis)', figsize=(12, 4))
    return df

