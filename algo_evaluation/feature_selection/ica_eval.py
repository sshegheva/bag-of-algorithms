"""
http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html
"""

import numpy as np
from sklearn.decomposition import FastICA


def run_pca(data, n_components):
    features, weights, labels = data
    ica = FastICA(n_components=n_components)
    ica.fit(features)
    return ica

