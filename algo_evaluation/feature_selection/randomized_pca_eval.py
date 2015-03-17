"""
http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.RandomizedPCA.html
"""
import numpy as np
from sklearn.decomposition import RandomizedPCA


def run_pca(data, n_components):
    features, weights, labels = data
    pca = RandomizedPCA(n_components=n_components)
    pca.fit(features)
    return pca