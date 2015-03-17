"""
http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

>>> import numpy as np
>>> from sklearn.decomposition import PCA
>>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
>>> pca = PCA(n_components=2)
>>> pca.fit(X)
PCA(copy=True, n_components=2, whiten=False)
>>> print(pca.explained_variance_ratio_)
[ 0.99244...  0.00755...]
"""
import numpy as np
from sklearn.decomposition import PCA


def run_pca(data, n_components):
    features, weights, labels = data
    pca = PCA(n_components=n_components)
    pca.fit(features)
    return pca.score(features)



