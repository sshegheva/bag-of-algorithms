"""
http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html
"""
import pandas as pd
from time import time
from sklearn.decomposition import FastICA


def transform(data, n_components=3):
    features, weights, labels = data
    start = time()
    ica = FastICA(n_components=n_components)
    transformed = ica.fit_transform(features)
    elapsed = time() - start
    df = pd.DataFrame(transformed)
    df['label'] = labels.values
    return df, elapsed

