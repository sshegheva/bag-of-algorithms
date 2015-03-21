"""
http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
"""
from time import time
import pandas as pd
from sklearn.decomposition import PCA


def rank_features(data, n_components, display=False):
    features, weights, labels = data
    feature_names = features.columns.tolist()
    start = time()
    pca = PCA(n_components=n_components)
    pca.fit(features)
    elapsed = time() - start
    variances = pca.explained_variance_
    variances = zip(feature_names, variances)
    df = pd.DataFrame(variances, columns=['feature', 'variance']).set_index('feature').sort('variance', ascending=False)
    df['time'] = elapsed
    df['algo'] = 'pca'
    if display:
        plot_rank(df)
    return df


def plot_rank(df):
    df['variance'].plot(kind='bar', title='Higgs Feature Rank (largest variance)', figsize=(10, 4))


def transform(data, n_components=3):
    features, weights, labels = data
    feature_names = features.columns.tolist()
    start = time()
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(features)
    elapsed = time() - start
    variances = pca.explained_variance_
    variances = sorted(zip(feature_names, variances), key=lambda x: x[1])
    df = pd.DataFrame(transformed, columns=[f for f, _ in variances[:n_components]])
    df['label'] = labels.values
    return df, elapsed



