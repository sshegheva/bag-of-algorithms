from time import time
import numpy as np
import pandas as pd
from sklearn import random_projection
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from algo_evaluation.supervised.decision_tree import DecisionTree


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
    dt = DecisionTree(data, min_samples_split=60)
    train_features = dt.dataset['training']['features']
    test_features = dt.dataset['test']['features']
    dt.clf.fit(train_features, dt.dataset['training']['labels'])
    baseline_accuracy = dt.clf.score(test_features, dt.dataset['test']['labels'],
                                sample_weight=dt.dataset['test']['weights'])
    scores = []
    for component in range(1, n_components):
        estimator = random_projection.SparseRandomProjection()
        estimator.n_components = component
        transformed_train_features = estimator.fit(train_features).transform(train_features)
        transformed_test_features = estimator.transform(test_features)
        dt.clf.fit(transformed_train_features, dt.dataset['training']['labels'])
        accuracy = dt.clf.score(transformed_test_features, dt.dataset['test']['labels'],
                                sample_weight=dt.dataset['test']['weights'])
        error = 1 - accuracy
        scores.append([component, accuracy, error])
    df = pd.DataFrame.from_records(scores,
                                   columns=['components', 'classification_accuracy', 'classification_error'])
    df['baseline'] = 1 - baseline_accuracy
    return df