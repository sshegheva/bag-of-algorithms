from time import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import random_projection
from algo_evaluation.datasets import split_dataset
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.svm import LinearSVC


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


def estimate_components(data, iterations=10):
    features, weights, labels = data
    n_components = features.shape[1]
    baseline_estimator = LinearSVC()
    dataset = split_dataset(features, weights, labels)
    train_features = dataset['training']['features']
    test_features = dataset['test']['features']
    start = time()
    baseline_estimator.fit(train_features, dataset['training']['labels'])
    elapsed = time() - start
    baseline_accuracy = baseline_estimator.score(test_features, dataset['test']['labels'],
                                sample_weight=dataset['test']['weights'])
    scores = []
    baseline_record = [[n, 0, baseline_accuracy, elapsed] for n in range(1, n_components)]
    [scores.append(b) for b in baseline_record]
    for component in range(1, n_components):
        estimator = random_projection.SparseRandomProjection()
        estimator.n_components = component
        start = time()
        for iter in range(1, iterations):
            transformed_train_features = estimator.fit_transform(train_features)
            transformed_test_features = estimator.transform(test_features)
            baseline_estimator.fit(transformed_train_features, dataset['training']['labels'])
            accuracy = baseline_estimator.score(transformed_test_features, dataset['test']['labels'],
                                    sample_weight=dataset['test']['weights'])
            scores.append([component, iter, accuracy, time() - start])
    df = pd.DataFrame.from_records(scores,
                                   columns=['components', 'iteration', 'classification_accuracy', 'time'])
    return df

def plot_component_estimation(df):
    sns.lmplot("components", "classification_accuracy", hue="iteration", data=df, size=4, order=1)
    plt.title('Randomized Projections Component Estimation')