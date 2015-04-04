"""
http://scikit-learn.org/0.11/auto_examples/mixture/plot_gmm_classifier.html#example-mixture-plot-gmm-classifier-py
"""
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import mixture
from sklearn.cross_validation import cross_val_score

higgs_estimators = {'gmm_higgs_2': mixture.GMM(n_components=2),
                    'gmm_higgs_8': mixture.GMM(n_components=8)}

bid_estimators = {'gmm_converters_2': mixture.GMM(n_components=2),
                  'gmm_converters_3': mixture.GMM(n_components=3),
                  'gmm_converters_8': mixture.GMM(n_components=8)}


def evaluate_gmm_generic(estimator, data, metric):
    t0 = time()
    features, _, labels = data
    estimator.fit(features)
    elapsed = time() - t0
    return metric(labels, estimator.predict(features)), elapsed


def evaluate_gmm(data, estimators):
    records = []
    for name, estimator in estimators.items():
        score, elapsed_time = evaluate_gmm_generic(estimator=estimator,
                                                       data=data,
                                                       metric=metrics.v_measure_score)
        records.append([name, score, elapsed_time])
    df = pd.DataFrame.from_records(records, columns=['estimator', 'v-measure', 'time'])
    return df


def estimate_clusters(data):
    features, _, _ = data
    scores = []
    estimator = mixture.GMM()
    n_clusters = features.shape[1]
    for n in range(1, n_clusters):
        estimator.n_components = n
        estimator.fit(features)
        score = np.mean(estimator.score(features))
        scores.append([n, score])
    df = pd.DataFrame.from_records(scores, columns=['clusters', 'score'])
    df['algo'] = 'gmm'
    return df