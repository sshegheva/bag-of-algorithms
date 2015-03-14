
"""
Evaluate K nearest neighbours
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from algo_evaluation.datasets import split_dataset, load_higgs_train



class KNN:
    def __init__(self, data, n_neighbours=3, power_parameter=2):
        features, weights, labels = data
        self.clf = KNeighborsClassifier(n_neighbors=n_neighbours, p=power_parameter)
        self.predictions, self.trnaccuracy, self.tstaccuracy = None, None, None
        self.dataset = split_dataset(features, weights, labels)

    def train(self):
        """
        Train K nearest neighbours on the higgs dataset
        """
        self.clf = self.clf.fit(self.dataset['training']['features'], self.dataset['training']['labels'])

    def predict(self):
        """
        Predict label using KNN
        :return:
        """
        self.predictions = self.clf.predict(self.dataset['test']['features'])

    def evaluate(self):
        self.trnaccuracy = self.clf.score(self.dataset['training']['features'],
                                          self.dataset['training']['labels'],
                                          sample_weight=self.dataset['training']['weights'])
        self.tstaccuracy = self.clf.score(self.dataset['test']['features'],
                                          self.dataset['test']['labels'],
                                          sample_weight=self.dataset['test']['weights'])


def run_knn(data, n_neighbours=50, power_parameter=2):
    """
    Run and evaluate KNN with default settings
    """
    dt = KNN(data=data, n_neighbours=n_neighbours, power_parameter=power_parameter)
    dt.train()
    dt.predict()
    dt.evaluate()
    return dt.trnaccuracy, dt.tstaccuracy


def estimate_best_n_neighbours():
    """
    Run KNN classifier with multiple settings of
    n_neighbours and plot the accuracy function of n_neighbours
    :return: the best n_neighbours setting
    """
    n_neighbours_range = np.arange(1, 26, 2)
    data = load_higgs_train()
    records = [[n_neighbours] + list(run_knn(data=data, n_neighbours=n_neighbours))
               for n_neighbours in n_neighbours_range]
    columns = ['n_neighbours', 'training_score', 'test_score']
    df = pd.DataFrame.from_records(records, columns=columns, index=columns[0])
    return df


def estimate_best_power():
    """
    Run KNN classifier with multiple settings of
    power parameter for distance metric
    """
    p_range = {1: 'manhattan', 2: 'euclidean', 3: 'minkowski'}
    data = load_higgs_train()
    records = [[p_range[p]] + list(run_knn(data=data, power_parameter=p))
               for p in p_range]
    columns = ['metric', 'training_score', 'test_score']
    df = pd.DataFrame.from_records(records, columns=columns, index=columns[0])
    return df


def plot_accuracy_function(neighbour_df, p_df, smoothing_factor=5):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharex=False, sharey=False)
    pd.rolling_mean(neighbour_df, smoothing_factor).plot(ax=axes[0],
                                                         title='Accuracy f(n_neighbours)')
    p_df.plot(ax=axes[1], kind='barh', title='Accuracy f(metric)')



