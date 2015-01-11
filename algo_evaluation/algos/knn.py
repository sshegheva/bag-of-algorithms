
"""
Evaluate K nearest neighbours
"""
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from algo_evaluation.datasets import split_dataset, load_higgs_train
from algo_evaluation import LOGGER


class KNN:
    def __init__(self, data, n_neighbours=3):
        features, weights, labels = data
        self.clf = KNeighborsClassifier(n_neighbors=n_neighbours)
        LOGGER.info('Created KNN classifier with %s neighbours', n_neighbours)
        self.predictions, self.trnaccuracy, self.tstaccuracy = None, None, None
        self.dataset = split_dataset(features, weights, labels)

    def train(self):
        """
        Train K nearest neighbours on the higgs dataset
        """
        self.clf = self.clf.fit(self.dataset['training']['features'], self.dataset['training']['labels'])
        LOGGER.info('Trained decision tree classifier')

    def predict(self):
        """
        Predict label using KNN
        :return:
        """
        self.predictions = self.clf.predict(self.dataset['test']['features'])
        LOGGER.info('Generated predictions')

    def evaluate(self):
        self.trnaccuracy = self.clf.score(self.dataset['training']['features'],
                                          self.dataset['training']['labels'],
                                          sample_weight=self.dataset['training']['weights'])
        self.tstaccuracy = self.clf.score(self.dataset['test']['features'],
                                          self.dataset['test']['labels'],
                                          sample_weight=self.dataset['test']['weights'])
        LOGGER.info('Training Weighted Accuracy score = %s', self.trnaccuracy)
        LOGGER.info('Test Weighted Accuracy score = %s', self.tstaccuracy)


def run_knn(data, n_neighbours):
    """
    Run and evaluate KNN with default settings
    """
    dt = KNN(data=data, n_neighbours=n_neighbours)
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
    n_neighbours_range = np.arange(1.0, 20.0, 2.0)
    data = load_higgs_train()
    records = [[n_neighbours] + list(run_knn(data=data, n_neighbours=n_neighbours))
               for n_neighbours in n_neighbours_range]
    LOGGER.info('Performed evaluation of the n_neighbours setting choice')
    columns = ['n_neighbours', 'training_score', 'test_score']
    df = pd.DataFrame.from_records(records, columns=columns, index=columns[0])
    LOGGER.info(df)
    return df


def plot_accuracy_function(df):
    smooth_df = pd.rolling_mean(df, 5)
    smooth_df.plot(title='Accuracy change as a function of n_neighbours (smoothed)')



