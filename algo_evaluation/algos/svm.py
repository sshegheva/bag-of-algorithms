"""
Evaluate Support Vector Machines
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import svm
from algo_evaluation.datasets import split_dataset, load_higgs_train
from algo_evaluation import LOGGER
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report


class SVM:
    def __init__(self, data, regularization_term=1.0, gamma=0.0):
        features, weights, labels = data
        self.gamma = gamma
        self.c = regularization_term
        self.clf = svm.SVC(C=self.c, gamma=self.gamma)
        self.predictions, self.trnaccuracy, self.tstaccuracy = None, None, None
        self.dataset = split_dataset(features, weights, labels)

    def train(self):
        """
        Train support vector machines on the higgs dataset
        :return: classifier
        """
        self.clf = self.clf.fit(self.dataset['training']['features'], self.dataset['training']['labels'])

    def predict(self):
        """
        Predict label using svm
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


def run_svm(data, regularization_term=1.0, gamma=0.0):
    """
    Run and evaluate svm with default settings
    """
    dt = SVM(data=data, regularization_term=regularization_term, gamma=gamma)
    dt.train()
    dt.predict()
    dt.evaluate()
    return dt.trnaccuracy, dt.tstaccuracy


def estimate_best_gamma():
    """
    Run svm classifier with multiple settings of
    gamma and plot the accuracy function of gamma
    :return: the best gamma setting
    """
    gamma_range = np.arange(0.0, 1.0, 0.2)
    data = load_higgs_train()
    records = [[gamma] + list(run_svm(data=data, gamma=gamma))
               for gamma in gamma_range]
    LOGGER.info('Performed evaluation of the gamma setting choice')
    columns = ['gamma', 'training_score', 'test_score']
    df = pd.DataFrame.from_records(records, columns=columns, index=columns[0])
    LOGGER.info(df)
    return df


def estimate_best_c():
    """
    Run svm classifier with multiple settings of
    C and plot the accuracy function of C
    :return: the best C setting
    """
    c_range = [10**n for n in range(4)]
    data = load_higgs_train()
    records = [[c] + list(run_svm(data=data, regularization_term=c))
               for c in c_range]
    LOGGER.info('Performed evaluation of the C setting choice')
    columns = ['C', 'training_score', 'test_score']
    df = pd.DataFrame.from_records(records, columns=columns, index=columns[0])
    LOGGER.info(df)
    return df


def estimate_dataset_size():
    data = load_higgs_train()
    features, weights, labels = data
    records = []
    for n in range(1, 10):
        f = features[:n * len(features)/10]
        w = weights[:n * len(weights)/10]
        l = labels[:n * len(labels)/10]
        records.append([len(f)] + list(run_svm((f, w, l))))
    columns = ['sample_size', 'training_score', 'test_score']
    df = pd.DataFrame.from_records(records, columns=columns, index=columns[0])
    return df

def grid_search_best_parameter(data):
    features, weights, labels = data
    labels = np.array([1 if l == 'b' else 0 for l in labels])
    trnfeatures, tstfeatures, trnweights, tstweights, trnlabels, tstlabels = split_dataset(features, weights, labels)
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']
    reports = {}
    for score in scores:
        LOGGER.info("# Tuning hyper-parameters for %s" % score)
        clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5, scoring=score)
        clf.fit(trnfeatures, trnlabels)

        LOGGER.info("Best parameters set found on development set:")
        LOGGER.info(clf.best_estimator_)
        LOGGER.info("Grid scores on development set:")
        for params, mean_score, scores in clf.grid_scores_:
            LOGGER.info("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params))

        LOGGER.info("Detailed classification report:")
        LOGGER.info("The model is trained on the full development set.")
        LOGGER.info("The scores are computed on the full evaluation set.")
        y_true, y_pred = tstlabels, clf.predict(tstfeatures)
        reports[score] = classification_report(y_true, y_pred)
    return reports


def plot_accuracy_function(size_df, smoothing_factor=5):
    pd.rolling_mean(size_df, smoothing_factor).plot(title='Accuracy f(dataset size) ')




