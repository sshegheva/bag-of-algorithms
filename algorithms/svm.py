"""
Evaluate Support Vector Machines
"""
import random

import numpy as np
import pandas as pd

from sklearn import svm
from datasets import split_dataset, load_higgs_train
from algorithms import LOGGER
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report


class SVM:
    def __init__(self, data, c_error_term=1.0, gamma=0.0):
        features, weights, labels = data
        self.feature_names = features.columns.tolist()
        self.gamma = gamma
        self.c = c_error_term
        self.clf = svm.SVC(C=self.c, gamma=self.gamma)
        LOGGER.info('Created SVM classifier with C = %s and gamma = %s', self.c, self.gamma)
        self.predictions, self.trnaccuracy, self.tstaccuracy = None, None, None
        self.trnfeatures, self.tstfeatures, \
            self.trnweights, self.tstweights, \
            self.trnlabels, self.tstlabels = split_dataset(features, weights, labels)

    def train(self):
        """
        Train support vector machines on the higgs dataset
        :return: classifier
        """
        self.clf = self.clf.fit(self.trnfeatures, self.trnlabels)
        LOGGER.info('Trained decision tree classifier')

    def predict(self):
        """
        Predict label using svm
        :return:
        """
        self.predictions = self.clf.predict(self.tstfeatures)
        LOGGER.info('Generated predictions')

    def evaluate(self):
        self.trnaccuracy = self.clf.score(self.trnfeatures,
                                          self.trnlabels,
                                          sample_weight=self.trnweights)
        self.tstaccuracy = self.clf.score(self.tstfeatures,
                                          self.tstlabels,
                                          sample_weight=self.tstweights)
        LOGGER.info('Training Weighted Accuracy score = %s', self.trnaccuracy)
        LOGGER.info('Test Weighted Accuracy score = %s', self.tstaccuracy)


def run_svm(data, gamma=0.0):
    """
    Run and evaluate svm with default settings
    """
    dt = SVM(data=data, gamma=gamma)
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
    records = [[c] + list(run_svm(data=data, c=c))
               for c in c_range]
    LOGGER.info('Performed evaluation of the C setting choice')
    columns = ['C', 'training_score', 'test_score']
    df = pd.DataFrame.from_records(records, columns=columns, index=columns[0])
    LOGGER.info(df)
    return df


def grid_search_best_parameter(data):
    features, weights, labels = data
    labels = np.array([1 if l == 'b' else 0 for l in labels])
    trnfeatures, tstfeatures, trnweights, tstweights, trnlabels, tstlabels = split_dataset(features, weights, labels)
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']

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
        LOGGER.info(classification_report(y_true, y_pred))


def plot_accuracy_function(df):
    smooth_df = pd.rolling_mean(df, 5)
    smooth_df.plot(title='Accuracy change as a function of gamma (smoothed)')



