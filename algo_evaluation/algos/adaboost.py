
"""
Evaluate Ada Boost classifier
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from algo_evaluation.datasets import split_dataset, load_higgs_train
from algo_evaluation.parameter_search import grid_search_best_parameter


class AdaBoost:
    def __init__(self, data, n_estimators=100, learning_rate=1.0):
        features, weights, labels = data
        self.clf = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
        self.predictions, self.trnaccuracy, self.tstaccuracy = None, None, None
        self.dataset = split_dataset(features, weights, labels)

    def train(self):
        """
        Train Ada Boost on the higgs dataset
        """
        self.clf = self.clf.fit(self.dataset['training']['features'], self.dataset['training']['labels'])

    def predict(self):
        """
        Predict label using Ada Boost
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


def run_AdaBoost(data, n_estimators=100, learning_rate=1.0):
    """
    Run and evaluate Ada Boost with default settings
    """
    dt = AdaBoost(data=data, n_estimators=n_estimators, learning_rate=learning_rate)
    dt.train()
    dt.predict()
    dt.evaluate()
    return dt.trnaccuracy, dt.tstaccuracy


def estimate_best_n_estimators():
    """
    Run Ada Boost classifier with multiple settings of
    n_estimators and plot the accuracy function of n_estimators
    :return: the best n_estimators setting
    """
    n_estimators_range = np.arange(30, 120, 5)
    data = load_higgs_train()
    records = [[n_estimator] + list(run_AdaBoost(data=data, n_estimators=n_estimator))
               for n_estimator in n_estimators_range]
    columns = ['n_estimators', 'training_score', 'test_score']
    df = pd.DataFrame.from_records(records, columns=columns, index=columns[0])
    return df


def estimate_best_learning_rate():
    """
    Run Ada Boost classifier with multiple settings of
    learning_rate and plot the accuracy function of learning rate
    :return: the best learning rate setting
    """
    learning_rate_range = np.arange(0.2, 2.0, 0.2)
    data = load_higgs_train()
    records = [[rate] + list(run_AdaBoost(data=data, learning_rate=rate))
               for rate in learning_rate_range]
    columns = ['learning_rate', 'training_score', 'test_score']
    df = pd.DataFrame.from_records(records, columns=columns, index=columns[0])
    return df


def plot_accuracy_functions(estimators_df, learning_rate_df, smoothing_factor=5):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharex=False, sharey=True)
    pd.rolling_mean(estimators_df, smoothing_factor).plot(ax=axes[0],
                                                          title='Accuracy f(n_estimators)')
    pd.rolling_mean(learning_rate_df, smoothing_factor).plot(ax=axes[1],
                                                          title='Accuracy f(learning_rate)')


def plot_accuracy_function(df, title, smoothing_factor=5):
    smooth_df = pd.rolling_mean(df, smoothing_factor)
    smooth_df.plot(title=title)


def grid_search_tradeoff_estimators_learning_rate(raw_data):
    features, weights, labels = raw_data
    dataset = split_dataset(features, weights, labels)
    dataset['training']['labels'] = [1 if l == 's' else 0 for l in dataset['training']['labels']]
    dataset['test']['labels'] = [1 if l == 's' else 0 for l in dataset['test']['labels']]
    tunning_parameters = {'n_estimators': np.arange(50, 100, 5),
                          'learning_rate': np.arange(0.2, 2.2, .2)}
    scores = ['precision', 'recall']
    report = grid_search_best_parameter(dataset, AdaBoostClassifier, tunning_parameters, scores=scores)
    return report


def tradeoff_estimators_learning_rate(raw_data):
    tunning_parameters = {'n_estimators': np.arange(50, 100, 5),
                          'learning_rate': np.arange(0.2, 2.2, .2)}
    data = []
    for n in tunning_parameters['n_estimators']:
        for l in tunning_parameters['learning_rate']:
            #LOGGER.debug('Running ada boost with %s n_estimators and %s learning rate', n, l)
            trn_acc, tst_acc = run_AdaBoost(raw_data, n_estimators=n, learning_rate=l)
            record = [n, l, trn_acc, tst_acc]
            data.append(record)
    df = pd.DataFrame.from_records(data,
                                   columns=['n_estimators', 'learning_rate', 'training_accuracy', 'test_accuracy'])
    return df


def plot_tradeoff(trades_df):
    plt.figure(figsize=(6, 4))
    plt.pcolor(trades_df[['n_estimators', 'learning_rate', 'test_accuracy']])
    plt.title('Accuracy trade off as a result of tuning \n n_estimators and learning rate')
    plt.xlabel('Learning Rate')
    plt.ylabel('N estimators')
