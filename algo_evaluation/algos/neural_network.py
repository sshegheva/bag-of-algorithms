import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.optimization import HillClimber, StochasticHillClimber
from pybrain.optimization.populationbased.ga import GA
from pybrain.structure.modules import SoftmaxLayer
from pybrain.tools.shortcuts import buildNetwork

from algo_evaluation.datasets import load_higgs_train, split_dataset
from algo_evaluation import LOGGER

np.random.seed(42)


class NeuralNetwork:
    def __init__(self, data, learning_rate=0.1, momentum=0.1, n_hidden_units=2):
        self.features, self.weights, labels = data
        self.labels = np.array([1 if l == 's' else 0 for l in labels])
        self._prepare_data()
        self._build_network(n_hidden_units)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.n_hidden_units = n_hidden_units

    def _prepare_data(self):
        self.dataset = split_dataset(self.features, self.weights, self.labels)
        classes = set(self.labels)

        def training_set():
            ds = ClassificationDataSet(self.dataset['training']['features'].shape[1], 1, nb_classes=len(classes))
            for i in range(self.dataset['training']['features'].shape[0]):
                ds.addSample(self.dataset['training']['features'][i],
                             self.dataset['training']['labels'][i])
            return ds

        def test_set():
            ds = ClassificationDataSet(self.features.shape[1], 1, nb_classes=len(classes))
            for i in range(self.dataset['test']['features'].shape[0]):
                ds.addSample(self.dataset['test']['features'][i],
                             self.dataset['test']['labels'][i])
            return ds

        self.trndata = training_set()
        self.tstdata = test_set()
        self.tstdata._convertToOneOfMany()
        self.trndata._convertToOneOfMany()

    def _build_network(self, n_hidden_units):
        self.fnn = buildNetwork(self.trndata.indim, n_hidden_units, self.trndata.outdim, outclass=SoftmaxLayer)

    def _create_trainer(self, learning_rate, momentum):
        self.trainer = BackpropTrainer(self.fnn, dataset=self.trndata,
                                       momentum=momentum,
                                       verbose=False,
                                       weightdecay=0.01,
                                       learningrate=learning_rate)

    def train(self, train_epoch=5):
        self._create_trainer(self.learning_rate, self.momentum)
        self.trainer.trainEpochs(train_epoch)

    def learn_weights(self, max_evaluations, algoritm):
        alg = algoritm(self.trndata.evaluateModuleMSE, self.fnn, maxEvaluations=max_evaluations, verbose=False)
        alg.minimize = True
        return alg.learn()

    def predict(self, dataset=None):
        if dataset is None:
            dataset = self.tstdata
        out = self.fnn.activateOnDataset(dataset)
        out = out.argmax(axis=1)
        return out

    def estimate_error(self):
        trnerror = percentError(self.trainer.testOnClassData(dataset=self.trndata), self.trndata['class'])
        tsterror = percentError(self.trainer.testOnClassData(dataset=self.tstdata), self.tstdata['class'])
        return self.trainer.totalepochs, trnerror, tsterror

    def train_accuracy(self):
        return accuracy_score(y_pred=self.predict(self.trndata),
                              y_true=self.trndata['class'])

    def test_accuracy(self):
        return accuracy_score(y_pred=self.predict(self.tstdata),
                              y_true=self.tstdata['class'])


def run_neural_net(data, learning_rate=0.1):
    """
    Run and evaluate neural networks
    """
    nn = NeuralNetwork(data, learning_rate=learning_rate)
    nn.train()
    nn.predict()
    return nn.estimate_error()


def evaluate_hill_climbing(data, max_evaluation_range=xrange(1, 100, 10)):
    acc_data = []
    for max_eval in max_evaluation_range:
        start = time.time()
        nn = NeuralNetwork(data=data)
        weights = nn.learn_weights(max_evaluations=max_eval, algoritm=HillClimber)
        #LOGGER.debug('hc %s', weights)
        trnacc = nn.train_accuracy()
        tstacc = nn.test_accuracy()
        elapsed = time.time() - start
        acc_data.append(['hill_climbing', elapsed, max_eval, trnacc, tstacc])
    return pd.DataFrame.from_records(acc_data,
                                     columns=['algo', 'time', 'max_evaluations', 'trnacc', 'tstacc'],
                                     index=['max_evaluations'])


def evaluate_genetic_algorithm(data, max_evaluation_range=xrange(1, 100, 10)):
    acc_data = []
    for max_eval in max_evaluation_range:
        start = time.time()
        nn = NeuralNetwork(data=data)
        weights = nn.learn_weights(max_evaluations=max_eval, algoritm=GA)
        #LOGGER.debug('ga %s', weights)
        trnacc = nn.train_accuracy()
        tstacc = nn.test_accuracy()
        elapsed = time.time() - start
        acc_data.append(['genetic_algorithm', elapsed, max_eval, trnacc, tstacc])
    return pd.DataFrame.from_records(acc_data,
                                     columns=['algo', 'time', 'max_evaluations', 'trnacc', 'tstacc'],
                                     index=['max_evaluations'])


def evaluate_simulated_annealing(data, max_evaluation_range=xrange(1, 100, 10)):
    acc_data = []
    for max_eval in max_evaluation_range:
        start = time.time()
        nn = NeuralNetwork(data=data)
        #weights = nn.learn_weights(max_evaluations=max_eval, algoritm=StochasticHillClimber)
        LOGGER.debug('sa %s', weights)
        trnacc = nn.train_accuracy()
        tstacc = nn.test_accuracy()
        elapsed = time.time() - start
        acc_data.append(['simulated_annealing', elapsed, max_eval, trnacc, tstacc])
    return pd.DataFrame.from_records(acc_data,
                                     columns=['algo', 'time', 'max_evaluations', 'trnacc', 'tstacc'],
                                     index=['max_evaluations'])


def estimate_training_iterations(n_iterations=10, learning_rate_range=tuple([0.001, 0.01, 0.1, 1.0])):
    data = load_higgs_train()

    def estimate_error(nn):
        error_data = []
        for i in range(n_iterations):
            nn.train(train_epoch=1)
            total_epochs, trn, tst = nn.estimate_error()
            error_data.append([nn.learning_rate, total_epochs, trn, tst])
        err_df = pd.DataFrame.from_records(error_data, columns=['learning_rate', 'iteration', 'training_error', 'test_error'])
        return err_df

    dfs = [estimate_error(NeuralNetwork(data, l)) for l in learning_rate_range]

    df = pd.concat(dfs)

    df['training_accuracy'] = 1 - df['training_error'] / 100
    df['test_accuracy'] = 1 - df['test_error'] / 100
    return df


def estimate_hidden_units(hidden_units_range=xrange(1, 100), sample_size=None):
    data = load_higgs_train(sample_size=sample_size)

    def estimate_error(nn):
        nn.train()
        total_epochs, trn, tst = nn.estimate_error()
        return [nn.n_hidden_units, total_epochs, trn, tst]

    data = [estimate_error(NeuralNetwork(data, n_hidden_units=l)) for l in hidden_units_range]

    df = pd.DataFrame.from_records(data, columns=['hidden_units', 'iteration', 'training_error', 'test_error'])

    df['training_accuracy'] = 1 - df['training_error'] / 100
    df['test_accuracy'] = 1 - df['test_error'] / 100
    return df


def plot_accuracy_function(df, smooth_factor=5):
    lr = df['learning_rate'].unique().tolist()
    rows = int(math.ceil(len(lr)/2))
    columns = 2
    fig, axes = plt.subplots(nrows=rows, ncols=columns, figsize=(12, 6), sharex=False, sharey=True)
    n = 0  # store the counter for retrieving learning rates
    for r in range(rows):
        for c in range(columns):
            if len(lr) > n:
                sub_df = df[df['learning_rate'] == lr[n]].set_index('iteration')
                smooth_df = pd.rolling_mean(sub_df[['training_accuracy', 'test_accuracy']], smooth_factor)
                title = 'Accuracy f (iterations) \n learning rate = {}'.format(lr[n])
                if rows == 1:
                    smooth_df.plot(ax=axes[c], title=title)
                else:
                    smooth_df.plot(ax=axes[r][c], title=title)
            n += 1


def plot_weight_learning_accuracy(hc_df, ga_df, sa_df, smoothing_factor=5):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), sharex=False, sharey=False)
    df = pd.concat([hc_df, ga_df, sa_df])
    df.groupby(by='algo')['trnacc'].plot(ax=axes[0],
                                         legend=True,
                                         kind='line',
                                         title='Accuracy f(evaluations)')
    df.groupby(by='algo')['time'].plot(ax=axes[1],
                                       legend=True,
                                       kind='line',
                                       title='Running time f(evaluations)')
    return df


def compare_weight_learning_optimized(data, max_evaluation_range=xrange(1, 100, 10)):
    hc_df = evaluate_hill_climbing(data, max_evaluation_range)
    ga_df = evaluate_genetic_algorithm(data, max_evaluation_range)
    sa_df = evaluate_simulated_annealing(data, max_evaluation_range)
    return plot_weight_learning_accuracy(hc_df, ga_df, sa_df)

