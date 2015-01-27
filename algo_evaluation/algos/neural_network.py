import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.optimization import HillClimber
from pybrain.optimization.populationbased.ga import GA
from pybrain.structure.connections import FullConnection
from pybrain.structure.modules import SigmoidLayer, SoftmaxLayer
from pybrain.structure.networks import FeedForwardNetwork

from algo_evaluation.datasets import load_higgs_train, split_dataset


class NeuralNetwork:
    def __init__(self, data, learning_rate=0.1, momentum=0.1):
        self.features, self.weights, labels = data
        self.labels = np.array([1 if l == 's' else 0 for l in labels])
        self._prepare_data()
        self._build_network()
        self.learning_rate = learning_rate
        self.momentum = momentum

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
        #self.tstdata, self.trndata = ds.splitWithProportion(TEST_DATA_SPLIT)
        #self.tstdata._convertToOneOfMany()
        #self.trndata._convertToOneOfMany()

    def _build_network(self):
        self.fnn = FeedForwardNetwork()
        inLayer = SigmoidLayer(self.trndata.indim)
        hiddenLayer = SigmoidLayer(int(self.trndata.indim * 0.5))
        outLayer = SoftmaxLayer(self.trndata.outdim)

        self.fnn.addInputModule(inLayer)
        self.fnn.addModule(hiddenLayer)
        self.fnn.addOutputModule(outLayer)

        in_to_hidden = FullConnection(inLayer, hiddenLayer)
        hidden_to_out = FullConnection(hiddenLayer, outLayer)

        self.fnn.addConnection(in_to_hidden)
        self.fnn.addConnection(hidden_to_out)
        self.fnn.sortModules()

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
        return algoritm(self.tstdata.evaluateModuleMSE, self.fnn, maxEvaluations=max_evaluations).learn()

    def predict(self, dataset=None):
        if dataset is None:
            dataset = self.tstdata
        return self.fnn.activateOnDataset(dataset)

    def estimate_error(self):
        trnerror = percentError(self.trainer.testOnClassData(dataset=self.trndata), self.trndata['class'])
        tsterror = percentError(self.trainer.testOnClassData(dataset=self.tstdata), self.tstdata['class'])
        return self.trainer.totalepochs, trnerror, tsterror

    def train_accuracy(self):
        return accuracy_score(y_pred=self.predict(self.trndata),
                              y_true=self.trndata['target'],
                              sample_weight=self.dataset['training']['weights'])

    def test_accuracy(self):
        return accuracy_score(y_pred=self.predict(self.tstdata),
                              y_true=self.tstdata['target'],
                              sample_weight=self.dataset['test']['weights'])


def run_neural_net(data, learning_rate=0.1):
    """
    Run and evaluate neural networks
    """
    nn = NeuralNetwork(data, learning_rate=learning_rate)
    nn.train()
    nn.predict()
    return nn.estimate_error()


def evaluate_hill_climbing(data, max_evaluation_range=xrange(10, 100, 10)):
    acc_data = []
    for max_eval in max_evaluation_range:
        nn = NeuralNetwork(data=data)
        nn.learn_weights(max_evaluations=max_eval, algoritm=HillClimber)
        trnacc = nn.train_accuracy()
        tstacc = nn.test_accuracy()
        acc_data.append([max_eval, trnacc, tstacc])
    return pd.DataFrame.from_records(acc_data,
                                     columns=['max_evaluations', 'trnacc', 'tstacc'],
                                     index=['max_evaluations'])


def evaluate_genetic_algorithm(data, max_evaluation_range=xrange(10, 100, 10)):
    acc_data = []
    for max_eval in max_evaluation_range:
        nn = NeuralNetwork(data=data)
        nn.learn_weights(max_evaluations=max_eval, algoritm=GA)
        trnacc = nn.train_accuracy()
        tstacc = nn.test_accuracy()
        acc_data.append([max_eval, trnacc, tstacc])
    return pd.DataFrame.from_records(acc_data,
                                     columns=['max_evaluations', 'trnacc', 'tstacc'],
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

