import numpy as np
import pandas as pd
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer

from datasets import load_higgs_train
from algorithms import LOGGER, TEST_DATA_SPLIT


class NeuralNetwork:
    def __init__(self, data,):
        self.features, self.weights, labels = data
        self.labels = np.array([1 if l == 'b' else 0 for l in labels])
        self.sample_size = self.features.shape[0]
        LOGGER.info('Loaded features %s', self.features.shape)
        LOGGER.info('Loaded weights %s', self.weights.shape)
        LOGGER.info('Loaded labels %s', self.labels.shape)
        self._prepare_data()
        self._create_trainer()

    def _prepare_data(self):
        classes = set(self.labels)
        ds = ClassificationDataSet(self.features.shape[1], 1, nb_classes=len(classes))
        for i in range(self.sample_size):
            ds.addSample(self.features.iloc[i], self.labels[i])
        self.trndata, self.tstdata = ds.splitWithProportion(TEST_DATA_SPLIT)
        LOGGER.info("Number of training patterns: %s", len(self.trndata))
        LOGGER.info("Input and output dimensions: %s, %s", self.trndata.indim, self.trndata.outdim)

    def _create_trainer(self):
        fnn = buildNetwork(self.trndata.indim, 5, self.trndata.outdim, outclass=SoftmaxLayer)
        self.trainer = BackpropTrainer(fnn, dataset=self.trndata, momentum=0.1, verbose=True, weightdecay=0.01)

    def train(self, train_epoch):
        self.trainer.trainEpochs(train_epoch)

    def evaluate(self):
        trnerror = percentError(self.trainer.testOnClassData(), self.trndata['class'] )
        tsterror = percentError(self.trainer.testOnClassData(dataset=self.tstdata), self.tstdata['class'] )
        LOGGER.info("epoch: %4d", self.trainer.totalepochs)
        LOGGER.info("train error: %5.2f%%", trnerror)
        LOGGER.info("test error: %5.2f%%", tsterror)
        return self.trainer.totalepochs, trnerror, tsterror


def estimate_training_iterations():
    data = load_higgs_train()
    nn = NeuralNetwork(data)
    error_data = []
    for i in range(20):
        nn.train(train_epoch=i)
        total_epochs, trnerror, tsterror = nn.evaluate()
        error_data.append([total_epochs, trnerror, tsterror])
    df = pd.DataFrame.from_records(error_data, columns=['iteration', 'training_error', 'test_error'])
    return df


def plot_accuracy_function(df):
    smooth_df = pd.rolling_mean(df, 5)
    smooth_df.plot(title='Error as a function of training epochs (smoothed)')

