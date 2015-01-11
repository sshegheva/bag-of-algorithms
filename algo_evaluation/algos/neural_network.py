import numpy as np
import pandas as pd
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.validation import Validator
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer

from algo_evaluation.datasets import load_higgs_train
from algo_evaluation import LOGGER, TEST_DATA_SPLIT


class NeuralNetwork:
    def __init__(self, data,):
        self.features, self.weights, labels = data
        self.labels = np.array([1 if l == 's' else 0 for l in labels])
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
        self.fnn = buildNetwork(self.trndata.indim, 2, self.trndata.outdim, outclass=SoftmaxLayer)
        self.trainer = BackpropTrainer(self.fnn, dataset=self.trndata, momentum=0.1, verbose=True, weightdecay=0.01)

    def train(self, train_epoch):
        self.trainer.trainEpochs(train_epoch)

    def predict(self, dataset):
        return self.fnn.activateOnDataset(dataset)

    def estimate_error(self):
        trnerror = percentError(self.trainer.testOnClassData(), self.trndata['class'])
        tsterror = percentError(self.trainer.testOnClassData(dataset=self.tstdata), self.tstdata['class'])
        LOGGER.info("epoch: %4d", self.trainer.totalepochs)
        LOGGER.info("train error: %5.4f", trnerror)
        LOGGER.info("test error: %5.4f", tsterror)
        return self.trainer.totalepochs, trnerror, tsterror

    def estimate_accuracy(self):
        training_predictions = self.predict(self.trndata)
        test_predictions = self.predict(self.tstdata)
        trnaccuracy = Validator.classificationPerformance(training_predictions, self.trndata['target'])
        tstaccuracy = Validator.classificationPerformance(test_predictions, self.tstdata['target'])
        LOGGER.info("train accuracy: %5.4f", trnaccuracy)
        LOGGER.info("test accuracy: %5.4f", tstaccuracy)
        return self.trainer.totalepochs, trnaccuracy, tstaccuracy




def estimate_training_iterations():
    data = load_higgs_train()
    nn = NeuralNetwork(data)
    error_data = []
    accuracy_data = []
    for i in range(20):
        nn.train(train_epoch=1)
        total_epochs, trnerror, tsterror = nn.estimate_error()
        total_epochs, trnaccuracy, tstaccuracy = nn.estimate_accuracy()
        error_data.append([total_epochs, trnerror, tsterror])
        accuracy_data.append([total_epochs, trnaccuracy, tstaccuracy])
    err_df = pd.DataFrame.from_records(error_data, columns=['iteration', 'training_error', 'test_error'])
    acc_df = pd.DataFrame.from_records(error_data, columns=['iteration', 'training_accuracy', 'test_accuracy'])
    return err_df, acc_df


def plot_accuracy_function(df, title):
    smooth_df = pd.rolling_mean(df, 5)
    smooth_df.plot(title=title)

