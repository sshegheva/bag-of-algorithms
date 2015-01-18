import numpy as np
import pandas as pd
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.validation import Validator
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.connections import FullConnection
from pybrain.structure.modules import SigmoidLayer
from pybrain.structure.networks import FeedForwardNetwork

from algo_evaluation.datasets import load_higgs_train
from algo_evaluation import LOGGER, TEST_DATA_SPLIT


class NeuralNetwork:
    def __init__(self, data,):
        self.features, self.weights, labels = data
        self.labels = np.array([1 if l == 's' else 0 for l in labels])
        self.sample_size = self.features.shape[0]
        self._prepare_data()
        self._create_trainer()

    def _prepare_data(self):
        classes = set(self.labels)
        ds = ClassificationDataSet(self.features.shape[1], 1, nb_classes=len(classes))
        for i in range(self.sample_size):
            ds.addSample(self.features.iloc[i], self.labels[i])
        self.trndata, self.tstdata = ds.splitWithProportion(TEST_DATA_SPLIT)

    def _create_trainer(self):
        self.fnn = FeedForwardNetwork()
        inLayer = SigmoidLayer(self.trndata.indim)
        hiddenLayer = SigmoidLayer(int(self.trndata.indim))
        outLayer = SigmoidLayer(self.trndata.outdim)

        self.fnn.addInputModule(inLayer)
        self.fnn.addModule(hiddenLayer)
        self.fnn.addOutputModule(outLayer)

        in_to_hidden = FullConnection(inLayer, hiddenLayer)
        hidden_to_out = FullConnection(hiddenLayer, outLayer)

        self.fnn.addConnection(in_to_hidden)
        self.fnn.addConnection(hidden_to_out)
        self.fnn.sortModules()

        self.trainer = BackpropTrainer(self.fnn, dataset=self.trndata,
                                       momentum=0.1,
                                       verbose=False,
                                       weightdecay=0.01,
                                       learningrate=1.0)

    def train(self, train_epoch=5):
        self.trainer.trainEpochs(train_epoch)

    def predict(self, dataset=None):
        if dataset is None:
            dataset = self.tstdata
        return self.fnn.activateOnDataset(dataset)

    def estimate_error(self):
        trnerror = percentError(self.trainer.testOnClassData(dataset=self.trndata), self.trndata['class'])
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


def run_neural_net(data):
    """
    Run and evaluate neural networks
    """
    nn = NeuralNetwork(data)
    nn.train()
    nn.predict()
    nn.estimate_error()


def estimate_training_iterations():
    data = load_higgs_train()
    nn = NeuralNetwork(data)
    error_data = []
    for i in range(5):
        nn.train(train_epoch=1)
        total_epochs, trnerror, tsterror = nn.estimate_error()
        error_data.append([total_epochs, trnerror, tsterror])
    err_df = pd.DataFrame.from_records(error_data, columns=['iteration', 'training_error', 'test_error'])
    return err_df


def plot_accuracy_function(df, title):
    smooth_df = pd.rolling_mean(df, 5)
    smooth_df.plot(title=title)

