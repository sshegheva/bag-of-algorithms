import numpy as np
import pandas as pd
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.connections import FullConnection
from pybrain.structure.modules import SigmoidLayer, SoftmaxLayer
from pybrain.structure.networks import FeedForwardNetwork
from sklearn.metrics import accuracy_score
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
        self.tstdata, self.trndata = ds.splitWithProportion(TEST_DATA_SPLIT)
        self.tstdata._convertToOneOfMany()
        self.trndata._convertToOneOfMany()

    def _create_trainer(self):
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

        self.trainer = BackpropTrainer(self.fnn, dataset=self.trndata,
                                       momentum=0.99,
                                       verbose=False,
                                       weightdecay=0.01,
                                       learningrate=0.001)

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
        LOGGER.info("train error: %s", trnerror)
        LOGGER.info("test error: %s", tsterror)
        return self.trainer.totalepochs, trnerror, tsterror


def run_neural_net(data):
    """
    Run and evaluate neural networks
    """
    nn = NeuralNetwork(data)
    nn.train()
    nn.predict()
    nn.estimate_error()
    return nn


def estimate_training_iterations():
    data = load_higgs_train()
    nn = NeuralNetwork(data)
    data = []
    for i in range(50):
        nn.train(train_epoch=5)
        total_epochs, trn, tst = nn.estimate_error()
        data.append([total_epochs, trn, tst])
    err_df = pd.DataFrame.from_records(data, columns=['iteration', 'training_error', 'test_error'])
    return err_df


def plot_accuracy_function(df, title):
    smooth_df = pd.rolling_mean(df, 5)
    smooth_df.plot(title=title)

