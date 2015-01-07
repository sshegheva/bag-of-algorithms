from sklearn import tree
from datasets import split_dataset, load_higgs_train
from evaluation import evaluation_score
from algorithms import LOGGER


class DecisionTree:
    def __init__(self):
        features, weights, labels = load_higgs_train()
        self.feature_names = features.columns.tolist()
        self.clf = tree.DecisionTreeClassifier()
        self.predictions = None
        self.score = None
        self.features_train, self.features_test, self.labels_train, self.labels_test = split_dataset(features, labels)

    def train(self):
        """
        Train the decision trees on the higgs dataset
        :return: classifier
        """
        self.clf = self.clf.fit(self.features_train, self.labels_train)
        LOGGER.info('Trained decision tree classifier')

    def predict(self):
        """
        Predict label using decision tree classifier
        :return:
        """
        self.predictions = self.clf.predict(self.features_test)
        LOGGER.info('Generated predictions')

    def evaluate(self):
        self.score = evaluation_score(self.labels_test, self.predictions)
        LOGGER.info('Accuracy score = %s', self.score)

