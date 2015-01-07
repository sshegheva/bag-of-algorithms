import pandas as pd
from sklearn import tree
from datasets import split_dataset, load_higgs_train
from evaluation import evaluation_score
from algorithms import LOGGER


class DecisionTree:
    def __init__(self, min_samples_split=40):
        features, weights, labels = load_higgs_train()
        self.feature_names = features.columns.tolist()
        self.min_samples_split = min_samples_split
        self.clf = tree.DecisionTreeClassifier(min_samples_split=self.min_samples_split)
        LOGGER.info('Created classifier with min_samples_split = %s', self.min_samples_split)
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


def run_decision_tree(min_samples_split=40):
    """
    Run and evaluate decision trees with default settings
    """
    dt = DecisionTree(min_samples_split)
    dt.train()
    dt.predict()
    dt.evaluate()
    return dt.score


def estimate_best_min_samples_split():
    """
    Run the decision tree classifier with multiple settings of
    min_sample_split and plot the accuracy function of min_sample_split
    :return: the best min_sample_split setting
    """
    min_split_range = xrange(2, 120, 2)
    data = {min_sample: run_decision_tree(min_sample) for min_sample in min_split_range}
    LOGGER.info('Performed evaluation of the min sample split choice')
    df = pd.DataFrame.from_dict(data, orient='index').sort(ascending=False).reset_index()
    df = df.rename(columns={0: 'accuracy score', 'index': 'min_sample_split'}).set_index('min_sample_split')
    LOGGER.info(df)
    # plot the accuracy curve
    df.plot(title='Accuracy change as a function of min_samples_split')
    pd.rolling_mean(df, 5).plot(title='Accuracy change as a function of min_samples_split (smoothed)')


