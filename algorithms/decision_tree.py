import pandas as pd
from sklearn import tree
from datasets import split_dataset, load_higgs_train
from evaluation import evaluation_score
from algorithms import LOGGER


class DecisionTree:
    def __init__(self, min_samples_split=40, max_depth=None):
        features, weights, labels = load_higgs_train()
        self.feature_names = features.columns.tolist()
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.clf = tree.DecisionTreeClassifier(min_samples_split=self.min_samples_split)
        LOGGER.info('Created classifier with min_samples_split = %s and max depth %s', self.min_samples_split, self.max_depth)
        self.training_results, self.predictions = None, None
        self.training_score, self.test_score = None, None
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
        self.training_results = self.clf.predict(self.features_train)
        self.predictions = self.clf.predict(self.features_test)
        LOGGER.info('Generated predictions')

    def evaluate(self):
        self.training_score = evaluation_score(self.labels_train, self.training_results)
        self.test_score = evaluation_score(self.labels_test, self.predictions)
        LOGGER.info('Training Accuracy score = %s', self.training_score)
        LOGGER.info('Test Accuracy score = %s', self.test_score)


def run_decision_tree(min_samples_split=40, max_depth=None):
    """
    Run and evaluate decision trees with default settings
    """
    dt = DecisionTree(min_samples_split=min_samples_split, max_depth=max_depth)
    dt.train()
    dt.predict()
    dt.evaluate()
    return dt.training_score, dt.test_score


def estimate_best_min_samples_split():
    """
    Run the decision tree classifier with multiple settings of
    min_sample_split and plot the accuracy function of min_sample_split
    :return: the best min_sample_split setting
    """
    min_split_range = xrange(2, 120, 2)
    data = [[min_sample] + list(run_decision_tree(min_samples_split=min_sample)) for min_sample in min_split_range]
    LOGGER.info('Performed evaluation of the min sample split choice')
    columns = ['min_sample_split', 'training_score', 'test_score']
    df = pd.DataFrame.from_records(data, columns=columns, index=columns[0])
    LOGGER.info(df)
    # plot the accuracy curve
    df['error'] = (df['training_score'] - df['test_score']) * (df['training_score'] - df['test_score'])
    smooth_df = pd.rolling_mean(df, 5)
    smooth_df = smooth_df[['training_score', 'test_score']]
    smooth_df.plot(title='Accuracy change as a function of min_samples_split (smoothed)')


def estimate_best_max_depth():
    """
    Run the decision tree classifier with multiple settings of
    max depth and plot the accuracy function of max depth
    :return: the best max depth setting
    """
    max_depth_range = xrange(2, 120, 2)
    data = [[depth] + list(run_decision_tree(max_depth=depth)) for depth in max_depth_range]
    LOGGER.info('Performed evaluation of the max death')
    columns = ['max_depth', 'training_score', 'test_score']
    df = pd.DataFrame.from_records(data, columns=columns, index=columns[0])
    LOGGER.info(df)
    # plot the accuracy curve
    df['error'] = (df['training_score'] - df['test score']) * (df['training_score'] - df['test_score'])
    smooth_df = pd.rolling_mean(df, 5)
    smooth_df = smooth_df[['training_score', 'test_score']]
    smooth_df.plot(title='Accuracy change as a function of max depth (smoothed)')


