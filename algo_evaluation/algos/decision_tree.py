import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from algo_evaluation.datasets import split_dataset, load_higgs_train


class DecisionTree:
    def __init__(self, data, criterion='gini', min_samples_split=60):
        features, weights, labels = data
        self.clf = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=min_samples_split)
        self.predictions, self.trnaccuracy, self.tstaccuracy = None, None, None
        self.dataset = split_dataset(features, weights, labels)

    def train(self):
        """
        Train the decision trees on the higgs dataset
        :return: classifier
        """
        self.clf = self.clf.fit(self.dataset['training']['features'], self.dataset['training']['labels'])

    def predict(self):
        """
        Predict label using decision tree classifier
        """
        self.predictions = self.clf.predict(self.dataset['test']['features'])

    def evaluate(self):
        self.trnaccuracy = self.clf.score(self.dataset['training']['features'],
                                          self.dataset['training']['labels'],
                                          sample_weight=self.dataset['training']['weights'])
        self.tstaccuracy = self.clf.score(self.dataset['test']['features'],
                                          self.dataset['test']['labels'],
                                          sample_weight=self.dataset['test']['weights'])


def run_decision_tree(data, criterion='gini', min_samples_split=60):
    """
    Run and evaluate decision trees with default settings
    """
    dt = DecisionTree(data=data, criterion=criterion, min_samples_split=min_samples_split)
    dt.train()
    dt.predict()
    dt.evaluate()
    return dt.trnaccuracy, dt.tstaccuracy


def estimate_best_min_samples_split():
    """
    Run the decision tree classifier with multiple settings of
    min_sample_split and plot the accuracy function of min_sample_split
    :return: the best min_sample_split setting
    """
    min_split_range = xrange(2, 120, 2)
    data = load_higgs_train()
    records = [[min_sample] + list(run_decision_tree(data=data, criterion='gini', min_samples_split=min_sample))
               + list(run_decision_tree(data=data, criterion='entropy', min_samples_split=min_sample))
               for min_sample in min_split_range]
    columns = ['min_sample_split', 'gini_training_score', 'gini_test_score',
               'entropy_training_score', 'entropy_test_score']
    df = pd.DataFrame.from_records(records, columns=columns, index=columns[0])
    return df


def plot_accuracy_function(df, smoothing_factor=5):
    gini_df = df[['gini_training_score', 'gini_test_score']]
    entropy_df = df[['entropy_training_score', 'entropy_test_score']]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharex=False, sharey=True)
    pd.rolling_mean(gini_df, smoothing_factor).plot(ax=axes[0],
                                                    title='Accuracy f(min_samples_split)\n with Gini split')
    pd.rolling_mean(entropy_df, smoothing_factor).plot(ax=axes[1],
                                                       title='Accuracy f(min_samples_split)\n with entropy split')



