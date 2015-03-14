
"""
Optimize classifier settings
"""
import seaborn as sns
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from algo_evaluation.optimization.hill_climbing import hillclimb
from algo_evaluation.optimization.simulated_annealing import simulated_annealing
from algo_evaluation.optimization.genetic_optimize import genetic_optimize
from algo_evaluation.optimization import mimic

DEFAULT_EXPERIMENT_SETTINGS = dict()
DEFAULT_EXPERIMENT_SETTINGS['rhc'] = {'evaluations': 1000}
DEFAULT_EXPERIMENT_SETTINGS['sa'] = {'T': 1000}
DEFAULT_EXPERIMENT_SETTINGS['ga'] = {'generations': 1000}
DEFAULT_EXPERIMENT_SETTINGS['mm'] = {'evaluations': 100}


class ClassifierOptimization:
    def __init__(self, data):
        self.features, _, self.labels = data
        # domain is specified starting from zero, need to adjust when submitting to
        # fitness function
        self.max_depth_range = (0, 100)
        self.min_max_depth_value = 10
        self.min_samples_split_range = (0, 50)
        self.min_min_samples_value = 2
        self.min_samples_leaf_range = (0, 10)
        self.min_min_samples_leaf_value = 1
        self.domain = self.create_domain()

    def create_domain(self):
        return [self.max_depth_range, self.min_samples_split_range, self.min_samples_leaf_range]

    def compute_classification_error(self, solution):
        md, ms, msl = solution
        md += self.min_max_depth_value
        ms += self.min_min_samples_value
        msl += self.min_min_samples_leaf_value
        clf = DecisionTreeClassifier(max_depth=md, min_samples_split=ms, min_samples_leaf=msl)
        scores = cross_val_score(clf, self.features, self.labels, cv=5)
        mean_score = scores.mean()
        return 1 - mean_score


def baseline_dt(data):
    features, _, labels = data
    clf = DecisionTreeClassifier()
    scores = cross_val_score(clf, features, labels, cv=10)
    mean_score = scores.mean()
    return mean_score


def plot_evaluation(df):
    sns.set(style="ticks")
    sns.lmplot("evaluations", "optimal_value", data=df, col='algo', hue='algo')


def compare_all(data, experiment_settings=DEFAULT_EXPERIMENT_SETTINGS):
    opt_problem = ClassifierOptimization(data)
    domain = opt_problem.domain
    start = time.time()
    rhc = hillclimb(domain=domain,
                    costf=opt_problem.compute_classification_error,
                    max_evaluations=experiment_settings['rhc']['evaluations'])
    rhc['optimal_value'] += 1
    rhc['time'] = time.time() - start
    rhc.set_index('evaluations', inplace=True)
    start = time.time()
    sa = simulated_annealing(domain=domain,
                             costf=opt_problem.compute_classification_error,
                             T=experiment_settings['sa']['T'])
    sa.set_index('temperature', inplace=True)
    sa['optimal_value'] += 1
    sa['time'] = time.time() - start
    start = time.time()
    ga = genetic_optimize(domain=domain,
                          costf=opt_problem.compute_classification_error,
                          maxiter=experiment_settings['ga']['generations'])
    ga.set_index('generations', inplace=True)
    ga['optimal_value'] += 1
    ga['time'] = time.time() - start
    start = time.time()
    mm = mimic.run_mimic(domain=domain,
                         fitness_function=opt_problem.compute_classification_error,
                         evaluations=experiment_settings['mm']['evaluations'])
    mm['optimal_value'] += 1
    mm['time'] = time.time() - start
    return rhc, sa, ga, mm