import math
import seaborn as sns
import matplotlib.pyplot as plt
from algo_evaluation.optimization.problems.plot_optimal_values import plot_optimal_values
from algo_evaluation.optimization.hill_climbing import hillclimb
from algo_evaluation.optimization.simulated_annealing import simulated_annealing
from algo_evaluation.optimization.genetic_optimize import genetic_optimize
from algo_evaluation.optimization import mimic

DEFAULT_EXPERIMENT_SETTINGS = dict()
DEFAULT_EXPERIMENT_SETTINGS['rhc'] = {'evaluations': 1000}
DEFAULT_EXPERIMENT_SETTINGS['sa'] = {'T': 1000}
DEFAULT_EXPERIMENT_SETTINGS['ga'] = {'generations': 1000}
DEFAULT_EXPERIMENT_SETTINGS['mm'] = {'evaluations': 1000}


class WaldoOpt:
    def __init__(self, waldo_df):
        self.waldo_df = waldo_df
        self.waldo_location_map = self.create_waldo_lookup()
        self.domain = self.waldo_domain()

    def create_waldo_lookup(self):
        loc_map = {}
        for i, record in self.waldo_df.iterrows():
            key = "B%dP%d" % (record.Book-1, record.Page-1)
            loc_map[key] = (record.X, record.Y)
        return loc_map

    def waldo_domain(self):
        min_book, max_book = 0.0, 6.0
        min_page, max_page = 0.0, 11.0
        domain = []
        for i in range(len(self.waldo_df)):
            domain.append((min_book, max_book))
            domain.append((min_page, max_page))
        return domain

    @staticmethod
    def calculate_distance(x1, y1, x2, y2):
        """
            Returns the Euclidean distance between points (x1, y1) and (x2, y2)
        """
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def waldo_coord(self, book, page):
        key = "B%dP%d" % (book, page)
        return self.waldo_location_map[key] if key in self.waldo_location_map else None

    def compute_fitness(self, solution):
        """
            Computes the distance that the Waldo-seeking solution covers.

            Lower distance is better, so the GA should try to minimize this function.
        """
        solution_fitness = 0.0
        solution_pair = zip(*(iter(solution),) * 2)
        for index in range(1, len(solution_pair)):
            book1, page1 = solution_pair[index]
            book2, page2 = solution_pair[index-1]
            w1 = self.waldo_coord(book1, page1)
            w2 = self.waldo_coord(book2, page2)
            if w1 and w2:
                solution_fitness += self.calculate_distance(w1[0], w1[1], w2[0], w2[1])
        return solution_fitness


def plot_evaluation(df):
    sns.set(style="ticks")
    sns.lmplot("evaluations", "optimal_value", data=df, col='algo', hue='algo')


def compare_all(waldo_df, experiment_settings=DEFAULT_EXPERIMENT_SETTINGS):
    opt_problem = WaldoOpt(waldo_df)
    domain = opt_problem.domain
    rhc = hillclimb(domain=domain,
                    costf=opt_problem.compute_fitness,
                    max_evaluations=experiment_settings['rhc']['evaluations'])
    rhc.set_index('evaluations', inplace=True)
    sa = simulated_annealing(domain=domain,
                             costf=opt_problem.compute_fitness,
                             T=experiment_settings['sa']['T'])
    sa.set_index('temperature', inplace=True)
    ga = genetic_optimize(domain=domain,
                          costf=opt_problem.compute_fitness,
                          maxiter=experiment_settings['ga']['generations'])
    ga.set_index('generations', inplace=True)
    mm = mimic.run_mimic(domain=domain,
                         fitness_function=opt_problem.compute_fitness,
                         evaluations=experiment_settings['mm']['evaluations'])
    return rhc, sa, ga, mm