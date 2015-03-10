import networkx as nx
import pandas as pd
import numpy as np
import random
import operator
from scipy import stats
from sklearn.metrics import mutual_info_score

np.set_printoptions(precision=4)


class Mimic(object):
    """
    Usage: from mimicry import Mimic

    :param domain: list of tuples containing the min and max value for each parameter to be optimized, for a bit
    string, this would be [(0, 1)]*bit_string_length

    :param fitness_function: callable that will take a single instance of your optimization parameters and return
    a scalar fitness score

    :param samples: Number of samples to generate from the distribution each iteration

    :param percentile: Percentile of the distribution to keep after each iteration, default is 0.90

    """

    def __init__(self, domain, fitness_function, samples=1000, percentile=0.90):

        self.domain = domain
        self.samples = samples
        initial_samples = np.array(self._generate_initial_samples())
        self.sample_set = SampleSet(initial_samples, fitness_function)
        self.fitness_function = fitness_function
        self.percentile = percentile

    def fit(self):
        """
        Run this to perform one iteration of the Mimic algorithm

        :return: A list containing the top percentile of data points
        """

        sample_fit_pairs = self.sample_set.get_percentile(self.percentile)
        samples = [s for s,f in sample_fit_pairs]
        self.distribution = Distribution(samples)
        self.sample_set = SampleSet(
            self.distribution.generate_samples(self.samples),
            self.fitness_function,
        )
        return self.sample_set.get_percentile(self.percentile)

    def _generate_initial_samples(self):
        return [self._generate_initial_sample() for i in xrange(self.samples)]

    def _generate_initial_sample(self):
        return [random.randint(self.domain[i][0], self.domain[i][1])
                for i in xrange(len(self.domain))]


class SampleSet(object):
    def __init__(self, samples, fitness_function, maximize=False):
        self.samples = samples
        self.fitness_function = fitness_function
        self.maximize = maximize

    def calculate_fitness(self):
        fitnesses = [(sample, self.fitness_function(sample)) for sample in self.samples]
        sorted_samples = sorted(fitnesses, key=operator.itemgetter(1), reverse=self.maximize)
        return sorted_samples

    def get_percentile(self, percentile):
        fit_samples = self.calculate_fitness()
        index = int(len(fit_samples) * percentile)
        return fit_samples[:index]


class Distribution(object):
    def __init__(self, samples):
        self.samples = samples
        self.complete_graph = self._generate_mutual_information_graph()
        self.spanning_graph = self._generate_spanning_graph()
        self._generate_bayes_net()

    def generate_samples(self, number_to_generate):
        root = 0
        sample_len = len(self.bayes_net.node)
        samples = np.zeros((number_to_generate, sample_len))
        values = self.bayes_net.node[root]["probabilities"].keys()
        probabilities = self.bayes_net.node[root]["probabilities"].values()
        dist = stats.rv_discrete(name="dist", values=(values, probabilities))
        samples[:, 0] = dist.rvs(size=number_to_generate)
        for parent, current in nx.bfs_edges(self.bayes_net, root):
            for i in xrange(number_to_generate):
                parent_val = samples[i, parent]
                current_node = self.bayes_net.node[current]
                cond_dist = current_node["probabilities"][int(parent_val)]
                values = cond_dist.keys()
                probabilities = cond_dist.values()
                dist = stats.rv_discrete(
                    name="dist",
                    values=(values, probabilities)
                )
                samples[i, current] = dist.rvs()

        return samples

    def _generate_bayes_net(self):
        # Pseudo Code
        # 1. Start at any node(probably 0 since that will be the easiest for
        # indexing)
        # 2. At each node figure out the conditional probability
        # 3. Add it to the new graph (which should probably be directed)
        # 4. Find unprocessed adjacent nodes
        # 5. If any go to 2
        #    Else return the bayes net'

        # Will it be possible that zero is not the root? If so, we need to pick
        # one
        root = 0

        samples = np.asarray(self.samples)

        self.bayes_net = nx.bfs_tree(self.spanning_graph, root)

        for parent, child in self.bayes_net.edges():

            parent_array = samples[:, parent]

            # Check if node is root
            if not self.bayes_net.predecessors(parent):
                parent_probs = np.histogram(parent_array,
                                            (np.max(parent_array)+1),
                                            )[0] / float(parent_array.shape[0])

                self.bayes_net.node[parent]["probabilities"] = dict(enumerate(parent_probs))

            child_array = samples[:, child]

            unique_parents = np.unique(parent_array)
            for parent_val in unique_parents:
                parent_inds = np.argwhere(parent_array == parent_val)
                sub_child = child_array[parent_inds]


                child_probs = np.histogram(sub_child,
                                           (np.max(sub_child)+1),
                                           )[0] / float(sub_child.shape[0])

                # If P(0) = 1 then child_probs = [1.]
                # must append zeros to ensure output consistency
                while child_probs.shape[0] < unique_parents.shape[0]:
                    child_probs = np.append(child_probs, 0.)
                self.bayes_net.node[child][parent_val] = dict(enumerate(child_probs))

            self.bayes_net.node[child] = dict(probabilities=self.bayes_net.node[child])

    def _generate_spanning_graph(self):
        return nx.prim_mst(self.complete_graph)

    def _generate_mutual_information_graph(self):
        samples = np.asarray(self.samples)
        complete_graph = nx.complete_graph(samples.shape[1])

        for edge in complete_graph.edges():
            mutual_info = mutual_info_score(
                samples[:, edge[0]],
                samples[:, edge[1]]
            )

            complete_graph.edge[edge[0]][edge[1]]['weight'] = -mutual_info

        return complete_graph


def run_mimic(domain, fitness_function, evaluations=1000):
    m = Mimic(domain, fitness_function)
    data = []
    for n in xrange(evaluations):
        results = m.fit()[0]
        [data.append([n, solution, optimal_value]) for solution, optimal_value in results]

    df = pd.DataFrame(data, columns=['iteration', 'solution', 'optimal_value'])
    return df