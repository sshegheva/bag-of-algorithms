import pandas as pd
import numpy as np
from mdptoolbox.mdp import ValueIteration
from mdptoolbox.mdp import PolicyIteration
from mdptoolbox.mdp import QLearning
from mdptoolbox.example import forest

from algo_evaluation.mdp.problems import backpacker_example


def solve_mdp(algorithm, P, R, discount, max_iter=10000):
    if algorithm.__name__ in [ValueIteration.__name__, PolicyIteration.__name__]:
        mdp = algorithm(transitions=P, reward=R, discount=discount, max_iter=max_iter)
    if algorithm.__name__ == QLearning.__name__:
        mdp = algorithm(transitions=P, reward=R, discount=discount, n_iter=max_iter)
        mdp.set_verbose()
    mdp.run()

    data = [algorithm.__name__, np.sum(mdp.V), np.mean(mdp.V),
            np.sum(mdp.R), np.mean(mdp.R),
            mdp.discount, mdp.max_iter,
            mdp.policy, mdp.time]
    return pd.Series(data, index=['algorithm', 'cumvalue', 'meanvalue',
                                  'cumrewards', 'meanrewards',
                                  'discount', 'max_iter', 'policy', 'time'])


def run_mdp(algorithm, P, R, discount, max_iter=10000):
    series = []
    for n in range(1, max_iter):
        d = solve_mdp(algorithm, P, R, discount, max_iter=n)
        series.append(d)
    df = pd.concat(series, axis=1)
    return df.T


def solve_by_value_iteration(mdp_problem):
    P, R = mdp_problem
    return solve_mdp(ValueIteration, P, R)


def solve_backpacker():
    mdp_problem = backpacker_example.backpacker()
    return solve_by_value_iteration(mdp_problem)


def solve_forest_example(num_simulations=500):
    P, R = forest(S=100, p=0.9)
    vi = run_mdp(ValueIteration, P, R, discount=0.9, max_iter=num_simulations)
    pi = run_mdp(PolicyIteration, P, R, discount=0.9, max_iter=num_simulations)
    #ql = run_mdp(QLearning, P, R, discount=0.9, max_iter=num_simulations)
    return vi, pi#, ql