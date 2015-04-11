import numpy as np
import pandas as pd
from mdptoolbox.mdp import ValueIteration
from mdptoolbox.mdp import PolicyIteration
from mdptoolbox.mdp import QLearning


def solve_mdp(algorithm, P, R, discount=0.9, max_iter=10000):
    if algorithm.__name__ in [ValueIteration.__name__, PolicyIteration.__name__]:
        mdp = algorithm(transitions=P, reward=R, discount=discount, max_iter=max_iter)
    if algorithm.__name__ == QLearning.__name__:
        mdp = algorithm(transitions=P, reward=R, discount=discount, n_iter=max_iter)
    mdp.max_iter = max_iter
    #mdp.setVerbose()
    mdp.run()
    n_states = R.shape[0]
    data = [algorithm.__name__, n_states, np.mean(mdp.V), np.sum(mdp.V),
            np.mean(mdp.R), np.sum(mdp.R),
            mdp.discount, mdp.max_iter,
            mdp.policy, mdp.time]
    return pd.Series(data, index=['algorithm', 'age', 'mean_values', 'accum_values',
                                  'mean_rewards', 'accum_rewards',
                                  'discount', 'max_iter', 'policy', 'time'])


def test_algorithm(algorithm, P, R, discount, num_sim=10000):
    series = []
    for n in range(1, num_sim):
        d = solve_mdp(algorithm, P, R, discount, max_iter=n)
        series.append(d)
    df = pd.concat(series, axis=1)
    return df.T