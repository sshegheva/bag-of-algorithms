import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mdptoolbox.mdp import QLearning


def solve_mdp_by_iteration(algorithm, P, R, discount=0.9, max_iter=10000):
    mdp = algorithm(transitions=P, reward=R, discount=discount, max_iter=max_iter)
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


def solve_mdp_by_qlearning(P, R, discount=0.9, max_iter=10000, verbose=False):
    mdp = QLearning(transitions=P, reward=R, discount=discount, n_iter=max_iter)
    mdp.max_iter = max_iter
    if verbose:
        mdp.setVerbose()
    mdp.run()
    n_states = R.shape[0]
    data = [QLearning.__name__, n_states, mdp.Q, mdp.V, np.mean(mdp.V), np.sum(mdp.V),
            mdp.mean_discrepancy, np.mean(mdp.mean_discrepancy),
            mdp.discount, mdp.max_iter,
            mdp.policy, mdp.time]
    return pd.Series(data, index=['algorithm', 'states', 'Q', 'values', 'mean_values',
                                  'accum_values', 'mean_discrepancy', 'mean_mean_discrepancy',
                                  'discount', 'max_iter', 'policy', 'time'])


def test_algorithm(algorithm, P, R, discount, num_sim=10000):
    series = []
    for n in range(1, num_sim):
        d = solve_mdp_by_iteration(algorithm, P, R, discount, max_iter=n)
        series.append(d)
    df = pd.concat(series, axis=1)
    return df.T


def plot_qlearn_performance(forest_df, ctr_df):
    f, ax = plt.subplots(1, 2, figsize=(12, 4))
    forest_df.groupby('algorithm').time.plot(ax=ax[0], legend=True, title='Algorithm Running Time')
    ax[0].set_xlabel('iterations')
    ax[0].set_ylabel('time (ms)')
    forest_discrepancy = pd.Series(forest_df[forest_df['algorithm'] == 'QLearning'].loc[0]['mean_discrepancy'])
    ctr_dicrepancy = pd.Series(ctr_df.loc[0]['mean_discrepancy'])
    disrepancy_df = pd.concat([forest_discrepancy, ctr_dicrepancy], axis=1)
    disrepancy_df.rename(columns={0: 'forest', 1: 'ctr'}, inplace=True)
    disrepancy_df.plot(ax=ax[1], legend=True, title='Q Matrix mean discrepancy')
    ax[1].set_xlabel('iterations')
    ax[1].set_ylabel('variation')