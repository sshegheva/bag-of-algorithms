import random
import numpy as np
import pandas as pd
import warnings
from scipy.stats import poisson
import matplotlib.pyplot as plt

from mdptoolbox.mdp import ValueIteration
from mdptoolbox.mdp import PolicyIteration
from mdptoolbox.mdp import QLearning

from algo_evaluation.mdp.simulations import solve_mdp

warnings.filterwarnings("ignore")


class BernoulliArm():
    def __init__(self, p):
        self.p = p

    def draw(self):
        if random.random() > self.p:
            return 0.0
        else:
            return 1.0


def simulate_user_click(n_ads=10, mu=7.0, display=True):
    rv = poisson.rvs(mu=mu, size=n_ads) / float(n_ads)
    means = pd.Series(rv)
    random.shuffle(means)
    actions = pd.Series(map(lambda mu: BernoulliArm(mu).draw(), means))
    if display:
        f, ax = plt.subplots(1, 2, figsize=(13, 3))

        means.hist(normed=True, ax=ax[0])
        means.plot(kind='kde', ax=ax[0])
        ax[0].set_xlabel('probability of clicking on add')
        ax[0].set_title('Probability click distribution per ad \n Fig. 2.1')

        actions.hist(normed=True, ax=ax[1])
        actions.plot(kind='kde', ax=ax[1])
        ax[1].set_xlabel('non-clicks/clicks')
        ax[1].set_title('Example of distribution of clicks vs ignores \n Fig. 2.2')
    return actions


def create_ctr_mdp(n_ads=20, mu=10, n_simulations=10000):
    n_states = 2
    P = np.array([np.identity(n_states) for _ in range(n_ads)])
    # reward is a measure of the success: it tells us
    # whether user clicked on an ad
    rv = poisson.rvs(mu=mu, size=n_ads) / float(n_ads)
    means = pd.Series(rv)
    random.shuffle(means)
    actions = [map(lambda mu: BernoulliArm(mu).draw(), means) for _ in range(n_simulations)]
    clicks_df = pd.DataFrame.from_records(actions)
    R = (clicks_df.sum() / clicks_df.count()).values
    fake_state_rewards = np.zeros(n_ads)
    R = np.array([R, fake_state_rewards])
    return P, R


def solve_ctr_mdp(transitions, rewards, num_simulations=1000, discount=0.99):
    P, R = transitions, rewards
    vi = solve_mdp.test_algorithm(ValueIteration, P, R, discount=discount, num_sim=num_simulations)
    pi = solve_mdp.test_algorithm(PolicyIteration, P, R, discount=discount, num_sim=num_simulations)
    df = pd.concat([vi, pi])
    return df


def test_qlearning_algorithm(transitions, rewards,
                             discount=0.9,
                             num_sim_range=(10000, 10050),
                             verbose=False):
    P, R = transitions, rewards
    min_value, max_value = num_sim_range
    series = []
    for n in range(min_value, max_value):
        s = solve_mdp.solve_mdp_by_qlearning(P, R, discount=discount, max_iter=n, verbose=verbose)
        series.append(s)
    df = pd.concat(series, axis=1)
    return df.T


def evaluate_best_policy_choice(df, rewards):
    R = pd.Series(rewards[0, :])
    series = []
    for pol in df.policy:
        s = pd.Series(['' for n in range(len(R))])
        idx = pol[0]
        s[idx] = '!!!'
        series.append(s)
    df = pd.concat([R] + series, axis=1)
    df.rename(columns={0: 'ctr'}, inplace=True)
    return df.sort('ctr', ascending=False).set_index('ctr')


def test_discount_factor(transitions, rewards, discount_factor_range=(0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99), num_sim=50):
    dfs = []
    P, R = transitions, rewards
    for factor in discount_factor_range:
        series = []
        for n in range(1, num_sim):
            vi = solve_mdp.solve_mdp_by_iteration(ValueIteration, P, R, discount=factor, max_iter=n)
            series.append(vi)
        df = pd.concat(series, axis=1).T
        dfs.append(df)
    return pd.concat(dfs)


def plot_values(df):
    f, ax = plt.subplots(1, 2, figsize=(12, 3))
    df.groupby('algorithm')['mean_values'].plot(legend=True, ax=ax[0])
    ax[0].set_ylabel('mean value')
    df.groupby('algorithm')['accum_values'].plot(legend=True, ax=ax[1])
    ax[1].set_ylabel('accumulated values')
    plt.suptitle('CTR MDP: Value and Policy Iteration Comparison ( Fig. 2.3-2.4 )')


def plot_time_and_discount(df_times, df_factor):
    f, ax = plt.subplots(1, 2, figsize=(11, 3))
    df_times.groupby('algorithm')['time'].plot(legend=True, ax=ax[0],
                                               title='Algorithm Iteration Time\nFig. 2.5')
    ax[0].set_ylabel('time')
    ax[0].set_xlabel('iterations')
    df_factor.groupby('discount')['accum_values'].plot(legend=True, ax=ax[1],
                                               title='Convergence with Discount Factor\nFig. 2.6')
    ax[1].set_ylabel('discount factor')
    ax[1].set_xlabel('iterations')