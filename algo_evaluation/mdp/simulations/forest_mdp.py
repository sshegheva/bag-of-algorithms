import pandas as pd
import matplotlib.pyplot as plt
from mdptoolbox.mdp import ValueIteration
from mdptoolbox.mdp import PolicyIteration
from mdptoolbox.example import forest

from algo_evaluation.mdp.simulations.solve_mdp import test_algorithm, solve_mdp_by_iteration, solve_mdp_by_qlearning


def solve_forest_example(forest_states_size=50, fire_prob=0.1, num_simulations=50, discount=0.9):
    P, R = forest(S=forest_states_size, p=fire_prob)
    vi = test_algorithm(ValueIteration, P, R, discount=discount, num_sim=num_simulations)
    pi = test_algorithm(PolicyIteration, P, R, discount=discount, num_sim=num_simulations)
    df = pd.concat([vi, pi])
    return df


def test_qlearning_algorithm(forest_states_size=50, fire_prob=0.1, discount=0.9):
    P, R = forest(S=forest_states_size, p=fire_prob)
    return solve_mdp_by_qlearning(P, R, discount=discount)


def test_forest_age(forest_age_range=(3, 10, 50, 100), num_sim=50):
    dfs = []
    for age in forest_age_range:
        series = []
        for n in range(1, num_sim):
            P, R = forest(S=age)
            vi = solve_mdp_by_iteration(ValueIteration, P, R, max_iter=n)
            series.append(vi)
        df = pd.concat(series, axis=1).T
        dfs.append(df)
    return pd.concat(dfs)


def test_discount_factor(discount_factor_range=(0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99), num_sim=50):
    dfs = []
    for factor in discount_factor_range:
        series = []
        for n in range(1, num_sim):
            P, R = forest(S=100)
            vi = solve_mdp_by_iteration(ValueIteration, P, R, discount=factor, max_iter=n)
            series.append(vi)
        df = pd.concat(series, axis=1).T
        dfs.append(df)
    return pd.concat(dfs)


def test_fire_probability(fireprob_range=(0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99), num_sim=50):
    dfs = []
    for factor in fireprob_range:
        series = []
        for n in range(1, num_sim):
            P, R = forest(S=100, p=factor)
            vi = solve_mdp_by_iteration(ValueIteration, P, R, max_iter=n)
            vi = vi.append(pd.Series(factor, index=['fire_probability']))
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
    plt.suptitle('Forest Management MDP: Value and Policy Iteration Comparison (Figures 1.1-1.2)')


def plot_time_and_states(df_times, df_states):
    f, ax = plt.subplots(1, 2, figsize=(11, 3))
    df_times.groupby('algorithm')['time'].plot(legend=True, ax=ax[0],
                                               title='Algorithm Iteration Time\nFigure 1.3')
    ax[0].set_ylabel('time')
    ax[0].set_xlabel('iterations')
    df_states.groupby(['age'])['accum_values'].plot(legend=True,  ax=ax[1],
                                                    title='Accumulated value with age\nFigure 1.4')
    ax[1].set_ylabel('accumulated values')
    ax[1].set_xlabel('forest age')


def plot_discount_factor_and_fire_probability(df_discount, df_prob):
    f, ax = plt.subplots(1, 2, figsize=(12, 4))
    df_discount.groupby('discount')['accum_values'].plot(legend=True, ax=ax[0],
                                               title='Convergence with Discount Factor\nFigure 1.5')
    ax[0].set_ylabel('discount factor')
    ax[0].set_xlabel('iterations')
    df_prob.groupby('fire_probability')['accum_values'].plot(legend=True,  ax=ax[1],
                                                    title='Convergence with Probability of Fire\nFigure 1.6')
    ax[1].set_ylabel('probability of fire')
    ax[1].set_xlabel('iterations')
    plt.tight_layout()
