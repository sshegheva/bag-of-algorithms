import pandas as pd
import matplotlib.pyplot as plt
from mdptoolbox.mdp import ValueIteration
from mdptoolbox.mdp import PolicyIteration
from mdptoolbox.example import forest

from algo_evaluation.mdp.simulations import solve_mdp
reload(solve_mdp)


def solve_forest_example(forest_states_size=50,  r1=50, r2=25, fire_prob=0.1, num_simulations=50, discount=0.9):
    P, R = forest(S=forest_states_size, r1=r1, r2=r2, p=fire_prob)
    vi = solve_mdp.test_algorithm(ValueIteration, P, R, discount=discount, num_sim=num_simulations)
    pi = solve_mdp.test_algorithm(PolicyIteration, P, R, discount=discount, num_sim=num_simulations)
    df = pd.concat([vi, pi])
    return df


def test_qlearning_algorithm(forest_states_size=50,
                             fire_prob=0.01, r1=50, r2=25,
                             discount=0.9,
                             num_sim_range=(10000, 10050),
                             verbose=False):
    P, R = forest(S=forest_states_size, r1=r1, r2=r2, p=fire_prob)
    min_value, max_value = num_sim_range
    series = []
    for n in range(min_value, max_value):
        s = solve_mdp.solve_mdp_by_qlearning(P, R, discount=discount, max_iter=n, verbose=verbose)
        series.append(s)
    df = pd.concat(series, axis=1)
    return df.T


def test_forest_age(forest_age_range=(3, 10, 50, 100), num_sim=50):
    dfs = []
    for age in forest_age_range:
        series = []
        for n in range(1, num_sim + 1):
            P, R = forest(S=age)
            vi = solve_mdp.solve_mdp_by_iteration(ValueIteration, P, R, max_iter=n)
            series.append(vi)
        df = pd.concat(series, axis=1).T
        dfs.append(df)
    return pd.concat(dfs)


def test_discount_factor(discount_factor_range=(0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99), num_sim=50):
    dfs = []
    for factor in discount_factor_range:
        series = []
        for n in range(1, num_sim + 1):
            P, R = forest(S=50)
            vi = solve_mdp.solve_mdp_by_iteration(ValueIteration, P, R, discount=factor, max_iter=n)
            series.append(vi)
        df = pd.concat(series, axis=1).T
        dfs.append(df)
    return pd.concat(dfs)


def test_qlearning_deterministic(fireprob_range=(0.0, 0.1, 0.2, 0.5, 1.0), num_sim=50):
    dfs = []
    for factor in fireprob_range:
        series = []
        for n in range(10000, 10000 + num_sim):
            P, R = forest(S=50, p=factor, r1=50, r2=25)
            vi = solve_mdp.solve_mdp_by_qlearning(P, R, max_iter=n)
            vi = vi.append(pd.Series(factor, index=['fire_probability']))
            series.append(vi)
        df = pd.concat(series, axis=1).T
        dfs.append(df)
    return pd.concat(dfs)


def test_qlearning_discounted_reward(discount_factor_range=(0.1, 0.3, 0.5, 0.9, 0.99), num_sim=50):
    dfs = []
    for factor in discount_factor_range:
        series = []
        for n in range(10000, 10000 + num_sim):
            P, R = forest(S=50, p=0.0, r1=50, r2=25)
            mdp = solve_mdp.solve_mdp_by_qlearning(P, R, discount=factor, max_iter=n)
            series.append(mdp)
        df = pd.concat(series, axis=1).T
        dfs.append(df)
    return pd.concat(dfs)


def test_fire_probability(fireprob_range=(0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99), num_sim=50):
    dfs = []
    for factor in fireprob_range:
        series = []
        for n in range(1, num_sim + 1):
            P, R = forest(S=50, p=factor)
            vi = solve_mdp.solve_mdp_by_iteration(ValueIteration, P, R, max_iter=n)
            vi = vi.append(pd.Series(factor, index=['fire_probability']))
            series.append(vi)
        df = pd.concat(series, axis=1).T
        dfs.append(df)
    return pd.concat(dfs)


def plot_values(df, title='Forest Management MDP: Value and Policy Iteration Comparison (Figures 1.1-1.2)'):
    f, ax = plt.subplots(1, 2, figsize=(12, 3))
    df.groupby('algorithm')['mean_values'].plot(legend=True, ax=ax[0])
    ax[0].set_ylabel('mean value')
    df.groupby('algorithm')['accum_values'].plot(legend=True, ax=ax[1])
    ax[1].set_ylabel('accumulated values')
    plt.suptitle(title)


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
    df_discount.set_index('max_iter').groupby('discount')['accum_values'].plot(legend=True, ax=ax[0],
                                               title='Convergence with Discount Factor\nFigure 1.5')
    ax[0].set_ylabel('accumulated reward')
    ax[0].set_xlabel('iterations')
    df_prob.set_index('max_iter').groupby('fire_probability')['accum_values'].plot(legend=True,  ax=ax[1],
                                                    title='Convergence with Probability of Fire\nFigure 1.6')
    ax[1].set_ylabel('accumulated reward')
    ax[1].set_xlabel('iterations')
    plt.tight_layout()
