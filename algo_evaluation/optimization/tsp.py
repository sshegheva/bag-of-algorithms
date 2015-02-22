import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from algo_evaluation.optimization.hill_climbing import evaluate_hc
from algo_evaluation.optimization.simulated_annealing import evaluate_sa


def make_monalisa_tsp(monalisa_df):
    return squareform(pdist(monalisa_df[['X', 'Y']]))


def plot_evaluation(df):
    sns.set(style="ticks")
    sns.lmplot("evaluations", "optimal_value", data=df, col='algo')


def compare_all(optimization_problem):
    hc_df = evaluate_hc(optimization_problem)
    sa_df = evaluate_sa(optimization_problem)
    df = pd.concat([hc_df, sa_df])
    return df
