import matplotlib.pyplot as plt
import pandas as pd


def plot_optimal_values(rhc_df, sa_df, ga_df, mm_df):
    f, ax = plt.subplots(2, 2, figsize=(10, 8))
    rhc_df['optimal_value'].plot(title='Hill Climber', ax=ax[0][0], legend=False, sharex=False)
    ax[0][0].set_ylabel("optimal value")
    sa_df['optimal_value'].plot(title='Simulated Annealing', logx=True, ax=ax[0][1], legend=False, sharex=False)
    ax[0][1].set_ylabel("optimal value")
    ga_df['optimal_value'].plot(title='Genetic Algorithm', ax=ax[1][0], legend=False, sharex=False)
    ax[1][0].set_ylabel("optimal value")
    mm_df.groupby('iteration').max()['optimal_value'].plot(title='MIMIC', ax=ax[1][1], legend=False, sharex=False)
    ax[1][1].set_ylabel("optimal value")
    plt.tight_layout()


def collect_and_plot_clocktimes(experiment_results):
    algos = ['RHC', 'SA', 'GA', 'MIMIC']
    domains = ['WaldoOpt', 'HyperOpt', 'CronOpt']
    data = []
    for dom in experiment_results:
        times = [algo['time'].mean() for algo in dom]
        data.append(times)
    df = pd.DataFrame.from_records(data, columns=algos)
    df.rename(index={i: domains[i] for i in range(len(df))}, inplace=True)
    df.plot(kind='barh', figsize=(8, 4))
    plt.xlabel('Time (sec)')
    return df