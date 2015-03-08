import matplotlib.pyplot as plt


def plot_optimal_values(rhc_df, sa_df, ga_df):
    f, ax = plt.subplots(2, 2, figsize=(10,8))
    rhc_df['optimal_value'].plot(title='Hill Climber', ax=ax[0][0], legend=False, sharex=False)
    ax[0][0].set_ylabel("optimal value")
    sa_df['optimal_value'].plot(title='Simulated Annealing', logx=True, ax=ax[0][1], legend=False, sharex=False)
    ax[0][1].set_ylabel("optimal value")
    ga_df['optimal_value'].plot(title='Genetic Algorithm', ax=ax[1][0], legend=False, sharex=False)
    ax[1][0].set_ylabel("optimal value")
    plt.tight_layout()
