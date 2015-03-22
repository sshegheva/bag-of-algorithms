import matplotlib.pyplot as plt


def plot_cluster_score(higgs_cluster_df, bid_cluster_df):
    f, ax = plt.subplots(1, 2, figsize=(10, 4))
    higgs_cluster_df['score'].plot(title='Higgs Clustering (ARI)', ax=ax[0], legend=False, sharex=False)
    ax[0].set_ylabel("ARI")
    ax[0].set_xlabel("clusters")
    bid_cluster_df['score'].plot(title='Converters Clustering (ARI)', ax=ax[1], legend=False, sharex=False)
    ax[1].set_ylabel("ARI")
    ax[1].set_xlabel("clusters")
    plt.tight_layout()
