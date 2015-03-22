import matplotlib.pyplot as plt


def plot_kmeans_cluster_score(higgs_cluster_df, bid_cluster_df):
    f, ax = plt.subplots(1, 2, figsize=(11, 4))
    higgs_cluster_df['score'].plot(title='KMeans Higgs Clustering (ARI)', ax=ax[0], legend=False, sharex=False)
    ax[0].set_ylabel("ARI")
    ax[0].set_xlabel("clusters")
    bid_cluster_df['score'].plot(title='Kmeans Converters Clustering (ARI)', ax=ax[1], legend=False, sharex=False)
    ax[1].set_ylabel("ARI")
    ax[1].set_xlabel("clusters")
    plt.tight_layout()


def plot_gmm_cluster_score(higgs_cluster_df, bid_cluster_df):
    f, ax = plt.subplots(1, 2, figsize=(11, 4))
    higgs_cluster_df['score'].plot(title='GMM Higgs Clustering', ax=ax[0], legend=False, sharex=False)
    ax[0].set_ylabel("loglikelihood")
    ax[0].set_xlabel("clusters")
    bid_cluster_df['score'].plot(title='GMM Converters Clustering', ax=ax[1], legend=False, sharex=False)
    ax[1].set_ylabel("loglikelihood")
    ax[1].set_xlabel("clusters")
    plt.tight_layout()
