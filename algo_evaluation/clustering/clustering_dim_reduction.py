"""
Reproduce your clustering experiments,
but on the data after you've run dimensionality reduction on it.
"""
from time import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import mixture
from algo_evaluation.feature_selection import pca_eval, rand_projections, ica_eval, trunk_svd
from algo_evaluation.feature_selection import reduce_dimensions
from algo_evaluation.clustering import kmeans_eval
from algo_evaluation.clustering import plot_cluster_estimation
from algo_evaluation.clustering import gmm_eval
from algo_evaluation.supervised.neural_network import run_neural_net


def reduce_higgs(higgs_data, n_components):
    return reduce_dimensions.run_higg_dimensionality_reduction(higgs_data, n_components=n_components)


def reduce_converters(bid_data, n_components):
    return reduce_dimensions.run_converters_dimensionality_reduction(bid_data, n_components=n_components)


def evaluate_higgs_clustering(higgs_data, n_components):
    transformations, times = reduce_higgs(higgs_data, n_components)
    dfs = []
    for name, transformation in transformations.iteritems():
        data = (transformation, higgs_data[1], higgs_data[2])
        df_k = kmeans_eval.estimate_clusters(data)
        df_k['reduction_algo'] = name
        df_k['reduction_time'] = times[name]

        df_gmm = gmm_eval.estimate_clusters(data)
        df_gmm['reduction_algo'] = name
        df_gmm['reduction_time'] = times[name]

        df = pd.concat([df_k, df_gmm])
        dfs.append(df)
    df = pd.concat(dfs)
    return df


def evaluate_conv_clustering(bid_data, n_components):
    transformations, times = reduce_converters(bid_data, n_components)
    dfs = []
    for name, transformation in transformations.iteritems():
        data = (transformation, bid_data[1], bid_data[2])
        df_k = kmeans_eval.estimate_clusters(data)
        df_k['reduction_algo'] = name
        df_k['reduction_time'] = times[name]

        df_gmm = gmm_eval.estimate_clusters(data)
        df_gmm['reduction_algo'] = name
        df_gmm['reduction_time'] = times[name]

        df = pd.concat([df_k, df_gmm])
        dfs.append(df)
    df = pd.concat(dfs)
    return df


def kmeans_transform(higgs_data, n_clusters, n_components, display=False):
    start = time()
    reduced_higgs_data, elapsed = pca_eval.transform(higgs_data, n_components=n_components)
    cluster_data = KMeans(n_clusters=n_clusters).fit_transform(reduced_higgs_data)
    elapsed = time() - start
    data = {'features': cluster_data, 'weights': higgs_data[1], 'labels': higgs_data[2]}
    if display and n_clusters == 2:
        df = pd.DataFrame.from_records(cluster_data, columns=['new_feature_' + str(n) for n in range(n_clusters)])
        df['label'] = higgs_data[2].values
        ax = df[df.label == 's'].plot(x='new_feature_0', y='new_feature_1',
                                      kind='scatter', color='darkgreen', label='signal')
        df[df.label == 'b'].plot(x='new_feature_0', y='new_feature_1',
                                 kind='scatter', color='darkred', ax=ax, label='background')
    return data, elapsed


def compare_cluster_runtime(data, n_clusters, n_components):
    t0 = time()
    features = data[0]
    KMeans(n_clusters=n_clusters).fit_transform(features)
    t1 = time() - t0
    t0 = time()
    mixture.GMM(n_components=n_clusters).fit(features)
    t2 = time() - t0
    reduced_higgs_data, t3 = pca_eval.transform(data, n_components=n_components)
    reduced_higgs_data, t4 = rand_projections.transform(data, n_components=n_components)
    reduced_higgs_data, t5 = ica_eval.transform(data, n_components=n_components)
    reduced_higgs_data, t6 = trunk_svd.transform(data, n_components=n_components)
    t0 = time()
    KMeans(n_clusters=n_clusters).fit_transform(reduced_higgs_data)
    t7 = time() - t0
    t0 = time()
    mixture.GMM(n_components=n_clusters).fit(reduced_higgs_data)
    t8 = time() - t0
    ser = pd.Series([t1, t2, t3, t4, t5, t6, t7, t8], index=['original Kmeans clustering',
                                                     'original GMM clustering',
                                                     'PCA', 'RCA', 'ICA', 'LSA',
                                                     'reduced Kmeans clustering',
                                                     'reduced GMM clustering'])
    ser.name = 'time'
    return ser


def plot_running_time(higgs_time, conv_time):
    f, ax = plt.subplots(1, 2, figsize=(12, 3))
    higgs_time.plot(kind='bar', ax=ax[0], title='Higgs Runtime Comparison')
    conv_time.plot(kind='bar', ax=ax[1], title='Converters Runtime Comparison')


def neural_net_post_clustering(data):
    return run_neural_net(data)

def plot_cluster_performance(df_higgs, df_conv, reduction_algo='PCA'):
    df_higgs_kmeans = df_higgs.query('algo == "kmeans"').query('reduction_algo == "{}"'.format(reduction_algo))
    df_higgs_gmm = df_higgs.query('algo == "gmm"').query('reduction_algo == "{}"'.format(reduction_algo))
    df_conv_kmeans = df_conv.query('algo == "kmeans"').query('reduction_algo == "{}"'.format(reduction_algo))
    df_conv_gmm = df_conv.query('algo == "gmm"').query('reduction_algo == "{}"'.format(reduction_algo))
    plot_cluster_estimation.plot_kmeans_cluster_score(df_higgs_kmeans, df_conv_kmeans)
    plot_cluster_estimation.plot_gmm_cluster_score(df_higgs_gmm, df_conv_gmm)