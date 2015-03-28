"""
Reproduce your clustering experiments,
but on the data after you've run dimensionality reduction on it.
"""
import pandas as pd
from algo_evaluation.feature_selection import reduce_dimensions
from algo_evaluation.clustering import kmeans_eval
from algo_evaluation.clustering import plot_cluster_estimation
from algo_evaluation.clustering import gmm_eval
reload(reduce_dimensions)


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


def plot_cluster_performance(df_higgs, df_conv, reduction_algo='PCA'):
    df_higgs_kmeans = df_higgs.query('algo == "kmeans"').query('reduction_algo == "{}"'.format(reduction_algo))
    df_higgs_gmm = df_higgs.query('algo == "gmm"').query('reduction_algo == "{}"'.format(reduction_algo))
    df_conv_kmeans = df_conv.query('algo == "kmeans"').query('reduction_algo == "{}"'.format(reduction_algo))
    df_conv_gmm = df_conv.query('algo == "gmm"').query('reduction_algo == "{}"'.format(reduction_algo))
    plot_cluster_estimation.plot_kmeans_cluster_score(df_higgs_kmeans, df_conv_kmeans)
    plot_cluster_estimation.plot_gmm_cluster_score(df_higgs_gmm, df_conv_gmm)