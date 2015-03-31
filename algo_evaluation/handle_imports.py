from algo_evaluation import datasets

from algo_evaluation.clustering import kmeans_eval
from algo_evaluation.clustering import plot_cluster_estimation
from algo_evaluation.clustering import gmm_eval
from algo_evaluation.clustering import clustering_dim_reduction as clustering

from algo_evaluation.feature_selection import pca_eval
reload(pca_eval)
from algo_evaluation.feature_selection import trunk_svd
reload(trunk_svd)
from algo_evaluation.feature_selection import rand_projections
reload(rand_projections)
from algo_evaluation.feature_selection import ica_eval
reload(ica_eval)

from algo_evaluation.supervised import neural_network_dim_reduction as nn
reload(nn)