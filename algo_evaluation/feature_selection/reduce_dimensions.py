import pandas as pd
import matplotlib.pyplot as plt
from algo_evaluation.feature_selection import pca_eval, rand_projections, ica_eval, trunk_svd


def run_higg_dimensionality_reduction(higgs_data, n_components):
    pca_trns, pca_elapsed = pca_eval.transform(higgs_data, n_components=n_components)
    rand_proj_trns, rand_proj_elapsed = rand_projections.transform(higgs_data, n_components=n_components)
    ica_trns, ica_elapsed = ica_eval.transform(higgs_data, n_components=n_components)
    lsa_trns, lsa_elapsed = trunk_svd.transform(higgs_data, n_components=n_components)
    transformation_time = pd.Series([pca_elapsed, rand_proj_elapsed, ica_elapsed, lsa_elapsed],
                                    index=['PCA', 'RCA', 'ICA', 'LSA'],
                                    name='transformation_time')
    return {'pca': pca_trns, 'rand_proj': rand_projections, 'ica': ica_trns, 'lsa': lsa_trns}, transformation_time


def run_converters_dimensionality_reduction(conv_data, n_components):
    pca_trns, pca_elapsed = pca_eval.transform(conv_data, n_components=n_components)
    rand_proj_trns, rand_proj_elapsed = rand_projections.transform(conv_data, n_components=n_components)
    ica_trns, ica_elapsed = ica_eval.transform(conv_data, n_components=n_components)
    lsa_trns, lsa_elapsed = trunk_svd.transform(conv_data, n_components=n_components)
    transformation_time = pd.Series([pca_elapsed, rand_proj_elapsed, ica_elapsed, lsa_elapsed],
                                    index=['PCA', 'RCA', 'ICA', 'LSA'],
                                    name='transformation_time')
    return {'pca': pca_trns, 'rand_proj': rand_projections, 'ica': ica_trns, 'lsa': lsa_trns}, transformation_time


def plot_transformation_time(higg_time, conv_time):
    f, ax = plt.subplots(1, 2, figsize=(12, 2))
    higg_time.plot(kind='barh', title='Higgs Transformation Time', ax=ax[0])
    conv_time.plot(kind='barh', title='Converters Transformation Time', ax=ax[1])