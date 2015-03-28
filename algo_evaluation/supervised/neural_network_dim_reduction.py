import pandas as pd
import matplotlib.pyplot as plt
from algo_evaluation.supervised import neural_network as nn
from algo_evaluation.feature_selection import reduce_dimensions

higgs_backprop_err = 100 * (1 - 0.47)
higgs_weight_learn_err = 100 * (1 - 0.7)


def reduce_higgs(higgs_data, n_components):
    return reduce_dimensions.run_higg_dimensionality_reduction(higgs_data, n_components=n_components)


def reduce_converters(bid_data, n_components):
    return reduce_dimensions.run_converters_dimensionality_reduction(bid_data, n_components=n_components)


def evaluate_nn_accuracy(higgs_data, n_components):
    transformations, times = reduce_higgs(higgs_data, n_components)
    results = []
    for name, transformation in transformations.iteritems():
        data = {'features': transformation, 'weights': higgs_data[1], 'labels': higgs_data[2]}
        nn_res = nn.run_neural_net(data)
        result = [name] + list(nn_res) + [times[name]]
        results.append(result)
    df = pd.DataFrame.from_records(results, columns=['algo', 'epocs', 'trnerr', 'tsterr', 'transformation_time'])
    return df


def plot_evaluation(df):
    f, ax = plt.subplots(1, 2, figsize=(12, 3))
    df_err = df[['algo', 'tstacc']].set_index('algo').T
    df_err['Backprop'] = higgs_backprop_err
    df_err['GA'] = higgs_weight_learn_err
    df_err.T.plot(ax=ax[0], kind='bar', title='Higgs Classification Error')
    df_t = df[['algo', 'transformation_time']]
    df_t.set_index('algo').plot(ax=ax[1], kind='bar', title='Dimensionality Reduction Time', color='darkgreen')