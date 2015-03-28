import pandas as pd
from algo_evaluation.supervised import neural_network as nn
from algo_evaluation.feature_selection import reduce_dimensions
reload(reduce_dimensions)

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
        result = [[name] + nn.run_neural_net(data) + times[name]]
        results.append(result)
    df = pd.DataFrame.from_records(results, columns=['algo', 'epocs', 'trnerr', 'tstacc', 'transformation_time'])
    return df