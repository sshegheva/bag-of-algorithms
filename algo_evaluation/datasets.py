"""
Utility to load datasets from existing data directories

Higgs Dataset:

    - all variables are floating point, except PRI_jet_num which is integer
    - variables prefixed with PRI (for PRImitives) are raw quantities
    about the bunch collision as measured by the detector.
    - variables prefixed with DER (for DERived) are quantities computed from the primitive features,
    which were selected by  the physicists of ATLAS
    it can happen that for some entries some variables are meaningless or cannot be computed;
    in this case, their value is -999.0 which is outside the normal range of all variables


Bidding Dataset:

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from algo_evaluation import BIDDING_DATA, HIGGS_DATA, LOGGER, TEST_DATA_SPLIT


def describe_higgs_raw():
    df = pd.read_csv(HIGGS_DATA['training'])
    derived = ['EventId', 'DER_mass_jet_jet', 'Label']
    primitive = ['EventId', 'PRI_jet_subleading_pt', 'Label']
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8), sharex=False, sharey=False)
    df[derived][df['Label'] == 's'].plot(kind='scatter',
                                          x=derived[0],
                                          y=derived[1],
                                          color='DarkBlue',
                                          label='Signal',
                                          ax=axes[0])
    df[derived][df['Label'] == 'b'].plot(kind='scatter',
                                         x=derived[0],
                                         y=derived[1],
                                         color='DarkRed',
                                         label='Background',
                                         ax=axes[0])
    df[primitive][df['Label'] == 's'].plot(kind='scatter',
                                          x=primitive[0],
                                          y=primitive[1],
                                          color='DarkBlue',
                                          label='Signal',
                                          ax=axes[1])
    df[primitive][df['Label'] == 'b'].plot(kind='scatter',
                                         x=primitive[0],
                                         y=primitive[1],
                                         color='DarkRed',
                                         label='Background',
                                         ax=axes[1])
    return df



def load_higgs_train(sample_size=None):
    """
    Load higgs dataset

    do some data cleanup:
    - remove data entries (anomalies) which are outside of normal range
    - pick only derived feature
    :return: dataframe
    """
    df = pd.read_csv(HIGGS_DATA['training'], nrows=sample_size)
    df = df.replace(-999.000, np.nan).dropna()
    df.set_index('EventId', inplace=True)
    columns = df.columns
    derived_features_names = [f for f in columns if f.startswith('DER')]
    derived_df = df[derived_features_names]
    features = derived_df[derived_features_names]
    weights = df['Weight']
    labels = df['Label']
    return features, weights, labels


def load_higgs_test():
    df = pd.read_csv(HIGGS_DATA['test'])
    LOGGER.info('Loaded higgs test dataset of size %s', len(df))
    return df


def load_bidding_train():
    df = pd.read_csv(BIDDING_DATA['training'], low_memory=False).dropna()
    features = df[df.columns.tolist()[:-1]]
    le = LabelEncoder()
    transformed = [le.fit_transform(features[f]) for f in features.columns]
    transformed_df = pd.DataFrame.from_records(transformed).transpose()
    labels = df['class']
    weights = np.ones(len(labels))
    return transformed_df, weights, labels


def load_bidding_test():
    df = pd.read_csv(BIDDING_DATA['test'])
    LOGGER.info('Loaded higgs test dataset of size %s', len(df))
    return df


def split_dataset(features, weights, labels):
    dataset = dict()
    trnfeatures, tstfeatures, \
        trnweights, tstweights, \
        trnlabels, tstlabels = train_test_split(features, weights, labels, test_size=TEST_DATA_SPLIT)
    dataset['training'] = {'features': trnfeatures, 'labels': trnlabels, 'weights': trnweights}
    dataset['test'] = {'features': tstfeatures, 'labels': tstlabels, 'weights': tstweights}
    return dataset


