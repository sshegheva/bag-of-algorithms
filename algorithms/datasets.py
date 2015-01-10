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
"""
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from algorithms import TEST_DATA, TRAINING_DATA, LOGGER


def load_higgs_train(sample_size=None):
    """
    Load higgs dataset

    do some data cleanup:
    - remove data entries (anomalies) which are outside of normal range
    - pick only derived feature
    :return: dataframe
    """
    df = pd.read_csv(TRAINING_DATA, nrows=sample_size)
    df = df.replace(-999.000, np.nan).dropna()
    df.set_index('EventId', inplace=True)
    LOGGER.info('Loaded higgs training dataset of size %s', len(df))
    columns = df.columns
    derived_features_names = [f for f in columns if f.startswith('DER')]
    derived_df = df[derived_features_names]
    features = derived_df[derived_features_names]
    weights = df['Weight']
    labels = df['Label']
    return features, weights, labels


def load_higgs_test():
    df = pd.read_csv(TEST_DATA)
    LOGGER.info('Loaded higgs test dataset of size %s', len(df))
    return df


def split_dataset(features, weights, labels):
    return train_test_split(features, weights, labels, test_size=0.33)