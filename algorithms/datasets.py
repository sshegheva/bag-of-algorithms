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


def load_higgs_train():
    df = pd.read_csv(TRAINING_DATA)
    df = df.replace(-999.000, np.nan).dropna()
    LOGGER.info('Loaded higgs training dataset of size %s', len(df))
    columns = df.columns
    features = df[columns[:-2]]
    weights = df[columns[-2]]
    labels = df[columns[-1]]
    return features, weights, labels


def load_higgs_test():
    df = pd.read_csv(TEST_DATA)
    LOGGER.info('Loaded higgs test dataset of size %s', len(df))
    return df


def split_dataset(features, labels):
    return train_test_split(features, labels, test_size=0.33)