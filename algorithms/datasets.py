"""
Utility to load datasets from existing data directories
"""
import pandas as pd
from sklearn.cross_validation import train_test_split
from algorithms import TEST_DATA, TRAINING_DATA, LOGGER


def load_higgs_train():
    df = pd.read_csv(TRAINING_DATA)
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