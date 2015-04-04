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
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from algo_evaluation import BIDDING_DATA, HIGGS_DATA, WALDO_DATA, MONA_LISA_DATA, SCHEDULE_DATA, LOGGER, TEST_DATA_SPLIT
from algo_evaluation.plotting.plot_waldo_data import plot_waldo_kde


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


def load_higgs_train(sample_size=None, verbose=True, scale=False, prune_features=True):
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
    if prune_features:
        derived_features_names = [f for f in columns if f.startswith('DER')]
        derived_df = df[derived_features_names]
        features = derived_df[derived_features_names]
    else:
        features = df[columns[:-2]]
    weights = df['Weight']
    labels = df['Label']
    if scale:
        scaled_features = preprocessing.scale(features)
        features = pd.DataFrame(scaled_features, columns=features.columns)
    if verbose:
        print 'Size of the dataset:', features.shape[0]
        print 'Number of features:', features.shape[1]
        print 'Number of positives (signal):', labels.value_counts()['s']
        print 'Number of negatives (background):', labels.value_counts()['b']
    return features, weights, labels


def scale_features(features):
    #normalized_features = preprocessing.normalize(features)
    standardized_features = preprocessing.scale(features)
    return standardized_features


def load_higgs_test():
    df = pd.read_csv(HIGGS_DATA['test'])
    LOGGER.info('Loaded higgs test dataset of size %s', len(df))
    return df


def _categorical_feature_transform(df, categorical_features):
    def transform_feature(df, feature):
        feature_dict = [{feature: n} for n in df[feature]]
        vec = DictVectorizer(sparse=False)
        new_features = vec.fit_transform(feature_dict)
        new_features = pd.DataFrame(new_features, columns=vec.get_feature_names())
        #df.reset_index(inplace=True)
        df = pd.concat([df, new_features], axis=1)
        return df.drop(feature, axis=1)
    for feature in categorical_features:
        df = transform_feature(df, feature)
    return df


def load_bidding_train(verbose=True, scale=False):
    df = pd.read_csv(BIDDING_DATA['training'], low_memory=False).dropna()
    categorical_features = ['os', 'browser', 'country', 'region', 'user_local_hour']
    #cols_to_drop = ['user_id', 'ipaddress', 'source_timestamp', 'creative_uid', 'sitename']
    #df.drop(cols_to_drop, axis=1, inplace=True)
    feature_columns = df.columns.tolist()[:-1]
    le = LabelEncoder()
    transformed = [le.fit_transform(df[f]) for f in feature_columns]
    #transformed = _categorical_feature_transform(df[feature_columns], categorical_features)
    #transformed = df[feature_columns]
    features_df = pd.DataFrame.from_records(transformed).T.astype(float)
    labels = df['class']
    weights = np.ones(len(labels))
    if verbose:
        print 'Size of the dataset:', features_df.shape[0]
        print 'Number of features:', features_df.shape[1]
        print 'Number of converters:', labels.value_counts()['converter']
        print 'Number of non-converters:', labels.value_counts()['non-converter']
        print 'Number of leads:', labels.value_counts()['lead']
    if scale:
        scaled_features = preprocessing.StandardScaler().fit_transform(features_df)
        features_df = pd.DataFrame(scaled_features, columns=feature_columns)

    return features_df, weights, labels


def load_bidding_test():
    df = pd.read_csv(BIDDING_DATA['test'])
    LOGGER.info('Loaded higgs test dataset of size %s', len(df))
    return df


def load_waldo_dataset(display=False):
    df = pd.read_csv(WALDO_DATA)
    if display:
        plot_waldo_kde(df)
    return df


def load_schedule_dataset(display=False):
    flights = {}
    for line in file(SCHEDULE_DATA):
        origin,dest,depart,arrive,price=line.strip().split(',')
        flights.setdefault((origin,dest),[])

         # Add details to the list of possible flights
        flights[(origin,dest)].append((depart,arrive,int(price)))
    return flights

def load_mona_lisa(sample_size=1000, display=False):
    n = 100000  #number of records in file
    header_skip = 6
    if sample_size is None:
        skip = []
    else:
        skip = list(sorted(random.sample(xrange(header_skip, n), n-sample_size)))
    df = pd.read_csv(MONA_LISA_DATA,
                     sep=' ',
                     skiprows=range(header_skip) + skip,
                     header=None,
                     names=['id', 'X', 'Y'],
                     index_col=0)
    df = df.dropna()
    if display:
        df.plot(x='X', y='Y',
                kind='hexbin',
                figsize=(6, 4),
                xticks=[], yticks=[],
                legend=False)
    return df


def split_dataset(features, weights, labels):
    dataset = dict()
    trnfeatures, tstfeatures, \
        trnweights, tstweights, \
        trnlabels, tstlabels = train_test_split(features, weights, labels, test_size=TEST_DATA_SPLIT)
    dataset['training'] = {'features': trnfeatures, 'labels': trnlabels, 'weights': trnweights}
    dataset['test'] = {'features': tstfeatures, 'labels': tstlabels, 'weights': tstweights}
    return dataset


