import os
import logging
import pandas as pd

pd.set_option('display.mpl_style', 'default')


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

PROJECT_PATH = os.path.abspath(__file__)
ROOT_PATH = os.path.abspath(os.path.join(PROJECT_PATH, '../..'))
DATA_PATH = os.path.join(ROOT_PATH, 'data')
HIGGS_DATA = {'training': os.path.join(DATA_PATH, 'training.csv')}
BIDDING_DATA = {'training': os.path.join(DATA_PATH, 'bidding_training.csv')}

# how to split test data from the training data
TEST_DATA_SPLIT = 0.33
