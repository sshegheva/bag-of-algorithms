import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

#pd.set_option('display.mpl_style', 'default')
sb.set_style("white")
plt.style.use("https://gist.githubusercontent.com/rhiever/a4fb39bfab4b33af0018/raw/9c857ed71cd8361b0b0da2c36404fa7245f3de5f/tableau20.mplstyle")


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
WALDO_DATA = os.path.join(DATA_PATH, 'whereis-waldo-locations.csv')
MONA_LISA_DATA = os.path.join(DATA_PATH, 'mona-lisa100k.tsp')

# how to split test data from the training data
TEST_DATA_SPLIT = 0.33
