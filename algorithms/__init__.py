import os
import logging

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

PROJECT_PATH = os.path.abspath(__file__)
ROOT_PATH = os.path.abspath(os.path.join(PROJECT_PATH, '../..'))
DATA_PATH = os.path.join(ROOT_PATH, 'data')
TRAINING_DATA = os.path.join(DATA_PATH, 'training/training.csv')
TEST_DATA = os.path.join(DATA_PATH, 'test/test.csv')
