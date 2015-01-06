from sklearn.metrics import accuracy_score
from algorithms import LOGGER


def evaluation_score(observations, predictions):
    score = accuracy_score(observations, predictions)
    LOGGER.info('Accuracy score %s', score)
    return score