import numpy as np


def backpacker():
    # reward matrix
    R = np.array([[-5, 0, -np.inf, -np.inf, -np.inf, -np.inf],
                  [0, -5, 0, 0, -np.inf, -np.inf],
                  [-np.inf, 0, -5, 0, -np.inf, 100],
                  [-np.inf, 0, 0, -5, 0, -np.inf],
                  [-np.inf, -np.inf, -np.inf, 0, -5, 100],
                  [-np.inf, -np.inf, 0, -np.inf, -np.inf, 0]])
    # transition probability matrix
    P = np.array([[1, 1, 0, 0, 0, 0],
                  [1, 1, 1, 1, 0, 0],
                  [0, 1, 1, 1, 0, 1],
                  [0, 1, 1, 1, 1, 0],
                  [0, 0, 0, 1, 1, 1],
                  [0, 0, 1, 0, 1, 1]])
    return P, R
