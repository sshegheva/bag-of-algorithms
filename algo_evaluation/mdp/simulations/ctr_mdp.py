import random
import numpy as np
import pandas as pd
from scipy.stats import poisson
import matplotlib.pyplot as plt

from algo_evaluation.mdp.simulations.solve_mdp import test_algorithm, solve_mdp


class BernoulliArm():
    def __init__(self, p):
        self.p = p

    def draw(self):
        if random.random() > self.p:
            return 0.0
        else:
            return 1.0


def simulate_user_click(n_ads=10.0, mu=7.0, display=True):
    rv = poisson.rvs(mu=mu, size=n_ads) / n_ads
    means = pd.Series(rv)
    random.shuffle(means)
    actions = pd.Series(map(lambda mu: BernoulliArm(mu).draw(), means))
    if display:
        f, ax = plt.subplots(1, 2, figsize=(13, 3))

        means.hist(normed=True, ax=ax[0])
        means.plot(kind='kde', ax=ax[0])
        ax[0].set_xlabel('probability of clicking on add')
        ax[0].set_title('Probability click distribution per ad')

        actions.hist(normed=True, ax=ax[1])
        actions.plot(kind='kde', ax=ax[1])
        ax[1].set_xlabel('non-clicks/clicks')
        ax[1].set_title('Example of distribution of clicks vs ignores')
    return actions


def create_ctr_mdp(n_ads=10, n_users=1000):
    """
    create markovian process to
    demonstrate the optimization of the click through rate
    :param n_ads:
    :param n_users:
    :return:
    """
    n_states = 1
    P = np.ones((n_ads, n_states, n_states))
    # reward is a measure of the success: it tells us
    # whether user clicked on an ad
    R = np.random.random()
    return P, R