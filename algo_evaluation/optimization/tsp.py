
# Code from Chapter 9 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# A demonstration of four methods of solving the Travelling Salesman Problem
import numpy as np
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist


def make_tsp(n_cities):
    positions = 2*np.random.rand(n_cities, 2)-1
    distances = np.zeros((n_cities, n_cities))

    for i in range(n_cities):
        for j in range(i+1, n_cities):
            distances[i, j] = np.sqrt((positions[i, 0] - positions[j, 0])**2 + (positions[i, 1] - positions[j, 1])**2)
            distances[j, i] = distances[i, j]

    return distances


def make_monalisa_tsp(monalisa_df):
    return squareform(pdist(monalisa_df[['X', 'Y']]))


def exhaustive(distances):
    nCities = np.shape(distances)[0]

    cityOrder = np.arange(nCities)

    distanceTravelled = 0
    for i in range(nCities-1):
        distanceTravelled += distances[cityOrder[i],cityOrder[i+1]]
    distanceTravelled += distances[cityOrder[nCities-1],0]

    for newOrder in permutation(range(nCities)):
        possibleDistanceTravelled = 0
        for i in range(nCities-1):
            possibleDistanceTravelled += distances[newOrder[i],newOrder[i+1]]
        possibleDistanceTravelled += distances[newOrder[nCities-1],0]

        if possibleDistanceTravelled < distanceTravelled:
            distanceTravelled = possibleDistanceTravelled
            cityOrder = newOrder

    return cityOrder, distanceTravelled


def permutation(order):
    order = tuple(order)
    if len(order)==1:
        yield order
    else:
        for i in range(len(order)):
            rest = order[:i] + order[i+1:]
            move = (order[i],)
            for smaller in permutation(rest):
                yield move + smaller


def greedy(distances):
    nCities = np.shape(distances)[0]
    distanceTravelled = 0

    # Need a version of the matrix we can trash
    dist = distances.copy()

    cityOrder = np.zeros(nCities)
    cityOrder[0] = np.random.randint(nCities)
    dist[:,cityOrder[0]] = np.Inf

    for i in range(nCities-1):
        cityOrder[i+1] = np.argmin(dist[cityOrder[i],:])
        distanceTravelled  += dist[cityOrder[i],cityOrder[i+1]]
        # Now exclude the chance of travelling to that city again
        dist[:,cityOrder[i+1]] = np.Inf

    # Now return to the original city
    distanceTravelled += distances[cityOrder[nCities-1],0]

    return cityOrder, distanceTravelled
