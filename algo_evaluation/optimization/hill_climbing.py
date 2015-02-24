import numpy as np
import pandas as pd
import time


def hill_climbing(distances, n_evaluations=1000):

    nCities = np.shape(distances)[0]

    cityOrder = np.arange(nCities)
    np.random.shuffle(cityOrder)

    cost = 0
    for i in range(nCities-1):
        cost += distances[cityOrder[i], cityOrder[i+1]]
    cost += distances[cityOrder[nCities-1], 0]

    for i in range(n_evaluations):
        # Choose cities to swap
        city1 = np.random.randint(nCities)
        city2 = np.random.randint(nCities)

        if city1 != city2:
            # Reorder the set of cities
            possibleCityOrder = cityOrder.copy()
            possibleCityOrder = np.where(possibleCityOrder == city1, -1, possibleCityOrder)
            possibleCityOrder = np.where(possibleCityOrder == city2, city1, possibleCityOrder)
            possibleCityOrder = np.where(possibleCityOrder == -1, city2, possibleCityOrder)

            # Work out the new distances
            # This can be done more efficiently
            new_cost = 0
            for j in range(nCities-1):
                new_cost += distances[possibleCityOrder[j], possibleCityOrder[j+1]]
            new_fitness = 1 / new_cost
            cost += distances[cityOrder[nCities-1], 0]
            fitness = 1 / cost

            if new_fitness > fitness:
                fitness = new_fitness
                cityOrder = possibleCityOrder

    return cityOrder, cost


def evaluate_hc(optimization_problem, evaluation_range=xrange(10, 1000, 10)):
    data = []
    for n in evaluation_range:
        start = time.time()
        solution, optimal_value = hill_climbing(optimization_problem, n_evaluations=n)
        elapsed = time.time() - start
        data.append(['hill_climbing', n, optimal_value, elapsed])
    df = pd.DataFrame.from_records(data, columns=['algo', 'evaluations', 'optimal_value', 'running_time'])
    return df