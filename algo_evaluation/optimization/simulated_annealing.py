import numpy as np
import pandas as pd
import time


def simulated_annealing(distances, T=500, c=0.8, n_tests=100):

    nCities = np.shape(distances)[0]

    cityOrder = np.arange(nCities)
    np.random.shuffle(cityOrder)

    cost = 0
    for i in range(nCities-1):
        cost += distances[cityOrder[i], cityOrder[i+1]]
    cost += distances[cityOrder[nCities-1], 0]

    while T > 1:
        for i in range(n_tests):
            # Choose cities to swap
            city1 = np.random.randint(nCities)
            city2 = np.random.randint(nCities)

            if city1 != city2:
                # Reorder the set of cities
                possibleCityOrder = cityOrder.copy()
                possibleCityOrder = np.where(possibleCityOrder==city1,-1,possibleCityOrder)
                possibleCityOrder = np.where(possibleCityOrder==city2,city1,possibleCityOrder)
                possibleCityOrder = np.where(possibleCityOrder==-1,city2,possibleCityOrder)

                # Work out the new distances
                # This can be done more efficiently
                new_cost = 0
                for j in range(nCities-1):
                    new_cost += distances[possibleCityOrder[j],possibleCityOrder[j+1]]
                new_fitnes = 1 / new_cost
                cost += distances[cityOrder[nCities-1], 0]
                fitness = 1 / cost

                if new_fitnes > fitness or (fitness - new_fitnes) < T*np.log(np.random.rand()):
                    fitness = new_fitnes
                    cityOrder = possibleCityOrder

            # Annealing schedule
            T *= c

    return cityOrder, cost


def evaluate_sa(optimization_problem, tests_range=xrange(10, 1000, 10)):
    data = []

    def evaluate_temperature(temp):
        for t in tests_range:
            start = time.time()
            solution, optimal_value = simulated_annealing(optimization_problem, T=temp, n_tests=t)
            elapsed = time.time() - start
            data.append(['simulated_annealing', t, temp, optimal_value, elapsed])
        df = pd.DataFrame.from_records(data,
                                       columns=['algo', 'evaluations', 'temperature', 'optimal_value', 'running_time'])
        return df
    dfs = [evaluate_temperature(t) for t in [10, 100, 500, 1000]]
    return pd.concat(dfs)