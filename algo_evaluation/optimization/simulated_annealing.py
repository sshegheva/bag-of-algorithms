import numpy as np
import pandas as pd
import time


def simulated_annealing(distances, T=10, c=0.8, n_evaluations=10):

    nCities = np.shape(distances)[0]

    cityOrder = np.arange(nCities)
    np.random.shuffle(cityOrder)

    distanceTravelled = 0
    for i in range(nCities-1):
        distanceTravelled += distances[cityOrder[i],cityOrder[i+1]]
    distanceTravelled += distances[cityOrder[nCities-1], 0]

    while T > 1:
        for i in range(n_evaluations):
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
                newDistanceTravelled = 0
                for j in range(nCities-1):
                    newDistanceTravelled += distances[possibleCityOrder[j],possibleCityOrder[j+1]]
                distanceTravelled += distances[cityOrder[nCities-1],0]

                if newDistanceTravelled < distanceTravelled or (distanceTravelled - newDistanceTravelled) < T*np.log(np.random.rand()):
                    distanceTravelled = newDistanceTravelled
                    cityOrder = possibleCityOrder

            # Annealing schedule
            T *= c

    return cityOrder, distanceTravelled


def evaluate_sa(optimization_problem, evaluation_range=xrange(0, 10, 1)):
    data = []
    for n in evaluation_range:
        start = time.time()
        solution, optimal_value = simulated_annealing(optimization_problem, n_evaluations=n)
        elapsed = time.time() - start
        data.append(['simulated_annealing', n, optimal_value, elapsed])
    df = pd.DataFrame.from_records(data, columns=['algo', 'evaluations', 'optimal_value', 'running_time'])
    return df