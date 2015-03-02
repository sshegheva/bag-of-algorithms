import numpy as np
import pandas as pd
import random
from algo_evaluation.optimization.problems.schedule_problem import schedulecost, people


def pbil(domain, costf, popsize=100, eta=0.005, iterations=100):

    stringlen = len(domain)
    p = 0.5*np.ones(stringlen)
    best = np.zeros(iterations, dtype=float)

    for count in range(iterations):
        # Generate samples
        population = np.random.rand(popsize, stringlen)
        for i in range(stringlen):
            population[:, i] = np.where(population[:, i] < p[i], 1, 0)

        print population
        # Evaluate fitness
        fitness = costf(population)
        print fitness

        # Pick best
        best[count] = np.max(fitness)
        bestplace = np.argmax(fitness)
        fitness[bestplace] = 0
        secondplace = np.argmax(fitness)

        # Update vector
        p = p*(1-eta) + eta*((population[bestplace,:]+population[secondplace,:])/2)

        if (np.mod(count,100)==0):
            print count, best[count]

def evaluate_pbil():
    domain = [(0,8)] * len(people) *2
    return pbil(domain, schedulecost)

