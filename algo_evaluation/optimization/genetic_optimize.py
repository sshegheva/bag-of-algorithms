import pandas as pd
import random


def genetic_optimize(domain, costf, popsize=50, step=1, mutprob=0.2, elite=0.2, maxiter=1000):
  # Mutation Operation
    def mutate(vec):
        mutation = v
        i = random.randint(0, len(domain)-1)
        if random.random() < 0.5 and vec[i] > domain[i][0]:
            mutation = vec[0:i]+[vec[i]-step]+vec[i+1:]
        elif vec[i] < domain[i][1]:
            mutation = vec[0:i]+[vec[i]+step]+vec[i+1:]
        return mutation

    # Crossover Operation
    def crossover(r1, r2):
        i = random.randint(1, len(domain)-2)
        result = r1[0:i]+r2[i:]
        return result

    # Build the initial population
    pop=[]
    data = []
    for i in range(popsize):
        vec=[random.randint(domain[i][0],domain[i][1])
             for i in range(len(domain))]
        pop.append(vec)
    # How many winners from each generation?
    topelite=int(elite*popsize)

    # Main loop
    for i in range(maxiter):
        scores = [(costf(v), v) for v in pop]
        scores.sort()
        ranked=[v for (s,v) in scores]

        # Start with the pure winners
        pop=ranked[0:topelite]

        # Add mutated and bred forms of the winners
        while len(pop) < popsize:
            if random.random()<mutprob:

                # Mutation
                c=random.randint(0,topelite)
                pop.append(mutate(ranked[c]))
            else:

                # Crossover
                c1=random.randint(0,topelite)
                c2=random.randint(0,topelite)
                cross = crossover(ranked[c1], ranked[c2])
                pop.append(cross)
        current_best_score = scores[0][0]
        data.append([popsize, i, current_best_score])

    df = pd.DataFrame.from_records(data, columns=['population_size', 'generation', 'cost'])
    return df