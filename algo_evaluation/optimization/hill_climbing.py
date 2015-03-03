import random
import pandas as pd


def hillclimb(domain, costf, max_evaluations=1000):
  # Create a random solution
  sol=[random.randint(domain[i][0], domain[i][1])
      for i in range(len(domain))]
  n_evaluations = 0
  data = []
  while n_evaluations < max_evaluations:
    # Create list of neighboring solutions
    neighbors=[]
    for j in range(len(domain)):
      # One away in each direction
      if sol[j]>domain[j][0]:
        neighbors.append(sol[0:j]+[sol[j]+1]+sol[j+1:])
      if sol[j]<domain[j][1]:
        neighbors.append(sol[0:j]+[sol[j]-1]+sol[j+1:])

    # See what the best solution amongst the neighbors is
    current=costf(sol)
    best=current
    for j in range(len(neighbors)):
      cost=costf(neighbors[j])
      if cost<best:
        best=cost
        sol=neighbors[j]
      n_evaluations +=1
      data.append([n_evaluations, best])
  df = pd.DataFrame.from_records(data, columns=['evaluations', 'cost'])
  df['optimal_value'] = 1 / df['cost']
  return df