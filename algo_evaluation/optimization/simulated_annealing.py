import math
import random
import pandas as pd


def simulated_annealing(domain, costf, T=10000.0, cool=0.99, step=1):
  # Initialize the values randomly
  vec=[float(random.randint(domain[i][0],domain[i][1]))
       for i in range(len(domain))]

  data = []
  while T > 1:
    # Choose one of the indices
    i=random.randint(0,len(domain)-1)
    # Choose a direction to change it
    dir=random.randint(-step,step)

    # Create a new list with one of the values changed
    vecb=vec[:]
    vecb[i]+=dir
    if vecb[i]<domain[i][0]: vecb[i]=domain[i][0]
    elif vecb[i]>domain[i][1]: vecb[i]=domain[i][1]

    # Calculate the current optimal values
    energy_before = costf(vec)
    energy_after = costf(vecb)
    p = pow(math.e, (-energy_before - energy_after)/T)
    rd = random.random()
    if (energy_after < energy_before) or (rd < p):
      vec=vecb
      data.append([T, energy_after, vec])
    else:
      data.append([T, energy_before, vec])

    # Decrease the temperature
    T *= cool
  df = pd.DataFrame.from_records(data, columns=['temperature', 'cost', 'solution'])
  df['optimal_value'] = -1 * df['cost']
  return df
