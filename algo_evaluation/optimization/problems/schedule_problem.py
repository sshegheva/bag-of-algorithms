import time
from algo_evaluation.datasets import load_schedule_dataset
from algo_evaluation.optimization.hill_climbing import hillclimb
from algo_evaluation.optimization.simulated_annealing import simulated_annealing
from algo_evaluation.optimization.genetic_optimize import genetic_optimize


people = [('Seymour','BOS'),
          ('Franny','DAL'),
          ('Zooey','CAK'),
          ('Walt','MIA'),
          ('Buddy','ORD'),
          ('Les','OMA')]
# Laguardia
destination='LGA'


flights = load_schedule_dataset()


def getminutes(t):
  x=time.strptime(t,'%H:%M')
  return x[3]*60+x[4]

def printschedule(r):
  for d in range(len(r)/2):
    name=people[d][0]
    origin=people[d][1]
    out=flights[(origin,destination)][int(r[d])]
    ret=flights[(destination,origin)][int(r[d+1])]
    print '%10s%10s %5s-%5s $%3s %5s-%5s $%3s' % (name,origin,
                                                  out[0],out[1],out[2],
                                                  ret[0],ret[1],ret[2])

def schedulecost(sol):
  totalprice=0
  latestarrival=0
  earliestdep=24*60
  for d in range(len(sol)/2):
    # Get the inbound and outbound flights
    origin=people[d][1]
    outbound=flights[(origin, destination)][int(sol[d])]
    returnf=flights[(destination, origin)][int(sol[d+1])]

    # Total price is the price of all outbound and return flights
    totalprice+=outbound[2]
    totalprice+=returnf[2]

    # Track the latest arrival and earliest departure
    if latestarrival<getminutes(outbound[1]): latestarrival=getminutes(outbound[1])
    if earliestdep>getminutes(returnf[0]): earliestdep=getminutes(returnf[0])

  # Every person must wait at the airport until the latest person arrives.
  # They also must arrive at the same time and wait for their flights.
  totalwait=0
  for d in range(len(sol)/2):
    origin=people[d][1]
    outbound=flights[(origin,destination)][int(sol[d])]
    returnf=flights[(destination,origin)][int(sol[d+1])]
    totalwait+=latestarrival-getminutes(outbound[1])
    totalwait+=getminutes(returnf[0])-earliestdep

  # Does this solution require an extra day of car rental? That'll be $50!
  if latestarrival>earliestdep: totalprice+=50

  return totalprice+totalwait


def evaluate_rhc():
    domain = [(0,8)] * len(people) *2
    df = hillclimb(domain, schedulecost)
    df['optimal_value'] = 1 / df['cost']
    return df

def evaluate_sa():
    domain = [(0,8)] * len(people) *2
    df = simulated_annealing(domain, schedulecost)
    df['optimal_value'] = 1 / df['cost']
    return df

def evaluate_ga():
    domain = [(0,8)] * len(people) *2
    df = genetic_optimize(domain, schedulecost)
    df['optimal_value'] = 1 / df['cost']
    return df