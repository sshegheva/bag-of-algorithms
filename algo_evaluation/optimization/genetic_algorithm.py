import math
import random
import pandas as pd
import matplotlib.pyplot as plt


def calculate_distance(x1, y1, x2, y2):
    """
        Returns the Euclidean distance between points (x1, y1) and (x2, y2)
    """
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def compute_fitness(solution, optimization_problem):
    """
        Computes the distance that the Waldo-seeking solution covers.

        Lower distance is better, so the GA should try to minimize this function.
    """
    solution_fitness = 0.0

    for index in range(1, len(solution)):
        w1 = solution[index]
        w2 = solution[index - 1]
        solution_fitness += calculate_distance(optimization_problem[w1][0], optimization_problem[w1][1],
                                               optimization_problem[w2][0], optimization_problem[w2][1])

    return solution_fitness


def mutate_agent(agent_genome, max_mutations=3):
    """
        Applies 1 - `max_mutations` point mutations to the given Waldo-seeking path.

        A point mutation swaps the order of two locations in the Waldo-seeking path.
    """
    agent_genome = list(agent_genome)
    num_mutations = random.randint(1, max_mutations)

    for mutation in range(num_mutations):
        swap_index1 = random.randint(0, len(agent_genome) - 1)
        swap_index2 = swap_index1

        while swap_index1 == swap_index2:
            swap_index2 = random.randint(0, len(agent_genome) - 1)

        agent_genome[swap_index1], agent_genome[swap_index2] = agent_genome[swap_index2], agent_genome[swap_index1]

    return tuple(agent_genome)

def shuffle_mutation(agent_genome):
    """
        Applies a single shuffle mutation to the given Waldo-seeking path.

        A shuffle mutation takes a random sub-section of the path and moves it to
        another location in the path.
    """
    agent_genome = list(agent_genome)

    start_index = random.randint(0, len(agent_genome) - 1)
    length = random.randint(2, 20)

    genome_subset = agent_genome[start_index:start_index + length]
    agent_genome = agent_genome[:start_index] + agent_genome[start_index + length:]

    insert_index = random.randint(0, len(agent_genome) + len(genome_subset) - 1)
    agent_genome = agent_genome[:insert_index] + genome_subset + agent_genome[insert_index:]

    return tuple(agent_genome)


def generate_random_population(pop_size, population_seed):
    """
        Generates a list with `pop_size` number of random paths.
    """
    random_population = []
    for agent in range(pop_size):
        new_random_agent = population_seed
        random.shuffle(new_random_agent)
        random_population.append(tuple(new_random_agent))
    return random_population

def plot_trajectory(agent_genome):
    """
        Create a visualization of the given Waldo-seeking path.
    """
    agent_xs = []
    agent_ys = []
    agent_fitness = compute_fitness(agent_genome)

    for waldo_loc in agent_genome:
        agent_xs.append(waldo_location_map[waldo_loc][0])
        agent_ys.append(waldo_location_map[waldo_loc][1])

    plt.figure()
    plt.title("Fitness: %f" % (agent_fitness))
    plt.plot(agent_xs[:18], agent_ys[:18], "-o", markersize=7)
    plt.plot(agent_xs[17:35], agent_ys[17:35], "-o", markersize=7)
    plt.plot(agent_xs[34:52], agent_ys[34:52], "-o", markersize=7)
    plt.plot(agent_xs[51:], agent_ys[51:], "-o", markersize=7)
    plt.plot(agent_xs[0], agent_ys[0], "^", color="#1f77b4", markersize=15)
    plt.plot(agent_xs[-1], agent_ys[-1], "v", color="#d62728", markersize=15)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def run_genetic_algorithm(optimization_problem,
                          generations=10000,
                          population_size=100):
    """
        The core of the Genetic Algorithm.
    """
    data = []
    # Create a random population of `population_size` number of solutions.
    population = generate_random_population(population_size, optimization_problem.keys())

    # For `generations` number of repetitions...
    for generation in range(int(generations)):
        
        # Compute the fitness of the entire current population
        population_fitness = {}

        for agent_genome in population:
            if agent_genome in population_fitness:
                continue

            population_fitness[agent_genome] = compute_fitness(agent_genome, optimization_problem)

        # Take the 10 shortest Waldo-seeking paths and produce 10 offspring each from them
        new_population = []
        for rank, agent_genome in enumerate(sorted(population_fitness, key=population_fitness.get)[:10]):
            #if (generation % 1000 == 0 or generation == 9999) and rank == 0:
            if rank == 0:
                fit = population_fitness[agent_genome]
                data.append([generation, fit, agent_genome])
                #plot_trajectory(agent_genome)

            # Create 1 exact copy of each top 10 Waldo-seeking path
            new_population.append(agent_genome)

            # Create 4 offspring with 1-3 mutations
            for offspring in range(4):
                new_population.append(mutate_agent(agent_genome, 3))
                
            # Create 5 offspring with a single shuffle mutation
            for offspring in range(5):
                new_population.append(shuffle_mutation(agent_genome))

        # Replace the old population with the new population of offspring
        for i in range(len(population))[::-1]:
            del population[i]

        population = new_population

    df = pd.DataFrame.from_records(data, columns=['generation', 'optimal_value', 'path'])
    return df


def evaluate_ga(waldo_optimization_problem, generations=1000, population_size=100):

    return run_genetic_algorithm(optimization_problem=waldo_optimization_problem,
                          generations=generations,
                          population_size=population_size)