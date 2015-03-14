import pandas as pd
import numpy as np
import time
from algo_evaluation.optimization.hill_climbing import hillclimb
from algo_evaluation.optimization.simulated_annealing import simulated_annealing
from algo_evaluation.optimization.genetic_optimize import genetic_optimize
from algo_evaluation.optimization import mimic

DEFAULT_EXPERIMENT_SETTINGS = dict()
DEFAULT_EXPERIMENT_SETTINGS['rhc'] = {'evaluations': 1000}
DEFAULT_EXPERIMENT_SETTINGS['sa'] = {'T': 1000}
DEFAULT_EXPERIMENT_SETTINGS['ga'] = {'generations': 100}
DEFAULT_EXPERIMENT_SETTINGS['mm'] = {'evaluations': 100}


class CronSchedule:
    def __init__(self, n_jobs=50, n_resources=10, names=False):
        self.n_jobs = n_jobs
        self.n_resources = n_resources
        self.hour_range = (0.0, 23.0)
        self.minute_range = (0.0, 59.0)
        self.cron_tasks_df = self.create_toy_schedule()
        if names:
            self.cron_tasks_df.rename(columns={x: 'resource_' + str(x) for x in range(self.n_resources)}, inplace=True)
            self.cron_tasks_df.rename(index={x: 'job_' + str(x) for x in range(self.n_jobs)}, inplace=True)
        self.domain = self.create_domain()

    def create_domain(self):
        domain = []
        for i in range(len(self.cron_tasks_df)):
            domain.append(self.hour_range)
            domain.append(self.minute_range)
        return domain

    def create_toy_schedule(self):
        df = pd.DataFrame(data=np.random.randint(2, size=(self.n_jobs, self.n_resources)))
        df['time'] = np.random.randint(100, size=self.n_jobs)
        return df

    def time_overlap(self, start_1, end_1, start_2, end_2):
        overlap = max(0, min(end_1, end_2) - max(start_1, start_2))
        return overlap

    def total_resources(self, job_1_id, job_2_id):
        return (self.cron_tasks_df.iloc[job_1_id] & self.cron_tasks_df.iloc[job_2_id]).sum()

    def compute_fitness(self, solution):
        """
           compute the overlap between jobs and resources
        """
        solution_fitness = 0.0
        solution_pair = zip(*(iter(solution),) * 2)
        for index in range(1, len(solution_pair)):
            hour_1, min_1 = solution_pair[index]
            hour_2, min_2 = solution_pair[index-1]
            start_1 = hour_1 + (min_1 / 60)
            end_1 = start_1 + float(self.cron_tasks_df.iloc[index]['time']) / 60
            start_2 = hour_2 + (min_2 / 60)
            end_2 = start_2 + float(self.cron_tasks_df.iloc[index-1]['time']) / 60
            resources = self.total_resources(index, index-1)
            overlap = self.time_overlap(start_1, end_1, start_2, end_2)
            solution_fitness += -1 * resources * overlap
        return solution_fitness


def compare_all(experiment_settings=DEFAULT_EXPERIMENT_SETTINGS):
    opt_problem = CronSchedule()
    domain = opt_problem.domain
    start = time.time()
    rhc = hillclimb(domain=domain,
                    costf=opt_problem.compute_fitness,
                    max_evaluations=experiment_settings['rhc']['evaluations'])
    rhc.set_index('evaluations', inplace=True)
    rhc['time'] = time.time() - start
    start = time.time()
    sa = simulated_annealing(domain=domain,
                             costf=opt_problem.compute_fitness,
                             T=experiment_settings['sa']['T'])
    sa.set_index('temperature', inplace=True)
    sa['time'] = time.time() - start
    start = time.time()
    ga = genetic_optimize(domain=domain,
                          costf=opt_problem.compute_fitness,
                          maxiter=experiment_settings['ga']['generations'])
    ga.set_index('generations', inplace=True)
    ga['time'] = time.time() - start
    start = time.time()
    mm = mimic.run_mimic(domain=domain,
                         fitness_function=opt_problem.compute_fitness,
                         evaluations=experiment_settings['mm']['evaluations'])
    mm['time'] = time.time() - start
    return rhc, sa, ga, mm