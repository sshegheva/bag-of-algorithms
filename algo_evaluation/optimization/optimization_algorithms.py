from pybrain.optimization.populationbased.ga import GA
from pybrain.optimization import HillClimber
from pybrain.optimization import StochasticHillClimber
from pybrain.datasets import SupervisedDataSet


def genetic_algorithm(ef, init_value):
    ga = GA(evaluator=ef, initEvaluable=init_value, minimize=True)
    print ga
    return ga.learn()


def hill_climber(ds):
    learner = HillClimber(evaluator=ds.evaluateModuleMSE(), initEvaluable=ds.getSample(), minimize=True)
    return learner.learn()


def simulated_annealing():
    ef = None   # evaluation function
    sa = StochasticHillClimber(evaluator=ef, max_evaluations=10, minimize=True)
    return sa.learn()


def waldo_supervised_ds(waldo_df):
    ds = SupervisedDataSet(2, 2)
    for row in range(len(waldo_df)):
        ds.addSample(inp=waldo_df.iloc[row][['Book', 'Page']], target=waldo_df.iloc[row][['X', 'Y']])
    return ds