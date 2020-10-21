"""

Algorithms for Hyperparameter-Optimization.
Self implemented or necessary wrappers/ preprocessing for packages

"""
import numpy as np

from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import json



class BayesOpt:
    """

    """
    def __init__(self,
                 parameter_specs,
                 objective,
                 iterations=50,
                 initial_samples=5,
                 logger=True,
                 random_seed=42,
                 verbose=2,
                 ):
        """
        args:
            parameter_specs(dict):
            objective(function):
            iterations(int):
            initial_samples(int):
            logger(bool):
            random_seed(int):
            verbose(int):
        """

        self.specs = parameter_specs
        self.objective = objective

        # parameters for optimizer
        self.iterations = iterations
        self.initial = initial_samples
        self.logger = logger
        self.seed = random_seed
        self.verbose = verbose

    def create_iterables(self,):
        """ creating iterable list of parameter-combinations/
            list of interval-specifications for parameters """
        # creating dict with interval specifiactions for each parameter

        specs_interval = {key: (np.amin(value), np.amax(value))
                          for key, value in self.specs.items()}
        # sort keys alphabetically
        bounds = dict(sorted(specs_interval.items(), key=lambda x: x[0].lower()))
        return bounds

    def optimize(self, path, filename, bound_trafo):
        assert (self.logger and (path is not None) and (filename is not None),
                'Filename and path are necessary when the logger is active')
        pbounds = self.create_iterables()

        if bound_trafo:
            bounds_transformer = SequentialDomainReductionTransformer()
            optimizer = BayesianOptimization(
                f=self.objective,
                pbounds=pbounds,
                verbose=self.verbose,
                random_state=self.seed,
                bounds_transformer=bounds_transformer,
            )

        else:
            optimizer = BayesianOptimization(
                f=self.objective,
                pbounds=pbounds,
                verbose=self.verbose,
                random_state=self.seed,
            )

        if self.logger:
            logger = JSONLogger(path=path + filename + '.txt')
            optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

        optimizer.maximize(init_points=self.initial,
                           n_iter=self.iterations)

        # write bound-trajectories in separate log-file,
        # if bounds have been transformed
        if bound_trafo:
            for i, p in enumerate(pbounds.keys()):
                bounds = {p: {}}
                bounds[p]['min'] = [b[i][0] for b in bounds_transformer.bounds]
                bounds[p]['max'] = [b[i][1] for b in bounds_transformer.bounds]

            with open(path + filename + '_bounds.txt', "a") as f:
                f.write(json.dumps(bounds))

