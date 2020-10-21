"""

Algorithms for Hyperparameter-Optimization.
Self implemented or necessary wrappers/ preprocessing for packages

"""
import numpy as np
import multiprocessing as mp
from multiprocessing import Lock, Process, Queue, current_process, Pool
import queue # imported for using queue.Empty exception

class ParameterSearch:
    """

    """
    def __init__(self,
                 environment,
                 agent_kwargs,
                 parameter_specs,
                 objective,
                 max_episode_steps=10000,
                 num_parameters=None,
                 parallel=True,
                 test_env=None,
                 config_name=None,
                 saver=None,
                 enable_dict=True,
                 distribution='uniform',
                 dtype='int',
                 ):
        """
        args:
            env: environment on which the rl-agent is performing
            agent_kwargs(dict): parameters from the used agent
            hyper_specs(dict): specifications for parameters to be optimized
            train_steps(int):
            test_steps(int):
            max_episode_steps(int):
            search_type(str):
            num_parameters(int):
            buffer(int): number of best models/configs that are saved
            parallel(bool): execute optimization parallel or not
            test_env:
            saver(int):

        """
        self.env = environment
        self.test_env = test_env or environment
        self.agent_kwargs = agent_kwargs
        self.specs = parameter_specs
        self.objective = objective

        self.num_params = num_parameters
        self.max_episode_steps = max_episode_steps

        self.parallel = parallel
        self.config_name = config_name
        self.saver = saver
        self.enable_dict = enable_dict
        self.distribution = distribution
        self.dtype = dtype

    def create_iterables(self, algorithm):
        """ creating iterable list of parameter-combinations/
            list of interval-specifications for parameters """
        # create meshgrid for given parameters
        if algorithm == 'grid':
            grids = np.meshgrid(*[v for v in self.specs.values()])
            specs_flat = self.specs.copy()
        # sample random-values out of an given interval (or min,max values)
        elif algorithm == 'random':
            assert self.num_params is not None, 'for random-search number of '\
                                                'samples have to be known'
            # samples parameters and writes them in self.specs
            self.sample_parameters()
            grids = np.meshgrid(*[v for v in self.specs.values()])
            specs_flat = self.specs.copy()

        # creating dict with interval specifiactions for each parameter
        for k, item in enumerate(self.specs.items()):
            # scale grid values
            # todo scale problem for random vars
            # getting flat parameter-values
            specs_flat[item[0]] = grids[k].flatten()

        parameters = list(specs_flat.keys())
        num_total = len(specs_flat[parameters[0]])
        # create list with all agent-configurations
        configs = []
        for i in range(num_total):
            if self.dtype == 'int':
                configs.append(
                    {key: int(values[i]) for key, values in specs_flat.items()}
                )
            else:
                pass
        return configs

    def sample_parameters(self):
        specs_interval = {}
        for key in self.specs.keys():
            specs_interval[key] = np.asarray([np.amin(self.specs[key]),
                                              np.amax(self.specs[key])])
            a = specs_interval[key][0]
            b = specs_interval[key][1]
            if self.distribution == 'uniform':
                if self.dtype == 'int':
                    self.specs[key] = np.random.randint(a, b,
                                                        int(self.num_params))
                else:
                    self.specs[key] = (b - a) * \
                                      np.random.random_sample(
                                          int(self.num_params)) + a

            elif self.distribution == 'lognormal':
                pass
            # todo andere verteilungen wenn nützlich

    def execute(self, parameters):
        """" parallel execution of func"""
        if self.parallel:
            num_tasks = len(parameters)
            num_procs = mp.cpu_count() - 4
            config_queue = Queue()
            result_queue = Queue()
            results = []
            procs = []

            # fill queue
            for t in range(num_tasks):
                config_queue.put(parameters[t])
            # starting p processes
            for i in range(num_procs):
                p = Process(target=self.do_job, args=(config_queue,
                                                      result_queue))
                procs.append(p)
                p.start()
            # closing all processes
            for p in procs:
                p.join()
            # convert result queue to list
            while not result_queue.empty():
                results.append(result_queue.get())

            return results

        else:
            results = []
            for p in parameters:
                results.append(self.objective(**p))
            return results

    def do_job(self, input_queue, result_queue):
        while True:
            try:
                params = input_queue.get_nowait()
                result = self.objective(**params)
            except queue.Empty:
                break
            else:
                result_queue.put(result)

        return True

        # print('exe \n', func, parameters[0:2])
        # if self.parallel:
        #     p = Pool(mp.cpu_count() - 2)
        #     results = p.map(func, parameters)
        #     p.close()
        #     p.join()
        # else:
        #     results = []
        #     for p in parameters:
        #         results.append(func(p))

    def grid_search(self, ):
        # self.saver = SavingResults(algorithm='grid')
        parameter_configs = self.create_iterables(algorithm='grid')
        agent_metrics = self.execute(parameter_configs)
        #if self.saver is not None:
        #    self.saver.save_results(agent_metrics, 'results')

    def random_search(self, ):
        # self.saver = SavingResults(algorithm='random')
        parameter_configs = self.create_iterables(algorithm='random')
        agent_metrics = self.execute(parameter_configs)
        #if self.saver is not None:
        #    self.saver.save_results(agent_metrics, 'results')
