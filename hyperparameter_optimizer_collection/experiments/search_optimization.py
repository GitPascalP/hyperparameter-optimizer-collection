import numpy as np

from algorithm_collection.basic_parameter_search import ParameterSearch
from utilities.setting_environment import set_env
from utilities.hp_opt_utils import load_json, go_dir_up, SavingResults
from objectives import get_experiment_specs

# directory with agent configurations
config_path = go_dir_up() + 'agent_configurations/'
print('choose agent ...')
config_name = input()

if config_name == 'def':
    config_name = 'tf_default_agent'
elif config_name == 'opt':
    config_name = 'optimized_agent_config'

#setting up environments
env = set_env(time_limit=True, training=True)
test_env = set_env(time_limit=False, training=False)
agent_config = load_json(config_path, config_name)

epsilon_decay = {'type': 'decaying',
                 'decay': 'polynomial',
                 'decay_steps': 50000,
                 'unit': 'timesteps',
                 'initial_value': 1.0,
                 'decay_rate': 5e-2,
                 'final_value': 5e-2,
                 'power': 3.0}

# selecting experiment
print('choose experiment ... ')
name = input()
objective, specs = get_experiment_specs(name)
print('search type ... ')
algorithm = input()
logger = SavingResults(algorithm=algorithm, name=name, up=2)


if algorithm == 'grid':
    seeker = ParameterSearch(environment=env,
                             agent_kwargs=agent_config,
                             parameter_specs=specs,
                             objective=objective,
                             config_name=name,
                             test_env=test_env,
                             saver=logger,
                             enable_dict=True,
                             parallel=False
                             )
    seeker.grid_search()

elif algorithm == 'random':
    print('# parameters ... ')
    num_params = input()
    seeker = ParameterSearch(environment=env,
                             agent_kwargs=agent_config,
                             parameter_specs=specs,
                             objective=objective,
                             config_name=name,
                             num_parameters=num_params,
                             test_env=test_env,
                             saver=logger,
                             parallel=False
                             )
    seeker.random_search()
