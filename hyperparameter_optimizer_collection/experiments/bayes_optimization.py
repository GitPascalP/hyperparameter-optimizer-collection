import numpy as np

from algorithm_collection.bayesian_optimization import BayesOpt
from utilities.setting_environment import set_env
from utilities.hp_opt_utils import load_json, go_dir_up, SavingResults
from objectives import get_experiment_specs

# directory with agent configurations
config_path = go_dir_up() + 'agent_configurations/'
print('agent ...')
config_name = input()

#setting up environments
env = set_env(time_limit=True, training=True)
test_env = set_env(time_limit=False, training=False)
agent_config = load_json(config_path, config_name)

"""
possible hyperparameter for agent_config:

    memory: size of replay-buffer
    batch_size: size of mini-batch used for training
    network: net-architect for dqn
    update_frequency: Frequency of updates
    start_updating: memory warm-up steps
    learning_rate for optimizer
    discount: gamma/ discount of future rewards
    target_sync_frequency: Target network gets updated 'sync_freq' steps
    target_update_weight: weight for target-network update

"""
# selecting experiment/ objective function
print('choose experiment/objective ... ')
name = input()
objective, specs = get_experiment_specs(name)
print('iterations ... ')
iterations = input()
print('bound transformation ... (y / n)')
bound_trafo_flag = input()

hp_opt = BayesOpt(
                 parameter_specs=specs,
                 objective=objective,
                 iterations=int(iterations),
)

# saves metrics to folder outside the repository (have to be created)
abs_path = go_dir_up(2) + '/saves/results/'
if bound_trafo_flag == 'y':
    hp_opt.optimize(path=abs_path, filename=name + '_trafo', bound_trafo=True)
elif bound_trafo_flag == 'n':
    hp_opt.optimize(path=abs_path, filename=name, bound_trafo=False)
