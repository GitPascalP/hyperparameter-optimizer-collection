"""

Objective-functions/rl-agents that are going to be optimized

hyperparameter for agent_config:

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

import numpy as np
from utilities.setting_environment import set_env, TensorforceModel, \
    KerasRLModel
from utilities.hp_opt_utils import load_json, go_dir_up
import json

env = set_env(time_limit=True, training=True)
test_env = set_env(time_limit=False, training=False)

epsilon_decay = {'type': 'decaying',
                 'decay': 'polynomial',
                 'decay_steps': 50000,
                 'unit': 'timesteps',
                 'initial_value': 1.0,
                 'decay_rate': 5e-2,
                 'final_value': 5e-2,
                 'power': 3.0}

print('agent (should be the same as for optimization) ...')
input_config = input()

# list of used names for experiments (has to be extended)
used_names = [
    'test_objective_bayes',
    'test_objective_grid'
]

agent_config = load_json(go_dir_up(1) + 'agent_configurations/',
                         name=input_config)

# request to read and return objective to experiment scripts
def get_experiment_specs(name):
    assert name in used_names, 'experiment-name has to be in list'

    # return corresponding objective and specs
    if name == 'test_objective_bayes':
        return test_objective_bayes_discrete, test_objective_bayes_specs
    elif name == 'test_objective_grid':
        return test_objective_grid, test_objective_grid_specs
    else:
        pass


# ############### objectives for tensorforce models ###############
def test_objective_bayes(batch_size, decay_rate, discount, learning_rate):
    """
    args:
        parameter set for optimization
    returns:
        target value for bayessian optimization
    """
    # setting parameters in config dictionary
    epsilon_decay.update({'decay_rate': decay_rate})
    parameter_config = {
        'batch_size': batch_size,
        'discount': discount,
        'exploration': epsilon_decay,
        'learning_rate': learning_rate,
        }

    agent_kwargs = agent_config.copy()
    agent_kwargs.update(parameter_config)
    # define new model, train and test it
    tf_model = TensorforceModel(env,
                                agent_kwargs,
                                max_episode_steps=10000,
                                test_env=test_env,
                                logger=False)

    tf_model.train_agent(train_steps=500000, progress_bar=True)
    metrics = tf_model.test_agent(eval_steps=1000000)
    # return reward per step as target
    target_metric = metrics['rew_per_step']
    return target_metric


# intervals for possible parameter values
test_objective_bayes_specs = {'batch_size': np.array([20, 100]),
                              'decay_rate': np.array([5e-3, 5e-1]),
                              'discount': np.array([0.90, 0.99]),
                              'learning_rate': np.array([1e-5, 1e-3]),
                              }


# work-around for using discrete values with the bayesian-opt. toolbox
def test_objective_bayes_discrete(batch_size,
                                  decay_rate, discount,
                                  learning_rate,
                                  ):
    batch_size_int = int(batch_size)
    return test_objective_bayes(batch_size=batch_size_int,
                                decay_rate=decay_rate,
                                discount=discount,
                                learning_rate=learning_rate,
                                )


""" ############### random/grid search ###############"""


def test_objective_grid(memory, batch_size, target_sync_frequency):
    """   for random/grid search   """
    # setting parameters in config dictionary
    parameter_config = {'memory': memory,
                        'batch_size': batch_size,
                        'target_sync_frequency': target_sync_frequency,
                        }
    agent_kwargs = agent_config.copy()
    agent_kwargs.update(parameter_config)
    # define new model, train and test it
    tf_model = TensorforceModel(env,
                                agent_kwargs,
                                max_episode_steps=10000,
                                test_env=test_env,
                                logger=False)

    tf_model.train_agent(train_steps=500000, progress_bar=True)
    metrics = tf_model.test_agent(eval_steps=1000000)

    # write configurations and metrics in json file
    data = dict(target=metrics['rew_per_step'], params=parameter_config)
    path = go_dir_up(2) + 'saves/' + 'results' + '/'
    # save_to_json(data, path + self.name)
    with open(path + 'neon_bible.json', "a") as f:
        f.write(json.dumps(data) + "\n")
    # return reward per step as target
    return metrics


# possible values for parameters used in grid search (random search uses
# max/min value as interval
test_objective_grid_specs = {'memory': np.array([150000, 200000, 250000], dtype=int),
                             'batch_size': np.array([15, 20, 25, 35, 50], dtype=int),
                             'target_sync_frequency': np.array([800, 1000, 1200]),
                             }
