from utilities.setting_environment import set_env, TensorforceModel
from utilities.hp_opt_utils import load_json, save_to_json, go_dir_up
import time


# directory with agent configurations
config_path = go_dir_up() + '/agent_configurations/'
# put in name of config file without file-ending
print('config to be loaded: ')
config_name = input()

# name to save the metrics and agent under
print('experiment/objective name ...')
exp_name = input()

#setting up environments
env = set_env(time_limit=True, training=True)
test_env = set_env(time_limit=False, training=False)
agent_config = load_json(config_path, config_name)

model_path = go_dir_up(2) + '/saves/model/'
results_path = go_dir_up(2) + '/saves/results/'

train_steps = 500000
test_steps = 1000000

# setting up rl-model
tf_model = TensorforceModel(env=env,
                            agent_configuration=agent_config,
                            max_episode_steps=10000,
                            test_env=test_env,
                            model_directory=model_path,
                            model_name=exp_name,
                            )

start_time = time.time()
tf_model.train_agent(train_steps, progress_bar=True)
end_time = time.time()
metrics = tf_model.test_agent(test_steps)

# saves metrics to folder outside the repository (have to be created)
data_to_save = dict(config=agent_config, metric=metrics,
                    train_steps=train_steps, test_steps=test_steps,
                    duration=(start_time-end_time)/60)
save_to_json(data_to_save, results_path + exp_name)

print(f'\n Execution time of tensorforce dqn-training is:'
      f' 'f'{(end_time-start_time)/60:.2f} seconds \n ')