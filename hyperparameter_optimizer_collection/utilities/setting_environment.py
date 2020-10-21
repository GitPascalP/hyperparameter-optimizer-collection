import numpy as np
from tqdm import tqdm
from pprint import pprint

import gym_electric_motor as gem
from gym_electric_motor.constraint_monitor import ConstraintMonitor
from gym_electric_motor.reference_generators import \
    MultipleReferenceGenerator,\
    WienerProcessReferenceGenerator
from gym_electric_motor.visualization import MotorDashboard
from gym_electric_motor.physical_systems import ConstantSpeedLoad
from gym.spaces import Discrete, Box
from gym.wrappers import FlattenObservation, TimeLimit
from gym import ObservationWrapper, Wrapper

from tensorforce import Environment, Agent, Runner
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory


class SqdCurrentMonitor:
    """
    monitor for squared currents:
    i_sd**2 + i_sq**2 < 1.5 * nominal_limit
    """

    def __call__(self, state, observed_states, k, physical_system):
        self.I_SD_IDX = physical_system.state_names.index('i_sd')
        self.I_SQ_IDX = physical_system.state_names.index('i_sq')
        sqd_currents = state[self.I_SD_IDX] ** 2 + state[self.I_SQ_IDX] ** 2
        return sqd_currents > 1


class EpsilonWrapper(ObservationWrapper):
    """
    Changes Epsilon in a flattened observation to cos(epsilon)
    and sin(epsilon)
    """

    def __init__(self, env, epsilon_idx, i_sd_idx, i_sq_idx):
        super(EpsilonWrapper, self).__init__(env)
        self.EPSILON_IDX = epsilon_idx
        self.I_SQ_IDX = i_sq_idx
        self.I_SD_IDX = i_sd_idx
        new_low = np.concatenate((self.env.observation_space.low[
                                  :self.EPSILON_IDX], np.array([-1.]),
                                  self.env.observation_space.low[
                                  self.EPSILON_IDX:], np.array([0.])))
        new_high = np.concatenate((self.env.observation_space.high[
                                   :self.EPSILON_IDX], np.array([1.]),
                                   self.env.observation_space.high[
                                   self.EPSILON_IDX:], np.array([1.])))

        self.observation_space = Box(new_low, new_high)

    def observation(self, observation):
        cos_eps = np.cos(observation[self.EPSILON_IDX] * np.pi)
        sin_eps = np.sin(observation[self.EPSILON_IDX] * np.pi)
        currents_squared = observation[self.I_SQ_IDX] ** 2 + observation[
            self.I_SD_IDX] ** 2
        observation = np.concatenate((observation[:self.EPSILON_IDX],
                                      np.array([cos_eps, sin_eps]),
                                      observation[self.EPSILON_IDX + 1:],
                                      np.array([currents_squared])))
        return observation


class AppendNLastOberservationsWrapper(Wrapper):

    def __init__(self, env, N):
        super().__init__(env)
        self._N = N
        self._current_step = 0
        self._obs = None
        new_low = self.env.observation_space.low
        new_high = self.env.observation_space.high
        for i in range(self._N):
            new_low = np.concatenate((new_low, self.env.observation_space.low))
            new_high = np.concatenate(
                (new_high, self.env.observation_space.high))
        self.observation_space = Box(new_low, new_high)

    def step(self, action):
        obs, rew, term, info = self.env.step(action)
        if self._current_step < self._N:
            self._current_step += 1
            self._obs[
            self._current_step * self.env.observation_space.shape[0]:(
                                                                                 self._current_step + 1) *
                                                                     self.env.observation_space.shape[
                                                                         0]] = obs
        else:
            valid_obs = self._obs[self.env.observation_space.shape[0]:]
            self._obs = np.concatenate((valid_obs, obs))
        return self._obs, rew, term, info

    def reset(self, **kwargs):
        self._current_step = 0
        obs = self.env.reset()
        for i in range(self._N):
            obs = np.concatenate(
                (obs, np.zeros(self.env.observation_space.shape)))
        self._obs = obs
        return self._obs


class AppendNLastActionsWrapper(Wrapper):
    def __init__(self, env, N):
        super().__init__(env)
        self._N = N
        self._current_step = 0
        self._obs = None
        new_low = self.env.observation_space.low
        new_high = self.env.observation_space.high
        for i in range(self._N):
            new_low = np.concatenate((new_low, [0]))
            new_high = np.concatenate((new_high, [0]))
        self.observation_space = Box(new_low, new_high)

    def step(self, action):
        obs, rew, term, info = self.env.step(action)
        self._obs[:self.env.observation_space.shape[0]] = obs
        if self._N > 0:
            if self._current_step < self._N:
                self._obs[
                self.env.observation_space.shape[0] + self._current_step:
                self.env.observation_space.shape[0] + (
                            self._current_step + 1)] = action
                self._current_step += 1
            else:
                valid_actions = self._obs[
                                self.env.observation_space.shape[0] + 1:]
                self._obs[
                self.env.observation_space.shape[0]:-1] = valid_actions
                self._obs = np.concatenate((self._obs[:-1], [action]))
        return self._obs, rew, term, info

    def reset(self, **kwargs):
        self._current_step = 0
        obs = self.env.reset()
        for i in range(self._N):
            obs = np.concatenate((obs, [0]))
        self._obs = obs
        return self._obs


class NormalizeObservation(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, rew, term, info = self.env.step(action)
        return obs / np.linalg.norm(obs), rew, term, info

    def reset(self, **kwargs):
        obs = self.env.reset()
        return obs / np.linalg.norm(obs)


def set_env(time_limit=True, gamma=0.99, N=0, M=0, training=True,
            callbacks=[]):
    # define motor arguments
    motor_parameter = dict(p=3,  # [p] = 1, nb of pole pairs
                           r_s=17.932e-3,  # [r_s] = Ohm, stator resistance
                           l_d=0.37e-3,  # [l_d] = H, d-axis inductance
                           l_q=1.2e-3,  # [l_q] = H, q-axis inductance
                           psi_p=65.65e-3,
                           # [psi_p] = Vs, magnetic flux of the permanent magnet
                           )
    u_sup = 350
    nominal_values = dict(omega=4000 * 2 * np.pi / 60,
                          i=230,
                          u=u_sup
                          )
    limit_values = dict(omega=4000 * 2 * np.pi / 60,
                        i=1.5 * 230,
                        u=u_sup
                        )
    q_generator = WienerProcessReferenceGenerator(reference_state='i_sq')
    d_generator = WienerProcessReferenceGenerator(reference_state='i_sd')
    rg = MultipleReferenceGenerator([q_generator, d_generator])
    tau = 1e-5
    max_eps_steps = 10000

    if training:
        motor_initializer = {'random_init': 'uniform',
                             'interval': [[-230, 230], [-230, 230],
                                          [-np.pi, np.pi]]}
        # motor_initializer={'random_init': 'gaussian'}
        reward_function = gem.reward_functions.WeightedSumOfErrors(
            observed_states=['i_sq', 'i_sd'],
            reward_weights={'i_sq': 10, 'i_sd': 10},
            constraint_monitor=SqdCurrentMonitor(),
            gamma=gamma,
            reward_power=1)
    else:
        motor_initializer = {'random_init': 'gaussian'}
        reward_function = gem.reward_functions.WeightedSumOfErrors(
            observed_states=['i_sq', 'i_sd'],
            reward_weights={'i_sq': 0.5, 'i_sd': 0.5},  # comparable reward
            constraint_monitor=SqdCurrentMonitor(),
            gamma=0.99,  # comparable reward
            reward_power=1)

    # creating gem environment
    env = gem.make(  # define a PMSM with discrete action space
        "PMSMDisc-v1",
        # visualize the results
        visualization=MotorDashboard(plots=['i_sq', 'i_sd', 'reward']),
        # parameterize the PMSM and update limitations
        motor_parameter=motor_parameter,
        limit_values=limit_values, nominal_values=nominal_values,
        # define the random initialisation for load and motor
        load='ConstSpeedLoad',
        load_initializer={'random_init': 'uniform', },
        motor_initializer=motor_initializer,
        reward_function=reward_function,

        # define the duration of one sampling step
        tau=tau, u_sup=u_sup,
        # turn off terminations via limit violation, parameterize the rew-fct
        reference_generator=rg, ode_solver='euler',
        callbacks=callbacks,
    )

    # appling wrappers and modifying environment
    env.action_space = Discrete(7)
    eps_idx = env.physical_system.state_names.index('epsilon')
    i_sd_idx = env.physical_system.state_names.index('i_sd')
    i_sq_idx = env.physical_system.state_names.index('i_sq')

    if time_limit:
        gem_env = TimeLimit(AppendNLastActionsWrapper(
            AppendNLastOberservationsWrapper(
                EpsilonWrapper(FlattenObservation(env), eps_idx, i_sd_idx,
                               i_sq_idx), N), M),
                            max_eps_steps)
    else:
        gem_env = AppendNLastActionsWrapper(AppendNLastOberservationsWrapper(
            EpsilonWrapper(FlattenObservation(env), eps_idx, i_sd_idx,
                           i_sq_idx), N), M)
    return gem_env


class TensorforceModel:
    """
    class to access all key-features of tensorforce, build, test and training
    routines.
    """
    def __init__(self,
                 env,
                 agent_configuration,
                 max_episode_steps,
                 ext_agent=None,
                 test_env=None,
                 logger=True,
                 model_directory=None,
                 model_name='rl_agent',
                 **kwargs):
        """
        args:
            env: gym-like environment on which tf is trained
            agent_configuration(dict): parameters for rl-agent
            max_episode_steps(int): step limit of one training episode
            ext_agent: externally defined tf-agent
            logger:
            test_env: gym-like environment on which the trained tf-agent is
                      tested (if None env is used)

        """
        self.env = env
        self.test_env = test_env or env

        self.max_steps = max_episode_steps

        self.agent_config = agent_configuration
        self.agent = ext_agent or None
        self.tf_env = self.set_environment()

        self.logger = logger
        self.save_dir = model_directory
        self.model_name = model_name

    def set_environment(self, **kwargs):
        tf_env = Environment.create(environment=self.env,
                                    max_episode_timesteps=self.max_steps,
                                    **kwargs)
        return tf_env

    def train_agent(self, train_steps, progress_bar=False,
                    optimizer='adam', **kwargs):
        """
        args:
            train_steps(int): number of steps to train the agent

        """
        if self.agent is None:
            if optimizer == 'adam':
                self.agent = Agent.create(agent=self.agent_config,
                                          environment=self.tf_env)
            elif optimizer == 'rmsprop':
                pass

        runner = Runner(agent=self.agent,
                        environment=self.tf_env,
                        **kwargs)
        runner.run(num_timesteps=train_steps, use_tqdm=progress_bar)
        if self.logger:
            runner.agent.save(directory=self.save_dir, filename=self.model_name)
        runner.close

    def test_agent(self, eval_steps, visual=False, **kwargs):
        """
        args:
            eval_steps(int): number of steps to evaluate the agent
            visual(bool): True for environment visualization during testing
        """
        # test agent
        tau = 1e-5
        steps = 1000000
        rewards = []
        states = []
        references = []

        obs = self.test_env.reset()
        terminal = False
        cum_rew = 0
        step_counter = 0
        eps_rew = 0
        for step in tqdm(range(eval_steps)):
            if visual:
                self.test_env.render()
            actions = self.agent.act(obs, evaluation=True)
            obs, reward, terminal, _ = self.test_env.step(action=actions)
            rewards.append(cum_rew)
            cum_rew += reward
            eps_rew += reward

            if terminal:
                obs = self.test_env.reset()
                rewards.append(eps_rew)
                terminal = False
                eps_rew = 0

        metrics = {'episode_reward': eps_rew,
                   'cumulated_reward': cum_rew,
                   'rew_per_step': cum_rew/eval_steps,
                   'steps': eval_steps}

        return metrics


class KerasRLModel:
    """
    class to access all key-features of keras-rl2, build, test and training
    routines.
    """
    def __init__(self,
                 env,
                 agent_config,
                 model=None,
                 memory=None,
                 policy=None,
                 max_episode_steps=10000,
                 window_length=1,
                 optimizer='adam',
                 test_env=None,
                 logger=True,
                 saving_directory=None,
                 saving_name='rl_agent',
                 ):
        # todo add docstring
        """


        """
        self.env = env
        self.test_env = test_env or env

        #agent configuration
        self.gamma = agent_config['discount']
        self.learning_starts = agent_config['start_updating']
        self.learning_rate = agent_config['learning_rate']
        self.batch_size = agent_config['batch_size']
        self.target_model_update = agent_config['target_sync_frequency']

        self.model = model or self.build_model(window_length=window_length,
                                               nb_actions=env.action_space.n)
        self.memory = memory or \
                      SequentialMemory(limit=agent_config['memory'],
                                       window_length=window_length)
        self.policy = policy or \
                      LinearAnnealedPolicy(EpsGreedyQPolicy(), 'eps', 1, 0.05, 0, 50000)

        self.max_episode_steps = max_episode_steps
        self.optimizer = optimizer
        self.agent = self.get_agent(nb_actions=env.action_space.n)

        self.agent_config = agent_config

        self.logger = logger
        self.save_dir = saving_directory
        self.agent_name = saving_name

    def build_model(self, window_length, nb_actions):
        """  """
        model = Sequential()
        model.add(Flatten(
            input_shape=(window_length,) + self.env.observation_space.shape))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(nb_actions, activation='linear'))

        return model

    def get_agent(self, nb_actions):
        agent = DQNAgent(
                model=self.model,
                policy=self.policy,
                nb_actions=nb_actions,
                memory=self.memory,
                gamma=self.gamma,
                batch_size=self.batch_size,
                target_model_update=self.target_model_update,
                nb_steps_warmup=self.learning_starts,
                enable_double_dqn=True
            )
        return agent

    def calc_agent(self, train_steps, test_episodes, save=False,
                   verbose=1, render=False):
        """
        args:
            train_steps(int): number of steps to train the agent
            save(bool): save models and weights
            verbose(int):
        """
        if self.optimizer == 'adam':
            self.agent.compile(Adam(lr=self.learning_rate),
                               metrics=['mse']
                               )
        elif self.optimizer == 'rmsprop':
            pass
            # todo implement rmsprop

        self.agent.fit(self.env,
                       nb_steps=train_steps,
                       action_repetition=1,
                       verbose=verbose,
                       visualize=False,
                       nb_max_episode_steps=self.max_episode_steps,
                       log_interval=train_steps
                       )

        if save:
            path = self.agent_name
            # todo add own paths and names, check dtypes
            self.agent.save_weights(path + '_weights' + '.hdf5', overwrite=True)
            self.model.save_weights(path + '_model' + '.hdf5')

        log = self.agent.test(self.test_env,
                              nb_episodes=test_episodes,
                              nb_max_episode_steps=100000,
                              visualize=render
                              )

        cum_rew = np.sum([r for r in log.history['episode_reward']])
        total_steps = np.sum([s for s in log.history['nb_steps']])

        rew_per_step = np.float(np.round(cum_rew / total_steps, decimals=5))
        total_steps = int(total_steps)

        metrics = dict(reward_per_step=rew_per_step, number_steps=total_steps)

        return metrics

        # metrics = {'episode_reward': eps_rew,
        #            'cumulated_reward': cum_rew,
        #            'rew_per_step': cum_rew/eval_steps,
        #            'steps': eval_steps}
        #
        # return metrics


    # def test_agent(self, eval_steps, render=False, load=False, **kwargs):
    #     """
    #     args:
    #         eval_steps(int): number of steps to evaluate the agent
    #         visual(bool): True for environment visualization during testing
    #     """
    #     if load:
    #         pass
    #         # self.agent.compile(Adam(lr=self.learning_rate),
    #         #                    metrics=['mse']
    #         #                    )
    #         # self.agent.load_weights('save_dqn_keras.hdf5')
    #     else:
    #         self.agent.compile(Adam(lr=self.learning_rate),
    #                            metrics=['mse']
    #                            )
    #
    #     self.agent.test(self.env,
    #                     nb_episodes=20,
    #                     nb_max_episode_steps=eval_steps,
    #                     visualize=render
    #                     )
    #     # todo how to get metrics
    #
    #     # metrics = {'episode_reward': eps_rew,
    #     #            'cumulated_reward': cum_rew,
    #     #            'rew_per_step': cum_rew/eval_steps,
    #     #            'steps': eval_steps}
    #     #
    #     # return metrics
