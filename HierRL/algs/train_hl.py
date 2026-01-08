import os
import sys
from pathlib import Path
import wandb
import torch
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (BaseCallback,
                                                CheckpointCallback,
                                                CallbackList)

from HierRL.models.maskable_policies import MaskableDQNPolicy
from HierRL.algs.maskable_dqn import MaskableDQN
from HierRL.algs.variable_step_dqn import VariableStepDQN
# from double_dqn import DoubleDQN
# from per import PERDQN
from HierRL.algs.bootstrap_dqn import get_bootstrap_model


class QValueLoggingCallback(BaseCallback):

  def __init__(self, log_freq, verbose=0):
    super().__init__(verbose)
    self.log_freq = log_freq

  def _on_step(self) -> bool:
    # Log Q-values every `log_freq` steps
    if isinstance(self.model, DQN):
      if self.num_timesteps % self.log_freq == 0:
        # Sample random states from the replay buffer
        replay_data = self.model.replay_buffer.sample(
            100, env=self.model._vec_normalize_env)

        # Compute Q-values using the Q-network
        with torch.no_grad():
          q_values = self.model.q_net(replay_data.observations)
          mean_q_value = q_values.mean().item()

        # Log to TensorBoard
        self.logger.record("q_values/mean_q_value", mean_q_value)
        if self.verbose > 0:
          print(f"Step {self.num_timesteps}: Mean Q-value = {mean_q_value:.4f}")

    return True


def train_hl_new(env_factory,
                 env_kwargs,
                 eval_callback,
                 model_type,
                 controller_save_path,
                 seed,
                 num_envs=1,
                 params=None,
                 **kwargs):
  # Make environment
  vec_env = DummyVecEnv(
      [lambda: env_factory(**env_kwargs) for _ in range(num_envs)])

  # Prepare policy info
  if model_type == 'MaskableDQN':
    policy_dict = dict(base_dim=params['base_dim'],
                       n_actions=params['n_actions'],
                       net_arch=params['net_arch'])
    model_dict = dict(base_dim=params['base_dim'],
                      n_actions=params['n_actions'])
    policy = MaskableDQNPolicy
  else:
    policy_dict = dict(net_arch=params['net_arch'])
    model_dict = {}
    policy = 'MlpPolicy'

  if model_type == 'VariableStepDQN':
    save_freq = 1_000
  else:
    save_freq = 50_000
  checkpoint_callback = CheckpointCallback(
      save_freq=save_freq,  # Save model every 100,000 steps
      save_path=controller_save_path,  # Folder to save models
      name_prefix="dqn_recent",  # File name prefix
      save_replay_buffer=False,  # Save replay buffer (optional)
      save_vecnormalize=True  # Save VecNormalize (if used)
  )
  q_value_callback = QValueLoggingCallback(log_freq=10_000)
  callback = CallbackList(
      [eval_callback, q_value_callback, checkpoint_callback])
  # print('provided hl pref r: ', hl_pref_r)
  # print('provided pbrs r: ', pbrs_r)
  # print_env = vec_env.envs[0]
  # print(type(print_env))
  # while hasattr(print_env, 'env'):
  #   print_env = print_env.env
  #   print(type(print_env))
  # if hasattr(vec_env.envs[0].unwrapped, 'base_env'):
  #   print('environment hl pref r: ',
  #         vec_env.envs[0].unwrapped.base_env.hl_pref_r)
  #   print('environment pbrs r: ', vec_env.envs[0].unwrapped.base_env.pbrs_r)
  # else:
  #   print('environment hl pref r: ', vec_env.envs[0].unwrapped.hl_pref_r)
  #   print('environment pbrs r: ', vec_env.envs[0].unwrapped.pbrs_r)
  # print('eval hl pref: ', eval_hl_pref)
  # print('learning rate: ', default_params['learning_rate'])
  # return
  train_hl_impl(vec_env,
                controller_save_path,
                params=params,
                model_type=model_type,
                seed=seed,
                policy=policy,
                policy_kwargs=policy_dict,
                model_kwargs=model_dict,
                callback=callback,
                verbose=1,
                **kwargs)


def train_hl_impl(env,
                  save_path,
                  params,
                  model_type='DQN',
                  seed=0,
                  policy='MlpPolicy',
                  policy_kwargs=None,
                  model_kwargs=None,
                  callback=None,
                  verbose=1,
                  tensorboard_log=None,
                  **kwargs):

  if tensorboard_log is None:
    if 'eureka' in save_path:
      tensorboard_log = f'{Path(save_path).parent}/summaries'
    else:
      tensorboard_log = f'{Path(save_path).parent}/tb_logs/{Path(save_path).name}'

  # if 'rw4t' in save_path:
  #   env_type = 'rw4t'
  # elif 'oc' in save_path:
  #   env_type = 'oc'
  # else:
  #   raise NotImplementedError
  # if tensorboard_log is None:
  #   tensorboard_log = f'{Path(save_path).parent}/../../../tensorboard_logs/' + \
  #     f'{env_type}/{Path(Path(save_path).parent).name}/{Path(save_path).name}'

  # Run the environment checker to ensure everything is okay
  if model_type != 'MaskableDQN':
    if hasattr(env, "num_envs"):
      check_env(env.envs[0])
    else:
      check_env(env)

  wandb.init(project="hrd_hl",
             config=params,
             sync_tensorboard=True,
             name=Path(save_path).name)

  bootstrap_model_path = params.get('bootstrap_model_path', None)

  # model = PPO(policy=policy,
  #             env=vec_env,
  #             verbose=1,
  #             ent_coef=default_params['ent_coef'],
  #             policy_kwargs=policy_dict,
  #             n_steps=2048,
  #             batch_size=default_params['batch_size'],
  #             learning_rate=default_params['learning_rate'],
  #             clip_range=default_params['clip_range'],
  #             gamma=0.99,
  #             tensorboard_log="./tensorboard_logs/")

  # Determine model type
  if model_type == 'DQN':
    m = DQN
  elif model_type == 'MaskableDQN':
    assert policy == MaskableDQNPolicy, \
      'Maskable DQN is only compatible with MaskableDQNPolicy'
    m = MaskableDQN
  elif model_type == 'VariableStepDQN':
    m = VariableStepDQN
    # Check if info has the right format
    if hasattr(env, "num_envs"):
      _obs, info = env.envs[0].reset()
    else:
      _obs, info = env.reset()
    assert 'num_steps' in info, \
      'You need to provide the number of internal steps for executing the ' + \
      'current option in "info"'

  # Redirect stdout and stderr outputs to an output file
  if 'redirect_output' in kwargs and kwargs['redirect_output']:
    print('Redirecting output to file')
    original_stdout = sys.stdout  # Save the original stdout
    original_stderr = sys.stderr  # Save the original stderr
    os.makedirs(save_path, exist_ok=False)
    log_file = open(f'{save_path}/output.txt', 'w')
    sys.stdout = log_file  # Redirect stdout to the file
    sys.stderr = log_file  # Redirect stderr to stdout

  # Instantiate model
  if bootstrap_model_path is not None:
    m = get_bootstrap_model(m)
  model = m(policy=policy,
            env=env,
            verbose=verbose,
            exploration_fraction=params['exploration_fraction'],
            exploration_initial_eps=params['exploration_initial_eps'],
            exploration_final_eps=params['exploration_final_eps'],
            buffer_size=params['buffer_size'],
            learning_starts=params['learning_starts'],
            learning_rate=params['learning_rate'],
            gamma=params['gamma'],
            batch_size=params['batch_size'],
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            seed=seed)
  # Add model params
  if model_type == 'MaskableDQN':
    assert model_kwargs is not None, \
      'You need to provided model_kwargs with base_dim and n_actions'
    assert 'base_dim' in model_kwargs and 'n_actions' in model_kwargs, \
      'You need to provided model_kwargs with base_dim and n_actions'
    model.set_dims(model_kwargs['base_dim'], model_kwargs['n_actions'])
  if bootstrap_model_path is not None:
    model.fill_replay_buffer(bootstrap_model_path, fill_portion=0.2)
  # Start RL
  model.learn(total_timesteps=params['total_timesteps'],
              callback=callback,
              tb_log_name=tensorboard_log)

  # Close the file when done
  if 'redirect_output' in kwargs and kwargs['redirect_output']:
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()
