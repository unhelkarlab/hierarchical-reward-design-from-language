import os
import sys
from pathlib import Path
from stable_baselines3 import PPO, DDPG, TD3, HerReplayBuffer
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize
from HierRL.envs.rw4t.rw4t_ll import EntropyAnnealingCallback
from HierRL.algs.bootstrap_dqn import get_bootstrap_model

import wandb


def train_ll_new(algo,
                 env_name,
                 env_factory,
                 env_kwargs,
                 eval_callback,
                 save_path,
                 seed,
                 policy='MlpPolicy',
                 pre_trained_path=None,
                 demos_folder=None,
                 env=None,
                 num_envs=1,
                 params=None,
                 **kwargs):
  total_timesteps = params['total_timesteps']

  # Define the environment
  if env is None:
    vec_env = DummyVecEnv(
        [lambda: env_factory(**env_kwargs) for _ in range(num_envs)])
  else:
    print('Using provided env')
    vec_env = DummyVecEnv([lambda: env])

  if algo == 'her':
    pass
    # vec_env = SubprocVecEnv(
    #     [lambda: env_factory(**env_kwargs) for _ in range(NUM_WORKERS)])
    # print('Using vec norm')
    # env = VecNormalize(vec_env,
    #                    norm_obs=True,
    #                    norm_reward=False,
    #                    clip_obs=5.0,
    #                    norm_obs_keys=['observation', 'desired_goal'])

  # Define callbacks
  if env_name == 'pnp':
    ts = 1_000_000
  else:
    ts = int(total_timesteps * 0.5)
  entropy_annealing_callback = EntropyAnnealingCallback(
      initial_ent_coef=params['initial_ent_coef'],
      final_ent_coef=0.01,
      total_timesteps=ts,
  )
  checkpoint_callback = CheckpointCallback(
      save_freq=100_000,  # Save model every 100,000 steps
      save_path=save_path,  # Folder to save models
      name_prefix=f"{algo}_recent",  # File name prefix
      save_vecnormalize=True  # Save VecNormalize (if used)
  )
  if algo == 'ppo':
    callback = CallbackList(
        [eval_callback, entropy_annealing_callback, checkpoint_callback])
  elif algo == 'ddpg' or 'td3' or 'her':
    callback = CallbackList([eval_callback, checkpoint_callback])
  else:
    raise NotImplementedError

  train_ll_impl(algo,
                vec_env,
                env_name,
                save_path,
                params,
                train_seed=seed,
                policy=policy,
                callback=callback,
                pre_trained_path=pre_trained_path,
                demos_folder=demos_folder,
                option_to_use=env_kwargs['option_to_use'],
                **kwargs)


def train_ll_impl(algo,
                  env,
                  env_name,
                  save_path,
                  params,
                  pre_trained_path,
                  demos_folder,
                  option_to_use=None,
                  train_seed=0,
                  policy='MlpPolicy',
                  callback=None,
                  verbose=1,
                  tensorboard_log=None,
                  **kwargs):

  if tensorboard_log is None:
    if 'eureka' in save_path:
      tensorboard_log = f'{Path(save_path).parent}/summaries'
      policy_save_folder = Path(save_path).parts[-1]
      if 'option' in policy_save_folder:
        tensorboard_log = f'{tensorboard_log}/{policy_save_folder}'
    else:
      tensorboard_log = f'{Path(save_path).parent}/tb_logs/{Path(save_path).name}'

  # learning_rate=3e-4,
  # batch_size=64,
  # total_timesteps=1_500_000,
  # gamma=1,
  # n_steps=2048,

  learning_rate = params['learning_rate']
  batch_size = params['batch_size']
  total_timesteps = params['total_timesteps']
  gamma = params['gamma']

  wandb.init(project="hrd_ll",
             config=params,
             sync_tensorboard=True,
             name=Path(save_path).name)

  if algo == 'ppo':
    n_steps = params['n_steps']
    ent_coef = params['initial_ent_coef']
  elif algo == 'ddpg' or algo == 'td3':
    buffer_size = params['buffer_size']
    learning_starts = params['learning_starts']
    if algo == 'td3':
      target_policy_noise = params['target_policy_noise']
      target_noise_clip = params['target_noise_clip']
  elif algo == 'her':
    print('we are here')
    buffer_size = params['buffer_size']
    learning_starts = params['learning_starts']
    tau = params['tau']
    replay_buffer_kwargs = params['replay_buffer_kwargs']
    action_noise = params['action_noise']

  if 'policy_kwargs' in params:
    policy_kwargs = params['policy_kwargs']
  else:
    policy_kwargs = dict(net_arch=[64, 64])
  print('params: ', params)
  print('policy: ', policy)

  # Run the environment checker to ensure everything is okay
  # if hasattr(env, "num_envs"):
  #   check_env(env.envs[0])
  # else:
  #   check_env(env)

  print('Environment is valid')
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
  print('Instantiating model')
  if algo == 'ppo':
    model = PPO(policy=policy,
                env=env,
                verbose=verbose,
                ent_coef=ent_coef,
                policy_kwargs=policy_kwargs,
                n_steps=n_steps,
                batch_size=batch_size,
                learning_rate=learning_rate,
                gamma=gamma,
                tensorboard_log=tensorboard_log,
                seed=train_seed)
    # Load pretrained model
    if pre_trained_path is not None:
      temp_model = PPO.load(pre_trained_path, env=env)
      model.policy.load_state_dict(temp_model.policy.state_dict())
      print('Loaded pretrained model from: ', pre_trained_path)
  elif algo == 'ddpg':
    print('DDPG')
    if demos_folder is not None:
      assert option_to_use is not None
      m = get_bootstrap_model(DDPG)
    else:
      m = DDPG

    model = m(policy=policy,
              env=env,
              learning_rate=learning_rate,
              buffer_size=buffer_size,
              learning_starts=learning_starts,
              batch_size=batch_size,
              gamma=gamma,
              tensorboard_log=tensorboard_log,
              policy_kwargs=policy_kwargs,
              verbose=verbose,
              seed=train_seed)
  elif algo == 'td3':
    print('TD3')
    if demos_folder is not None:
      assert option_to_use is not None
      m = get_bootstrap_model(TD3)
    else:
      m = TD3

    model = m(policy=policy,
              env=env,
              learning_rate=learning_rate,
              buffer_size=buffer_size,
              learning_starts=learning_starts,
              batch_size=batch_size,
              gamma=gamma,
              target_policy_noise=target_policy_noise,
              target_noise_clip=target_noise_clip,
              tensorboard_log=tensorboard_log,
              policy_kwargs=policy_kwargs,
              verbose=verbose,
              seed=train_seed)
  elif algo == 'her':
    print('HER')
    assert policy == 'MultiInputPolicy'
    model = DDPG(policy=policy,
                 env=env,
                 learning_rate=learning_rate,
                 buffer_size=buffer_size,
                 learning_starts=learning_starts,
                 batch_size=batch_size,
                 tau=tau,
                 gamma=gamma,
                 action_noise=action_noise,
                 replay_buffer_class=HerReplayBuffer,
                 replay_buffer_kwargs=replay_buffer_kwargs,
                 tensorboard_log=tensorboard_log,
                 policy_kwargs=policy_kwargs,
                 verbose=verbose,
                 seed=train_seed)

  # if algo == 'ddpg' or algo == 'td3':
  #   if demos_folder is not None:
  #     demos = load_demonstrations_with_reward(demos_folder=demos_folder,
  #                                             option_to_use=option_to_use)
  #     model.fill_replay_buffer_from_demos(demonstrations=demos,
  #                                         target_num_transitions=int(
  #                                             buffer_size * 0.15))

  # Train model
  print('Training model')
  model.learn(total_timesteps=total_timesteps,
              callback=callback,
              tb_log_name=tensorboard_log)

  # Close the file when done
  if 'redirect_output' in kwargs and kwargs['redirect_output']:
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()
