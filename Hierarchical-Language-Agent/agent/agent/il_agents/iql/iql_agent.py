import os
import shutil
import time
import pickle
from tqdm import tqdm
from copy import deepcopy
import datetime
import hydra
import wandb
import types
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from omegaconf import OmegaConf, DictConfig

from agent.config import OvercookedExp1
from agent.read_demonstrations import read_multiple_files
from agent.il_agents.agent_base import IL_Agent
from agent.executor.low import EnvState
from agent.executor.high import HighTask
from agent.mind.prompt_local import MOVE_TO_HT, ALL_MOVES
from agent.gameenv_single_concur import GameEnv_Single_Concur
from agent.il_agents.demonstrator_agent import get_priority_str, all_env_seeds
from agent.gameenv_single import GameEnv_Single, get_env

from aic_ml.baselines.IQLearn.agent.softq import SoftQ
from aic_ml.baselines.IQLearn.agent.softq_models import (SimpleQNetwork,
                                                         SingleQCriticDiscrete)
from aic_ml.baselines.IQLearn.agent.sac_models import DiscreteActor
from aic_ml.baselines.IQLearn.agent.sac_discrete import SAC_Discrete
from aic_ml.baselines.IQLearn.iql_offline import iq_offline
from aic_ml.baselines.IQLearn.dataset.memory import Memory
from aic_ml.baselines.IQLearn.utils.logger import Logger


class IQL_Agent(IL_Agent):

  def __init__(self, use_intent=True) -> None:
    super().__init__()
    # Whether to use the previous intent to predict the next macro action
    self.use_intent = use_intent
    self.prev_intent_idx = 0
    self.intent_hist = [ALL_MOVES[0]]

  def load_model(self, model_path, cfg_path, input_size):
    obs_dim = input_size
    action_dim = len(ALL_MOVES)
    discrete_obs = False
    q_net_base = SimpleQNetwork
    cfg = OmegaConf.load(cfg_path)
    self.model = SoftQ(cfg, obs_dim, action_dim, discrete_obs, q_net_base)
    self.model.load(model_path)

  def load_model_direct(self, model):
    self.model = model

  def step(self, env_state: EnvState, env_tensor):
    while True:
      if self._task is None:
        if self.use_intent:
          pred = self.model.choose_action(torch.cat(
              (env_tensor, torch.tensor([self.prev_intent_idx]))),
                                          sample=True)
        else:
          pred = self.model.choose_action(env_tensor, sample=True)
        pred = pred.item()
        self.prev_intent_idx = pred
        self.cur_intent = ALL_MOVES[pred]
        self.intent_hist.append(self.cur_intent)
        print('Move: ', self.cur_intent)
        self._task = deepcopy(MOVE_TO_HT[self.cur_intent])
        self.new_task = True
      else:
        self.new_task = False

      state, move, msg = self._task(env_state)
      if state == HighTask.Working:  # working
        return move, None
      elif state == HighTask.Failed:  # reassign task
        print(f"Move Failed: {move}")
        self._task = None
        return (0, 0), None
      else:
        self._task = None

  def __call__(self, env_state: EnvState, env_tensor, chat=''):
    return self.step(env_state, env_tensor)

  def get_intent_hist(self):
    return self.intent_hist[-5:]


# Read the dataset and preprocess the values
def read_datasets(fname_list,
                  concatenate,
                  filter,
                  write=True,
                  save_name='il_agents/overcooked_ring.pkl'):

  # Get features and labels from the dataset
  def concatenate_values(row, next=False):
    if not next:
      return np.concatenate(
          (row['features'], np.array([row['prev_macro_idx']])), axis=None)
    else:
      return np.concatenate(
          (row['next_features'], np.array([row['macro_idx']])), axis=None)

  demos = read_multiple_files(fname_list, filter)
  expert_dataset = {
      'states': [],
      'actions': [],
      'rewards': [],
      'next_states': [],
      'dones': [],
      'lengths': []
  }
  for fname in demos:
    df = demos[fname]
    df = df.dropna()  # Drop rows with any missing values
    df.loc[:, 'prev_macro_idx'] = df['prev_macro_idx'].astype(int)

    if concatenate:
      df = df.assign(X=df.apply(concatenate_values, axis=1))
      X = df['X'].values
      df = df.assign(X_next=df.apply(concatenate_values, args=(True, ), axis=1))
      X_next = df['X_next'].values
    else:
      X = df['features'].values
      X_next = df['next_features'].values
    X = [np.array(arr) for arr in X]
    X_next = [np.array(arr) for arr in X_next]
    y = df['macro_idx'].values
    r = df['reward'].values
    d = df['done'].values

    expert_dataset['states'].append(X)
    expert_dataset['actions'].append(y)
    expert_dataset['rewards'].append(r)
    expert_dataset['next_states'].append(X_next)
    expert_dataset['dones'].append(d)
    expert_dataset['lengths'].append(len(y))

  if write:
    with open(save_name, 'wb') as handle:
      pickle.dump(expert_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

  return expert_dataset, len(X[0])


def get_dirs(base_dir="",
             alg_name="gail",
             env_name="HalfCheetah-v2",
             msg="default"):

  base_log_dir = os.path.join(base_dir, "result/")

  ts_str = datetime.datetime.fromtimestamp(
      time.time()).strftime("%Y-%m-%d_%H-%M-%S")
  log_dir_root = os.path.join(base_log_dir, env_name, alg_name, msg, ts_str)
  save_dir = os.path.join(log_dir_root, "model")
  log_dir = os.path.join(log_dir_root, "log")
  os.makedirs(save_dir)
  os.makedirs(log_dir)

  return log_dir, save_dir


def save(agent, output_dir, fname):
  if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
  file_path = os.path.join(output_dir, fname)
  agent.save(file_path)


def evaluate(agent, priority, use_intent, seeds=[51, 52, 53]):
  c_reward = 0
  for seed in seeds:
    overcooked_env = get_env(OvercookedExp1, priority=priority, seed=seed)
    iql_agent = IQL_Agent(use_intent)
    iql_agent.load_model_direct(agent)
    game = GameEnv_Single(env=overcooked_env,
                          max_timesteps=1000,
                          agent_type='iql',
                          agent_model=iql_agent,
                          play=False)
    c_reward += game.execute_agent(fps=3, sleep_time=0, fname='', write=False)
  return c_reward / len(seeds)


def move_config(log_dir, dest_dir):

  def parse_folder_name(folder_name):
    return datetime.datetime.strptime(folder_name, '%Y-%m-%d_%H-%M-%S')

  folders = [
      folder for folder in os.listdir(log_dir)
      if os.path.isdir(os.path.join(log_dir, folder))
  ]
  sorted_folders = sorted(folders, key=parse_folder_name)
  last_folder_name = sorted_folders[-1]
  config_path = os.path.join(log_dir, last_folder_name, 'log/config.yaml')
  dest_path = os.path.join(dest_dir, 'config.yaml')
  shutil.copy(config_path, dest_path)


def convert_to_utf8(config_path, output_path):
  with open(config_path, 'rb') as f:
    content = f.read()
  with open(output_path, 'wb') as f:
    f.write(content.decode('latin1').encode('utf-8'))


# config_path specifies the directory where configuration files are located
# config_name specifies the base configuration file name
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train_iql(cfg: DictConfig):
  # Set some config variables
  p = additional_config['priority_str']
  cfg.n_traj = additional_config['num_demos']
  cfg.data_path = f'il_agents/iql/{p}_{skip_name}_{prev_name}/{cfg.n_traj}demos/overcooked_ring_{cfg.n_traj}demos.pkl'
  cfg.max_explore_step = additional_config['steps']

  # Get config variables
  env_name = cfg.env_name
  seed = cfg.seed
  batch_size = cfg.mini_batch_size
  agent_name = cfg.iql_agent_name
  max_step = int(cfg.max_explore_step)

  # set seeds
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  # device
  device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
  cfg.device = device_name
  device = torch.device(device_name)

  # Get logging and output directories
  log_dir, output_dir = get_dirs(cfg.base_dir, cfg.alg_name, cfg.env_name,
                                 f"{cfg.tag}")

  # save config
  config_path = os.path.join(log_dir, "config.yaml")
  with open(config_path, "w") as outfile:
    OmegaConf.save(config=cfg, f=outfile)

  # Initialize weights and biases
  dict_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
  run_name = f"{p}_{cfg.n_traj}{cfg.iql_agent_name}_{skip_name}_{prev_name}"
  wandb.init(project=env_name + '_offline',
             name=run_name,
             entity='...',
             sync_tensorboard=True,
             reinit=True,
             config=dict_config)

  # Set up agent
  obs_dim = additional_config['input_size']
  action_dim = len(ALL_MOVES)
  discrete_obs = False
  if cfg.iql_agent_name == 'softq':
    q_net_base = SimpleQNetwork
    use_target = False
    do_soft_update = False
    agent = SoftQ(cfg, obs_dim, action_dim, discrete_obs, q_net_base)
  elif cfg.iql_agent_name == 'sacd':
    use_target = True
    do_soft_update = True
    critic_base = SingleQCriticDiscrete
    actor = DiscreteActor(obs_dim, action_dim, cfg.hidden_policy)
    agent = SAC_Discrete(cfg, obs_dim, action_dim, discrete_obs, critic_base,
                         actor)
  else:
    raise NotImplementedError

  # Load demonstrations data
  expert_memory_replay = Memory(int(cfg.n_sample), seed)
  if (cfg.data_path.endswith("torch") or cfg.data_path.endswith("pt")
      or cfg.data_path.endswith("pkl") or cfg.data_path.endswith("npy")):
    path_iq_data = os.path.join(cfg.base_dir, cfg.data_path)
    num_trajs = cfg.n_traj
  else:
    print(f"Data path not exists: {cfg.data_path}")
  expertdata = expert_memory_replay.load(path_iq_data,
                                         num_trajs=num_trajs,
                                         sample_freq=1,
                                         seed=seed + 42)
  batch_size = min(batch_size, expert_memory_replay.size())

  # Setup logging
  writer = SummaryWriter(log_dir=log_dir)
  print(f'--> Saving logs at: {log_dir}')
  logger = Logger(log_dir,
                  log_frequency=cfg.log_interval,
                  writer=writer,
                  save_tb=True,
                  agent=agent_name,
                  run_name=f"{env_name}_{run_name}")

  # Train IQL agent
  best_eval_returns = -np.inf
  for update_steps in tqdm(range(1, max_step + 1)):
    expert_batch = expert_memory_replay.get_samples(batch_size, device)

    agent.iq_offline = types.MethodType(iq_offline, agent)
    losses = agent.iq_offline(expert_batch, logger, update_steps, use_target,
                              do_soft_update, cfg.method_regularize,
                              cfg.method_div)

    if update_steps % cfg.log_interval == 0:
      # Log losses
      for key, loss in losses.items():
        print(f'{key}: {loss}')
        writer.add_scalar("loss/" + key, loss, global_step=update_steps)

    if update_steps % cfg.eval_interval == 0 or update_steps == max_step:
      # Eval agent and save best model so far
      eval_return = evaluate(agent, priority, use_intent)
      print('Eval return: ', eval_return)
      logger.log('eval/episode_reward', eval_return, update_steps)

      eval_returns.append(eval_return)
      last_three_returns = eval_returns[-3:]
      avg_return = sum(last_three_returns) / len(last_three_returns)
      logger.log('eval/last3_avg_episode_reward', avg_return, update_steps)
      logger.dump(update_steps, ty='eval')

      if eval_return >= best_eval_returns:
        best_eval_returns = eval_return
        wandb.run.summary["best_returns"] = best_eval_returns
        if additional_config['save']:
          save(agent, additional_config['output_dir'],
               additional_config['fname'])

  wandb.finish()
  return


use_intent = True
filter = True
prev_name = 'noprev' if not use_intent else 'withprev'
skip_name = 'skip' if filter else 'noskip'
num_demos_list = [1]  # [1, 3, 5, 10]
num_demos_2_steps = {1: 300000, 3: 300000, 5: 400000, 10: 300000}
additional_config = {}
priority = [['David Soup'], ['Alice Soup']]
eval_returns = []


def main():

  for num_demos in num_demos_list:
    p = get_priority_str(priority)
    directory = f'demonstrations_new/{p}'
    train_env_seeds = all_env_seeds[:num_demos]
    file_names = [f'{p}_demo_env{s}_agent0.txt' for s in train_env_seeds]
    file_names = [os.path.join(directory, f) for f in file_names]
    # print('Train files: ', file_names)
    os.makedirs(
        f'il_agents/iql/{p}_{skip_name}_{prev_name}/{str(num_demos)}demos',
        exist_ok=True)
    expert_dataset, input_size = read_datasets(
        fname_list=file_names,
        concatenate=use_intent,
        filter=filter,
        write=False,
        save_name=
        f'il_agents/iql/{p}_{skip_name}_{prev_name}/{str(num_demos)}demos/overcooked_ring_{str(num_demos)}demos.pkl'
    )

    additional_config['priority_str'] = p
    additional_config['num_demos'] = num_demos
    additional_config['steps'] = num_demos_2_steps[num_demos]
    additional_config['input_size'] = input_size
    additional_config[
        'output_dir'] = f'il_agents/iql/{p}_{skip_name}_{prev_name}/{str(num_demos)}demos'
    additional_config['fname'] = f'best_softq_{str(num_demos)}demos'
    additional_config['save'] = True

    # train_iql()
    # eval_returns.clear()

    # move_config('result/Overcooked-Single/iql/default',
    #             additional_config['output_dir'])
    # convert_to_utf8(
    #     f'il_agents/iql/{p}_{skip_name}_{prev_name}/{str(num_demos)}demos/config.yaml',
    #     f'il_agents/iql/{p}_{skip_name}_{prev_name}/{str(num_demos)}demos/config_utf.yaml'
    # )

    all_evals = []
    eval_env_seeds = all_env_seeds[-3:]
    suffixes = ['fast' + str(i + 1) for i in range(5)]
    for env_seed in eval_env_seeds:
      for suffix in suffixes:
        overcooked_env = get_env(OvercookedExp1,
                                 priority=priority,
                                 seed=env_seed)
        iql_agent = IQL_Agent(use_intent)
        iql_agent.load_model(
            f'il_agents/iql/{p}_{skip_name}_{prev_name}/{str(num_demos)}demos/best_softq_{str(num_demos)}demos',
            f'il_agents/iql/{p}_{skip_name}_{prev_name}/{str(num_demos)}demos/config_utf.yaml',
            input_size=input_size)
        game = GameEnv_Single(env=overcooked_env,
                              max_timesteps=1000,
                              agent_type='iql',
                              agent_model=iql_agent,
                              play=False)
        c_rewards = game.execute_agent(
            fps=3,
            sleep_time=0,
            fname=
            f'il_agents/iql/{p}_{skip_name}_{prev_name}/{str(num_demos)}demos/test_env{str(env_seed)}_{suffix}',
            write=False)
        all_evals.append(c_rewards)

    all_evals_np = np.array(all_evals)
    print('mean: ', np.mean(all_evals_np))
    print('std: ', np.std(all_evals_np))


if __name__ == "__main__":
  main()
