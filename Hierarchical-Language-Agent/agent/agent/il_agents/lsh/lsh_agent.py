import os
import numpy as np
import random
import torch
from copy import deepcopy
import dill

from agent.gameenv_single import GameEnv_Single, get_env
from agent.config import OvercookedExp1
from agent.read_demonstrations import read_multiple_files
from agent.mind.lsh import LSHTable
from agent.il_agents.agent_base import IL_Agent
from agent.executor.low import EnvState
from agent.executor.high import HighTask
from agent.mind.prompt_local import MOVE_TO_HT, ALL_MOVES


# Read the dataset and preprocess the values
def read_datasets(fname_list):

  # Get features and labels from the dataset
  def concatenate_values(row):
    return np.concatenate((row['f_state'], np.array([row['prev_macro_idx']])),
                          axis=None)

  demos = read_multiple_files(fname_list)
  X_tensor = None
  y_tensor = None
  for fname in demos:
    df = demos[fname]
    df = df.dropna()  # Drop rows with any missing values
    df.loc[:, 'prev_macro_idx'] = df['prev_macro_idx'].astype(int)

    df = df.assign(X=df.apply(concatenate_values, axis=1))
    X = df['X'].values
    X = [torch.tensor(arr) for arr in X]
    if X_tensor is not None:
      X_tensor = torch.cat((X_tensor, torch.stack(X).float()), dim=0)
    else:
      X_tensor = torch.stack(X).float()

    y = df['macro_idx'].values
    if y_tensor is not None:
      y_tensor = np.concatenate((y_tensor, y))
    else:
      y_tensor = y

  return X_tensor, y_tensor


def insert_training_data(X, y):
  lsh_table = LSHTable(K=35, L=3)
  for i in range(X.shape[0]):
    lsh_table.insert(X[i], y[i])

  return lsh_table


class LSH_Agent(IL_Agent):

  def __init__(self):
    super().__init__()
    self.prev_intent_idx = 0
    self.lsh_table = None

  def load_model(self, lsh_path):
    with open(lsh_path, 'rb') as file:
      self.lsh_table = dill.load(file)

  def step(self, env_state: EnvState, env_tensor):
    while True:
      if self._task is None:
        actions_list = self.lsh_table.query(
            torch.cat((env_tensor, torch.tensor([self.prev_intent_idx]))))
        print('Action list: ', actions_list)
        actions_set = set(actions_list)
        if len(actions_set) > 0:
          action = random.choice(list(actions_set))
        else:
          action = random.choice(ALL_MOVES)
          action = ALL_MOVES.index(action)
        self.cur_intent = ALL_MOVES[action]
        print('Move: ', self.cur_intent)
        self._task = deepcopy(MOVE_TO_HT[self.cur_intent])

      state, move, msg = self._task(env_state)
      if state == HighTask.Working:  # working
        return move, None
      elif state == HighTask.Failed:  # reassign task
        print(f"Move Failed: {move}")
        self._task = None
        return (0, 0), None
      else:
        self._task = None


lsh_fname = 'il_agents/lsh/lsh_table.pkl'
# directory = 'demonstrations'
# file_names = os.listdir(directory)
# file_names = [os.path.join(directory, f) for f in file_names]
# X, y = read_datasets(fname_list=file_names)
# lsh_table = insert_training_data(X, y)
# with open(lsh_fname, 'wb') as file:
#   dill.dump(lsh_table, file)

env_seed = 0
overcooked_env = get_env(OvercookedExp1, seed=env_seed)
lsh_agent = LSH_Agent()
lsh_agent.load_model(lsh_fname)
game = GameEnv_Single(env=overcooked_env,
                      max_timesteps=1000,
                      agent_type='lsh',
                      agent_model=lsh_agent,
                      play=True)
game.execute_agent(fps=3, sleep_time=0.1, fname='', write=False)
