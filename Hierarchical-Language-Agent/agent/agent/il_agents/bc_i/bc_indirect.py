import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from copy import deepcopy
from sklearn.model_selection import train_test_split

from agent.gameenv_single import GameEnv_Single, get_env
from agent.config import OvercookedExp1
from agent.il_agents.bc.bc import create_loaders, train_model, BC_Model
from agent.read_demonstrations import read_multiple_files
from agent.il_agents.agent_base import IL_Agent
from agent.executor.low import EnvState
from agent.executor.high import HighTask
from agent.mind.prompt_local import MOVE_TO_HT, ALL_MOVES


def read_datasets(fname_list):
  # Get features and labels from the dataset
  def concatenate_values(row):
    return np.concatenate((row['f_state'], np.array([row['prev_macro_idx']])),
                          axis=None)

  demos = read_multiple_files(fname_list)
  combined_state_tensor = None
  same_macro_tensor = None
  state_tensor = None
  macro_tensor = None
  for fname in demos:
    df = demos[fname]
    df = df.dropna()  # Drop rows with any missing values
    df.loc[:, 'prev_macro_idx'] = df['prev_macro_idx'].astype(int)
    df = df.assign(
        same_macro_idx=np.where(df['macro_idx'] == df['prev_macro_idx'], 1, 0))

    df = df.assign(combined_state=df.apply(concatenate_values, axis=1))
    combined = df['combined_state'].values
    combined = [torch.tensor(arr) for arr in combined]
    if combined_state_tensor is not None:
      combined_state_tensor = torch.cat(
          (combined_state_tensor, torch.stack(combined).float()), dim=0)
    else:
      combined_state_tensor = torch.stack(combined).float()

    same = torch.tensor(df['same_macro_idx'].values).reshape(-1, 1)
    if same_macro_tensor is not None:
      same_macro_tensor = torch.cat((same_macro_tensor, same.float()), dim=0)
    else:
      same_macro_tensor = same.float()

    state = df['f_state'].values
    state = [torch.tensor(arr) for arr in state]
    if state_tensor is not None:
      state_tensor = torch.cat((state_tensor, torch.stack(state).float()),
                               dim=0)
    else:
      state_tensor = torch.stack(state).float()

    macro = torch.tensor(df['macro_idx'].values)
    if macro_tensor is not None:
      macro_tensor = torch.cat(
          (macro_tensor, F.one_hot(macro, num_classes=len(ALL_MOVES)).float()),
          dim=0)
    else:
      macro_tensor = F.one_hot(macro, num_classes=len(ALL_MOVES)).float()

  (combined_state_tensor_train, combined_state_tensor_test,
   same_macro_tensor_train,
   same_macro_tensor_test) = train_test_split(combined_state_tensor,
                                              same_macro_tensor,
                                              test_size=0.2,
                                              random_state=0)

  (state_tensor_train, state_tensor_test, macro_tensor_train,
   macro_tensor_test) = train_test_split(state_tensor,
                                         macro_tensor,
                                         test_size=0.2,
                                         random_state=0)
  return (combined_state_tensor_train, combined_state_tensor_test,
          same_macro_tensor_train, same_macro_tensor_test, state_tensor_train,
          state_tensor_test, macro_tensor_train, macro_tensor_test)


class BCI_Agent(IL_Agent):
  """
  Indirect BC Agent
  """

  def __init__(self) -> None:
    super().__init__()
    self.prev_intent_idx = 0

  def load_model(self, transition_model, t_model_input_size,
                 t_model_output_size, macro_model, m_model_input_size,
                 m_model_output_size):
    self.transition_model = BC_Model(t_model_input_size, t_model_output_size)
    self.transition_model.load_state_dict(torch.load(transition_model))
    self.transition_model.eval()

    self.macro_model = BC_Model(m_model_input_size, m_model_output_size)
    self.macro_model.load_state_dict(torch.load(macro_model))
    self.macro_model.eval()

  def step(self, env_state: EnvState, env_tensor):
    while True:
      if self._task is None:
        transition = self.transition_model(
            torch.cat((env_tensor, torch.tensor([self.prev_intent_idx]))))
        transition_prob = torch.sigmoid(transition)
        print('transition prob: ', transition_prob.item())
        if transition_prob.item() < 0.5:
          self.cur_intent = ALL_MOVES[self.prev_intent_idx]
        else:
          pred = self.macro_model(env_tensor)
          pred_ = pred.softmax(dim=-1)
          pred_label = torch.multinomial(pred_, num_samples=1).item()
          self.prev_intent_idx = pred_label
          self.cur_intent = ALL_MOVES[pred_label]
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


print('Read demonstrations...')
directory = 'demonstrations'
file_names = os.listdir(directory)
file_names = [os.path.join(directory, f) for f in file_names]
(combined_state_tensor_train, combined_state_tensor_test,
 same_macro_tensor_train, same_macro_tensor_test, state_tensor_train,
 state_tensor_test, macro_tensor_train,
 macro_tensor_test) = read_datasets(fname_list=file_names)

# print('combined_state_tensor_train: ', combined_state_tensor_train.shape)
# print('combined_state_tensor_test: ', combined_state_tensor_test.shape)
# print('same_macro_tensor_train: ', same_macro_tensor_train.shape)
# print('same_macro_tensor_test: ', same_macro_tensor_test.shape)
# print('state_tensor_train: ', state_tensor_train.shape)
# print('state_tensor_test: ', state_tensor_test.shape)
# print('macro_tensor_train: ', macro_tensor_train.shape)
# print('macro_tensor_test: ', macro_tensor_test.shape)

# print('Train bc transition...')
# transition_train_loader, transition_val_loader = create_loaders(
#     combined_state_tensor_train, combined_state_tensor_test,
#     same_macro_tensor_train, same_macro_tensor_test)

# train_model(100,
#             combined_state_tensor_train.shape[1],
#             same_macro_tensor_train.shape[1],
#             transition_train_loader,
#             transition_val_loader,
#             save_path='il_agents/bc_i/best_transition.pth',
#             wandb_name='overcooked_bc_transition',
#             save=True,
#             loss=nn.BCEWithLogitsLoss())

# print('Train bc macro pred...')
# macro_train_loader, macro_val_loader = create_loaders(state_tensor_train,
#                                                       state_tensor_test,
#                                                       macro_tensor_train,
#                                                       macro_tensor_test)

# train_model(200,
#             state_tensor_train.shape[1],
#             macro_tensor_train.shape[1],
#             macro_train_loader,
#             macro_val_loader,
#             save_path='il_agents/bc_i/best_macro_pred.pth',
#             wandb_name='overcooked_bc_macro_pred',
#             save=True)

env_seeds = [3, 4, 5, 6, 7, 8, 9]
suffix = 'fast1'

print('Evaluate models...')
for env_seed in env_seeds:
  overcooked_env = get_env(OvercookedExp1, seed=env_seed)
  bci_agent = BCI_Agent()
  bci_agent.load_model('il_agents/bc_i/best_transition.pth',
                       combined_state_tensor_train.shape[1],
                       same_macro_tensor_train.shape[1],
                       'il_agents/bc_i/best_macro_pred.pth',
                       state_tensor_train.shape[1], macro_tensor_train.shape[1])
  game = GameEnv_Single(env=overcooked_env,
                        max_timesteps=1000,
                        agent_type='bci',
                        agent_model=bci_agent,
                        play=False)
  game.execute_agent(
      fps=3,
      sleep_time=0.001,
      fname=f'il_agents/bc_i/results/test_env{str(env_seed)}_{suffix}',
      write=True)
