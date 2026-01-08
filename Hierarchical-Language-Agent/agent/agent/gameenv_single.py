import queue
import os
import time
import threading
from copy import deepcopy
import torch
from agent.config import OvercookedConfig
import web_experiment.exp_common.overcooked_helper as helper

# We need to change the params in helper before importing anything from
# gym_cooking as the scripts in gym_cooking also import from helper. The
# rest of game initialization is done in the init_user_data function below.
helper.CHOPPING_NUM_STEPS = OvercookedConfig.COOKING_TIME_SECONDS
helper.COOKED_BEFORE_FIRE_TIME_SECONDS = OvercookedConfig.COOKED_BEFORE_FIRE_TIME_SECONDS
helper.FIRE_PUTOUT_TIME_SECONDS = OvercookedConfig.FIRE_PUTOUT_TIME_SECONDS
helper.FIRE_RECOVER_GAP_TIME_SECONDS = OvercookedConfig.FIRE_RECOVER_GAP_TIME_SECONDS
helper.CHOPPING_NUM_STEPS = OvercookedConfig.CHOPPING_NUM_STEPS
helper.MAX_ORDER_LENGTH_SECONDS = OvercookedConfig.MAX_ORDER_LENGTH_SECONDS
helper.ORDER_EXPIRE_PUNISH = OvercookedConfig.ORDER_EXPIRE_PUNISH

from gym_cooking.misc.game.game import Game
from gym_cooking.utils.replay import Replay
from agent.mind.prompt_local import ALL_MOVES
from agent.executor.low import EnvState
from agent.mind.agent_new import get_agent, AgentSetting
from gym_cooking.envs.overcooked_environment import (OvercookedEnvironment,
                                                     MapSetting)

MAP_SETTINGS = dict(
    ring=dict(level="new1", ),
    bottleneck=dict(level="new3", ),
    partition=dict(level="new2"),
    quick=dict(
        level="new5",
        max_num_orders=4,
    ),
)

MAP_SEED = 0
AGENT_SEED = 0


class AgentInfo():

  def __init__(self, type, idx) -> None:
    self.type = type
    self.idx = idx
    self.agent = None
    self.q = queue.Queue()


class Transition():

  def __init__(self, state, features, action, macro_action, macro_idx,
               new_macro, reward, next_state, next_features, done,
               info) -> None:
    self.state = state
    self.features = features
    self.action = action
    self.macro_action = macro_action
    self.macro_idx = macro_idx
    self.new_macro = new_macro
    self.reward = reward
    self.next_state = next_state
    self.next_features = next_features
    self.done = done
    self.info = info

  def __str__(self):
    state_map = self.state['map'].tolist()
    state_orders = self.state['current_orders'].tolist()
    state_holdings = {
        key: value.tolist()
        for key, value in self.state['current_holdings'].items()
    }
    state_str = self.get_state_str(state_map, state_orders, state_holdings)

    next_state_map = self.next_state['map'].tolist()
    next_state_orders = self.next_state['current_orders'].tolist()
    next_state_holdings = {
        key: value.tolist()
        for key, value in self.next_state['current_holdings'].items()
    }
    next_state_str = self.get_state_str(next_state_map, next_state_orders,
                                        next_state_holdings)

    return state_str + '; ' + str(self.features.tolist()) + '; ' + str(
        self.action) + '; ' + str(self.macro_action) + '; ' + str(
            self.macro_idx) + '; ' + str(self.new_macro) + '; ' + str(
                self.reward) + '; ' + next_state_str + '; ' + str(
                    self.next_features.tolist()) + '; ' + str(
                        self.done) + '; ' + str(self.info)

  def get_state_str(self, game_map, orders, holdings):
    return str(game_map) + '|' + str(orders) + '|' + str(holdings['agent-1'])

  def get_header(self):
    return 'state; features; action; macro_action; macro_idx; new_macro; reward; next_state; next_features; done; info'


class GameEnv_Single(Game):

  def __init__(
      self,
      env,
      max_timesteps: int,
      agent_type: str,
      prev_macro_idx: int = 0,
      agent_seed: int = 0,
      agent_model=None,
      play: bool = False,
  ):
    super().__init__(env, play=play)
    self.max_timesteps = max_timesteps
    self.action_dict = {agent.name: None for agent in self.sim_agents}

    self.agent_info = AgentInfo(agent_type, 0)
    if agent_type in ['ai', 'sim_h', 'bc', 'lsh', 'iql', 'bci', 'futures']:
      self.agent_info.agent = agent_model

    self.on_init()
    if self.play:
      self.on_render()

    self.prev_macro_idx = prev_macro_idx
    self.all_obs = []
    self.all_hl_actions = []
    self.all_next_obs = []

  def execute_agent(self,
                    fps: float,
                    sleep_time: float,
                    fname: str,
                    write: bool = False):
    step = 0
    c_reward = 0
    while step < self.max_timesteps:
      old_state = self.env.get_current_state()
      old_features = self.env.get_current_features()
      info = self.env.get_ai_info()
      env = EnvState(world=info['world'],
                     agents=info['sim_agents'],
                     agent_idx=0,
                     order=info['order_scheduler'],
                     event_history=info['event_history'],
                     time=info['current_time'],
                     chg_grid=info['chg_grid'],
                     env=self.env)
      move, _ = self.agent_info.agent.step(
          env,
          torch.from_numpy(old_features).float())
      macro_action = self.agent_info.agent.get_cur_intent()
      if macro_action != 'None' and macro_action is not None:
        macro_idx = ALL_MOVES.index(macro_action)
      else:
        macro_idx = -1
      self.action_dict[self.sim_agents[0].name] = move
      ad = {
          k: v if v is not None else (0, 0)
          for k, v in self.action_dict.items()
      }
      state, reward, done, info = self.env.step(ad, passed_time=1 / fps)
      features = self.env.get_current_features()
      c_reward += reward
      if write:
        new_macro = True if (self.agent_info.agent.new_task is not None
                             and self.agent_info.agent.new_task) else False
        transition = Transition(old_state, old_features, move, macro_action,
                                macro_idx, new_macro, reward, state, features,
                                done, info)
        self.write_transition(transition, fname)
      self.add_obs_and_actions(old_features, self.prev_macro_idx, features,
                               macro_idx)
      self.prev_macro_idx = macro_idx
      step += 1
      if done:
        break
      if self.agent_info.type == 'futures' and self.agent_info.agent.done:
        # If the agent is simulating the future, then the agent knows when
        # the simulation should be done, not the environment.
        break
      if self.play:
        self.on_render()
      if sleep_time > 0:
        time.sleep(sleep_time)
    # print('Cumulative reward: ', c_reward)
    return c_reward

  def write_transition(self, transition: Transition, fname: str):
    header = ''
    if not os.path.isfile(fname):
      header += transition.get_header()
      header += '\n'

    data = str(transition)
    data += '\n'

    with open(fname, 'a+') as f:
      f.write(header)
      f.write(data)
      f.close()

  def state_to_tensor(self, state):
    state_map = torch.tensor(state['map'].tolist()).view(-1)
    state_orders = torch.tensor(state['current_orders'].tolist())
    state_holdings = torch.tensor(state['current_holdings']['agent-1'].tolist())
    state = torch.cat((state_map, state_orders, state_holdings))
    return state

  def add_obs_and_actions(self, features, prev_action, next_features,
                          next_action):
    obs = torch.cat(
        (torch.from_numpy(features).float(), torch.tensor([prev_action])))
    self.all_obs.append(obs)
    self.all_hl_actions.append(next_action)
    next_obs = torch.cat(
        (torch.from_numpy(next_features).float(), torch.tensor([next_action])))
    self.all_next_obs.append(next_obs)


def get_env(config: OvercookedConfig, priority=[], seed=MAP_SEED):
  map_set = MapSetting(**MAP_SETTINGS[config.game_map])
  map_set.user_recipy = config.user_recipy
  map_set.ai_recipy = config.ai_recipy
  map_set.max_num_timesteps = config.max_num_timesteps
  map_set.max_num_orders = config.max_num_orders
  map_set.num_agents = config.num_agents
  map_set.seed = seed
  map_set.priority = priority
  env = OvercookedEnvironment(map_set)
  env.reset()
  return env
