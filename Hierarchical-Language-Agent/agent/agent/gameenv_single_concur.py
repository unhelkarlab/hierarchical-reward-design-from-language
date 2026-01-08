from collections import Counter
import queue
import time
import os
import threading
from copy import deepcopy
from typing import List
import random
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

from agent.executor.high import OBJ_TO_GOODS_GS, OBJ_TO_GOODS_POT
from agent.gameenv_single import AgentInfo
from gym_cooking.misc.game.game import Game
from agent.mind.prompt_local import ALL_MOVES
from agent.executor.low import EnvState
from gym_cooking.envs.overcooked_environment import (OvercookedEnvironment,
                                                     MapSetting)
from agent.gameenv_single import Transition

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

p_2_order_name = {
    'A': 'Alice Soup',
    'B': 'Bob Soup',
    'C': 'Cathy Soup',
    'D': 'David Soup'
}

ORDER_NAMES = {
    "CookedLettuce-CookedOnion-Plate": "Alice Soup",
    "CookedLettuce-CookedTomato-Plate": "Bob Soup",
    "CookedOnion-CookedTomato-Plate": "Cathy Soup",
    "CookedLettuce-CookedOnion-CookedTomato-Plate": "David Soup",
}

ORDER_TO_ALL_NAMES = {
    'David Soup': [
        'CookingLettuce-CookingOnion-CookingTomato',
        'CookedLettuce-CookedOnion-CookedTomato',
        'CookedLettuce-CookedOnion-CookedTomato-Plate'
    ],
    'Cathy Soup': [
        'CookingOnion-CookingTomato', 'CookedOnion-CookedTomato',
        'CookedOnion-CookedTomato-Plate'
    ],
    'Bob Soup': [
        'CookingLettuce-CookingTomato', 'CookedLettuce-CookedTomato',
        'CookedLettuce-CookedTomato-Plate'
    ],
    'Alice Soup': [
        'CookingLettuce-CookingOnion', 'CookedLettuce-CookedOnion',
        'CookedLettuce-CookedOnion-Plate'
    ]
}


def get_priority_list(pref_str: str) -> List[str]:
  '''
  Convert a priority string to a priority list.
  A priority string has the format: 'A_B'
  A priority list has the format: ['Alice Soup', 'Bob Soup']
  '''
  p_list = pref_str.split('_')
  priority_list = []
  for p in p_list:
    priority_list.append(p_2_order_name[p])
  return priority_list


def get_order_names(env_state: EnvState):
  '''
  Get all current order names.
  '''
  current_orders = env_state.order.current_orders
  order_names = [order.full_name for order, _, _, _ in current_orders]
  return [ORDER_NAMES[name] for name in order_names]


def get_num_priority_orders(env: EnvState, priority_order: str,
                            order_names: List[str]):
  '''
  Get the number of a soup that the needs to prepare, calculated as total 
  number of orders for that soup minus the number of soups being cooked.
  '''
  if priority_order in order_names:
    order_counts = Counter(order_names)
    cooking_count = 0
    pots = env.world.get_all_gridsquares('Pot')
    for pot in pots:
      if (pot.holding is not None
          and pot.holding.full_name in ORDER_TO_ALL_NAMES[priority_order]):
        cooking_count += 1
    return order_counts[priority_order] - cooking_count
  else:
    return 0


def get_sit_pref(pref_list: List[str], env_state: EnvState,
                 last_msg: str) -> str:
  '''
  Generate command for HLA.
  '''
  assert len(env_state.agents) == 1
  if env_state.agents[0].holding is not None:
    # If the agent is holding something, then we let the agent finish its
    # current action before sending it another command
    return last_msg

  order_names = get_order_names(env_state)

  # Send robot a command to plate a soup if a soup finishes cooking
  pots = env_state.world.get_all_gridsquares('Pot')
  for pot in pots:
    if pot.holding is not None and pot.holding.is_cooked(
    ) and 'Fire' not in pot.holding.full_name and env_state.rch_map[
        pot.location[0]][pot.location[1]]:
      dish_name = OBJ_TO_GOODS_POT[pot.holding.full_name]
      return 'Plate ' + dish_name

  # Send robot a command to put out a fire
  for pot in pots:
    if pot.holding is not None and pot.holding.is_cooked(
    ) and 'Fire' in pot.holding.full_name and env_state.rch_map[
        pot.location[0]][pot.location[1]]:
      # print('pot holding: ', pot.holding.full_name)
      return 'Putout the fire'

  # Send robot a command to discard burned dishes
  for pot in pots:
    if pot.holding is not None and pot.holding.is_cooked(
    ) and 'Char' in pot.holding.full_name and env_state.rch_map[
        pot.location[0]][pot.location[1]]:
      # print('pot holding: ', pot.holding.full_name)
      return 'Discard charred soup'

  # Send robot a command to serve a soup if a soup is plated but not served
  counters = env_state.world.get_all_gridsquares('Counter')
  for counter in counters:
    if counter.holding is not None:
      holding_name = OBJ_TO_GOODS_GS[counter.holding.full_name]
      sub_name = holding_name.split(' ', 1)
      if sub_name[0] == 'Plated' and sub_name[1] in order_names:
        return 'Serve ' + sub_name[1]

  # Send robot a command to make a soup based on the current orders
  sit_pref_str = 'Make '
  for pref in pref_list:
    num_to_prep = get_num_priority_orders(env_state, pref, order_names)
    if num_to_prep > 0:
      return sit_pref_str + pref

  return sit_pref_str + order_names[0]


def pref_helper(pref_str: str) -> str:
  """
  Convert a compact representation of the human's preference to a textual 
  description.
  """

  p_list = pref_str.split('_')
  processed_p = '''
Here is the human's suggestion:
Only prepare or cook the following soups for the rest of the game:
'''
  for i in range(len(p_list)):
    line = f'Priority #{i+1}: '
    letters_list = list(p_list[i])
    for letter in letters_list:
      soup_name = p_2_order_name[letter]
      line += (soup_name + ', ')
    processed_p += (line + '\n')

  processed_p += 'End of list.\n'
  return processed_p


class GameEnv_Single_Concur(Game):

  def __init__(
      self,
      env,
      max_timesteps: int,
      agent_type: str,
      agent_model,
      agent_fps: float,
      game_fps: float,
      p_str: str = '',
      play: bool = False,
      write: bool = False,
      fname: str = '',
  ):
    super().__init__(env, play=play)
    self.max_timesteps = max_timesteps
    self.agent_fps = agent_fps
    self.game_fps = game_fps
    self.write = write
    self.fname = fname
    self.action_dict = {agent.name: None for agent in self.sim_agents}
    self._q_env = queue.Queue()

    self.agent_info = AgentInfo(agent_type, 0)
    self.agent_info.agent = agent_model
    self.success = False

    self.p_list = []
    if p_str != '':
      self.p_list = get_priority_list(p_str)
    self.last_msg = ''
    # Wait a few frames to send commands to acount for 1) lag between when robot
    # thinks it finishes a high level action when it actually finishes the
    # action and 2) human typing time.
    # frames_waited is the number of frames waited.
    self.frames_waited = 0
    self.wait_needed = 15

  def _run_env(self):
    seconds_per_step = 1 / self.game_fps
    self.on_render()
    info = self.env.get_ai_info()
    state = self.env.get_current_state()
    old_features = self.env.get_current_features()
    env = EnvState(world=info['world'],
                   agents=info['sim_agents'],
                   agent_idx=0,
                   order=info['order_scheduler'],
                   event_history=info['event_history'],
                   time=info['current_time'],
                   chg_grid=info['chg_grid'],
                   env=self.env)
    self.agent_info.q.put_nowait(('Env', {
        "EnvState":
        deepcopy(env),
        "EnvTensor":
        torch.from_numpy(old_features).float()
    }))

    step = 0
    c_reward = 0
    next_frame_time = time.perf_counter()
    while step < self.max_timesteps:
      if self.success or self.env.current_time >= self.env.arglist.max_num_timesteps:
        return
      need_log = True
      old_state = self.env.get_current_state()
      while not self._q_env.empty():
        event = self._q_env.get_nowait()
        event_type, args = event
        if event_type == 'Action':
          self.action_dict[self.sim_agents[0].name] = args['action']
          # need_log = True

      ad = {
          k: v if v is not None else (0, 0)
          for k, v in self.action_dict.items()
      }
      state, reward, done, info = self.env.step(ad,
                                                passed_time=seconds_per_step)
      features = self.env.get_current_features()
      c_reward += reward
      if self.write and need_log:
        move = list(ad.values())[0]
        macro_action = self.agent_info.agent.get_cur_intent()
        if macro_action != 'None' and macro_action is not None:
          # print('macro: ', macro_action)
          macro_idx = ALL_MOVES.index(macro_action)
        else:
          # print('macro action is none')
          macro_idx = -1
        new_macro = True if (hasattr(self.agent_info.agent, 'new_task')
                             and self.agent_info.agent.new_task) else False
        transition = Transition(old_state, old_features, move, macro_action,
                                macro_idx, new_macro, reward, state, features,
                                done, info)
        self.write_transition(transition)
      step += 1
      if done:
        self.success = True

      info = self.env.get_ai_info()
      state = self.env.get_current_state()
      env = EnvState(world=info['world'],
                     agents=info['sim_agents'],
                     agent_idx=0,
                     order=info['order_scheduler'],
                     event_history=info['event_history'],
                     time=info['current_time'],
                     chg_grid=info['chg_grid'],
                     env=self.env)
      old_features = self.env.get_current_features()
      if self.action_dict[self.sim_agents[0].name] is not None:
        self.agent_info.q.put(('Env', {
            "EnvState":
            deepcopy(env),
            "EnvTensor":
            torch.from_numpy(old_features).float()
        }))
        if self.p_list != []:
          msg = get_sit_pref(self.p_list, env, self.last_msg)
          self.agent_info.q.put(('Chat', dict(chat=msg)))
      self.action_dict = {agent.name: None for agent in self.sim_agents}
      self.on_render()

      if done:
        return
      next_frame_time += seconds_per_step
      sleep_time = next_frame_time - time.perf_counter()
      if sleep_time > 0:
        time.sleep(sleep_time)
      else:
        print('Negative sleep time')

    print('Cumulative reward: ', c_reward)
    return c_reward

  def _run_ai(self):
    time_per_step = 1 / self.agent_fps

    env = None
    state = None
    env_update = False
    chat = ''
    while True:
      loop_start_time = time.perf_counter()
      if self.success or self.env.current_time >= self.env.arglist.max_num_timesteps:
        break
      event = self.agent_info.q.get()
      while True:
        event_type, args = event
        if event_type == 'Env':
          env = args['EnvState']
          state = args['EnvTensor']
          env_update = True
        elif event_type == 'Chat':
          chat = args['chat']
        if not self.agent_info.q.empty():
          event = self.agent_info.q.get()
        else:
          break

      if chat != '':
        if (self.agent_info.agent._is_finished and self.frames_waited
            == self.wait_needed) or self.last_msg != chat:
          # Only send msg if the agent is done processing the previous msg
          # or if the human has a new command.
          # We need to wait a few frames before sending the command.
          # print('Sending message to agent...')
          self.agent_info.agent.high_level_infer(env, chat)
          self.last_msg = chat
          chat = ''
          self.frames_waited = 0
        elif self.agent_info.agent._is_finished and self.frames_waited < self.wait_needed:
          self.frames_waited += 1

      if env_update:
        move, _ = self.agent_info.agent.step(env, state)
        self._q_env.put(('Action', {"agent": "ai", "action": move}))
        env_update = False

      # sleep
      elapsed_time = time.perf_counter() - loop_start_time
      sleep_time = max(time_per_step - elapsed_time, 0)
      time.sleep(sleep_time)

  def on_execute(self):
    self.on_init()

    thread_env = threading.Thread(target=self._run_env, daemon=True)
    thread_env.start()
    self._run_ai()
    self.on_cleanup()

  def write_transition(self, transition: Transition):
    header = ''
    if not os.path.isfile(self.fname):
      header += transition.get_header()
      header += '\n'

    data = str(transition)
    data += '\n'

    with open(self.fname, 'a+') as f:
      f.write(header)
      f.write(data)
      f.close()

  def state_to_tensor(self, state):
    state_map = torch.tensor(state['map'].tolist()).view(-1)
    state_orders = torch.tensor(state['current_orders'].tolist())
    state_holdings = torch.tensor(state['current_holdings']['agent-1'].tolist())
    state = torch.cat((state_map, state_orders, state_holdings))
    return state


def get_env(config: OvercookedConfig, priority, seed=MAP_SEED):
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
