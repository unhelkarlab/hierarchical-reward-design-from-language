from copy import deepcopy
from dataclasses import dataclass, field
import gymnasium as gym
import numpy as np
import pygame
import time

from agent.executor.high import HighTask
from agent.executor.low import EnvState
from gym_cooking.utils.core import Ingredients, SoupType
from gym_cooking.envs.overcooked_environment import OvercookedEnvironment
from agent.executor.high import (HighTask, HTChop, HTAssemble, HTPutout, HTCook,
                                 HTPick, HTServe, HTDrop, HTWait,
                                 ALL_FRESH_FOOD, ALL_ASSEMBLE, ALL_SOUP,
                                 ALL_SALAD, HT_MAP)
from gym_cooking.misc.game.game import Game


@dataclass
class MapSetting:
  level: str
  user_recipy: bool = True  # whether user can see the recipy
  ai_recipy: bool = False  # whether ai can see the recipy
  max_num_timesteps: int = 100  # max number of timesteps
  max_num_orders: int = 1  # max number of orders
  seed: int = 0  # seed for the order scheduler
  priority: list = field(
      default_factory=lambda: [['David Salad']])  # orders to prioritize

  num_agents: int = 1  # fixed


class KitchenHLGPT(OvercookedEnvironment):
  '''
  A simplified Overcooked domain where one agent only needs to make one soup.
  '''

  def __init__(self,
               arglist,
               salad=True,
               serve=True,
               ez=True,
               detailed_hl_pref=False,
               dish_type='David'):
    super().__init__(arglist)

    if salad:
      self.moves_to_ht = {
          **{
              f"{HT_MAP['Chop']} {x}": HTChop(x)
              for x in ALL_FRESH_FOOD
          },
          **{
              f"{HT_MAP['Assemble']} {x}": HTAssemble(x)
              for x in ALL_ASSEMBLE
          },
          **{
              f"{HT_MAP['Putout']}": HTPutout()
          },
          #  **{f"{HT_MAP['Cook']} {x}": HTCook(x) for x in ALL_SALAD},
          **{
              f"{HT_MAP['Pick']} {x}": HTPick(x)
              for x in ALL_SALAD
          },
          **{
              f"{HT_MAP['Serve']} {x}": HTServe(x)
              for x in ALL_SALAD
          },
          **{
              f"{HT_MAP['Drop']}": HTDrop()
          },
          **{
              f"{HT_MAP['Wait']}": HTWait()
          }
      }
    else:
      self.moves_to_ht = {
          **{
              f"{HT_MAP['Chop']} {x}": HTChop(x)
              for x in ALL_FRESH_FOOD
          },
          **{
              f"{HT_MAP['Assemble']} {x}": HTAssemble(x)
              for x in ALL_ASSEMBLE
          },
          **{
              f"{HT_MAP['Putout']}": HTPutout()
          },
          **{
              f"{HT_MAP['Cook']} {x}": HTCook(x)
              for x in ALL_SOUP
          },
          **{
              f"{HT_MAP['Pick']} {x}": HTPick(x)
              for x in ALL_SOUP
          },
          **{
              f"{HT_MAP['Serve']} {x}": HTServe(x)
              for x in ALL_SOUP
          },
          **{
              f"{HT_MAP['Drop']}": HTDrop()
          },
          **{
              f"{HT_MAP['Wait']}": HTWait()
          }
      }
    self.every_moves = list(self.moves_to_ht.keys())
    self.serve = serve
    self.ez = ez
    self.detailed_hl_pref = detailed_hl_pref
    self.dish_type = dish_type
    self._set_moves_dict()

    # The high-level preference reward
    self.hl_pref_reward = 0.2
    # Step penalty
    self.step_penalty = -0.01
    # Completion reward
    self.completion_reward = 1

    self.c_task_reward = 0
    self.c_gt_hl_pref = 0

  def step(self, action_dict, option, passed_time=1):
    hl_pref_reward = 0
    gt_hl_pref_reward = 0
    if 'Eureka' in self.__class__.__name__:
      hl_pref_reward += self.get_high_level_pref_gpt(self.state,
                                                     self.prev_option, option)
      gt_hl_pref_reward += self.get_high_level_pref(self.state,
                                                    self.prev_option, option)
    else:
      hl_pref_reward += self.get_high_level_pref(self.state, self.prev_option,
                                                 option)
      gt_hl_pref_reward += hl_pref_reward
    state, reward, truncated, info = super().step(action_dict, passed_time)
    state = self.modify_state(state)
    reward = self.modify_task_reward(reward)
    done = self.check_done(reward)
    self.prev_option = option

    self.c_task_reward += reward
    self.c_gt_hl_pref += gt_hl_pref_reward
    info['c_task_reward'] = self.c_task_reward
    info['c_gt_hl_pref'] = self.c_gt_hl_pref

    # print('state (just chopped): ', state['just_chopped'])
    # print(f'rewards: {reward} (task), {hl_reward} (hl)')
    # print('done: ', done)
    # print('truncated: ', truncated)
    return state, (reward, hl_pref_reward), done, truncated, info

  def reset(self):
    obs, info = super().reset()
    self.c_task_reward = 0
    self.c_gt_hl_pref = 0
    obs = self.modify_state(obs)
    # Set prev_option to dummy option
    self.prev_option = self.all_moves_dict_with_wait['Wait']
    return obs, info

  def _set_moves_dict(self):
    # Create a dictionary that maps options to indices
    self.all_moves_dict_with_wait = {
        item: index
        for index, item in enumerate(self.every_moves)
    }
    self.all_moves_dict = self.all_moves_dict_with_wait.copy()
    del self.all_moves_dict['Wait']

    # If we are in easy mode, then remove all irrelevant options from the
    # dictionary and redo the indices
    if self.ez:
      # Create an options dictionary (with wait) without the irrelevant options
      all_moves_dict_with_wait_ez = self._get_relevant_moves()
      # Create an option dictionary (without wait) without the irrelevant
      # options
      all_moves_dict_ez = all_moves_dict_with_wait_ez.copy()
      del all_moves_dict_ez['Wait']
      # Set the created dictionaries as fields
      self.all_moves_dict = all_moves_dict_ez
      print('all moves: ', self.all_moves_dict)
      self.all_moves_dict_with_wait = all_moves_dict_with_wait_ez
      print('all moves with wait: ', self.all_moves_dict_with_wait)

  def check_done(self, reward):
    '''
    The agent is only done after completing the soup.
    '''
    return reward == self.completion_reward

  def modify_state(self, state):
    '''
    Modify the state so that the 'current_holdings' key directly maps to the
    agent's holdings.
    '''
    holdings = state['current_holdings']['agent-1']
    state['current_holdings'] = holdings
    return state

  def modify_task_reward(self, reward):
    '''
    Modify the task reward so that completing a full soup leads to a reward
    and the agent incurs a small negative penalty every step if the soup is
    not completed.
    '''
    if self.dish_type == 'Alice':
      soup_val = SoupType.alice.value
    elif self.dish_type == 'Bob':
      soup_val = SoupType.bob.value
    elif self.dish_type == 'Cathy':
      soup_val = SoupType.cathy.value
    elif self.dish_type == 'David':
      soup_val = SoupType.david.value
    else:
      raise NotImplementedError

    ingre_chopped = self.state['ingre_chopped']
    just_chopped = self.state['just_chopped']
    ingre_combined = self.state['ingre_combined']
    just_combined = self.state['just_combined']
    soup_plated = self.state['soup_plated']
    just_plated = self.state['just_plated']
    if just_chopped == Ingredients.tomato.value and ingre_chopped[
        Ingredients.tomato.value - 1] == 1:
      reward = self.step_penalty
    elif just_chopped == Ingredients.onion.value and ingre_chopped[
        Ingredients.onion.value - 1] == 1:
      reward = self.step_penalty
    elif just_chopped == Ingredients.lettuce.value and ingre_chopped[
        Ingredients.lettuce.value - 1] == 1:
      reward = self.step_penalty
    elif just_combined == soup_val and ingre_combined[soup_val - 1] == 1:
      reward = self.step_penalty
    elif just_plated == soup_val and soup_plated[soup_val - 1] == 1:
      if self.serve:
        reward = self.step_penalty
      else:
        reward = self.completion_reward
    elif reward > 0:
      print('Serving event!!!')
      reward = self.completion_reward
    else:
      reward = self.step_penalty

    return reward

  def get_high_level_pref(self, state, prev_option, option):
    '''
    Get the high-level preference reward.
    '''
    assert self.dish_type == 'Cathy' or self.dish_type == 'David'

    ingre_chopped = self.state['ingre_chopped']
    just_chopped = self.state['just_chopped']
    ingre_combined = self.state['ingre_combined']
    just_combined = self.state['just_combined']
    soup_plated = self.state['soup_plated']
    just_plated = self.state['just_plated']

    reward = 0
    if self.dish_type == 'Cathy':
      if just_chopped == Ingredients.onion.value and ingre_chopped[
          Ingredients.onion.value - 1] == 1:
        if self.prev_option == self.all_moves_dict[
            'Chop Onion'] and option == self.all_moves_dict[
                'Chop Tomato'] and ingre_chopped[Ingredients.tomato.value -
                                                 1] == 0:
          reward = self.hl_pref_reward

      if self.detailed_hl_pref:
        if just_chopped == Ingredients.tomato.value and ingre_chopped[
            Ingredients.tomato.value - 1] == 1:
          if self.prev_option == self.all_moves_dict[
              'Chop Tomato'] and option == self.all_moves_dict[
                  'Prepare Cathy Ingredients'] and ingre_combined[
                      SoupType.cathy.value - 1] == 0:
            reward = self.hl_pref_reward
        elif just_combined == SoupType.cathy.value and ingre_combined[
            SoupType.cathy.value - 1] == 1:
          if self.prev_option == self.all_moves_dict[
              'Prepare Cathy Ingredients'] and option == self.all_moves_dict[
                  'Plate Cathy Salad'] and soup_plated[SoupType.cathy.value -
                                                       1] == 0:
            reward = self.hl_pref_reward
        elif just_plated == SoupType.cathy.value and soup_plated[
            SoupType.cathy.value - 1] == 1:
          if self.prev_option == self.all_moves_dict[
              'Plate Cathy Salad'] and option == self.all_moves_dict[
                  'Serve Cathy Salad']:
            reward = self.hl_pref_reward

    if self.dish_type == 'David':
      if not self.detailed_hl_pref:
        if just_chopped == Ingredients.tomato.value and ingre_chopped[
            Ingredients.tomato.value - 1] == 1:
          if self.prev_option == self.all_moves_dict[
              'Chop Tomato'] and option == self.all_moves_dict[
                  'Chop Onion'] and ingre_chopped[Ingredients.onion.value -
                                                  1] == 0:
            return self.hl_pref_reward
        if just_chopped == Ingredients.onion.value and ingre_chopped[
            Ingredients.onion.value - 1] == 1:
          if self.prev_option == self.all_moves_dict[
              'Chop Onion'] and option == self.all_moves_dict[
                  'Chop Lettuce'] and ingre_chopped[Ingredients.lettuce.value -
                                                    1] == 0:
            return self.hl_pref_reward

      if self.detailed_hl_pref:
        if (just_chopped == Ingredients.tomato.value
            and self.prev_option == self.all_moves_dict['Chop Tomato']):
          if option == self.all_moves_dict['Chop Onion'] and ingre_chopped[
              Ingredients.onion.value - 1] == 0:
            return self.hl_pref_reward
          if option == self.all_moves_dict['Chop Lettuce'] and ingre_chopped[
              Ingredients.lettuce.value - 1] == 0:
            return self.hl_pref_reward
        if (just_chopped == Ingredients.onion.value
            and self.prev_option == self.all_moves_dict['Chop Onion']):
          if option == self.all_moves_dict['Chop Lettuce'] and ingre_chopped[
              Ingredients.lettuce.value - 1] == 0:
            return self.hl_pref_reward
          if option == self.all_moves_dict['Chop Tomato'] and ingre_chopped[
              Ingredients.tomato.value - 1] == 0:
            return self.hl_pref_reward
        if (just_chopped == Ingredients.lettuce.value
            and self.prev_option == self.all_moves_dict['Chop Lettuce']):
          if option == self.all_moves_dict['Chop Tomato'] and ingre_chopped[
              Ingredients.tomato.value - 1] == 0:
            return self.hl_pref_reward
          if option == self.all_moves_dict['Chop Onion'] and ingre_chopped[
              Ingredients.onion.value - 1] == 0:
            return self.hl_pref_reward
        if (just_chopped == Ingredients.tomato.value
            or just_chopped == Ingredients.onion.value
            or just_chopped == Ingredients.lettuce.value) and (
                ingre_chopped[Ingredients.tomato.value - 1] >= 1
                and ingre_chopped[Ingredients.lettuce.value - 1] >= 1
                and ingre_chopped[Ingredients.onion.value - 1] >= 1
            ) and ingre_combined[SoupType.david.value -
                                 1] == 0 and option == self.all_moves_dict[
                                     'Prepare David Ingredients']:
          return self.hl_pref_reward
        if just_combined == SoupType.david.value and ingre_combined[
            SoupType.david.value - 1] == 1:
          if self.prev_option == self.all_moves_dict[
              'Prepare David Ingredients'] and option == self.all_moves_dict[
                  'Plate David Salad'] and soup_plated[SoupType.david.value -
                                                       1] == 0:
            return self.hl_pref_reward
        if just_plated == SoupType.david.value and soup_plated[
            SoupType.david.value - 1] == 1:
          if self.prev_option == self.all_moves_dict[
              'Plate David Salad'] and option == self.all_moves_dict[
                  'Serve David Salad']:
            return self.hl_pref_reward

    return reward

  def _get_relevant_moves(self):
    '''
    Create an options dictionary (with wait) without the irrelevant options
    '''
    if self.dish_type == 'Alice':
      irrelevant_names = ['Tomato', 'Bob', 'Cathy', 'David', 'Putout', 'Drop']
    elif self.dish_type == 'Bob':
      irrelevant_names = ['Onion', 'Alice', 'Cathy', 'David', 'Putout', 'Drop']
    elif self.dish_type == 'Cathy':
      irrelevant_names = ['Lettuce', 'Alice', 'Bob', 'David', 'Putout', 'Drop']
    elif self.dish_type == 'David':
      irrelevant_names = ['Alice', 'Bob', 'Cathy', 'Putout', 'Drop']
    else:
      raise NotImplementedError
    if not self.serve:
      irrelevant_names.append('Serve')

    all_moves_dict_with_wait_ez = {}
    num_moves = 0
    for move in self.all_moves_dict_with_wait:
      relevant_move = True
      for name in irrelevant_names:
        if name in move:
          relevant_move = False
          break
      if relevant_move:
        all_moves_dict_with_wait_ez[move] = num_moves
        num_moves += 1
    return all_moves_dict_with_wait_ez

  def get_high_level_pref_gpt(self, state, prev_option, option):
    if not isinstance(state, dict): state = state.state_to_dict()
    reward, reward_dict = get_high_level_pref_gpt(state, prev_option, option)
    return reward


class EurekaOvercookedSimpleHL(KitchenHLGPT):

  def __init__(self,
               arglist,
               ez,
               masked,
               salad,
               serve,
               detailed_hl_pref,
               smdp=False,
               seed=0,
               render=False):
    super().__init__(arglist,
                     salad=salad,
                     serve=serve,
                     detailed_hl_pref=detailed_hl_pref,
                     ez=ez)

    # Set seed
    self.np_random = np.random.RandomState(seed)

    # Define HL action space
    self.action_space = gym.spaces.Discrete(len(self.all_moves_dict))

    # Define HL observation space.
    # Flattened map length:
    map_len = 24 * 5 * 4
    current_orders_len = 14
    current_holdings_len = 13
    # We'll have: map_len, current_orders, current_holdings, just_chopped,
    # ingre_chopped, option.
    self.obs_len = map_len + current_holdings_len + len(Ingredients) + len(
        Ingredients) - 1 + 2 * (len(SoupType) + len(SoupType) - 1) + len(
            self.all_moves_dict_with_wait)
    self.masked = masked
    if self.masked:
      self.obs_len += len(self.all_moves_dict)
    self.observation_space = gym.spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(self.obs_len, ),
                                            dtype=np.float32)

    # Render the env if needed
    self.if_render = render
    self.game = None
    self.sleep_time = 0.1

    # Keep track of the current option and its status
    self.option = None
    self.option_task = None
    self.num_ll_steps = 0
    if map_len / 24 <= 20:
      self.max_steps_ll = 10
    else:
      self.max_steps_ll = 30
    self.last_option_status = -5
    self.option_mask = np.zeros(len(self.all_moves_dict))

    # A list of all valid moves
    self.moves = [(1, 0), (0, 1), (-1, 0), (0, -1), (0, 0)]
    self.rand_move_prob = 0

    self.smdp = smdp

  def reset(self, seed=0):
    """
    Reset the base environment, then build and return the HL observation.
    """
    self.option = None
    self.option_task = None
    self.num_ll_steps = 0
    self.last_option_status = -5
    self.np_random = np.random.RandomState(seed)
    obs_dict, info = super().reset()
    self.option_mask = np.zeros(len(self.all_moves_dict))
    obs_dict['option_mask'] = self.option_mask
    if self.if_render:
      # If we have yet to instantiate an instance of Game, do that.
      # Otherwise, call __init__ to reinitialize.
      # Not the best practice but needs to be done.
      if self.game is None:
        self.game = Game(self, play=True)
      else:
        self.game.__init__(self, play=True)
      self.game.on_init()
      self.game.on_render()
    hl_obs = self._build_hl_observation(obs_dict)
    return hl_obs, info

  def _build_hl_observation_cnn(self, obs_dict):
    return obs_dict['map'].astype(np.float32)

  def _build_hl_observation(self, obs_dict):
    """
    Convert the dictionary obs:
      obs_dict['map']: ,
      obs_dict['current_orders']: ,
      obs_dict['current_holdings']: ,
      obs_dict['just_chopped']: (int),
      obs_dict['ingre_chopped']: (3 x 1)
    plus option
    into a single 1D float array.
    """
    # Flatten game map
    game_map = obs_dict['map'].flatten().astype(np.float32)

    # Cast current_oders, current_holdings, ingre_chopped, ingre_combined, and
    # soup_plated to float32
    # print(obs_dict['current_orders'])
    # current_orders = obs_dict['current_orders'].astype(np.float32)
    current_holdings = obs_dict['current_holdings'].astype(np.float32)
    ingre_chopped = obs_dict['ingre_chopped'].astype(np.float32)
    ingre_combined = obs_dict['ingre_combined'].astype(np.float32)
    soup_plated = obs_dict['soup_plated'].astype(np.float32)

    # One-hot encode just_chopped
    just_chopped = np.zeros(len(Ingredients), dtype=np.float32)
    just_chopped[obs_dict['just_chopped']] = 1.0

    # One-hot encode just_combined
    just_combined = np.zeros(len(SoupType), dtype=np.float32)
    just_combined[obs_dict['just_combined']] = 1.0

    # One-hot encode just_plated
    just_plated = np.zeros(len(SoupType), dtype=np.float32)
    just_plated[obs_dict['just_plated']] = 1.0

    # One-hot encode option
    prev_option = np.zeros(len(self.all_moves_dict_with_wait), dtype=np.float32)
    prev_option[self.prev_option] = 1.0

    # Concatenate all parts
    obs = np.concatenate([
        game_map, current_holdings, just_chopped, ingre_chopped, just_combined,
        ingre_combined, just_plated, soup_plated, prev_option
    ],
                         axis=0)
    if self.masked:
      if np.all(obs_dict['option_mask'] == 0):
        mask = np.ones_like(obs_dict['option_mask'])
      else:
        mask = obs_dict['option_mask']
      obs = np.concatenate([obs, mask.astype(np.float32)], axis=0)
    return obs

  def _build_ll_observation(self):
    '''
    Get the input state for the low level executor.
    '''
    info = self.get_ai_info()
    env_state = EnvState(world=info['world'],
                         agents=info['sim_agents'],
                         agent_idx=0,
                         order=info['order_scheduler'],
                         event_history=info['event_history'],
                         time=info['current_time'],
                         chg_grid=info['chg_grid'])
    return env_state

  def step(self, hl_action):
    """
    1) Initialize LL sub-policy based on the option.
    2) Run the LL sub-policy for one-step.
    3) Return (HL_observation, HL_reward, done, truncated, info).
    """
    if self.masked:
      if np.any(self.option_mask):
        assert hl_action == np.argmax(self.option_mask)

    done = False
    truncated = False
    ll_done = False
    ll_truncated = False
    action_dict = {}
    # Get action from LL policy
    if (self.smdp
        or self.masked) and self.last_option_status == HighTask.Failed:
      # If we model the as an SMDP, and if the previous action failed, then
      # choose a random action
      action_dict[self.sim_agents[0].name] = self.get_rand_action(
          self.rand_move_prob)
    else:
      # Otherwise, initialize low-level policy based on the option if need:
      # If we are modeling the task as an MDP, then update the option and
      # execute it every time step.
      # Otherwise, only update the option after the execution of the option
      # finishes.
      if (not self.smdp and not self.masked) or self.option_task is None:
        self.option = hl_action
        ht = [
            name for name, val in self.all_moves_dict.items()
            if val == hl_action
        ][0]
        task = deepcopy(self.moves_to_ht[ht])
        self.option_task = task
      # print('Current option: ', [
      #     name for name, val in self.all_moves_dict.items()
      #     if val == self.option
      # ][0])

      # Get the action with the low-level policy
      env_state = self._build_ll_observation()
      move_status, move, _msg = self.option_task(env_state)
      self.last_option_status = move_status
      # print('move status: ', move_status)
      # print('move: ', move)
      # print('msg: ', _msg)
      # Prepare action dict
      if move_status == HighTask.Working:
        action_dict[self.sim_agents[0].name] = move
      else:
        action_dict[self.sim_agents[0].name] = self.get_rand_action(
            self.rand_move_prob)
    action_dict = {
        k: v if v is not None else (0, 0)
        for k, v in action_dict.items()
    }
    # Step in the environment
    reward = (0, 0)
    obs, reward, done, truncated, info = self.step_helper(
        action_dict, hl_action, reward)
    # Check if LL is done (LL can be done if either its status is success or
    # it has reached the step limit) to prepare the info dict
    # info = {}
    if self.masked or self.smdp:
      # We don't need to check if LL is done if we model the task as an MDP
      # without explicit termination conditions
      self.num_ll_steps += 1
      if self.num_ll_steps == self.max_steps_ll:
        self.option_task = None
        self.num_ll_steps = 0
        ll_truncated = True
        self.last_option_status = -5
      if not ll_truncated and self.last_option_status != HighTask.Failed:
        next_env_state = self._build_ll_observation()
        task_copy = deepcopy(self.option_task)
        next_move_status, _next_move, _msg = task_copy(next_env_state)
        if next_move_status == HighTask.Success:
          self.option_task = None
          self.num_ll_steps = 0
          ll_done = True
          self.last_option_status = -5
    info['ll_done'] = ll_done
    info['ll_truncated'] = ll_truncated
    # print('ll done: ', ll_done)
    # print('ll truncated: ', ll_truncated)
    # Build HL observation
    self.option_mask = np.zeros(len(self.all_moves_dict))
    if not ll_done and not ll_truncated:
      self.option_mask[self.prev_option] = 1
    obs['option_mask'] = self.option_mask
    # print('option mask: ', self.option_mask)
    hl_next_obs = self._build_hl_observation(obs)
    return hl_next_obs, reward, done, truncated, info

  def step_helper(self, action_dict, hl_action, total_reward):
    # Check pygame
    if self.if_render:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          done = True
          truncated = True
          break
    # Step the base environment
    obs, reward, done, truncated, info = super().step(action_dict, hl_action)
    # Accumulate reward from each LL step
    total_reward = tuple(
        [acc + sub_r for acc, sub_r in zip(total_reward, reward)])
    # Update rendering
    if self.if_render:
      # Render the updated state in the window
      self.game.on_render()
      # A small pause to see what's going on
      time.sleep(self.sleep_time)
    return obs, total_reward, done, truncated, info

  def get_rand_action(self, rand_prob):
    if self.np_random.random() < rand_prob:
      rand_move_idx = self.np_random.choice(list(range(len(self.moves))))
    else:
      rand_move_idx = 4
    self.rand_move = self.moves[rand_move_idx]
    return self.rand_move

  def close(self):
    super().close()
    if self.if_render:
      self.game.on_cleanup()

from typing import Dict, Tuple
import math
def get_high_level_pref_gpt(state: Dict, prev_option: int, option: int) -> Tuple[float, Dict[str, float]]:
    '''
    state: the current state of the environment.
    prev_option: the last option (subtask) executed by the agent to reach the current state.
    option: the option (subtask) the agent is about to perform in the current state.
    '''
    
    reward = 0.0
    reward_components = {
        "tomato_to_onion": 0.0,
        "onion_to_lettuce": 0.0
    }

    # Define the numerical identifiers for the options to allow comparisons
    option_chop_tomato = 0  # 'Chop Tomato'
    option_chop_onion = 2   # 'Chop Onion'
    option_chop_lettuce = 1 # 'Chop Lettuce'

    # Access the number of each ingredient chopped from the state
    num_tomatoes_chopped = state['ingre_chopped'][Ingredients.tomato.value - 1]
    num_onions_chopped = state['ingre_chopped'][Ingredients.onion.value - 1]
    num_lettuce_chopped = state['ingre_chopped'][Ingredients.lettuce.value - 1]

    if prev_option == option_chop_tomato and option == option_chop_onion:
        # Check if the agent is moving from chopping tomato to chopping onion
        # Only reward this transition once for each ingredient needed
        if num_onions_chopped == 0:
            reward += 0.5
            reward_components["tomato_to_onion"] = 0.5

    elif prev_option == option_chop_onion and option == option_chop_lettuce:
        # Check if the agent is moving from chopping onion to chopping lettuce
        # Only reward this transition once for each ingredient needed
        if num_lettuce_chopped == 0:
            reward += 0.5
            reward_components["onion_to_lettuce"] = 0.5

    return reward, reward_components
