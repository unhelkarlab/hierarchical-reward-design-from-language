import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import os
import random
from copy import deepcopy
from collections import defaultdict
from typing import Optional
from collections.abc import Iterable

import rw4t.utils as utils
import rw4t.utils as rw4t_utils
from rw4t.map_config import maps, pref_dicts
from rw4t.rw4t_game import RW4T_Game


class RW4T_GameState:

  def __init__(self, obs: np.ndarray, pos: np.ndarray, holding: int,
               last_pickup: int, last_drop: int, option_mask: np.ndarray):
    '''
    :param obs: a 2D numpy of the current environment
    :param pos: a 1D numpy array of the agent's (x, y) position in the
                environment
    :param holding: an integer indicating what object the agent is holding
                    if any
    :param last_pickup: an integer indicating what object the agent just picked
                    up if any
    :param last_drop: an integer indicating what object the agent just dropped
                    if any
    :param optino_mask: a 1D array indicating the valid options to select next
                        (should not be used when computing rewards, this is only
                        used in some downstream algorithms)
    '''
    # Agent's x pos in bound
    assert pos[1] >= 0 and pos[1] < len(obs)
    # Agent's y pos in bound
    assert pos[0] >= 0 and pos[0] < len(obs[0])
    # holding, last_pickup, and last_drop should be a value in the Holding_Obj
    # Enum
    assert holding < len(rw4t_utils.Holding_Obj)
    assert last_pickup < len(rw4t_utils.Holding_Obj)
    assert last_drop < len(rw4t_utils.Holding_Obj)
    self.obs = obs
    self.pos = pos
    self.holding = holding
    self.last_pickup = last_pickup
    self.last_drop = last_drop
    self.option_mask = option_mask

  def state_to_dict(self):
    return {
        'map': np.array(self.obs, dtype=np.int32),
        'pos': np.array(self.pos, dtype=np.int32),
        'holding': self.holding,
        'last_pickup': self.last_pickup,
        'last_drop': self.last_drop,
        'option_mask': self.option_mask
    }


class RescueWorldFlatSAGPT(gym.Env):

  def __init__(self,
               map_name,
               low_level,
               hl_pref_r,
               pbrs_r,
               rw4t_game_params=dict(),
               init_pos=None,
               ez=True,
               pref_dict_name='',
               pref_dict=None,
               seed=None,
               option=None,
               action_duration=0,
               write=False,
               fname='',
               render=False):
    super(RescueWorldFlatSAGPT, self).__init__()
    if seed is not None:
      random.seed(seed)

    # Whether we are working with the low-level only
    self.low_level = low_level
    # Current option
    self.option = option

    # Define action and observation space
    self.init_map = maps[map_name]
    self.map_size = len(self.init_map)
    self.ll_action_space = spaces.Discrete(len(utils.RW4T_LL_Actions) - 1)
    if ez:
      self.rw4t_hl_actions = utils.RW4T_HL_Actions_EZ
      self.rw4t_hl_actions_with_dummy = utils.RW4T_HL_Actions_With_Dummy_EZ
    else:
      self.rw4t_hl_actions = utils.RW4T_HL_Actions
      self.rw4t_hl_actions_with_dummy = utils.RW4T_HL_Actions_With_Dummy
    self.hl_action_space = spaces.Discrete(len(self.rw4t_hl_actions))
    if self.low_level:
      self.action_space = self.ll_action_space
    else:
      self.action_space = self.hl_action_space
    self.map_space = spaces.Box(low=0,
                                high=len(utils.RW4T_State.__members__),
                                shape=(self.map_size, self.map_size),
                                dtype=np.int32)
    self.pos_space = spaces.Box(low=np.array([0, 0]),
                                high=np.array(
                                    [self.map_size - 1, self.map_size - 1]),
                                dtype=np.int32)
    self.holding_space = spaces.Discrete(len(rw4t_utils.Holding_Obj))
    self.last_pickup_space = spaces.Discrete(len(rw4t_utils.Holding_Obj))
    self.last_drop_space = spaces.Discrete(len(rw4t_utils.Holding_Obj))
    self.option_mask_space = spaces.MultiBinary(len(self.rw4t_hl_actions))
    self.observation_space = spaces.Dict({
        "map": self.map_space,
        "pos": self.pos_space,
        "holding": self.holding_space,
        "last_pickup": self.last_pickup_space,
        "last_drop": self.last_drop_space,
        "option_mask": self.option_mask_space
    })

    # Store preference info
    # pref_dict should consist of the following dictionaries:
    # 1. a dictionary whose key is 'objects' that maps integers to mappings of
    #    integers to integers. The first integer indicates the object type,
    #    the second integer indicates the landmark type, and the third integer
    #    indicates the upper bound on the number of objects to be dropped at the
    #    landmark.
    # 2. a list whose key is 'zones' that contains the type of zones the agent
    #    should avoid (indicated by integers)
    # 3. a number whose key is 'total_num' that is the total number of objects
    #    to drop
    if pref_dict_name == '' and pref_dict is None:
      split_map_name = map_name.rsplit('_', 1)[0]
      inferred_pref_dict_name = f'{split_map_name}_pref_dict'
      self.pref_dict = pref_dicts[inferred_pref_dict_name]
    elif pref_dict is not None:
      self.pref_dict = pref_dict
    else:
      self.pref_dict = pref_dicts[pref_dict_name]
    self.object_pref = self.pref_dict['objects']
    self.danger_pref = self.pref_dict['zones']
    self.total_num_to_drop = self.pref_dict['total_num']

    # Set position, holding, and state info
    self.init_pos = init_pos
    if 'valid_start_pos' in self.pref_dict and self.pref_dict[
        'valid_start_pos'] != []:
      self.valid_start_pos = self.pref_dict['valid_start_pos']
    else:
      self.valid_start_pos = None
    self.map = deepcopy(self.init_map)
    if not self.low_level:
      self.reset()
    else:
      self.option = self.get_valid_option()
      self.reset(options={'option': self.option})
    self.old_state = deepcopy(self.state)

    # Inits for action delay (currently should not be used with any delay)
    self.possible_durations = [0]
    self.action_duration = action_duration
    assert self.action_duration in self.possible_durations
    self.current_action = None
    self.action_start_time = None

    # Init rewards related info
    self.diversity_reward = 10
    self.pick_pseudo_reward = 10
    self.drop_pseudo_reward = 20
    self.pseudo_penalty = -3
    self.correct_obj_lm_reward = 30
    self.incorrect_obj_lm_reward = 0
    self.dropped_objs = defaultdict(lambda: defaultdict(int))

    self.danger_reward_per_second = -50
    self.time_in_danger_zone = 0
    self.danger_reward_per_entrance = -5
    self.prev_danger_zone = (-1, -1)
    self.reward_persistence = True

    # Other init
    if self.map_size == 10:
      self.max_timesteps = 200
    else:
      self.max_timesteps = 100
    self.max_timesteps_ll = 30
    self.max_time_in_seconds = 300
    self.write = write
    self.fname = fname
    self.first_write = True
    self.prev_option = self.rw4t_hl_actions_with_dummy.dummy.value
    self.c_task_reward = 0
    self.c_pseudo_reward = 0
    self.c_gt_hl_pref = 0
    self.c_gt_ll_pref = 0
    self.hl_pref_r = hl_pref_r
    self.pbrs_r = pbrs_r
    self.pbrs_factor = 0.5

    # State related inits
    self.last_pickup = rw4t_utils.RW4T_State.empty.value
    self.last_drop = rw4t_utils.RW4T_State.empty.value
    self.option_mask = np.zeros(len(self.rw4t_hl_actions))

    self.if_render = render
    if render:
      rw4t_game_params['env'] = self
      rw4t_game_params['play'] = True
      self.game = RW4T_Game(**rw4t_game_params)
      self.game.low_level = False  # whether to display the current "option"
      self.game.on_init()
      self.game.on_render()

  def step(self, ll_action, hl_action, passed_time=0):
    self.timestep += 1
    self.timestep_ll += 1
    self.timer += passed_time
    self.option = hl_action
    # Check if agent is in the danger zone
    new_in_danger_zone = False
    if self.check_danger():
      self.time_in_danger_zone += passed_time
      if self.prev_danger_zone != self.agent_pos:
        new_in_danger_zone = True
        self.prev_danger_zone = self.agent_pos
    else:
      self.prev_danger_zone = (-1, -1)

    pick_info = utils.RW4T_State.empty.value, ''
    drop_info = defaultdict(int)
    executed_action = None
    self.old_state = deepcopy(self.state)
    old_features = self.get_current_features()
    current_time = time.perf_counter()
    # print('ll action: ', ll_action)
    if ll_action != utils.RW4T_LL_Actions.idle.value and (
        self.current_action is None or ll_action != self.current_action):
      # If we don't have a current action (i.e. because we have completed the
      # action) or if the incoming action is not the same as the action,
      # set current action to this new action
      self.current_action = ll_action
      self.action_start_time = current_time
    if (self.current_action is not None and self.action_start_time
        is not None) and (current_time - self.action_start_time
                          >= self.action_duration):
      # If we've finished waiting for an action, then execute the action
      if self.current_action == utils.RW4T_LL_Actions.go_up.value:
        self.agent_pos = self.get_new_pos(utils.RW4T_LL_Actions.go_up.value)
      elif self.current_action == utils.RW4T_LL_Actions.go_down.value:
        self.agent_pos = self.get_new_pos(utils.RW4T_LL_Actions.go_down.value)
      elif self.current_action == utils.RW4T_LL_Actions.go_left.value:
        self.agent_pos = self.get_new_pos(utils.RW4T_LL_Actions.go_left.value)
      elif self.current_action == utils.RW4T_LL_Actions.go_right.value:
        self.agent_pos = self.get_new_pos(utils.RW4T_LL_Actions.go_right.value)
      elif self.current_action == utils.RW4T_LL_Actions.pick.value:
        pick_info = self.perform_pick()
      elif self.current_action == utils.RW4T_LL_Actions.drop.value:
        drop_info = self.perform_drop()
      else:
        raise NotImplementedError
      executed_action = self.current_action
      self.current_action = None

    # Prepare the values to be returned
    # State
    self.last_pickup = pick_info[0]
    if len(drop_info) == 0:
      self.last_drop = rw4t_utils.RW4T_State.empty.value
    else:
      assert len(drop_info) == 1
      for obj in drop_info:
        self.last_drop = obj
      for obj, loc in drop_info.items():
        # If the agent dropped off an object at a landmark
        if (loc == rw4t_utils.RW4T_State.school.value
            or loc == rw4t_utils.RW4T_State.hospital.value
            or loc == rw4t_utils.RW4T_State.park.value):
          self.all_drops.append((obj, loc))
    # Reward
    # reward = self.calculate_rewards(drop_info, new_in_danger_zone)
    task_reward = self.get_task_reward(ll_action, drop_info)
    pseudo_reward = self.get_pseudo_reward(hl_action, ll_action, pick_info,
                                           drop_info)
    ll_pref_reward = 0
    gt_ll_pref_reward = 0
    if 'LLGPT' in self.__class__.__name__:
      ll_pref_reward += self.get_low_level_pref_gpt(self.old_state, hl_action,
                                                    ll_action)
      gt_ll_pref_reward += self.get_low_level_pref(self.old_state, hl_action,
                                                   ll_action)
    elif 'FlatSAGPT' in self.__class__.__name__:
      ll_pref_reward += self.get_flat_sa_pref_gpt(self.old_state, ll_action)
      gt_ll_pref_reward += self.get_low_level_pref(self.old_state, hl_action,
                                                   ll_action)
    else:
      if self.hl_pref_r:
        ll_pref_reward += self.get_low_level_pref(self.old_state, hl_action,
                                                  ll_action)
        gt_ll_pref_reward += ll_pref_reward
      else:
        ll_pref_reward += self.get_flatsa_pref(self.old_state, ll_action)
        gt_ll_pref_reward += self.get_low_level_pref(self.old_state, hl_action,
                                                     ll_action)
    hl_pref_reward = 0
    gt_hl_pref_reward = 0
    pbrs_reward = 0
    # if self.hl_pref_r:
    if 'HLGPT' in self.__class__.__name__:
      hl_pref_reward += self.get_high_level_pref_gpt(self.old_state,
                                                     self.prev_option,
                                                     hl_action)
      gt_hl_pref_reward += self.get_high_level_pref(self.old_state,
                                                    self.prev_option, hl_action)
    elif 'FlatSAGPT' in self.__class__.__name__:
      hl_pref_reward += self.get_flat_sa_pref_gpt(self.old_state, ll_action)
      gt_hl_pref_reward += self.get_high_level_pref(self.old_state,
                                                    self.prev_option, hl_action)
    else:
      if self.hl_pref_r:
        hl_pref_reward += self.get_high_level_pref(self.old_state,
                                                   self.prev_option, hl_action)
        gt_hl_pref_reward += hl_pref_reward
      else:
        hl_pref_reward += self.get_flatsa_pref(self.old_state, ll_action)
        gt_hl_pref_reward += self.get_high_level_pref(self.old_state,
                                                      self.prev_option,
                                                      hl_action)
    pbrs_reward += self.get_high_level_pref_pbrs(hl_action)
    r = (task_reward, pseudo_reward, ll_pref_reward, hl_pref_reward,
         pbrs_reward)
    # print('task reward: ', task_reward)
    # print('pseudo reward: ', pseudo_reward)
    # print('ll pref: ', ll_pref_reward)
    # print('hl pref: ', hl_pref_reward)
    # print('pbrs pref: ', pbrs_reward)
    # Done and info
    info = {}
    self.option_mask = np.zeros(len(self.rw4t_hl_actions))
    if not self.low_level:
      done, truncated = self.check_done()
      ll_done, ll_truncated = self.check_done_ll(pseudo_reward)
      info['ll_done'] = ll_done
      info['ll_truncated'] = ll_truncated
      if ll_done or ll_truncated:
        self.timestep_ll = 0
      else:
        self.option_mask[self.option] = 1
    else:
      done, truncated = self.check_done_ll(pseudo_reward)
      if not done and not truncated:
        self.option_mask[self.option] = 1
    info['all_drops'] = self.all_drops
    # Write transitions
    if self.write and (done or executed_action is not None or self.first_write):
      self.first_write = False
      # Log a transition if the game is done or if the agent takes an action
      # As there can only be a reward when the action finishes an action,
      # we can skip the intermediate frames if there are any
      if 0 <= hl_action < len(self.rw4t_hl_actions):
        macro_action = utils.get_enum_name_by_value(self.rw4t_hl_actions,
                                                    hl_action)
      else:
        macro_action = 'None'
      macro_action_idx = hl_action
      new_features = self.get_current_features()
      transition = utils.RW4T_Transition(self.old_state, old_features,
                                         executed_action, macro_action,
                                         macro_action_idx, r, self.state,
                                         new_features, done, info)
      # print('Transition: ', transition)
      self.write_transition(transition)
    self.done = done
    self.c_task_reward += task_reward
    self.c_pseudo_reward += pseudo_reward
    self.c_gt_hl_pref += gt_hl_pref_reward
    self.c_gt_ll_pref += gt_ll_pref_reward
    info['c_task_reward'] = self.c_task_reward
    info['c_pseudo_reward'] = self.c_pseudo_reward
    info['c_gt_hl_pref'] = self.c_gt_hl_pref
    info['c_gt_ll_pref'] = self.c_gt_ll_pref
    self.prev_option = hl_action
    # print('info: ', info)
    self.state = RW4T_GameState(self.map, self.agent_pos, self.agent_holding,
                                self.last_pickup, self.last_drop,
                                self.option_mask)
    if self.if_render:
      # Render the updated state in your RW4T_Game window
      self.game.on_render()
      # A small pause to see what's going on
      time.sleep(0.5)
    return self.state.state_to_dict(), r, done, truncated, info

  def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
    '''
    Reset the environment.
    '''
    if seed is not None:
      random.seed(seed)
    self.done = False
    self.timer = 0
    self.timestep = 0
    self.timestep_ll = 0
    self.dropped_objs = defaultdict(lambda: defaultdict(int))
    self.c_task_reward = 0
    self.c_pseudo_reward = 0
    self.c_gt_hl_pref = 0
    self.c_gt_ll_pref = 0
    self.prev_option = self.rw4t_hl_actions_with_dummy.dummy.value
    self.map = deepcopy(self.init_map)
    self.all_drops = []
    if self.low_level:
      if (options is not None and 'option' in options
          and self.check_valid_option(options['option'])):
        self.option = options['option']
      else:
        self.option = self.get_valid_option()
      return self.ll_reset(self.option, seed)
    self.last_pickup = rw4t_utils.RW4T_State.empty.value
    self.last_drop = rw4t_utils.RW4T_State.empty.value
    self.option_mask = np.zeros(len(self.rw4t_hl_actions))
    if self.init_pos is None:
      self.agent_pos = self.choose_rand_start_pos()
    else:
      self.agent_pos = self.init_pos
    self.agent_holding = 0
    self.state = RW4T_GameState(self.map, self.agent_pos, self.agent_holding,
                                self.last_pickup, self.last_drop,
                                self.option_mask)
    return self.state.state_to_dict(), {}

  def ll_reset(
      self,
      option,
      seed: Optional[int] = None,
  ):
    '''
    Reset if we are only working with the low-level.
    '''
    obj_remove_counts = self.reset_map_ll(option)
    self.option_mask = np.zeros(len(self.rw4t_hl_actions))
    self.option_mask[option] = 1
    if self.check_is_pick_option(option):
      if sum(obj_remove_counts.values()) == 0:
        self.agent_pos = self.choose_rand_start_pos()
      else:
        random_drop_loc = random.choice(self.get_all_drop_locations())
        self.agent_pos = random_drop_loc
      # If the option is a pick, randomly choose a start position for the agent
      # if np.random.rand() < 0.2:
      #   random_drop_loc = random.choice(self.get_all_drop_locations())
      #   self.agent_pos = random_drop_loc
      # else:
      #   self.agent_pos = self.choose_rand_start_pos()
      # If the agent is at a drop location, randomly choose a viable last_drop
      # object.
      if self.agent_pos in self.get_all_drop_locations():
        objs_removed = [
            obj for obj, count in obj_remove_counts.items() if count > 0
        ]
        if len(objs_removed) > 0:
          self.last_drop = random.choice(objs_removed)
        else:
          self.last_drop = rw4t_utils.RW4T_State.empty.value
      else:
        self.last_drop = rw4t_utils.RW4T_State.empty.value
      self.last_pickup = rw4t_utils.RW4T_State.empty.value
      self.agent_holding = 0
      self.state = RW4T_GameState(self.map, self.agent_pos, self.agent_holding,
                                  self.last_pickup, self.last_drop,
                                  self.option_mask)
      return self.state.state_to_dict(), {}
    else:
      # If the option is a drop, move the agent to one of the object locations
      obj_type = rw4t_utils.HL_Action_2_Obj[option]
      obj_locs = self.get_all_obj_locations_with_types()[obj_type]
      chosen_obj_loc = random.choice(list(obj_locs))
      self.map[chosen_obj_loc[1], chosen_obj_loc[0]] = 0
      self.agent_pos = (chosen_obj_loc[0], chosen_obj_loc[1])
      self.agent_holding = obj_type
      self.last_pickup = obj_type
      self.last_drop = rw4t_utils.RW4T_State.empty.value
      self.state = RW4T_GameState(self.map, self.agent_pos, self.agent_holding,
                                  self.last_pickup, self.last_drop,
                                  self.option_mask)
      return self.state.state_to_dict(), {}

  def reset_map_ll(self, option):
    '''
    Reset the map if we are only working with the low-level.

    Returns:
    a dictionary that maps the object type to the number of that object being
    removed from the initial map.
    '''
    # Initialize the object counts with at least one for type 'option'
    circle_val = rw4t_utils.RW4T_State.circle.value
    square_val = rw4t_utils.RW4T_State.square.value
    triangle_val = rw4t_utils.RW4T_State.triangle.value
    counts = {circle_val: 0, square_val: 0, triangle_val: 0}
    counts[rw4t_utils.HL_Action_2_Obj[option]] = 1

    all_objs = self.get_all_obj_locations_with_types()
    # Randomize the number of objects for each type, while ensuring we don't
    # exceed max limit
    counts[circle_val] += random.randint(
        0,
        len(all_objs[circle_val]) - counts[circle_val])
    counts[square_val] += random.randint(
        0,
        len(all_objs[square_val]) - counts[square_val])
    counts[triangle_val] += random.randint(
        0,
        len(all_objs[triangle_val]) - counts[triangle_val])

    # Construct the a dict of the number of objects to remove from the map for
    # each object type
    obj_remove_counts = {
        key: len(all_objs[key]) - counts[key]
        for key in counts
    }

    # Remove objects from the map
    for obj in obj_remove_counts:
      remove_locations = random.sample(all_objs[obj], obj_remove_counts[obj])
      for loc in remove_locations:
        self.map[loc[1], loc[0]] = 0

    # Sanity test that the number of objects on the map matches the count dict
    new_all_objs = self.get_all_obj_locations_with_types()
    new_counts = {obj: len(new_all_objs[obj]) for obj in new_all_objs}
    assert new_counts == counts

    return obj_remove_counts

  def close(self):
    super().close()
    if self.if_render:
      self.game.on_cleanup()

  def get_new_pos(self, action):
    '''
    Get the new position of the agent after executing the given action.
    '''
    action_coord = utils.Action_2_Coord[action]
    new_pos_maybe = (self.agent_pos[0] + action_coord[0],
                     self.agent_pos[1] + action_coord[1])
    if self.check_in_bounds(
        new_pos_maybe) and new_pos_maybe not in self.get_obs_locations():
      return new_pos_maybe
    else:
      return self.agent_pos

  def check_in_bounds(self, coord):
    '''
    Check if the input coordinate is within bounds.
    '''
    if coord[0] >= 0 and coord[0] < self.map_size:
      if coord[1] >= 0 and coord[1] < self.map_size:
        return True
    return False

  def perform_pick(self):
    '''
    Execute the pick action, update the game map and agent's holding.
    '''
    picked_obj = utils.RW4T_State.empty.value, ''
    if self.agent_holding != utils.RW4T_State.empty.value:
      # The agent cannot pick up another object when holding something
      return picked_obj

    if self.map[self.agent_pos[1],
                self.agent_pos[0]] == utils.RW4T_State.circle.value:
      picked_obj = utils.RW4T_State.circle.value, 'circle'
      self.map[self.agent_pos[1], self.agent_pos[0]] = 0
      self.agent_holding = utils.RW4T_State.circle.value
    elif self.map[self.agent_pos[1],
                  self.agent_pos[0]] == utils.RW4T_State.square.value:
      picked_obj = utils.RW4T_State.square.value, 'square'
      self.map[self.agent_pos[1], self.agent_pos[0]] = 0
      self.agent_holding = utils.RW4T_State.square.value
    elif self.map[self.agent_pos[1],
                  self.agent_pos[0]] == utils.RW4T_State.triangle.value:
      picked_obj = utils.RW4T_State.triangle.value, 'triangle'
      self.map[self.agent_pos[1], self.agent_pos[0]] = 0
      self.agent_holding = utils.RW4T_State.triangle.value
    return picked_obj

  def perform_drop(self):
    '''
    Execute the drop action, update the game map and agent's holding.
    '''
    drop_info = defaultdict(int)
    if self.agent_holding == utils.RW4T_State.empty.value:
      # The agent cannot drop an object when it is not holding anything
      return drop_info

    if self.map[self.agent_pos[1],
                self.agent_pos[0]] == utils.RW4T_State.empty.value:
      self.map[self.agent_pos[1], self.agent_pos[0]] = self.agent_holding
      drop_info[self.agent_holding] = utils.RW4T_State.empty.value
      self.agent_holding = utils.RW4T_State.empty.value

    if self.map[self.agent_pos[1],
                self.agent_pos[0]] == utils.RW4T_State.school.value:
      self.dropped_objs[self.agent_holding][utils.RW4T_State.school.value] += 1
      drop_info[self.agent_holding] = utils.RW4T_State.school.value
      self.agent_holding = utils.RW4T_State.empty.value
    elif self.map[self.agent_pos[1],
                  self.agent_pos[0]] == utils.RW4T_State.hospital.value:
      self.dropped_objs[self.agent_holding][
          utils.RW4T_State.hospital.value] += 1
      drop_info[self.agent_holding] = utils.RW4T_State.hospital.value
      self.agent_holding = utils.RW4T_State.empty.value
    elif self.map[self.agent_pos[1],
                  self.agent_pos[0]] == utils.RW4T_State.park.value:
      self.dropped_objs[self.agent_holding][utils.RW4T_State.park.value] += 1
      drop_info[self.agent_holding] = utils.RW4T_State.park.value
      self.agent_holding = utils.RW4T_State.empty.value
    return drop_info

  def check_danger(self):
    '''
    Check if the agent is in one of the danger zones that should be avoided.
    '''
    if (self.map[self.agent_pos[1],
                 self.agent_pos[0]] == utils.RW4T_State.yellow_zone.value
        ) and utils.RW4T_State.yellow_zone.value in self.danger_pref:
      return True
    if (self.map[self.agent_pos[1],
                 self.agent_pos[0]] == utils.RW4T_State.orange_zone.value
        ) and utils.RW4T_State.orange_zone.value in self.danger_pref:
      return True
    if (self.map[self.agent_pos[1],
                 self.agent_pos[0]] == utils.RW4T_State.red_zone.value
        ) and utils.RW4T_State.red_zone.value in self.danger_pref:
      return True
    return False

  def calculate_rewards(self, drop_info, new_in_danger_zone):
    '''
    Args:
    - drop_info: a dictionary that maps what the agent is holding to the drop
                 off location
    - new_in_danger_zone: whether the agent just arrives at a danger zone
    '''
    reward = 0
    for obj in drop_info:
      # Check if the object is in the list of preferred objects
      if obj in self.object_pref:
        lm = drop_info[obj]
        # Check if the landmark is one of the preferred landmarks for that
        # object and we haven't exceeded the number of drops for that object-
        # landmark pair
        if (lm in self.object_pref[obj]
            and self.dropped_objs[obj][lm] <= self.object_pref[obj][lm]):
          reward += self.correct_obj_lm_reward
        else:
          reward += self.incorrect_obj_lm_reward
      else:
        reward += self.incorrect_obj_lm_reward
    if new_in_danger_zone:
      reward += self.danger_reward_per_entrance
    return reward

  def get_task_reward(self, curr_a, drop_info):
    '''
    Get the task reward.
    '''
    for obj in drop_info:
      # Check if the object is in the list of preferred objects
      if obj in self.object_pref:
        lm = drop_info[obj]
        # Check if the landmark is one of the preferred landmarks for that
        # object and we haven't exceeded the number of drops for that object-
        # landmark pair
        if (lm in self.object_pref[obj]
            and self.dropped_objs[obj][lm] <= self.object_pref[obj][lm]):
          return self.correct_obj_lm_reward
        else:
          return -1
      else:
        return -1
    return -1

  def get_high_level_pref(self, state, prev_o, curr_o):
    '''
    Get the reward associated with the high level preference function.
    '''
    if self.reward_persistence:
      return self.get_high_level_pref_persistence(curr_o)
    else:
      return self.get_high_level_pref_diversity(curr_o)

  def get_high_level_pref_diversity(self, curr_o):
    '''
    High level reward for working on different objects.
    '''
    useful_objs = self.get_useful_obj_locations_with_types()
    circle_val = rw4t_utils.RW4T_State.circle.value
    square_val = rw4t_utils.RW4T_State.square.value
    triangle_val = rw4t_utils.RW4T_State.triangle.value
    # If we just delivered a circle, the agent is rewarded to pick up another
    # type of object if the other type is available.
    if (self.prev_option == self.rw4t_hl_actions.deliver_circle.value
        and ((len(useful_objs[square_val]) > 0
              and curr_o == self.rw4t_hl_actions.go_to_square.value) or
             (len(useful_objs[triangle_val]) > 0
              and curr_o == self.rw4t_hl_actions.go_to_triangle.value))):
      # If the agent is standing at a drop location for circles and just
      # dropped a circle
      if (circle_val in self.object_pref
          and self.map[self.old_state.pos[1],
                       self.old_state.pos[0]] in self.object_pref[circle_val]
          and self.old_state.last_drop == rw4t_utils.RW4T_State.circle.value):
        return self.diversity_reward
    # If we just delivered a square, the agent is rewarded to pick up another
    # type of object if the other type is available.
    if (self.prev_option == self.rw4t_hl_actions.deliver_square.value
        and ((len(useful_objs[circle_val]) > 0
              and curr_o == self.rw4t_hl_actions.go_to_circle.value) or
             (len(useful_objs[triangle_val]) > 0
              and curr_o == self.rw4t_hl_actions.go_to_triangle.value))):
      # If the agent is standing at a drop location for squares and just
      # dropped a square
      if (square_val in self.object_pref
          and self.map[self.old_state.pos[1],
                       self.old_state.pos[0]] in self.object_pref[square_val]
          and self.old_state.last_drop == rw4t_utils.RW4T_State.square.value):
        return self.diversity_reward
    # # If we just delivered a triangle, the agent is rewarded to pick up another
    # # type of object if the other type is available.
    # if (self.prev_option == self.rw4t_hl_actions.deliver_triangle.value
    #     and ((len(useful_objs[circle_val]) > 0
    #           and curr_o == self.rw4t_hl_actions.go_to_circle.value) or
    #          (len(useful_objs[square_val]) > 0
    #           and curr_o == self.rw4t_hl_actions.go_to_square.value))):
    #   # If the agent is standing at a drop location for triangles and just
    #   # dropped a triangle
    #   if (triangle_val in self.object_pref
    #       and self.map[self.old_state.pos[1],
    #                    self.old_state.pos[0]] in self.object_pref[triangle_val]
    #       and self.old_state.last_drop == rw4t_utils.RW4T_State.triangle.value):
    #     return self.diversity_reward
    # No HL preference otherwise.
    return 0

  def get_high_level_pref_persistence(self, curr_o):
    '''
    High level reward for working on the same object.
    '''
    useful_objs = self.get_useful_obj_locations_with_types()
    circle_val = rw4t_utils.RW4T_State.circle.value
    square_val = rw4t_utils.RW4T_State.square.value
    triangle_val = rw4t_utils.RW4T_State.triangle.value
    # If we just delivered a circle, the agent is rewarded to pick up another
    # circle if another circle is available.
    if (self.prev_option == self.rw4t_hl_actions.deliver_circle.value
        and ((len(useful_objs[circle_val]) > 0
              and curr_o == self.rw4t_hl_actions.go_to_circle.value))):
      # If the agent is standing at a drop location for circles and just
      # dropped a circle
      if (circle_val in self.object_pref
          and self.map[self.old_state.pos[1],
                       self.old_state.pos[0]] in self.object_pref[circle_val]
          and self.old_state.last_drop == rw4t_utils.RW4T_State.circle.value):
        return self.diversity_reward
    # If we just delivered a square, the agent is rewarded to pick up another
    # square if another square is available.
    if (self.prev_option == self.rw4t_hl_actions.deliver_square.value
        and ((len(useful_objs[square_val]) > 0
              and curr_o == self.rw4t_hl_actions.go_to_square.value))):
      # If the agent is standing at a drop location for squares and just
      # dropped a square
      if (square_val in self.object_pref
          and self.map[self.old_state.pos[1],
                       self.old_state.pos[0]] in self.object_pref[square_val]
          and self.old_state.last_drop == rw4t_utils.RW4T_State.square.value):
        return self.diversity_reward
    # # If we just delivered a triangle, the agent is rewarded to pick up another
    # # triangle if another triangle is available.
    # if (self.prev_option == self.rw4t_hl_actions.deliver_triangle.value
    #     and ((len(useful_objs[triangle_val]) > 0
    #           and curr_o == self.rw4t_hl_actions.go_to_triangle.value))):
    #   # If the agent is standing at a drop location for triangles and just
    #   # dropped a triangle
    #   if (triangle_val in self.object_pref
    #       and self.map[self.old_state.pos[1],
    #                    self.old_state.pos[0]] in self.object_pref[triangle_val]
    #       and self.old_state.last_drop == rw4t_utils.RW4T_State.triangle.value):
    #     return self.diversity_reward
    # No HL preference otherwise.
    return 0

  def get_high_level_pref_pbrs(self, curr_o):
    '''
    PBRS on the high-level does not theoretically depend on the action.
    The action here is merely used to get the next state, so we can get the
    potential difference.
    '''
    # Initialize the a new state (we don't care about the mask here so it is
    # initialized to zeros)
    new_state = RW4T_GameState(self.map, self.agent_pos, self.agent_holding,
                               self.last_pickup, self.last_drop,
                               np.zeros(len(self.rw4t_hl_actions)))
    return (self.get_potential(new_state, curr_o) - self.get_potential(
        self.old_state, self.prev_option)) * self.pbrs_factor

  def get_potential(self, state: RW4T_GameState, prev_o):
    # Calculate the base potential of the current state.
    # The base potential is calculated as the total number of supplies on the
    # initial map - the number of supplies to be delivered in the current state.
    all_supplies = self.get_all_locations_by_types_custom_map(
        state.obs, [
            utils.RW4T_State.circle.value, utils.RW4T_State.square.value,
            utils.RW4T_State.triangle.value
        ])
    start_potential = self.total_num_to_drop - len(all_supplies)
    if state.holding != utils.RW4T_State.empty.value:
      start_potential -= 1

    # Maximum distance between the agent and any location on the map
    max_dist = 2 * (len(state.obs) - 1) + 1
    assert max_dist == 11

    # If the previous option was a "go to" type
    if (prev_o == utils.RW4T_HL_Actions.go_to_circle.value
        or prev_o == utils.RW4T_HL_Actions.go_to_square.value):
      if prev_o == utils.RW4T_HL_Actions.go_to_circle.value:
        obj = utils.RW4T_State.circle.value
      else:
        obj = utils.RW4T_State.square.value
      # If the agent is not holding anything, calculate the potential based on
      # the agent's distance to the closest supply indicated by the previous
      # optin. If there is no supplies of such type, return the base potential
      if state.holding == utils.RW4T_State.empty.value:
        all_supply_locs = self.get_all_locations_by_types_custom_map(
            state.obs, [obj])
        if len(all_supply_locs) > 0:
          min_dist = 100
          for supply_loc in all_supply_locs:
            if utils.dist_heuristic(state.pos, supply_loc) < min_dist:
              min_dist = utils.dist_heuristic(state.pos, supply_loc)
          min_dist += 1  # Account for the pick action at the end
          assert min_dist <= max_dist
          return start_potential + ((max_dist - min_dist) / max_dist) * 0.5
        else:
          return start_potential
      else:
        # If the agent is holding some supply, the potential is simply base
        # potential.
        return start_potential
    else:
      # If the previous option was a "deliver" type
      if prev_o == utils.RW4T_HL_Actions.deliver_circle.value:
        obj = utils.RW4T_State.circle.value
      else:
        obj = utils.RW4T_State.square.value
      # If the agent is holding an object, then calculate the distance to the
      # closest delivery location and use that to calculate the potential
      if state.holding == obj:
        min_dist = 100
        for dest in self.object_pref[obj]:
          all_dropoff_locatons = self.get_all_locations_by_types_custom_map(
              state.obs, [dest])
          for loc in all_dropoff_locatons:
            if utils.dist_heuristic(state.pos, loc) < min_dist:
              min_dist = utils.dist_heuristic(state.pos, loc)
        min_dist += 1  # Account for the drop action at the end
        assert min_dist <= max_dist
        return start_potential + 0.5 + ((max_dist - min_dist) / max_dist) * 0.5
      elif state.holding == utils.RW4T_State.empty.value:
        # If the agent is not holding anything, check if the agent there are
        # objects remaining on the map and set the potential accordingly.
        if len(all_supplies) > 0:
          return start_potential
        else:
          return self.total_num_to_drop
      else:
        # If the agent is holding a supply that is not the type indicated by
        # the previous option
        return start_potential

  def get_low_level_pref(self, state, curr_o, curr_a):
    '''
    Get the reward associated with the low level preference function.

    The current low-level pref is implemented as not going through danger zones
    when agent is delivering an object.
    '''
    # When the agent is holding an object (no matter whether its option is
    # delivery)
    # In RW4T, the agent should only pick up an object if it wishes to
    # deliver the object, so adding this check penalizes the agent when it is
    # delivering an object but not using the corresponding option
    state_dict = state.state_to_dict()
    all_danger_zones = set([])
    for zones in self.danger_pref.values():
      all_danger_zones.update(zones)
    if state_dict['holding'] != rw4t_utils.Holding_Obj.empty.value:
      if self.map[self.agent_pos[1], self.agent_pos[0]] in all_danger_zones:
        return self.danger_reward_per_entrance
    # When the agent is performing a delivery option
    if curr_o in self.danger_pref:
      zones_to_avoid = self.danger_pref[curr_o]
      if self.map[self.agent_pos[1], self.agent_pos[0]] in zones_to_avoid:
        return self.danger_reward_per_entrance
      else:
        return 0
    else:
      return 0

  def get_pseudo_reward(self, curr_o, curr_a, pick_info, drop_info):
    '''
    Get the pseudo reward for low level policy training.
    '''
    # Get the pseudo reward for picking an object.
    assert curr_o in rw4t_utils.HL_Action_2_Obj

    if pick_info[0] != rw4t_utils.RW4T_State.empty.value:
      if self.check_is_pick_option(
          curr_o) and rw4t_utils.HL_Action_2_Obj[curr_o] == pick_info[0]:
        return self.pick_pseudo_reward
      else:
        return self.pseudo_penalty

    # Get the pseudo reward for dropping an object.
    if len(drop_info) > 0:
      for obj in drop_info:
        # Check if the object is in the list of preferred objects
        if obj in self.object_pref:
          lm = drop_info[obj]
          # Check if the landmark is one of the preferred landmarks for that
          # object & we haven't exceeded the number of drops for that object-
          # landmark pair & the current option corresponds to the object dropped
          if (lm in self.object_pref[obj]
              and self.dropped_objs[obj][lm] <= self.object_pref[obj][lm]
              and not self.check_is_pick_option(curr_o)
              and rw4t_utils.HL_Action_2_Obj[curr_o] == obj):
            return self.drop_pseudo_reward
          else:
            return self.pseudo_penalty
        else:
          return self.pseudo_penalty

    return -1

  def get_flatsa_pref(self, state, action):
    '''
    In the flat reward function, we penalize the agent for entering a danger
    zone.

    The high-level preference cannot be represeted as a flat reward and so is
    not implemented in this function.
    '''
    state_dict = state.state_to_dict()
    all_danger_zones = set([])
    for zones in self.danger_pref.values():
      all_danger_zones.update(zones)
    if state_dict['holding'] != rw4t_utils.Holding_Obj.empty.value:
      if self.map[self.agent_pos[1], self.agent_pos[0]] in all_danger_zones:
        return self.danger_reward_per_entrance

    return 0

  def check_done(self):
    '''
    This function returns two booleans: whether the environment is terminated
    and whether the environment is truncated.
    '''
    done = False
    truncated = False
    # The episode is TRUNCATED if we reach the time limit
    if (self.timer >= self.max_time_in_seconds
        or self.timestep >= self.max_timesteps):
      truncated = True
      return done, truncated
    # THE episode is DONE if the agent has dropped a certain number of objects
    num_drops = 0
    for inner_dict in self.dropped_objs.values():
      num_drops += sum(inner_dict.values())
    if num_drops == self.total_num_to_drop:
      done = True
    return done, truncated

  def check_done_ll(self, pseudo_reward):
    if pseudo_reward > 0:
      return True, False
    elif self.timestep_ll >= self.max_timesteps_ll:
      return False, True
    else:
      return False, False

  def get_all_locations_by_type(self, location_type):
    '''
    Get all locations by type on the current map.
    '''
    all_objs = set([])
    rows, cols = np.where(self.map == location_type)
    all_objs.update(set(zip(cols, rows)))
    return list(all_objs)

  def get_all_locations_by_types_custom_map(self, map,
                                            location_types: Iterable):
    # Get all locations by types on a custom map
    all_objs = set([])
    for loc_type in location_types:
      rows, cols = np.where(map == loc_type)
      all_objs.update(set(zip(cols, rows)))
    return list(all_objs)

  def get_all_obj_locations(self):
    '''
    Get all object locations on the map.
    '''
    all_objs = set([])
    if utils.RW4T_State.circle.value in self.object_pref:
      rows, cols = np.where(self.map == utils.RW4T_State.circle.value)
      all_objs.update(set(zip(cols, rows)))

    if utils.RW4T_State.square.value in self.object_pref:
      rows, cols = np.where(self.map == utils.RW4T_State.square.value)
      all_objs.update(set(zip(cols, rows)))

    if utils.RW4T_State.triangle.value in self.object_pref:
      rows, cols = np.where(self.map == utils.RW4T_State.triangle.value)
      all_objs.update(set(zip(cols, rows)))

    return list(all_objs)

  def get_all_obj_locations_with_types(self):
    '''
    Get all object locations on the map.
    '''
    all_objs = {}
    all_objs[utils.RW4T_State.circle.value] = set([])
    all_objs[utils.RW4T_State.square.value] = set([])
    all_objs[utils.RW4T_State.triangle.value] = set([])
    if utils.RW4T_State.circle.value in self.object_pref:
      rows, cols = np.where(self.map == utils.RW4T_State.circle.value)
      all_objs[utils.RW4T_State.circle.value] = set(zip(cols, rows))

    if utils.RW4T_State.square.value in self.object_pref:
      rows, cols = np.where(self.map == utils.RW4T_State.square.value)
      all_objs[utils.RW4T_State.square.value] = set(zip(cols, rows))

    if utils.RW4T_State.triangle.value in self.object_pref:
      rows, cols = np.where(self.map == utils.RW4T_State.triangle.value)
      all_objs[utils.RW4T_State.triangle.value] = set(zip(cols, rows))

    return all_objs

  def get_useful_obj_locations(self):
    '''
    Get all object locations if it is useful for agents to go to these
    locations.
    An object is useful if the following is true:
    - the object is in the preference dictionary,
    - the agent has not reached the number of drops for that object.
    '''
    all_objs = set([])
    if utils.RW4T_State.circle.value in self.object_pref:
      circle_dict_pref = self.object_pref[utils.RW4T_State.circle.value]
      circle_dict_cur = self.dropped_objs[utils.RW4T_State.circle.value]
      if sum(circle_dict_cur.values()) < sum(circle_dict_pref.values()):
        rows, cols = np.where(self.map == utils.RW4T_State.circle.value)
        all_objs.update(set(zip(cols, rows)))

    if utils.RW4T_State.square.value in self.object_pref:
      square_dict_pref = self.object_pref[utils.RW4T_State.square.value]
      square_dict_cur = self.dropped_objs[utils.RW4T_State.square.value]
      if sum(square_dict_cur.values()) < sum(square_dict_pref.values()):
        rows, cols = np.where(self.map == utils.RW4T_State.square.value)
        all_objs.update(set(zip(cols, rows)))

    if utils.RW4T_State.triangle.value in self.object_pref:
      tri_dict_pref = self.object_pref[utils.RW4T_State.triangle.value]
      tri_dict_cur = self.dropped_objs[utils.RW4T_State.triangle.value]
      if sum(tri_dict_cur.values()) < sum(tri_dict_pref.values()):
        rows, cols = np.where(self.map == utils.RW4T_State.triangle.value)
        all_objs.update(set(zip(cols, rows)))

    return list(all_objs)

  def get_useful_obj_locations_with_types(self):
    '''
    Get all object locations if it is useful for agents to go to these
    locations.
    An object is useful if the following is true:
    - the object is in the preference dictionary,
    - the agent has not reached the number of drops for that object.
    '''
    all_objs = {}
    all_objs[utils.RW4T_State.circle.value] = set([])
    all_objs[utils.RW4T_State.square.value] = set([])
    all_objs[utils.RW4T_State.triangle.value] = set([])
    if utils.RW4T_State.circle.value in self.object_pref:
      circle_dict_pref = self.object_pref[utils.RW4T_State.circle.value]
      circle_dict_cur = self.dropped_objs[utils.RW4T_State.circle.value]
      if sum(circle_dict_cur.values()) < sum(circle_dict_pref.values()):
        rows, cols = np.where(self.map == utils.RW4T_State.circle.value)
        all_objs[utils.RW4T_State.circle.value] = set(zip(cols, rows))

    if utils.RW4T_State.square.value in self.object_pref:
      square_dict_pref = self.object_pref[utils.RW4T_State.square.value]
      square_dict_cur = self.dropped_objs[utils.RW4T_State.square.value]
      if sum(square_dict_cur.values()) < sum(square_dict_pref.values()):
        rows, cols = np.where(self.map == utils.RW4T_State.square.value)
        all_objs[utils.RW4T_State.square.value] = set(zip(cols, rows))

    if utils.RW4T_State.triangle.value in self.object_pref:
      tri_dict_pref = self.object_pref[utils.RW4T_State.triangle.value]
      tri_dict_cur = self.dropped_objs[utils.RW4T_State.triangle.value]
      if sum(tri_dict_cur.values()) < sum(tri_dict_pref.values()):
        rows, cols = np.where(self.map == utils.RW4T_State.triangle.value)
        all_objs[utils.RW4T_State.triangle.value] = set(zip(cols, rows))

    return all_objs

  def get_obs_locations(self):
    '''
    Get all obstacle locations on the map.
    '''
    rows, cols = np.where(self.map == utils.RW4T_State.obstacle.value)
    return list(zip(cols, rows))

  def get_all_drop_locations(self):
    '''
    Get all drop locations on the map.
    '''
    all_locs = set([])
    rows, cols = np.where(self.map == utils.RW4T_State.school.value)
    all_locs.update(set(zip(cols, rows)))

    rows, cols = np.where(self.map == utils.RW4T_State.hospital.value)
    all_locs.update(set(zip(cols, rows)))

    rows, cols = np.where(self.map == utils.RW4T_State.park.value)
    all_locs.update(set(zip(cols, rows)))

    return list(all_locs)

  def get_danger_locations(self):
    '''
    Get all danger zone locations that should be avoided by the agent.
    '''
    all_zones = set([])
    all_zone_types = [
        item for sublist in list(self.danger_pref.values()) for item in sublist
    ]
    if utils.RW4T_State.yellow_zone.value in all_zone_types:
      rows, cols = np.where(self.map == utils.RW4T_State.yellow_zone.value)
      all_zones.update(set(zip(cols, rows)))

    if utils.RW4T_State.orange_zone.value in all_zone_types:
      rows, cols = np.where(self.map == utils.RW4T_State.orange_zone.value)
      all_zones.update(set(zip(cols, rows)))

    if utils.RW4T_State.red_zone.value in all_zone_types:
      rows, cols = np.where(self.map == utils.RW4T_State.red_zone.value)
      all_zones.update(set(zip(cols, rows)))

    return list(all_zones)

  def get_all_avoid_locations(self):
    '''
    Get all locations that should be avoided by the agent, including both
    danger zones and obstacles.
    '''
    all_avoid_locations = self.get_danger_locations()
    all_avoid_locations.extend(self.get_obs_locations())
    return all_avoid_locations

  def get_relevant_avoid_locations(self, option):
    all_avoid_locations = []
    if option in self.pref_dict['zones']:
      zones_to_avoid = self.pref_dict['zones'][option]
      for zone in zones_to_avoid:
        all_avoid_locations.extend(self.get_all_locations_by_type(zone))
    all_avoid_locations.extend(self.get_obs_locations())
    return all_avoid_locations

  def write_transition(self, transition: utils.RW4T_Transition):
    header = ''
    if not os.path.isfile(self.fname):
      header += transition.get_header()
      header += '\n'
      dir_name = os.path.dirname(self.fname)
      os.makedirs(dir_name, exist_ok=True)

    data = str(transition)
    data += '\n'

    with open(self.fname, 'a+') as f:
      f.write(header)
      f.write(data)
      f.close()

  def choose_rand_start_pos(self):
    if self.valid_start_pos is not None:
      return random.choice(self.valid_start_pos)

    start_pos = None
    while start_pos is None:
      x = random.randint(0, self.map_size - 1)
      y = random.randint(0, self.map_size - 1)
      if self.init_map[y, x] == utils.RW4T_State.empty.value:
        start_pos = (x, y)
    return start_pos

  def get_current_state(self):
    return self.state

  def get_current_features(self):
    # Include information of each object on the map
    all_objs = []
    for row_idx in range(len(self.map)):
      for col_idx in range(len(self.map[row_idx])):
        if self.map[row_idx][col_idx] != utils.RW4T_State.empty.value:
          one_hot_obj = utils.enum_to_one_hot(self.map[row_idx][col_idx],
                                              len(utils.RW4T_State))
          x_diff, y_diff = (col_idx - self.agent_pos[0],
                            row_idx - self.agent_pos[1])
          obj = np.concatenate((one_hot_obj, [x_diff, y_diff]))
          all_objs.append(obj)
    features = np.concatenate(all_objs)

    # If there are fewer objects on the map now than at the start, use zeros
    # to represent the rest of the objects
    init_num_items = np.sum(self.init_map != utils.RW4T_State.empty.value)
    cur_num_items = np.sum(self.map != utils.RW4T_State.empty.value)
    assert init_num_items - cur_num_items >= 0
    zeroes = np.zeros(
        (init_num_items - cur_num_items) * (len(utils.RW4T_State) + 2))

    # Include the object the agent is holding
    holding = utils.enum_to_one_hot(self.agent_holding, len(utils.RW4T_State))
    features = np.concatenate((features, zeroes, holding))

    return features

  def generate_hl_action(self):
    '''
    If the agent is holding an object, select a goal location.
    Otherwise, select an object location. Used for generating demos.
    '''
    potential_types = []
    if self.agent_holding != rw4t_utils.RW4T_State.empty.value:
      # If the agent has picked up a supply
      if self.agent_holding not in self.object_pref:
        if np.any(self.map == utils.RW4T_State.school.value):
          potential_types.append(utils.RW4T_State.school.value)
        if np.any(self.map == utils.RW4T_State.hospital.value):
          potential_types.append(utils.RW4T_State.hospital.value)
        if np.any(self.map == utils.RW4T_State.park.value):
          potential_types.append(utils.RW4T_State.park.value)
      else:
        for dest in self.object_pref[self.agent_holding]:
          if self.object_pref[self.agent_holding][dest] > self.dropped_objs[
              self.agent_holding][dest]:
            potential_types.append(dest)
    else:
      # Agent is not holding anything, so it should pick up a supply that is
      # the same as the previously delivered type
      for obj in self.object_pref:
        for dest in self.object_pref[obj]:
          if self.object_pref[obj][dest] > self.dropped_objs[obj][dest]:
            if (obj == rw4t_utils.RW4T_State.circle.value and self.prev_option
                == self.rw4t_hl_actions_with_dummy.deliver_circle.value) or (
                    obj == rw4t_utils.RW4T_State.square.value
                    and self.prev_option
                    == self.rw4t_hl_actions_with_dummy.deliver_square.value
                ) or (
                    obj == rw4t_utils.RW4T_State.triangle.value
                    and self.prev_option
                    == self.rw4t_hl_actions_with_dummy.deliver_triangle.value):
              potential_types.append(obj)
              continue

      # No supply found to match previously delivered type (i.e. at the start
      # of the environment or if all supplies of the type have been delivered)
      if len(potential_types) == 0:
        if np.any(self.map == utils.RW4T_State.circle.value):
          potential_types.append(utils.RW4T_State.circle.value)
        if np.any(self.map == utils.RW4T_State.square.value):
          potential_types.append(utils.RW4T_State.square.value)
        if np.any(self.map == utils.RW4T_State.triangle.value):
          potential_types.append(utils.RW4T_State.triangle.value)
    if len(potential_types) == 0:
      potential_types.append(utils.RW4T_State.school.value)
      potential_types.append(utils.RW4T_State.hospital.value)
    # Find the closest obj in terms of Manhattan distance
    shortest_dist = float('inf')
    closest_goal = None
    closest_goal_type = -1
    for dest in potential_types:
      rows, cols = np.where(self.map == dest)
      row_col_pairs = list(zip(cols, rows))
      for pair in row_col_pairs:
        dist = utils.dist_heuristic(self.agent_pos, pair)
        if dist < shortest_dist:
          shortest_dist = dist
          closest_goal = pair
          closest_goal_type = dest
    if closest_goal_type in rw4t_utils.Location_2_HL_Action:
      # If location is an object location
      hl_action = rw4t_utils.Location_2_HL_Action[closest_goal_type]
    else:
      # If location is a delivery location
      if self.agent_holding == rw4t_utils.RW4T_State.circle.value:
        hl_action = self.rw4t_hl_actions.deliver_circle.value
      elif self.agent_holding == rw4t_utils.RW4T_State.square.value:
        hl_action = self.rw4t_hl_actions.deliver_square.value
      elif self.agent_holding == rw4t_utils.RW4T_State.triangle.value:
        hl_action = self.rw4t_hl_actions.deliver_triangle.value
      else:
        raise NotImplementedError
    return hl_action, closest_goal

  def generate_ll_actions(self, hl_action, goal=None):
    '''
    Generate sequence of low level actions based on the given hl action.
    Used for generating demos.
    '''
    # The agent cannot go to an object and pick it up if it is holding
    # something.
    if (self.agent_holding != utils.RW4T_State.empty.value
        and self.check_is_pick_option(hl_action)):
      return [rw4t_utils.RW4T_LL_Actions.idle.value]
    # The agent cannot go to a dropoff location to drop off the object if it is
    # not holding anything.
    if (self.agent_holding == utils.RW4T_State.empty.value
        and not self.check_is_pick_option(hl_action)):
      return [rw4t_utils.RW4T_LL_Actions.idle.value]
    coords_2_dir = {
        (-1, 0): 'go_up',
        (1, 0): 'go_down',
        (0, -1): 'go_left',
        (0, 1): 'go_right'
    }
    # If a goal is not specified, find one based on the high level action
    if goal is None:
      loc_val = utils.HL_Action_2_Location[hl_action]
      shortest_dist = float('inf')
      closest_goal = None
      rows, cols = np.where(self.map == loc_val)
      row_col_pairs = list(zip(cols, rows))
      for pair in row_col_pairs:
        dist = utils.dist_heuristic(self.agent_pos, pair)
        if dist < shortest_dist:
          shortest_dist = dist
          closest_goal = pair
      # If there is a no goal that matches the high level action, return idle.
      if closest_goal is None:
        return [rw4t_utils.RW4T_LL_Actions.idle.value]
      else:
        goal = closest_goal
    # Generate a path to the goal
    avoid_locations = self.get_relevant_avoid_locations(hl_action)
    all_shortest_paths = rw4t_utils.find_all_shortest_path_bfs(
        self.agent_pos,
        goal,
        avoid_locations=avoid_locations,
        map_size=self.map_size)
    assert len(all_shortest_paths) >= 1
    path = all_shortest_paths[0]
    # path = random.choice(all_shortest_paths)
    assert path is not None
    # Generate a list of low level actions to go to that goal
    ll_actions = []
    for path_idx in range(1, len(path)):
      cur_pos = path[path_idx - 1]
      next_pos = path[path_idx]
      coord_diff = (next_pos[0] - cur_pos[0], next_pos[1] - cur_pos[1])
      action = coords_2_dir[coord_diff]
      ll_actions.append(rw4t_utils.RW4T_LL_Actions[action].value)
    # Add pick or drop depending on the type of goal
    if self.check_is_pick_option(hl_action):
      ll_actions.append(rw4t_utils.RW4T_LL_Actions['pick'].value)
    else:
      ll_actions.append(rw4t_utils.RW4T_LL_Actions['drop'].value)
    return ll_actions

  def get_env_description(self):
    '''
    Get a text description of the environment.
    '''
    all_obj_description = ''
    for row_idx in range(len(self.map)):
      for col_idx in range(len(self.map[row_idx])):
        if self.map[row_idx][col_idx] != utils.RW4T_State.empty.value:
          obj_name = utils.get_enum_name_by_value(utils.RW4T_State,
                                                  self.map[row_idx][col_idx])
          x_diff, y_diff = (col_idx - self.agent_pos[0],
                            row_idx - self.agent_pos[1])
          one_obj = f'{obj_name}: {y_diff} rows and {x_diff} columns away\n'
          all_obj_description += one_obj
    return all_obj_description

  def get_env_description_coord(self):
    '''
    Get a text description of the environment with coordinates.
    '''
    all_obj_description = ''
    for row_idx in range(len(self.map)):
      for col_idx in range(len(self.map[row_idx])):
        if self.map[row_idx][col_idx] != utils.RW4T_State.empty.value:
          obj_name = utils.get_enum_name_by_value(utils.RW4T_State,
                                                  self.map[row_idx][col_idx])
          one_obj = f'{obj_name} at ({col_idx}, {row_idx})\n'
          all_obj_description += one_obj
    return all_obj_description

  def get_agent_description(self):
    '''
    Get a text description of the agnet (i.e. what object the agent is holding).
    '''
    if self.agent_holding != utils.RW4T_State.empty.value:
      return utils.get_enum_name_by_value(utils.RW4T_State, self.agent_holding)
    else:
      return 'None'

  def get_agent_description_with_coords(self):
    '''
    Get a text description of the agnet (i.e. what object the agent is holding
    and where the agent is ).
    '''
    agent_desc = ''
    if self.agent_holding != utils.RW4T_State.empty.value:
      agent_desc += utils.get_enum_name_by_value(utils.RW4T_State,
                                                 self.agent_holding)
    agent_desc += \
      f'\nYou are currently at: ({self.agent_pos[0]}, {self.agent_pos[1]})'
    return agent_desc

  def get_ll_action_valid(self, ll_action):
    if ll_action in utils.Action_2_Coord:
      action_coord = utils.Action_2_Coord[ll_action]
      new_pos_maybe = (self.agent_pos[0] + action_coord[0],
                       self.agent_pos[1] + action_coord[1])
      return (self.check_in_bounds(new_pos_maybe)
              and new_pos_maybe not in self.get_obs_locations())
    elif ll_action == utils.RW4T_LL_Actions.pick.value:
      return (self.agent_holding == utils.RW4T_State.empty.value
              and self.map[self.agent_pos[1]][self.agent_pos[0]]
              != utils.RW4T_State.empty.value)
    elif ll_action == utils.RW4T_LL_Actions.drop.value:
      return (self.agent_holding != utils.RW4T_State.empty.value
              and self.map[self.agent_pos[1]][self.agent_pos[0]]
              == utils.RW4T_State.empty.value)
    else:
      return True

  def check_valid_option(self, option):
    '''
    Check whether an option is valid for the current map.
    '''
    if option not in rw4t_utils.HL_Action_2_Obj:
      return False

    obj = rw4t_utils.HL_Action_2_Obj[option]
    all_objs = self.get_all_obj_locations_with_types()
    return len(all_objs[obj]) > 0

  def get_valid_options(self):
    '''
    Get a list of valid options for the current map.
    '''
    valid_options = []
    all_objs = self.get_all_obj_locations_with_types()
    for obj in all_objs:
      if len(all_objs[obj]) > 0:
        valid_options.extend(rw4t_utils.Obj_2_Valid_Options[obj])
    return valid_options

  def get_valid_option(self):
    '''
    Sample a single valid option from a list of valid options.
    '''
    return random.choice(self.get_valid_options())

  def check_is_pick_option(self, option):
    return (option == self.rw4t_hl_actions.go_to_circle.value
            or option == self.rw4t_hl_actions.go_to_square.value
            or (self.rw4t_hl_actions == utils.RW4T_HL_Actions
                and option == self.rw4t_hl_actions.go_to_triangle.value))

  def get_high_level_pref_gpt(self, state, prev_option, option):
    pass

  def get_low_level_pref_gpt(self, state, option, action):
    pass

  def get_flat_sa_pref_gpt(self, state, action):
    if not isinstance(state, dict): state = state.state_to_dict()
    reward, reward_dict = get_user_pref_reward(state, action)
    return reward



from typing import Dict, Tuple
import math
def get_user_pref_reward(state: Dict, action: int) -> Tuple[float, Dict[str, float]]:
    '''
    Calculate the user preference reward for the current state and action.

    :param state: The state of the environment as a dictionary.
    :param action: The action taken by the agent.
    :return: A tuple consisting of the user preference reward and a
             dictionary with the individual reward components.
    '''
    pos = state['pos']
    holding = state['holding']
    current_pos_state = state['map'][pos[1], pos[0]]

    # Initialize reward components
    reward = 0
    reward_components = {
        "match_previous_type": 0,
        "avoid_yellow_zone_while_delivering": 0
    }

    # Check user preference on object pickup (matching previous delivered type)
    if action == rw4t_utils.RW4T_LL_Actions.pick.value and holding == rw4t_utils.Holding_Obj.empty.value:
        # Determine the type of the object at the current position
        if current_pos_state in [rw4t_utils.RW4T_State.circle.value, rw4t_utils.RW4T_State.square.value, rw4t_utils.RW4T_State.triangle.value]:
            current_object_type = current_pos_state

            # Check if the agent previously delivered this type
            # For demonstration: assuming last_delivered_obj_type is a parameter in the state indicating the previously delivered type
            # This detail would ideally be part of the agent's internal memory or state
            last_delivered_obj_type = state.get('last_delivered_obj_type', None)

            if last_delivered_obj_type is not None and current_object_type == last_delivered_obj_type:
                reward_components["match_previous_type"] = 1
                reward += 1

    # Check user preference to avoid yellow zones while delivering
    if holding != rw4t_utils.Holding_Obj.empty.value:
        if current_pos_state == rw4t_utils.RW4T_State.yellow_zone.value:
            reward_components["avoid_yellow_zone_while_delivering"] = -1
            reward -= 1

    return reward, reward_components
