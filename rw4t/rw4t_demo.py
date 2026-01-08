import heapq
import time
from copy import deepcopy
from typing import List
import random
import numpy as np
import inflect

import map_config
from rw4t_env import RW4TEnv
from rw4t.rw4t_game import RW4T_Game
import rw4t.utils as rw4t_utils
from rw4t.map_config import pref_dicts


class RW4T_Demo(RW4T_Game):
  '''
  Demo agent for RW4T.
  '''

  def __init__(
      self,
      env: RW4TEnv,
      seed: int,
      game_fps: float = 2,
      play: bool = True,
  ):
    RW4T_Game.__init__(self, env, play=play)
    # Game frame rate
    self.fps = game_fps
    # The agent's current goal. The goal will be the row and column indices
    # of an object or goal
    self.cur_goal = None
    # The path to the goal. The path will be a list of row and column indices
    # from the start to the goal, inclusive.
    self.path = None
    # The name of the current hl action
    self.hl_action = None
    # A list of low level actions based on the hl action
    self.ll_actions = []
    # Index of the current unexecuted low level action
    self.ll_action_idx = 0
    # Initialize and render game
    self.on_init()
    self.on_render()
    random.seed(seed)

  def execute_agent(self, sleep_time=0):
    '''
    Execute the agent until done.
    '''
    c_reward = 0
    done = truncated = False
    while not (done or truncated):
      if self.cur_goal is None:
        # Select goal
        self.hl_action, self.cur_goal = self.env.generate_hl_action()
        assert self.hl_action is not None
        # Generate low level actions
        self.ll_actions = self.env.generate_ll_actions(self.hl_action,
                                                       self.cur_goal)

      action = self.ll_actions[self.ll_action_idx]
      _obs, reward, done, truncated, _info = self.env.step(
          action, passed_time=1 / self.fps, hl_action=self.hl_action)

      self.ll_action_idx += 1
      if self.ll_action_idx == len(self.ll_actions):
        self.reset()
      c_reward += reward
      self.on_render()
      time.sleep(sleep_time)
    return c_reward

  def reset(self):
    '''
    Reset when a HL action is complete.
    '''
    self.cur_goal = None
    self.path = None
    self.hl_action = None
    self.ll_actions = []
    self.ll_action_idx = 0

  def a_star_search(self, goal):
    '''
    Perform a* search from the agent's current position to the goal location.
    Returns a list of row and column indices for the path from start to goal,
    while avoiding the danger zones.
    The path starts from the start location and ends at the goal location.
    '''
    # Note: all locations will be row and column indices in this function
    start = (self.env.agent_pos[1], self.env.agent_pos[0])
    goal = (goal[1], goal[0])
    avoid_locations = self.env.get_all_avoid_locations()
    # Convert to rows and columns
    avoid_locations = [(b, a) for a, b in avoid_locations]
    # print('avoid locations: ', avoid_locations)

    # Open list (priority queue) with the starting pos
    open_list = []
    heapq.heappush(open_list, (0, start))

    # came_from dictionary to track the path
    came_from = {}
    came_from[start] = None

    # cost to reach each point
    g_score = {start: 0}

    while open_list:
      current = heapq.heappop(open_list)[1]
      if current == goal:
        # Reconstruct path
        path = []
        while current is not None:
          path.append(current)
          current = came_from[current]
        path.reverse()
        return path

      x, y = current
      # List of possible directions
      neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
      for neighbor in neighbors:
        nx, ny = neighbor
        if (0 <= nx < self.env.map_size and 0 <= ny < self.env.map_size
            and neighbor not in avoid_locations):
          tentative_g_score = g_score[current] + 1
          if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
            g_score[neighbor] = tentative_g_score
            est_score = tentative_g_score + self.dist_heuristic(neighbor, goal)
            heapq.heappush(open_list, (est_score, neighbor))
            came_from[neighbor] = current

    return None

  def dist_heuristic(self, a, b):
    # Manhattan distance heuristic for grid pathfinding
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def eval_demo(num_episodes: int = 100,
              map_num: int = 8,
              init_pos=None,
              rw4t_game_params=dict(),
              seed=0,
              render=False):
  # Make RW4T env
  pref_dict = pref_dicts[f'six_by_six_{map_num}_train_pref_dict']
  env = RW4TEnv(map_name=f'six_by_six_{map_num}_train_map',
                low_level=False,
                hl_pref_r=True,
                pbrs_r=False,
                pref_dict=pref_dict,
                seed=seed,
                rw4t_game_params=rw4t_game_params,
                init_pos=init_pos,
                action_duration=0,
                write=False,
                fname=f'rw4t_demos_manual/manual_control_{seed}.txt',
                render=render)

  all_rewards_sum = None
  for ep in range(num_episodes):
    # Reset the environment
    env.reset()
    done = False
    truncated = False

    # Reset the demo agent's internal status
    cur_goal = None
    hl_action = None
    ll_actions = []
    ll_action_idx = 0

    # Run one episode
    while not (done or truncated):
      # Select agent
      if cur_goal is None:
        # Select goal
        hl_action, cur_goal = env.generate_hl_action()
        assert hl_action is not None
        # Generate low level actions
        ll_actions = env.generate_ll_actions(hl_action, cur_goal)
      action = ll_actions[ll_action_idx]

      # Step the environment forward
      obs, reward, done, truncated, info = env.step(action, hl_action)

      # Accumulate rewards
      if all_rewards_sum is None:
        all_rewards_sum = tuple(0 for _ in reward)
      all_rewards_sum = tuple([
          acc_sub_r + sub_r for acc_sub_r, sub_r in zip(all_rewards_sum, reward)
      ])

      # Bookkeep the agent's internal status
      ll_action_idx += 1
      if ll_action_idx == len(ll_actions):
        cur_goal = None
        hl_action = None
        ll_actions = []
        ll_action_idx = 0

  all_rewards_avg = tuple(
      [rewards_sum / num_episodes for rewards_sum in all_rewards_sum])
  print("Avg cumulative rewards: ", all_rewards_avg)
  print('Avg task reward: ', all_rewards_avg[0])
  print('Avg pseudo reward: ', all_rewards_avg[1])
  print('Avg low-level pref: ', all_rewards_avg[2])
  hl_pref_reward = all_rewards_avg[3]
  print('Avg high-level pref: ', hl_pref_reward)
  print('Avg low-level reward (pseudo + low-level pref): ',
        all_rewards_avg[1] + all_rewards_avg[2])
  print('Avg high-level reward (task + high-level pref): ',
        all_rewards_avg[0] + hl_pref_reward)
  print('Avg all rewards (task +  HL + LL): ',
        all_rewards_avg[0] + all_rewards_avg[2] + hl_pref_reward)


if __name__ == '__main__':
  eval_demo()

  # map_size = 10
  # p = inflect.engine()
  # map_size_word = p.number_to_words(map_size)
  # version = 'v2_train'
  # pref_dict_name = f'{map_size_word}_by_{map_size_word}_{version[1:]}_pref_dict'
  # pref_dict = map_config.pref_dicts[pref_dict_name]
  # map_name = f'{map_size_word}_by_{map_size_word}_{version[1:]}_map'
  # seeds = rw4t_utils.rw4t_seeds[:20]

  # for seed in seeds:
  #   env = RW4TEnv(
  #       map_name=map_name,
  #       pref_dict=pref_dict,
  #       seed=seed,
  #       action_duration=0,
  #       write=False,
  #       fname=
  #       f'rw4t_demos_bfs/{map_size}by{map_size}/{version}/bfs_control_{seed}.txt'
  #   )
  #   demo = RW4T_Demo(env, seed, play=True)
  #   demo.execute_agent(sleep_time=1)
