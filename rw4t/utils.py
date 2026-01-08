from enum import Enum
import pygame
import random
import torch
import torch.nn.functional as F
import numpy as np
import heapq
from collections import deque


class RW4T_LL_Actions(Enum):
  go_left = 0
  go_down = 1
  go_right = 2
  go_up = 3
  pick = 4
  drop = 5
  idle = 6


class RW4T_HL_Actions(Enum):
  go_to_circle = 0
  deliver_circle = 1
  go_to_square = 2
  deliver_square = 3
  go_to_triangle = 4
  deliver_triangle = 5
  # go_to_school = 3
  # go_to_hospital = 4
  # go_to_park = 5


class RW4T_HL_Actions_With_Dummy(Enum):
  go_to_circle = 0
  deliver_circle = 1
  go_to_square = 2
  deliver_square = 3
  go_to_triangle = 4
  deliver_triangle = 5
  dummy = 6


class RW4T_HL_Actions_EZ(Enum):
  go_to_circle = 0
  deliver_circle = 1
  go_to_square = 2
  deliver_square = 3


class RW4T_HL_Actions_With_Dummy_EZ(Enum):
  go_to_circle = 0
  deliver_circle = 1
  go_to_square = 2
  deliver_square = 3
  dummy = 4


class RW4T_State(Enum):
  empty = 0
  circle = 1
  square = 2
  triangle = 3
  obstacle = 4
  yellow_zone = 5
  orange_zone = 6
  red_zone = 7
  school = 8
  hospital = 9
  park = 10


class Holding_Obj(Enum):
  empty = 0
  circle = 1
  square = 2
  triangle = 3


class RW4T_Dir(Enum):
  west = 0
  south = 1
  east = 2
  north = 3


Dir_2_Coord = {
    RW4T_Dir.west.value: (-1, 0),
    RW4T_Dir.south.value: (0, 1),
    RW4T_Dir.east.value: (1, 0),
    RW4T_Dir.north.value: (0, -1)
}

Key_2_Action = {
    pygame.K_w: RW4T_LL_Actions.go_up.value,
    pygame.K_a: RW4T_LL_Actions.go_left.value,
    pygame.K_s: RW4T_LL_Actions.go_down.value,
    pygame.K_d: RW4T_LL_Actions.go_right.value,
    pygame.K_p: RW4T_LL_Actions.pick.value,
    pygame.K_o: RW4T_LL_Actions.drop.value
}

Action_2_Coord = {
    RW4T_LL_Actions.go_up.value: (0, -1),
    RW4T_LL_Actions.go_down.value: (0, 1),
    RW4T_LL_Actions.go_left.value: (-1, 0),
    RW4T_LL_Actions.go_right.value: (1, 0)
}

Location_2_HL_Action = {
    RW4T_State.circle.value: RW4T_HL_Actions.go_to_circle.value,
    RW4T_State.square.value: RW4T_HL_Actions.go_to_square.value,
    RW4T_State.triangle.value: RW4T_HL_Actions.go_to_triangle.value,
    # RW4T_State.school.value: RW4T_HL_Actions.go_to_school.value,
    # RW4T_State.hospital.value: RW4T_HL_Actions.go_to_hospital.value,
    # RW4T_State.park.value: RW4T_HL_Actions.go_to_park.value
}
HL_Action_2_Location = {v: k for k, v in Location_2_HL_Action.items()}

HL_Action_2_Obj = {
    RW4T_HL_Actions.go_to_circle.value: RW4T_State.circle.value,
    RW4T_HL_Actions.go_to_square.value: RW4T_State.square.value,
    RW4T_HL_Actions.go_to_triangle.value: RW4T_State.triangle.value,
    RW4T_HL_Actions.deliver_circle.value: RW4T_State.circle.value,
    RW4T_HL_Actions.deliver_square.value: RW4T_State.square.value,
    RW4T_HL_Actions.deliver_triangle.value: RW4T_State.triangle.value
}

HL_Action_2_HL_Name = {
    RW4T_HL_Actions.go_to_circle.name: 'pick up a circle',
    RW4T_HL_Actions.go_to_square.name: 'pick up a square',
    RW4T_HL_Actions.go_to_triangle.name: 'pick up a triangle',
    # RW4T_HL_Actions.go_to_school.name: 'drop off at the school',
    # RW4T_HL_Actions.go_to_hospital.name: 'drop off at the hospital',
    # RW4T_HL_Actions.go_to_park.name: 'drop off at the park'
}
HL_Name_2_HL_Action = {v: k for k, v in HL_Action_2_HL_Name.items()}

HL_Action_2_Prog_Action = {
    RW4T_HL_Actions.go_to_circle.name: "pick('circle')",
    RW4T_HL_Actions.go_to_square.name: "pick('square')",
    RW4T_HL_Actions.go_to_triangle.name: "pick('triangle')",
    # RW4T_HL_Actions.go_to_school.name: "drop('school')",
    # RW4T_HL_Actions.go_to_hospital.name: "drop('hospital')",
    # RW4T_HL_Actions.go_to_park.name: "drop('park')",
}

Obj_2_Valid_Options = {
    RW4T_State.circle.value:
    [RW4T_HL_Actions.go_to_circle.value, RW4T_HL_Actions.deliver_circle.value],
    RW4T_State.square.value:
    [RW4T_HL_Actions.go_to_square.value, RW4T_HL_Actions.deliver_square.value],
    RW4T_State.triangle.value: [
        RW4T_HL_Actions.go_to_triangle.value,
        RW4T_HL_Actions.deliver_triangle.value
    ]
}


def get_enum_name_by_value(enum_class, value):
  for member in enum_class:
    if member.value == value:
      return member.name
  return None  # Return None if no match is found


def enum_to_one_hot(enum_value, num_classes):
  '''
  Function to convert an enum value to a one-hot encoding
  '''
  one_hot = np.zeros(num_classes, dtype=int)
  one_hot[enum_value] = 1
  return one_hot


def row_to_state_desc(row):
  '''
  Generate a text desription of the state given the features in a dataset.
  '''
  env_desc = 'State: \n'
  features = row['features']
  obj_features = features[:len(features) - len(RW4T_State)]
  reshaped_obj_features = np.array(obj_features).reshape(
      -1,
      len(RW4T_State) + 2)
  obj_names = [
      RW4T_State(np.argmax(obj[:-2])).name for obj in reshaped_obj_features
  ]
  obj_locs = [(int(obj[-1]), int(obj[-2])) for obj in reshaped_obj_features]
  for idx in range(len(obj_names)):
    if obj_names[idx] != RW4T_State.empty.name:
      env_desc += f'{obj_names[idx]}: {obj_locs[idx][0]} rows and {obj_locs[idx][-1]} columns away\n'

  holding_features = features[-len(RW4T_State):]
  env_desc += f'You are holding: {RW4T_State(np.argmax(np.array(holding_features))).name}\n'
  return env_desc


def row_to_desc_ll(row):
  '''
  Generate a text description for a (s, a) pair as in context examples.
  a is a low level action.
  '''
  env_desc = row_to_state_desc(row)
  action = row['action']
  action_name = get_enum_name_by_value(RW4T_LL_Actions, action)
  env_desc += f'The expert\'s action at this state is: {action_name}\n\n'
  return env_desc


def row_to_desc_hl(row, prompt_style):
  '''
  Generate a text description for a (s, a) pair as in context examples.
  a is a high level action.
  '''
  env_desc = row_to_state_desc(row)
  action = row['macro_action']
  if prompt_style == 'lang':
    name = HL_Action_2_HL_Name[action]
  else:
    name = HL_Action_2_Prog_Action[action]
  env_desc += f'The expert\'s action at this state is: {name}\n\n'
  return env_desc


seed = 42
random.seed(seed)
rw4t_seeds = random.sample(range(1, 1001), 100)
print('RW4T env seeds: ', rw4t_seeds)

composite_skills = ['pick', 'go left', 'go down', 'go right', 'go up']


class RW4T_Transition():

  def __init__(self, state, features, action, macro_action, macro_idx, reward,
               next_state, next_features, done, info) -> None:
    self.state = state
    self.features = features
    self.action = action
    self.macro_action = macro_action
    self.macro_idx = macro_idx
    self.reward = reward
    self.next_state = next_state
    self.next_features = next_features
    self.done = done
    self.info = info

  def __str__(self):
    state_map = self.state.obs.tolist()
    state_pos = list(self.state.pos)
    state_holding = self.state.holding
    state_str = self.get_state_str(state_map, state_pos, state_holding)

    next_state_map = self.next_state.obs.tolist()
    next_state_pos = list(self.next_state.pos)
    next_state_holding = self.next_state.holding
    next_state_str = self.get_state_str(next_state_map, next_state_pos,
                                        next_state_holding)

    return state_str + '; ' + str(self.features.tolist()) + '; ' + str(
        self.action) + '; ' + str(self.macro_action) + '; ' + str(
            self.macro_idx) + '; ' + str(
                self.reward) + '; ' + next_state_str + '; ' + str(
                    self.next_features.tolist()) + '; ' + str(
                        self.done) + '; ' + str(self.info)

  def get_state_str(self, game_map, agent_pos, agent_holding):
    return str(game_map) + '|' + str(agent_pos) + '|' + str(agent_holding)

  def get_header(self):
    return ('state; features; action; macro_action; macro_idx; reward; ' +
            'next_state; next_features; done; info')


def convert_obs_to_tensor(env_state, hl=True, one_hot=False):
  obs_map = torch.tensor(env_state.obs.tolist()).flatten()
  if one_hot:
    obs_map = F.one_hot(obs_map, num_classes=len(RW4T_State)).flatten()
  if hl:
    obs_pos = torch.tensor(list(env_state.pos))
    obs_tensor = torch.cat((obs_map, obs_pos), dim=0).float()
  else:
    obs_dir = torch.tensor([env_state.dir])
    if one_hot:
      obs_dir = F.one_hot(obs_dir, num_classes=len(RW4T_Dir)).flatten()
    obs_pos = torch.tensor(list(env_state.pos))
    obs_tensor = torch.cat((obs_map, obs_dir, obs_pos), dim=0).float()
  return obs_tensor


def generate_ll_actions(env_state, hl_action):
  '''
  Generate low level actions given the name of a hl_action.
  '''
  recipe = []
  agent_dir = env_state.dir
  # print('agent dir: ', agent_dir)
  hl_val = RW4T_HL_Actions[hl_action].value

  if 'left' in hl_action:
    num_turns = hl_val - agent_dir
  elif 'down' in hl_action:
    num_turns = hl_val - agent_dir
  elif 'right' in hl_action:
    num_turns = hl_val - agent_dir
  elif 'up' in hl_action:
    num_turns = hl_val - agent_dir
  else:
    num_turns = 0
  if num_turns > 0:
    direction = 'turn_left'
  elif num_turns < 0:
    direction = 'turn_right'
  for i in range(abs(num_turns)):
    recipe.append(direction)

  if 'go' in hl_action:
    recipe.append('move_forward')
  if 'pick' in hl_action:
    recipe.append('pick')
  return recipe


def a_star_search(start, goal, avoid_locations=[], map_size=6):
  '''
  Perform a* search from the agent's current position to the goal location.
  Returns a list of row and column indices for the path from start to goal.
  The path starts from the start location and ends at the goal location.
  '''
  # Note: all locations will be row and column indices in this function
  # Convert to rows and columns
  start = (start[1], start[0])
  goal = (goal[1], goal[0])
  avoid_locations = [(b, a) for a, b in avoid_locations]

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
      if (0 <= nx < map_size and 0 <= ny < map_size
          and neighbor not in avoid_locations):
        tentative_g_score = g_score[current] + 1
        if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
          g_score[neighbor] = tentative_g_score
          est_score = tentative_g_score + dist_heuristic(neighbor, goal)
          heapq.heappush(open_list, (est_score, neighbor))
          came_from[neighbor] = current

  return None


def dist_heuristic(a, b):
  # Manhattan distance heuristic for grid pathfinding
  return abs(a[0] - b[0]) + abs(a[1] - b[1])


def find_all_shortest_path_bfs(start, goal, avoid_locations=[], map_size=6):
  '''
  Perform bfs search from the agent's current position to the goal location.
  Returns a list of shortest paths from start to goal.
  The path starts from the start location and ends at the goal location.
  '''
  # Convert to rows and columns
  start = (start[1], start[0])
  # print('start: ', start)
  goal = (goal[1], goal[0])
  # print('goal: ', goal)
  avoid_locations = [(b, a) for a, b in avoid_locations]
  # print('avoid locations: ', avoid_locations)

  # Queue holds tuples of (current_position, current_path)
  queue = deque([(start, [start])])
  # Track visited nodes to avoid re-exploration
  visited = set()
  shortest_paths = []
  shortest_length = float('inf')

  while queue:
    # print('shortest paths: ', shortest_paths)
    current_pos, path = queue.popleft()
    x, y = current_pos

    # If the goal is reached
    if current_pos == goal:
      if len(path) < shortest_length:
        # Reset shortest length and shortest paths list
        shortest_length = len(path)
        shortest_paths = [path]
      elif len(path) == shortest_length:
        # Add the current path to the list
        shortest_paths.append(path)
      continue

    # Mark the current node as visited (visited at the shortest possible distance)
    visited.add(current_pos)

    # Explore alll possible directions
    neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    for neighbor in neighbors:
      nx, ny = neighbor
      if (0 <= nx < map_size and 0 <= ny < map_size
          and neighbor not in avoid_locations and neighbor not in visited):
        queue.append((neighbor, path + [neighbor]))

  return shortest_paths


def pref_to_str(kit_pref, danger_pref):
  kit_pref_str = 'The human prefers you to pick up the objects at the following locations: '
  if len(kit_pref) == 0:
    kit_pref_str += 'None\n'
  else:
    counter = 0
    for kit in kit_pref:
      kit_pref_str += f'row {kit[1]}, column {kit[0]}'
      if counter < len(kit_pref) - 1:
        kit_pref_str += '; '
      else:
        kit_pref_str += '\n'

  danger_pref_str = 'The human prefers you to avoid the danger zones at the following locations: '
  if len(danger_pref) == 0:
    danger_pref_str += 'None\n'
  else:
    counter = 0
    for danger in danger_pref:
      danger_pref_str += f'row {danger[1]}, column {danger[0]}'
      if counter < len(danger_pref) - 1:
        danger_pref_str += '; '
      else:
        danger_pref_str += '\n'

  return kit_pref_str + danger_pref_str
