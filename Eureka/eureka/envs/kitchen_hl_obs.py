'''
Background:
1) Ingredients:
class Ingredients(Enum):
  empty = 0
  tomato = 1
  onion = 2
  lettuce = 3

2) Salad types:
class SoupType(Enum):
  no_soup = 0
  alice = 1
  bob = 2
  cathy = 3
  david = 4

3) All available options:
{'Chop Tomato': 0, 'Chop Lettuce': 1, 'Chop Onion': 2, 'Prepare David Ingredients': 3, 'Plate David Salad': 4}

4) All available actions:
{0: (0, -1),
 1: (0, 1),
 2: (1, 0),
 3: (-1, 0),
 4: (0, 0)}
If the agent is standing next to a counter, performing an action in the direction of the counter interacts with the counter.
For example, if the agent is standing under a counter, performing action 0 (goes up) interacts with the counter above the agent.
'''

import gymnasium as gym


class KitchenHL(gym.Env):

  def get_plain_state(self, raw_info):
    '''
    The output of this function will be the input state in the generated reward
    function.

    The state is a dictionary that maps object names to their locations on the
    map.

    If the object 'obj' is at location (x, y), then state['obj'][y, x] == 1.
    Otherwise, state['obj'][y, x] == 0.
    '''
    num_rows = self.world_size[1]
    num_cols = self.world_size[0]
    state_dict = {}

    # Process Grid Squares Map
    GRIDSQUARES = [
        "Floor", "Counter", "Cutboard", "Bin", "Pot", "FreshTomatoTile",
        "FreshOnionTile", "FreshLettuceTile", "PlateTile"
    ]
    gridsquares_map = raw_info['gridsquare']
    for gridsquare_type in GRIDSQUARES:
      grid_map = gridsquares_map[gridsquare_type].T
      assert grid_map.shape == (num_rows, num_cols)
      state_dict[gridsquare_type] = grid_map

    # Process Object Map
    OBJECTS = ['FreshTomato', 'FreshLettuce', 'FreshOnion'] + [
        'ChoppingTomato', 'ChoppingOnion', 'ChoppingLettuce'
    ] + ['ChoppedTomato', 'ChoppedOnion', 'ChoppedLettuce'] + ['Plate']
    objects_map = raw_info['objects']
    for obj_type in OBJECTS:
      obj_map = objects_map[obj_type].T
      assert obj_map.shape == (num_rows, num_cols)
      state_dict[obj_type] = obj_map

    # Process Agent Map
    agent_map = raw_info['agent_map']['agent-1'].T
    assert agent_map.shape == (num_rows, num_cols)
    state_dict['agent'] = agent_map

    return state_dict
