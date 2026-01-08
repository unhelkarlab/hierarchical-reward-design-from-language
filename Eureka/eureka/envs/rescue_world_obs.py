'''
Background:

1) Initial game map example
init_map = np.array(
    [[
        rw4t_utils.RW4T_State.empty.value, rw4t_utils.RW4T_State.empty.value,
        rw4t_utils.RW4T_State.circle.value, rw4t_utils.RW4T_State.empty.value,
        rw4t_utils.RW4T_State.yellow_zone.value,
        rw4t_utils.RW4T_State.school.value
    ],
     [
         rw4t_utils.RW4T_State.empty.value,
         rw4t_utils.RW4T_State.yellow_zone.value,
         rw4t_utils.RW4T_State.yellow_zone.value,
         rw4t_utils.RW4T_State.empty.value,
         rw4t_utils.RW4T_State.yellow_zone.value,
         rw4t_utils.RW4T_State.empty.value
     ],
     [
         rw4t_utils.RW4T_State.empty.value, rw4t_utils.RW4T_State.empty.value,
         rw4t_utils.RW4T_State.empty.value, rw4t_utils.RW4T_State.square.value,
         rw4t_utils.RW4T_State.yellow_zone.value,
         rw4t_utils.RW4T_State.empty.value
     ],
     [
         rw4t_utils.RW4T_State.empty.value, rw4t_utils.RW4T_State.circle.value,
         rw4t_utils.RW4T_State.empty.value, rw4t_utils.RW4T_State.empty.value,
         rw4t_utils.RW4T_State.empty.value, rw4t_utils.RW4T_State.empty.value
     ],
     [
         rw4t_utils.RW4T_State.yellow_zone.value,
         rw4t_utils.RW4T_State.yellow_zone.value,
         rw4t_utils.RW4T_State.empty.value, rw4t_utils.RW4T_State.empty.value,
         rw4t_utils.RW4T_State.yellow_zone.value,
         rw4t_utils.RW4T_State.yellow_zone.value
     ],
     [
         rw4t_utils.RW4T_State.empty.value, rw4t_utils.RW4T_State.empty.value,
         rw4t_utils.RW4T_State.empty.value, rw4t_utils.RW4T_State.empty.value,
         rw4t_utils.RW4T_State.square.value, rw4t_utils.RW4T_State.empty.value
     ]])

2) rw4t.utils:
class RW4T_LL_Actions(Enum):
  go_left = 0
  go_down = 1
  go_right = 2
  go_up = 3
  pick = 4
  drop = 5
  idle = 6


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
'''

import numpy as np
import gymnasium as gym

import rw4t.utils as rw4t_utils


class RW4T_GameState:

  def __init__(self, obs: np.ndarray, pos: np.ndarray, holding: int,
               option_mask: np.ndarray):
    '''
    :param obs: a 2D numpy of the current environment
    :param pos: a 1D numpy array of the agent's (x, y) position in the
                environment
    :param holding: an integer indicating what object the agent is currently
                    holding if any.
                    This parameter only has a non-empty value AFTER the agent
                    performs a 'pick up ...' option and BEFORE it performs a
                    'deliver ...' option.
    :param option_mask: a 1D array indicating the valid options to select next
                        (should not be used when computing rewards, this is only
                        used in some downstream algorithms)
    '''
    # Y pos in bound
    assert pos[1] >= 0 and pos[1] < len(obs)
    # X pos in bound
    assert pos[0] >= 0 and pos[0] < len(obs[0])
    # holding should be a value in the Holding_Obj Enum
    assert holding < len(rw4t_utils.Holding_Obj)
    self.obs = obs
    self.pos = pos
    self.holding = holding
    self.option_mask = option_mask

  def state_to_dict(self):
    return {
        'map': np.array(self.obs, dtype=np.int32),
        'pos': np.array(self.pos, dtype=np.int32),
        'holding': self.holding,
        'option_mask': self.option_mask
    }


class RW4TEnv(gym.Env):

  def get_state(self):
    state = RW4T_GameState(self.map, self.agent_pos, self.agent_holding,
                           self.option_mask)
    state_dict = state.state_to_dict()
    return state_dict
