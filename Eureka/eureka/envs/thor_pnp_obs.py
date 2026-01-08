'''
Background:

1) utils:
PnP_LL_Actions = [
    "MoveAhead",
    "RotateLeft",
    "RotateRight",
    "PickupNearestTarget",
    "PutHeldOnReceptacle",
]

class PnP_HL_Actions(Enum):
    pick_apple = 0
    pick_egg = 1
    drop_apple = 2
    drop_egg = 3

class PnP_HL_Actions_With_Dummy(Enum):
    pick_apple = 0
    pick_egg = 1
    drop_apple = 2
    drop_egg = 3
    dummy = 4
'''

import random
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from typing import Dict, List, Optional

from HierRL.envs.ai2thor.pnp_training_utils import (PnP_HL_Actions,
                                                    PnP_HL_Actions_With_Dummy,
                                                    PnP_LL_Actions)
from HierRL.envs.ai2thor.pnp_config import avoid_stool


class ThorPickPlaceEnv(gym.Env):
  """
  Pick-and-place environment on top of AI2-THOR, using the Gymnasium API.

  Episode structure:
    - reset() loads a kitchen scene, curates it (move/disable/spawn a few
      items), and returns an observation.
    - step(a) applies either low-level nav (Move/Rotate/Look) or a simple HL
      manipulation (Pickup nearest target / Put on nearest receptacle / Drop).
    - reward is currently 0/1 placeholder (see _compute_reward_and_done).
  """
  metadata = {"render_modes": ["rgb_array"]}

  def __init__(
      self,
      scene: str = "FloorPlan20",  # scene id
      pref_dict: Dict[str, List[int]] = avoid_stool,  # preference dictionary
      visibilityDistance: float = 1,  # meters for "visible" flag (not reach)
      grid_size: float = 0.25,  # movement step in meters
      snap_to_grid: bool = True,  # keep motion aligned to grid
      rotate_step_degrees: int = 90,  # degree per rotate action
      render_depth: bool = False,
      render_instance_masks: bool = False,
      target_types=('Apple',
                    'Egg'),  # categories of objects that the agent can pick
      receptacle_types=("SinkBasin", ),  # categories we allow "PutObject" on
      max_steps: int = None,
      low_level: bool = False,  # whether we are working with low-level only
      hl_pref_r=None,
      option: PnP_HL_Actions = None,
      seed: Optional[int] = None,
      render: bool = True):
    super().__init__()
    # Save config
    self.scene = scene
    self.max_steps = max_steps
    self.target_types = set(target_types)
    self.receptacle_types = set(receptacle_types)
    self._rng = random.Random(seed)

    h, w = 600, 600
    platform = None if render else CloudRendering
    self.need_render = render
    self.controller = Controller(
        width=w,
        height=h,
        scene=self.scene,
        gridSize=grid_size,
        snapToGrid=snap_to_grid,
        rotateStepDegrees=rotate_step_degrees,
        renderDepthImage=render_depth,
        renderInstanceSegmentation=render_instance_masks,
        visibilityDistance=visibilityDistance,
        platform=platform)
    self.controller.step(action="Initialize", gridSize=grid_size)

    # Observation: dictionary-based state space.
    self.observation_space = spaces.Dict({
        "apple_1_pos":
        spaces.Box(-3.0, 3.0, (2, ), dtype=np.float32),
        "apple_2_pos":
        spaces.Box(-3.0, 3.0, (2, ), dtype=np.float32),
        "egg_1_pos":
        spaces.Box(-3.0, 3.0, (2, ), dtype=np.float32),
        "egg_2_pos":
        spaces.Box(-3.0, 3.0, (2, ), dtype=np.float32),
        "stool_pos":
        spaces.Box(-3.0, 3.0, (2, ), dtype=np.float32),
        "sink_pos":
        spaces.Box(-3.0, 3.0, (2, ), dtype=np.float32),
        "agent_pos":
        spaces.Box(-3.0, 3.0, (2, ), dtype=np.float32),  # x and z pos
        "agent_rot":
        spaces.Box(0.0, 1.0, (4, ),
                   dtype=np.float32),  # y rot (one-hot encoded)
        "apple_1_state":
        spaces.Discrete(3),  # 0 = on table, 1 = held, 2 = in sink
        "apple_2_state":
        spaces.Discrete(3),  # 0 = on table, 1 = held, 2 = in sink
        "egg_1_state":
        spaces.Discrete(3),  # 0 = on table, 1 = held, 2 = in sink
        "egg_2_state":
        spaces.Discrete(3),  # 0 = on table, 1 = held, 2 = in sink
    })

    # Whether we are working with the low-level only
    self.low_level = low_level

    # Adjust task/subtask horizons
    if max_steps is not None:
      self.max_steps = max_steps
    else:
      if self.low_level:
        self.max_steps = 100
      else:
        self.max_steps = 500

    # Define action spaces
    self.pnp_ll_actions = PnP_LL_Actions
    self.pnp_hl_actions = PnP_HL_Actions
    self.pnp_hl_actions_with_dummy = PnP_HL_Actions_With_Dummy

    # Low level action space: iThor environment commands
    self.ll_action_space = spaces.Discrete(len(self.pnp_ll_actions))
    self.hl_action_space = spaces.Discrete(len(self.pnp_hl_actions))

    # High level action space: Options (pick up/drop specific items)
    # Option values
    self.option = option

    # Initialize environment
    self._setup_env()
    if self.low_level:
      if self.option is None:
        self.option = random.choice(list(self.pnp_hl_actions)).value
      self.action_space = self.ll_action_space
      self.reset(options={'option': self.option})
    else:
      # Replace option with dummy value for high level training
      self.option = self.pnp_hl_actions_with_dummy.dummy.value
      self.action_space = self.hl_action_space
      self.reset()

    self.steps = 0

    # Set preferences
    self.pref_dict = pref_dict

    # Rewards initialization
    self.hl_pref_r = hl_pref_r

    self._per_step_reward = -0.01
    self._obj_drop_reward = 10.0
    self._obj_pick_reward = 10.0
    self._wrong_obj_pick_reward = -5.0
    self._dist_shaping_factor = -0.05
    self._ll_penalty = -1
    self._ll_radius = 1.5
    self._hl_diversity_reward = 5.0

    self.prev_option = self.pnp_hl_actions_with_dummy.dummy.value
    self.c_task_reward = 0
    self.c_pseudo_reward = 0
    self.c_gt_hl_pref = 0
    self.c_gt_ll_pref = 0

    # Used for determining successful placement into receptacle
    self._drop_success = False
    self._pick_apple_success = False
    self._pick_egg_success = False
