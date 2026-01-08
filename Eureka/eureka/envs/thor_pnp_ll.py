import time
from copy import deepcopy
import random
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from typing import Dict, Any, Optional, Tuple

from HierRL.envs.ai2thor.pnp_utils import (_dist, _dist_xz,
                                           _move_object_to_point,
                                           _place_object_at_point,
                                           _disable_all_objects_of_type,
                                           _spawn_pickable_object_of_type,
                                           _spawn_points_above_receptacle)
from HierRL.envs.ai2thor.pnp_training_utils import (PnP_HL_Actions,
                                                    PnP_HL_Actions_With_Dummy,
                                                    PnP_LL_Actions)
from HierRL.envs.ai2thor.pnp_config import avoid_stool

from typing import Dict, List


class ThorPickPlaceEnvLL(gym.Env):
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

    self._per_step_reward_ll = -0.01
    self._per_step_reward_hl = -0.1
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

  # ---------- Core RL API ----------
  def reset(self,
            *,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None):
    super().reset(seed=seed)
    self.steps = 0
    self._drop_success = False
    self._pick_apple_success = False
    self._pick_egg_success = False
    self.option = self.pnp_hl_actions_with_dummy.dummy.value
    self.prev_option = self.pnp_hl_actions_with_dummy.dummy.value

    self.c_task_reward = 0
    self.c_pseudo_reward = 0
    self.c_gt_hl_pref = 0
    self.c_gt_ll_pref = 0

    if self.low_level:
      if options is None or 'option' not in options:
        self.option = random.choice(list(self.pnp_hl_actions)).value
      else:
        self.option = options['option']
      # print('Option for LL training after reset: ', self.option)
      ok = self.ll_reset(option=self.option, seed=seed)

      if not ok:
        # Hard reset environment (slower) if low level reset fails
        print("[ERROR] First reset failed, trying again from hard reset...")
        self._setup_env()
        ok = self.ll_reset(option=self.option, seed=seed)
        if not ok:
          raise Exception("Hard environment reset failed. Something is wrong!")

    else:
      ok = self.hl_reset(seed=seed)

      if not ok:
        # Hard reset environment (slower) if high level reset fails
        print("[ERROR] First reset failed, trying again from hard reset...")
        self._setup_env()
        ok = self.hl_reset(seed=seed)
        if not ok:
          raise Exception("Hard environment reset failed. Something is wrong!")

    obs = self._get_obs()
    info = {}
    return obs, info

  def ll_reset(self, option, seed: Optional[int] = None) -> bool:
    '''
    Perform this reset if we are only working with the low-level policy
    '''
    # print('Low-level reset with option', option)
    # If agent is holding something, drop it
    if self._held_object() is not None:
      sink = [
          obj for obj in self.controller.last_event.metadata["objects"]
          if obj["objectType"] == "SinkBasin"
      ][0]

      ev = self.controller.step(action="PutObject",
                                objectId=sink["objectId"],
                                forceAction=True,
                                placeStationary=True)

      ok = ev.metadata.get("lastActionSuccess", False)
      if not ok:
        print("Low level reset failed:", ev.metadata.get("errorMessage", ""))
        return False

    # Spawn agent at random position/orientation in the environment
    open_agent_waypoints = self.controller.step(
        action="GetReachablePositions").metadata["actionReturn"]
    spawnpoint = random.choice(open_agent_waypoints)
    agent_y_rot = random.choice([0, 90, 180, 270])
    self.controller.step(action="Teleport",
                         position=spawnpoint,
                         rotation=dict(x=0, y=agent_y_rot, z=0),
                         horizon=0,
                         standing=True)

    # Depending on pick option, put one of those objects on their hardcoded
    # spawnpoint:
    # Spawn two apples at random positions on dining table
    # apple = self._check_obj('Apple')
    src_height = 0.982677161693573  # apple["position"]["y"]

    apple_pos_1 = {"x": 0.45, "y": src_height, "z": 1.5}
    apple_pos_2 = {"x": 0.25, "y": src_height, "z": 1.5}
    egg_pos_1 = {"x": -0.1, "y": src_height, "z": 1.5}
    egg_pos_2 = {"x": 0.05, "y": src_height, "z": 1.5}

    apple_spawn_points = [apple_pos_1, apple_pos_2]
    egg_spawn_points = [egg_pos_1, egg_pos_2]

    apples = [
        obj for obj in self.controller.last_event.metadata["objects"]
        if obj["objectType"] == "Apple"
    ]

    eggs = [
        obj for obj in self.controller.last_event.metadata["objects"]
        if obj["objectType"] == "Egg"
    ]

    # Move all objects to their original spawnpoints
    poses = []
    for obj, sp in zip(apples + eggs, apple_spawn_points + egg_spawn_points):
      poses.append({
          "objectName": obj["name"],
          "position": sp,
          "rotation:": {
              "x": 0,
              "y": 0,
              "z": 0
          }
      })
    poses.extend(self.other_obj_poses)

    ev = self.controller.step(action="SetObjectPoses", objectPoses=poses)
    ok = ev.metadata.get("lastActionSuccess", False)
    if not ok:
      print("Low level reset failed:", ev.metadata.get("errorMessage", ""))
      return False

    apples = [
        obj for obj in self.controller.last_event.metadata["objects"]
        if obj["objectType"] == "Apple"
    ]

    eggs = [
        obj for obj in self.controller.last_event.metadata["objects"]
        if obj["objectType"] == "Egg"
    ]

    sink = [
        obj for obj in self.controller.last_event.metadata["objects"]
        if obj["objectType"] == "SinkBasin"
    ][0]

    if option == self.pnp_hl_actions.pick_apple.value:
      # Choose one of the apples to keep on the table
      idx = random.randint(0, len(apples) - 1)
      apples.pop(idx)
    elif option == self.pnp_hl_actions.pick_egg.value:
      # Choose one of the eggs to keep on the table
      idx = random.randint(0, len(eggs) - 1)
      eggs.pop(idx)
    elif (option != self.pnp_hl_actions.drop_apple.value
          and option != self.pnp_hl_actions.drop_egg.value):
      print('This option is not supported:', option)
      raise NotImplementedError

    all_objs = apples + eggs
    random.shuffle(all_objs)

    # Choose random number of objs to put in sink
    num_sink_objs = random.randint(0, len(all_objs))
    # spawn_points = _spawn_points_above_receptacle(self.controller,
    #                                               sink["objectId"])
    # random.shuffle(spawn_points)

    # for obj, point in zip(all_objs[:num_sink_objs],
    #                       spawn_points[:num_sink_objs]):
    #   t = obj["objectType"]
    #   ev = self.controller.step(action="TeleportObject",
    #                             objectId=obj["objectId"],
    #                             position=point,
    #                             rotation={
    #                                 'x': 0.0,
    #                                 'y': 0.0,
    #                                 'z': 0.0
    #                             },
    #                             forceAction=True)
    #   ok = ev.metadata.get("lastActionSuccess", False)
    #   if not ok:
    #     print("Low level reset failed:", ev.metadata.get("errorMessage", ""))
    #     return False

    # Pick and then place onto sink
    for obj in all_objs[:num_sink_objs]:
      ev = self.controller.step(action="PickupObject",
                                objectId=obj["objectId"],
                                forceAction=True,
                                manualInteract=False)

      ok = ev.metadata.get("lastActionSuccess", False)
      if not ok:
        print("Low level reset failed:", ev.metadata.get("errorMessage", ""))
        return False

      ev = self.controller.step(action="PutObject",
                                objectId=sink["objectId"],
                                forceAction=True,
                                placeStationary=True)

      ok = ev.metadata.get("lastActionSuccess", False)
      if not ok:
        print("Low level reset failed:", ev.metadata.get("errorMessage", ""))
        return False

    # If place option:
    # Put one of the objects (apples/eggs) in the agent's hand depending on option
    target_type = None
    if option == self.pnp_hl_actions.drop_apple.value:
      target_type = "Apple"
    elif option == self.pnp_hl_actions.drop_egg.value:
      target_type = "Egg"
    elif (option != self.pnp_hl_actions.pick_apple.value
          and option != self.pnp_hl_actions.pick_egg.value):
      print('This option is not supported: ', option)
      raise NotImplementedError

    if target_type is not None:
      objs = [
          obj for obj in self.controller.last_event.metadata["objects"]
          if obj["objectType"] == target_type
      ]
      obj_pick = random.choice(objs)
      ev = self.controller.step(action="PickupObject",
                                objectId=obj_pick["objectId"],
                                forceAction=True,
                                manualInteract=False)

      ok = ev.metadata.get("lastActionSuccess", False)
      if not ok:
        print("Low level reset failed:", ev.metadata.get("errorMessage", ""))
        return False

    return True

  def hl_reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
    '''
    Perform this reset if we are only working with the low-level policy
    '''
    # If agent is holding something, drop it
    if self._held_object() is not None:
      sink = [
          obj for obj in self.controller.last_event.metadata["objects"]
          if obj["objectType"] == "SinkBasin"
      ][0]

      ev = self.controller.step(action="PutObject",
                                objectId=sink["objectId"],
                                forceAction=True,
                                placeStationary=True)

      ok = ev.metadata.get("lastActionSuccess", False)
      if not ok:
        print("High level reset failed:", ev.metadata.get("errorMessage", ""))
        return False

    # Depending on pick option, put one of those objects on their hardcoded
    # spawnpoint:
    # Spawn two apples at random positions on dining table
    # apple = self._check_obj('Apple')
    src_height = 0.982677161693573  # apple["position"]["y"]

    apple_pos_1 = {"x": 0.45, "y": src_height, "z": 1.5}
    apple_pos_2 = {"x": 0.25, "y": src_height, "z": 1.5}
    egg_pos_1 = {"x": -0.1, "y": src_height, "z": 1.5}
    egg_pos_2 = {"x": 0.05, "y": src_height, "z": 1.5}

    apple_spawn_points = [apple_pos_1, apple_pos_2]
    egg_spawn_points = [egg_pos_1, egg_pos_2]

    apples = [
        obj for obj in self.controller.last_event.metadata["objects"]
        if obj["objectType"] == "Apple"
    ]

    eggs = [
        obj for obj in self.controller.last_event.metadata["objects"]
        if obj["objectType"] == "Egg"
    ]

    # Move all objects to their original spawnpoints
    poses = []
    for obj, sp in zip(apples + eggs, apple_spawn_points + egg_spawn_points):
      poses.append({
          "objectName": obj["name"],
          "position": sp,
          "rotation:": {
              "x": 0,
              "y": 0,
              "z": 0
          }
      })
    poses.extend(self.other_obj_poses)

    ev = self.controller.step(action="SetObjectPoses", objectPoses=poses)
    ok = ev.metadata.get("lastActionSuccess", False)
    if not ok:
      print("Low level reset failed:", ev.metadata.get("errorMessage", ""))
      return False

    # Spawn agent at random position/orientation in the environment
    open_agent_waypoints = self.controller.step(
        action="GetReachablePositions").metadata["actionReturn"]
    sample_waypoints = []
    for p in open_agent_waypoints:
      if p['x'] >= -0.25 and p['x'] <= 0.25 and p['z'] == 2.0:
        sample_waypoints.append(p)
    spawnpoint = random.choice(sample_waypoints)
    # print('Agent is spawning at spawnpoint: ', spawnpoint)
    # agent_y_rot = random.choice([0, 90, 180, 270])
    ev = self.controller.step(action="Teleport",
                              position=spawnpoint,
                              rotation=dict(x=0, y=180, z=0),
                              horizon=0,
                              standing=True)
    ok = ev.metadata.get("lastActionSuccess", False)
    if not ok:
      print("Low level reset failed:", ev.metadata.get("errorMessage", ""))
      return False

    return True

  def step(self, ll_action: int, hl_action: int):
    self.option = hl_action
    self.action = ll_action
    self.old_state = deepcopy(self._get_obs())
    self.steps += 1

    action = self.pnp_ll_actions[ll_action]
    # print('Action in env: ', action)

    if action in {"PickupNearestTarget", "PutHeldOnReceptacle", "DropHeld"}:
      self._do_high_level(action)
    else:
      self._do_low_level(action)

    obs = self._get_obs()
    info = {
        "lastActionSuccess":
        self.controller.last_event.metadata.get("lastActionSuccess", False),
        "errorMessage":
        self.controller.last_event.metadata.get("errorMessage", ""),
    }

    if not self.low_level:
      info['ll_done'] = self._compute_subtask_done()

    # Reward: +1 when target ends up on desired receptacle
    # (simple success condition)
    reward, terminated = self._compute_reward_and_done()
    truncated = self.steps >= self.max_steps

    (task_reward, pseudo_reward, ll_pref_reward, hl_pref_reward,
     gt_ll_pref_reward, gt_hl_pref_reward) = reward

    self.done = terminated
    self.c_task_reward += task_reward
    self.c_pseudo_reward += pseudo_reward
    self.c_gt_hl_pref += gt_hl_pref_reward
    self.c_gt_ll_pref += gt_ll_pref_reward
    info['c_task_reward'] = self.c_task_reward
    info['c_pseudo_reward'] = self.c_pseudo_reward
    info['c_gt_hl_pref'] = self.c_gt_hl_pref
    info['c_gt_ll_pref'] = self.c_gt_ll_pref
    self.prev_option = hl_action

    if self.need_render:
      time.sleep(0.2)
    return obs, (task_reward, pseudo_reward, ll_pref_reward,
                 hl_pref_reward), terminated, truncated, info

  def render(self):
    return self._get_obs()

  def close(self):
    self.controller.stop()

  # ---------- Helpers ----------
  def _setup_env(self):
    self.controller.reset(scene=self.scene)

    # Move stool
    stool = self._check_obj('Stool')
    pos = stool["position"]  # dict with "x","y","z"
    new_pos = {"x": pos["x"] - 0.7, "y": pos["y"], "z": pos["z"] - 0.1}
    _move_object_to_point(self.controller, stool, new_pos)

    # Disable objects
    _disable_all_objects_of_type(self.controller, 'Pot')
    _disable_all_objects_of_type(self.controller, 'Ladle')
    _disable_all_objects_of_type(self.controller, 'Mug')

    # Spawn two apples at fixed positions
    apple = self._check_obj('Apple')
    src_height = apple["position"]["y"]
    apple_pos_1 = {"x": 0.45, "y": src_height, "z": 1.5}
    apple_pos_2 = {"x": 0.25, "y": src_height, "z": 1.5}
    apple_pos_list = [apple_pos_1, apple_pos_2]
    _spawn_pickable_object_of_type(self.controller,
                                   'Apple',
                                   num_to_spawn=2,
                                   spawn_points=apple_pos_list)

    # Spawn two eggs at fixed positions
    self._check_obj('Egg')
    egg_pos_1 = {"x": -0.1, "y": src_height, "z": 1.5}
    egg_pos_2 = {"x": 0.05, "y": src_height, "z": 1.5}
    egg_pos_list = [egg_pos_1, egg_pos_2]
    _spawn_pickable_object_of_type(self.controller,
                                   'Egg',
                                   num_to_spawn=2,
                                   spawn_points=egg_pos_list)

    # Save the poses of objects other than apples and eggs
    other_objs = [
        obj for obj in self.controller.last_event.metadata["objects"]
        if (obj.get("pickupable") or obj.get("moveable"))
        and obj["objectType"] not in ("Apple", "Egg")
    ]

    self.other_obj_poses = []
    for obj in other_objs:
      if not obj.get("pickupable"):
        self.other_obj_poses.append({
            "objectName": obj["name"],
            "position": obj["position"],
            "rotation": obj["rotation"]
        })

  def _get_obs(self) -> np.ndarray:
    """
    Dictionary of
      object_pos: x and z coordinates of target objs, receptacle objs,
                  and stool concatenated in one array
      agent_pos: x and z coordinates of the agent
      agent_rot: y rotation of the agent (given in degrees)
    """
    # frame = self.controller.last_event.frame
    obj_data = self.controller.last_event.metadata["objects"]
    agent_data = self.controller.last_event.metadata["agent"]

    # Collect positions in flat lists first.
    apple_pos, egg_pos, stool_pos, sink_pos = [], [], [], []
    apple_states = []
    egg_states = []

    for obj in obj_data:
      t = obj["objectType"]
      if t == "Apple":
        apple_pos.extend([obj["position"]["x"], obj["position"]["z"]])
        apple_states.append(self._compute_obj_state(obj))
      elif t == "Egg":
        egg_pos.extend([obj["position"]["x"], obj["position"]["z"]])
        egg_states.append(self._compute_obj_state(obj))
      elif t == "Stool":
        stool_pos = [obj["position"]["x"], obj["position"]["z"]]
      elif t == "SinkBasin":
        sink_pos = [obj["position"]["x"], obj["position"]["z"]]

    obs = {
        "apple_1_pos":
        np.array(apple_pos[:2], dtype=np.float32),
        "apple_2_pos":
        np.array(apple_pos[2:], dtype=np.float32),
        "egg_1_pos":
        np.array(egg_pos[:2], dtype=np.float32),
        "egg_2_pos":
        np.array(egg_pos[2:], dtype=np.float32),
        "stool_pos":
        np.array(stool_pos, dtype=np.float32),
        "sink_pos":
        np.array(sink_pos, dtype=np.float32),
        "agent_pos":
        np.array([agent_data["position"]["x"], agent_data["position"]["z"]],
                 dtype=np.float32),
        "agent_rot":
        np.zeros(4, dtype=np.float32),
        "apple_1_state":
        np.array(apple_states[0], dtype=np.uint32),
        "apple_2_state":
        np.array(apple_states[1], dtype=np.uint32),
        "egg_1_state":
        np.array(egg_states[0], dtype=np.uint32),
        "egg_2_state":
        np.array(egg_states[1], dtype=np.uint32),
    }

    # One-hot encode y-rotation (assuming multiples of 90°)
    rot_idx = int(round(agent_data["rotation"]["y"] / 90.0)) % 4
    obs["agent_rot"][rot_idx] = 1.0

    return obs

  def _compute_obj_state(self, obj: Dict) -> int:
    '''
    Compute the object's state:
    0 = on the dining table
    1 = held in hand
    2 = in the receptacle
    '''
    # Check if the object is being held
    if obj.get("isPickedUp", False):
      return 1

    # Next, check if the object is in the receptacle
    prs = obj.get("parentReceptacles")
    recep = prs[0] if prs else None
    if recep:
      if self._get_receptacle_type(recep) in self.receptacle_types:
        return 2

    # Otherwise, assume it's on the table
    return 0

  def _do_low_level(self, action: str):
    """
    Forward a low-level action string (e.g., 'MoveAhead') to THOR.
    """
    self.controller.step(action=action)

  def _do_high_level(self, action: str):
    """
    High-level actions implemented in terms of low-level THOR actions.
    """
    if action == "PickupNearestTarget":
      self._pickup_nearest_target()
    elif action == "PutHeldOnReceptacle":
      self._put_on_receptacle()
    elif action == "DropHeld":
      self.controller.step(action="DropHandObject")

  def _visible_objects(self):
    return [
        o for o in self.controller.last_event.metadata["objects"]
        if o.get("visible")
    ]

  def _held_object(self) -> Optional[Dict]:
    """
    Return the single held item if any, else None.
    """
    inv = self.controller.last_event.metadata["inventoryObjects"]
    return inv[0] if inv else None

  def _nearest_object_of_types(self, types):
    """
    Among all objects, pick the nearest whose objectType is in `types`.
    Uses Euclidean distance via your _dist helper (origin = agent).
    """
    objs = [
        obj for obj in self.controller.last_event.metadata["objects"]
        if obj["objectType"] in types
    ]
    # print('Target types: ', types)
    if not objs:
      return None

    origin = self.controller.last_event.metadata["agent"]
    return min(objs, key=lambda o: _dist(o, origin))

  def _nearest_receptacle_anywhere(self):
    '''
    Find the nearest receptacle of an allowed type, regardless of visibility.

    - Candidate objects = all scene objects whose type is in
      self._receptacle_types.
    - Distance metric = squared Euclidean in XZ plane to the agent.
    - Returns the closest such object, or None if no candidates exist.
    '''
    objs = self.controller.last_event.metadata["objects"]
    cands = [o for o in objs if o["objectType"] in self.receptacle_types]
    if not cands:
      return None
    agent = self.controller.last_event.metadata["agent"]

    def d2(o):
      dx = o["position"]["x"] - agent["position"]["x"]
      dz = o["position"]["z"] - agent["position"]["z"]
      return dx * dx + dz * dz

    return min(cands, key=d2)

  def _get_receptacle_type(self, rec_id: str) -> str:
    '''
    Extract the receptacle's object type from a THOR receptacle identifier.

    Args:
        rec_id (str): A receptacle ID string from THOR metadata.
                      These IDs often look like:
                        "Sink|-00.11|+00.89|-02.01|SinkBasin"
                        "DiningTable|+00.17|+00.01|+00.68"

    Behavior:
        - THOR object/receptacle IDs are composed of:
            <BaseName>|<x>|<y>|<z>|<SubPartName?>
        - Example:
            "Sink|...|SinkBasin" => Base = "Sink", SubPart = "SinkBasin"
            "DiningTable|..."    => Base = "DiningTable" (no subpart)
        - The final component may be a sub-receptacle (like "SinkBasin"),
          which is alphabetic. If so, we return that as the receptacle type.
        - Otherwise, we fall back to the first component (the base object type).

    Returns:
        str: The receptacle type, e.g. "SinkBasin" or "DiningTable".
    '''
    parts = rec_id.split("|")
    return parts[-1] if parts[-1].isalpha() else parts[0]

  def _nearest_object_anywhere(self, obj_type: str):
    '''
    Find the nearest object of a given type (e.g., 'Apple', 'Egg') that is not
    already in a target receptacle in the scene, regardless of visibility.

    Args:
        obj_type (str): The object type string to search for.

    Returns:
        dict or None: The metadata entry of the nearest object, or None if none
        found.
    '''
    # Gather all objects of the requested type from the scene
    objs = [
        o for o in self.controller.last_event.metadata["objects"]
        if o["objectType"] == obj_type
    ]
    if not objs:
      return None

    # Filter out objects that are already inside/on allowed receptacles
    filtered = []
    for o in objs:
      prs = o.get("parentReceptacles")
      if prs:
        rec_types = [self._get_receptacle_type(r) for r in prs]
        if any(rt in self.receptacle_types for rt in rec_types):
          continue
      filtered.append(o)
    if not filtered:
      return None

    # Get the agent position for distance calculation
    agent = self.controller.last_event.metadata["agent"]

    def d2(o):
      dx = o["position"]["x"] - agent["position"]["x"]
      dz = o["position"]["z"] - agent["position"]["z"]
      return dx * dx + dz * dz

    # Return the object with the smallest distance
    return min(filtered, key=d2)

  def _nearest_visible_of_types(self, types):
    """
    Among currently visible objects, pick the nearest whose objectType is
    in `types`.
    Uses Euclidean distance via your _dist helper (origin = agent).
    """
    vis = self._visible_objects()
    # print('Visible objects: ', [o['objectType'] for o in vis])
    candidates = [o for o in vis if o["objectType"] in types]
    # print('Target types: ', types)
    if not candidates:
      return None

    origin = self.controller.last_event.metadata["agent"]
    return min(candidates, key=lambda o: _dist(o, origin))

  def _pickup_nearest_target(self):
    """
    If not holding anything, attempt to pick up the nearest visible target type.
    """
    # print('Attempting to pick up nearest target')
    if self._held_object() is not None:
      # print('Already holding an object: ', self._held_object()['objectType'])
      return  # already holding
    obj = self._nearest_object_of_types(self.target_types)
    # print('Nearest target: ', obj['objectType'] if obj else None)

    # Check that object is not already in target receptacle
    if obj["parentReceptacles"] is not None and len(
        obj["parentReceptacles"]) > 0:
      parent_type = obj["parentReceptacles"][0].split("|")[-1]

      # If object already in target receptacle, don't pick up
      if parent_type in self.receptacle_types:
        return

    def _set_pick_success(ev):
      if ev.metadata['lastActionSuccess']:
        held = self._held_object()
        if held:
          if held['objectType'] == 'Apple':
            # print('Set pick apple success to True...')
            self._pick_apple_success = True
          elif held['objectType'] == 'Egg':
            # print('Set pick egg success to True...')
            self._pick_egg_success = True

    if obj['visible']:
      ev = self.controller.step(action="PickupObject", objectId=obj["objectId"])
      _set_pick_success(ev)
      return

    ACTIONS_2_ANTIACTIONS = {
        "MoveAhead": "MoveBack",
        "MoveBack": "MoveAhead",
        "MoveLeft": "MoveRight",
        "MoveRight": "MoveLeft",
    }

    # Try each move direction to see if a slightly different pose helps
    for action, anti_action in ACTIONS_2_ANTIACTIONS.items():
      ev = self.controller.step(action=action)
      if not ev.metadata["lastActionSuccess"]:
        continue  # skip if the move itself failed
      obj = self._get_object_by_id(obj['objectId'])
      if not obj['visible']:
        self.controller.step(action=anti_action)
      else:
        ev = self.controller.step(action="PickupObject",
                                  objectId=obj["objectId"])
        _set_pick_success(ev)
        self.controller.step(action=anti_action)
        break

    # If after all retries PutObject still fails, log the error message
    if not ev.metadata["lastActionSuccess"]:
      print('Pick object failed: ', ev.metadata.get("errorMessage", ""))

  def _put_on_receptacle(self):
    """
    Try to place the currently held object onto the nearest visible receptacle.

    Logic:
      1. If no object is held, return immediately.
      2. Find the nearest visible receptacle of allowed types.
         - If none are visible, drop the held object on the ground.
      3. If the receptacle is openable and closed (drawer, fridge, etc.), open it.
      4. Attempt to PutObject into/on the receptacle.
      5. If the PutObject fails:
         - As a simple recovery strategy, move the agent around
           (try each of the 4 cardinal moves).
         - After each move, recompute the nearest receptacle and retry PutObject.
         - If a retry succeeds, undo the move with the opposite action
           (so the agent ends up back where it started).
      6. If all retries fail, print the error message from THOR.
    """
    held = self._held_object()
    # print('Held object: ', held['objectType'] if held else None)
    if held is None:
      return

    rec = self._nearest_visible_of_types(self.receptacle_types)
    # print('Nearest receptacle: ', rec['objectType'] if rec else None)
    # If no available receptacle, just leave env unchanged
    if rec is None:
      # if nothing visible, just drop
      # self.controller.step(action="DropHandObject")
      return

    # Open if needed (drawers, fridge, etc.)
    if rec.get("openable") and not rec.get("isOpen"):
      self.controller.step(action="OpenObject", objectId=rec["objectId"])

    # First attempt to put the held object into/on receptacle
    ev = self.controller.step(action="PutObject",
                              objectId=rec["objectId"],
                              forceAction=True,
                              placeStationary=True)

    # Second attempt: Try to turn towards the receptacle and try again
    if not ev.metadata["lastActionSuccess"]:
      agent = self.controller.last_event.metadata["agent"]
      yaw = np.degrees(
          np.arctan2(rec["position"]["x"] - agent["position"]["x"],
                     rec["position"]["z"] - agent["position"]["z"]))
      yaw = (yaw + 360) % 360  # convert to [0, 360]
      curr_yaw = agent["rotation"]["y"] % 360
      yaw_diff = (yaw - curr_yaw + 540) % 360 - 180  # in [-180, 180]
      turn = yaw_diff / 90.0
      if turn < -0.4:
        self.controller.step(action="RotateLeft")
      elif turn > 0.4:
        self.controller.step(action="RotateRight")
      ev = self.controller.step(action="PutObject", objectId=rec["objectId"])

    # If first attempt failed, try to “wiggle” the agent around
    if not ev.metadata["lastActionSuccess"]:
      ACTIONS_2_ANTIACTIONS = {
          "MoveAhead": "MoveBack",
          "MoveBack": "MoveAhead",
          "MoveLeft": "MoveRight",
          "MoveRight": "MoveLeft",
      }

      # Try each move direction to see if a slightly different pose helps
      for action, anti_action in ACTIONS_2_ANTIACTIONS.items():
        ev = self.controller.step(action=action)
        if not ev.metadata["lastActionSuccess"]:
          continue  # skip if the move itself failed
        rec = self._nearest_visible_of_types(self.receptacle_types)
        if rec is None:
          self.controller.step(action=anti_action)
          continue
        ev = self.controller.step(action="PutObject", objectId=rec["objectId"])
        if ev.metadata["lastActionSuccess"]:
          self.controller.step(action=anti_action)
          break
        self.controller.step(action=anti_action)

      # If after all retries PutObject still fails, log the error message
      if not ev.metadata["lastActionSuccess"]:
        # print('PutObject failed: ', ev.metadata.get("errorMessage", ""))
        pass
      else:
        self._drop_success = True
    else:
      self._drop_success = True

    return self._drop_success

  def _get_object_by_id(self, oid: str) -> Optional[Dict]:
    """
    Return the object metadata dict for the given object_id,
    or None if not found.
    """
    for o in self.controller.last_event.metadata["objects"]:
      if o["objectId"] == oid:
        return o
    return None

  def _check_obj(self, object_type):
    """
    Ensure exactly one object of the given type is present; return it.
    """
    objects = [
        o for o in self.controller.last_event.metadata["objects"]
        if o["objectType"] == object_type
    ]
    assert len(objects) == 1, f'Expected exactly one {object_type} in the scene'
    return objects[0]

  def _get_obs_after_step(self, state, action):
    '''
    Compute the observation resulting from applying an action to a given state.
    You can call this helper method in your code by doing env._get_obs_after_step(...).

    Parameters
    ----------
    state : Dict
        The current environment state before taking the action.
    action : int
        The (low-level) action to be applied to the state.

    Returns
    -------
    Dict
        The resulting observation after the action has been applied.
    '''
    return self._get_obs()

  def _compute_ll_reward_and_done(self) -> Tuple[float, bool]:
    # Each step has negative reward
    pseudo_reward = self._per_step_reward_ll
    done = False
    held_object = self._held_object()
    if held_object is not None:
      held_object = held_object["objectType"]

    if self.option == self.pnp_hl_actions.pick_apple.value:
      # Done + pos reward if successful pick and apple in hand
      if held_object == "Apple":
        pseudo_reward = self._obj_pick_reward
        done = True
      # Negative reward (and done) if pick up egg
      elif held_object == "Egg":
        pseudo_reward = self._wrong_obj_pick_reward
        done = True
      else:
        nearest_obj = self._nearest_object_anywhere('Apple')
        agent = self.controller.last_event.metadata["agent"]
        pseudo_reward += self._dist_shaping_factor * _dist_xz(
            agent, nearest_obj)

    elif self.option == self.pnp_hl_actions.pick_egg.value:
      # Done + pos reward if successful pick and egg in hand
      if held_object == "Egg":
        pseudo_reward = self._obj_pick_reward
        done = True
      # Negative reward (and done) if pick up apple
      elif held_object == "Apple":
        pseudo_reward = self._wrong_obj_pick_reward
        done = True
      else:
        nearest_obj = self._nearest_object_anywhere('Egg')
        agent = self.controller.last_event.metadata["agent"]
        pseudo_reward += self._dist_shaping_factor * _dist_xz(
            agent, nearest_obj)

    elif (self.option == self.pnp_hl_actions.drop_apple.value
          or self.option == self.pnp_hl_actions.drop_egg.value):
      # Done + pos reward if successful drop
      if self._drop_success:
        pseudo_reward = self._obj_drop_reward
        done = True
        self._drop_success = False
      else:
        nearest_obj = self._nearest_receptacle_anywhere()
        agent = self.controller.last_event.metadata["agent"]
        pseudo_reward += self._dist_shaping_factor * _dist_xz(
            agent, nearest_obj)

    else:
      print('Using option: ', self.option)
      raise NotImplementedError

    # print('pseudo reward: ', pseudo_reward)
    return pseudo_reward, done

  def _compute_subtask_done(self) -> bool:
    '''
    Check if the current subtask/option is done.

    Logic:
    - For pick options: done if the correct object is in hand, or if the wrong
      object is in hand (failed pick).
    - For drop options: done if the held object was just dropped successfully.
    '''
    done = False
    held_object = self._held_object()
    if held_object is not None:
      held_object = held_object["objectType"]

    if self.option == self.pnp_hl_actions.pick_apple.value:
      if self._pick_apple_success:
        done = True

    elif self.option == self.pnp_hl_actions.pick_egg.value:
      if self._pick_egg_success:
        done = True

    elif (self.option == self.pnp_hl_actions.drop_apple.value
          or self.option == self.pnp_hl_actions.drop_egg.value):
      if self._drop_success:
        done = True

    else:
      print('Using option: ', self.option)
      raise NotImplementedError

    return done

  def _compute_reward_and_done(self) -> Tuple[float, bool]:
    """
    Compute the reward signal and termination condition for the environment.

    - Task reward: sparse reward when objects are placed correctly, small
                   penalty each step.
    - Pseudo reward: reward for the low-level policy.
    - Preference rewards: extra signals based on preference functions (LL/HL).
    - done: True if all target objects are in one of the allowed receptacles.
    """
    task_reward = 0.0
    pseudo_reward = 0.0
    done = True

    if self.low_level:
      # Delegate to the LL reward function if we are running in low-level mode
      pseudo_reward, done = self._compute_ll_reward_and_done()
    else:
      # ----------- Check completion (done condition) -----------
      # For done: look at all target objects and check if they are in a
      # receptacle
      obj_data = self.controller.last_event.metadata["objects"]
      for obj in obj_data:
        if obj["objectType"] in self.target_types:
          obj_receptacles = obj["parentReceptacles"]
          if not (obj_receptacles is not None and
                  obj_receptacles[0].split("|")[-1] in self.receptacle_types):
            done = False

      # ----------- Task reward -----------
      # For reward: Each step has negative reward
      task_reward = self._per_step_reward_hl
      if self._drop_success:
        # Add a positive reward if an object was just put into a receptacle
        task_reward = self._obj_drop_reward

      # Reset _drop_success flag since we only want to check the immediate step after the drop
      self._drop_success = False
      self._pick_apple_success = False
      self._pick_egg_success = False

    # ----------- Preference-based rewards -----------
    ll_pref_reward = 0.0
    gt_ll_pref_reward = 0.0
    hl_pref_reward = 0.0
    gt_hl_pref_reward = 0.0

    if 'LLGPT' in self.__class__.__name__:
      ll_pref_reward += self.get_low_level_pref_gpt(self.old_state, self.option,
                                                    self.action)
      gt_ll_pref_reward += self._get_low_level_pref(self.old_state, self.option,
                                                    self.action)
    elif 'FlatSAGPT' in self.__class__.__name__:
      ll_pref_reward += self.get_flat_sa_pref_gpt(self.old_state, self.action)
      gt_ll_pref_reward += self._get_low_level_pref(self.old_state, self.option,
                                                    self.action)
    else:
      if self.hl_pref_r:
        ll_pref_reward += self._get_low_level_pref(self.old_state, self.option,
                                                   self.action)
        gt_ll_pref_reward += ll_pref_reward
      else:
        # print('Getting flat sa pref...')
        ll_pref_reward += self._get_flatsa_pref(self.old_state, self.action)
        gt_ll_pref_reward += self._get_low_level_pref(self.old_state,
                                                      self.option, self.action)

    if 'HLGPT' in self.__class__.__name__:
      hl_pref_reward += self.get_high_level_pref_gpt(self.old_state,
                                                     self.prev_option,
                                                     self.option)
      gt_hl_pref_reward += self._get_high_level_pref(self.old_state,
                                                     self.prev_option,
                                                     self.option)
    elif 'FlatSAGPT' in self.__class__.__name__:
      hl_pref_reward += self.get_flat_sa_pref_gpt(self.old_state, self.action)
      gt_hl_pref_reward += self._get_high_level_pref(self.old_state,
                                                     self.prev_option,
                                                     self.option)
    else:
      if self.hl_pref_r:
        hl_pref_reward += self._get_high_level_pref(self.old_state,
                                                    self.prev_option,
                                                    self.option)
        gt_hl_pref_reward += hl_pref_reward
      else:
        hl_pref_reward += self._get_flatsa_pref(self.old_state, self.action)
        gt_hl_pref_reward += self._get_high_level_pref(self.old_state,
                                                       self.prev_option,
                                                       self.option)

    rewards = (task_reward, pseudo_reward, ll_pref_reward, hl_pref_reward,
               gt_ll_pref_reward, gt_hl_pref_reward)
    # print('ll pref reward: ', ll_pref_reward)
    return rewards, done

  def _get_low_level_pref(self, state, option, action):
    """
    Hand-crafted low-level preference function.

    Logic:
    - For each object to avoid (from pref_dict), check if the current option
      is affected by it.
    - Simulate the next state after taking the given action.
    - Measure agent distance to the avoid-object.
    - If within a certain radius, return a penalty; else 0.
    """
    for obj_to_avoid in self.pref_dict.keys():
      # If this option is not constrained by the avoid-object, skip
      if (option not in self.pref_dict[obj_to_avoid]):
        return 0.0

      # Predict the new agent state after action
      new_obs = self._get_obs_after_step(state, action)
      agent_pos = new_obs['agent_pos']

      # Find the environment position of the avoid-object
      # stool_pos = new_obs['stool_pos']
      env_obj = self._nearest_object_anywhere(obj_to_avoid)
      env_obj_pos = (env_obj['position']['x'], env_obj['position']['z'])

      # Distance from agent to avoid-object
      dist_to_obj = ((agent_pos[0] - env_obj_pos[0])**2 +
                     (agent_pos[1] - env_obj_pos[1])**2)**0.5
      # print('Dist to stool: ', dist_to_obj)

      # Penalize if too close
      if dist_to_obj < self._ll_radius:
        return self._ll_penalty
      else:
        return 0.0

  def _get_flatsa_pref(self, state, action):
    """
    Hand-crafted flat state-action preference function.

    Logic:
    - Similar to _get_low_level_pref, but does not condition on options.
    - For each object to avoid, simulate the agent's next position after action.
    - Penalize if agent comes within a given radius of the avoid-object.
    """
    for obj_to_avoid in self.pref_dict.keys():
      # Predict the new agent state after action
      new_obs = self._get_obs_after_step(state, action)
      agent_pos = new_obs['agent_pos']

      # Find the environment position of the avoid-object
      # stool_pos = new_obs['stool_pos']
      env_obj = self._nearest_object_anywhere(obj_to_avoid)
      env_obj_pos = (env_obj['position']['x'], env_obj['position']['z'])

      # Distance from agent to avoid-object
      dist_to_obj = ((agent_pos[0] - env_obj_pos[0])**2 +
                     (agent_pos[1] - env_obj_pos[1])**2)**0.5
      # print('Dist to stool: ', dist_to_obj)

      # Penalize if too close
      if dist_to_obj < self._ll_radius:
        return self._ll_penalty
      else:
        return 0.0

  def _get_high_level_pref(self, state, prev_option, option):
    # If the agent just placed an apple, the agent is rewarded to pick up an
    # egg, if there are if there are still eggs left.
    hl_pref_reward = 0.0
    if (prev_option == self.pnp_hl_actions.drop_apple.value
        and option == self.pnp_hl_actions.pick_egg.value):
      if state["egg_1_state"] == 0 or state["egg_2_state"] == 0:
        hl_pref_reward += self._hl_diversity_reward
    # If the agent just placed an egg, the agent is rewarded to pick up an
    # apple, if there are if there are still apples left.
    if (prev_option == self.pnp_hl_actions.drop_egg.value
        and option == self.pnp_hl_actions.pick_apple.value):
      if state["apple_1_state"] == 0 or state["apple_2_state"] == 0:
        hl_pref_reward += self._hl_diversity_reward
    # if hl_pref_reward != 0:
    #   print('Prev option: ', prev_option)
    #   print('Option: ', option)
    #   print('HL pref reward: ', hl_pref_reward)
    return hl_pref_reward

  def get_high_level_pref_gpt(self, state, prev_option, option):
    pass

  def get_low_level_pref_gpt(self, state, option, action):
    pass

  def get_flat_sa_pref_gpt(self, state, action):
    pass
