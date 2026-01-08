from math import sqrt
from heapq import heappush, heappop
from typing import Optional, Tuple, Dict, List

from HierRL.envs.ai2thor.pnp_env import ThorPickPlaceEnv
from HierRL.envs.ai2thor.pnp_training_utils import (PnP_LL_Actions,
                                                    PnP_HL_Actions)


class ThorPickPlacePlanner:

  def __init__(self,
               env: ThorPickPlaceEnv,
               use_pref: bool = False,
               stop_dist: float = 0.3):
    self.env = env
    self.stop_dist = stop_dist
    self.active = False
    self.task = None  # {"mode": "pick"|"drop", "obj_type": "Apple"|"Egg"}
    self.phase = None  # "acquire"|"deliver" or "done"
    self.target_obj_id = None
    self.cached_target_pos = None  # last chosen target xyz
    self._receptacle_types = set(self.env.receptacle_types)

    # Low-level action indices
    self._aidx = {a: i for i, a in enumerate(PnP_LL_Actions)}

    # ---- Path planning cache ----
    self._path: List[Tuple[float, float]] = []  # [(x,z), ...]
    self._path_i: int = 0
    self._last_goal_q: Optional[Tuple[float, float]] = None  # quantized (x,z)
    self._grid = getattr(self.env, "grid_size", 0.25)

    # Set preference
    self.use_pref = use_pref

  def start_option(self, option_idx: int):
    name = PnP_HL_Actions(option_idx).name
    if name == "pick_apple":
      self.task = {"mode": "pick", "obj_type": "Apple"}
    elif name == "pick_egg":
      self.task = {"mode": "pick", "obj_type": "Egg"}
    elif name == "drop_apple":
      self.task = {"mode": "drop", "obj_type": "Apple"}
    elif name == "drop_egg":
      self.task = {"mode": "drop", "obj_type": "Egg"}
    else:
      self.task = None

    self.phase = "acquire" if self.task and self.task["mode"] == "pick" else (
        "deliver" if self.task else "done")
    self.active = self.task is not None
    self.target_obj_id = None
    self.cached_target_pos = None

  def is_done(self) -> bool:
    return not self.active or self.phase == "done"

  def predict(self, obs, deterministic=True) -> Tuple[Optional[int], Dict]:
    """
    Returns (action_idx, info). If plan finished, returns:
    (None, {"ll_done": True})
    """
    if not self.active or self.phase == "done" or self.task is None:
      return None, {"ll_done": True}

    mode = self.task["mode"]
    obj_type = self.task["obj_type"]

    held = self._held_object()

    # ---- Task routing ----
    if mode == "pick":
      # If already holding target, we are done
      if held and held["objectType"] == obj_type:
        self.phase = "done"
        self.active = False
        return None, {"ll_done": True}

      # Otherwise seek the object and try to pick
      return self._policy_seek_and_pick(obj_type)

    elif mode == "drop":
      # If not holding the target object, we are done
      if (held is None) or (held["objectType"] != obj_type):
        self.phase = "done"
        self.active = False
        return None, {"ll_done": True}
        # return self._policy_seek_and_pick(obj_type)

      # Now deliver: move to nearest allowed receptacle and place
      return self._policy_seek_and_put()

    # Fallback
    self.phase = "done"
    self.active = False
    return None, {"ll_done": True}

  # ---- Subpolicies (emit ONE atomic action index each call) ----
  # ========== Subpolicies ==========
  def _policy_seek_and_pick(self, obj_type: str) -> Tuple[Optional[int], Dict]:
    '''
    Low-level policy to navigate toward and pick up an object of a given type.

    Args:
        obj_type (str): The type of object to pick (e.g., "Apple").

    Returns:
        (action_idx, info):
          - action_idx (Optional[int]): Index of the next atomic action
            to take (RotateLeft/Right, MoveAhead, etc.).
          - info (dict): {"ll_done": bool} flag indicating whether the
            low-level option has finished.
    '''
    # 1. Pick or refresh the nearest available target of this type.
    #    (Objects in "solved" receptacles are ignored.)
    target = self.env._nearest_object_anywhere(obj_type)
    if target is None:
      self.phase = "done"
      self.active = False
      return None, {"ll_done": True}

    self.cached_target_pos = target["position"]

    # 2. If agent is already close enough to target or has exhausted the path
    if self._dist_xz(self._agent_pos(),
                     self.cached_target_pos) <= self.stop_dist or (
                         self._path and self._path_i >= len(self._path) - 1):
      # First, check if orientation is correct. If not, issue a turn.
      turn = self._turn_action_towards(self.cached_target_pos)
      if turn is not None:
        return turn, {"ll_done": False}
      # Already aligned => issue pickup action
      self._clear_path()
      return self._aidx["PickupNearestTarget"], {"ll_done": True}

    # 3. Otherwise: still far => issue one navigation action
    #    (computed via A* pathfinding with obstacle avoidance)
    nav_action = self._next_nav_action_towards(self.cached_target_pos)
    return nav_action, {"ll_done": False}

  def _policy_seek_and_put(self) -> Tuple[Optional[int], Dict]:
    '''
    Low-level policy to navigate toward a receptacle and put the held object.

    Returns:
        (action_idx, info):
          - action_idx (Optional[int]): Index of the next atomic action
            (RotateLeft/Right, MoveAhead, PutHeldOnReceptacle, etc.).
          - info (dict): {"ll_done": bool} flag indicating whether the
            low-level option has finished.
    '''
    # 1. Find the nearest valid receptacle (anywhere in the scene).
    rec = self.env._nearest_receptacle_anywhere()
    self.cached_target_pos = rec["position"]

    # 2. If agent is already close enough to receptacle or has exhausted the
    # path
    if self._dist_xz(self._agent_pos(),
                     self.cached_target_pos) <= self.stop_dist or (
                         self._path and self._path_i >= len(self._path) - 1):
      # Turn to face receptacle first, if needed
      turn = self._turn_action_towards(self.cached_target_pos)
      if turn is not None:
        return turn, {"ll_done": False}
      # Already aligned => issue put action and terminate this option
      self.phase = "done"
      self.active = False
      self._clear_path()
      return self._aidx["PutHeldOnReceptacle"], {"ll_done": True}

    # 3. Otherwise: still far => issue one navigation step toward receptacle
    nav_action = self._next_nav_action_towards(self.cached_target_pos)
    return nav_action, {"ll_done": False}

  # ========== One-step navigator with obstacle avoidance ==========
  def _next_nav_action_towards(self, target_xyz: Dict[str, float]) -> int:
    '''
    Emit ONE low-level navigation action (rotate or move) toward a target point.

    Logic:
      1. Ensure an up-to-date A* path exists (recompute if target changed).
      2. If path finished => just align to the target and step forward.
      3. Otherwise, follow the next waypoint:
         - If agent is already "at" the waypoint (within a tolerance),
           advance to the next one.
         - If facing waypoint, MoveAhead.
         - Else, rotate toward waypoint.

    Args:
        target_xyz (dict): Desired target location ({"x", "z"}).

    Returns:
        int: Index of the chosen atomic action (RotateLeft/Right or MoveAhead).
    '''
    # 1. Rebuild path if no path yet OR if target cell changed significantly
    goal_q = self._q2(target_xyz["x"], target_xyz["z"])
    if (not self._path) or (self._last_goal_q != goal_q):
      objs_to_avoid = self._objs_to_avoid()
      print('Objects to avoid: ', objs_to_avoid)
      if self.use_pref:
        ok = self._plan_path_to(target_xyz,
                                obstacle_types=set(objs_to_avoid),
                                obstacle_margin=0.5)
      else:
        ok = self._plan_path_to(target_xyz)
      if not ok:
        # fallback: keep turning to search
        return self._aidx["RotateRight"]

    # 2. If path has been exhausted => directly align to goal & move
    if self._path_i >= len(self._path) - 1:
      turn = self._turn_action_towards(target_xyz)
      return turn if turn is not None else self._aidx["MoveAhead"]

    # 3. Get next waypoint along the path
    wx, wz = self._path[self._path_i + 1]

    # If agent is already very close to this waypoint => skip ahead
    axz = self._agent_pos()
    if self._dist_xz(axz, {"x": wx, "z": wz}) < (self._grid * 0.25):
      self._path_i += 1
      # If that was the last one, align to goal
      if self._path_i >= len(self._path) - 1:
        turn = self._turn_action_towards(target_xyz)
        return turn if turn is not None else self._aidx["MoveAhead"]
      wx, wz = self._path[self._path_i + 1]

    # 4. Either rotate toward waypoint or MoveAhead if already aligned
    turn = self._turn_action_towards({"x": wx, "z": wz})
    return turn if turn is not None else self._aidx["MoveAhead"]

  def _plan_path_to(self,
                    target_xyz: Dict[str, float],
                    obstacle_types=None,
                    obstacle_margin=0.02) -> bool:
    '''
    Compute an A* path from the agent's current position to the cell nearest
    the target, while pruning out cells that collide with inflated obstacle
    footprints.

    Args:
        target_xyz (dict): Desired target position.

    Returns:
        bool: True if a path was found, False otherwise.
    '''
    # 1. Get all reachable cells
    cells = self._reachable_cells()
    if not cells:
      self._clear_path()
      return False

    # 2. Collect obstacle objects (non-pickupable, non-receptacles)
    objs = self.env.controller.last_event.metadata["objects"]
    if obstacle_types is None:
      obstacles = []
    else:
      obstacles = [o for o in objs if o["objectType"] in obstacle_types]
    # Inflate each obstacle to a rectangle footprint in XZ
    footprints = [
        self._object_footprint_rect(o, margin=obstacle_margin)
        for o in obstacles
    ]

    def cell_ok(p) -> bool:
      '''
      Check if a cell is not inside any inflated obstacle footprint.
      '''
      x, z = p["x"], p["z"]
      half = self._grid / 2
      for (cx, cz, hx, hz) in footprints:
        if (abs(x - cx) <= (hx + half)) and (abs(z - cz) <= (hz + half)):
          return False
      return True

    # 3. Prune out blocked cells
    pruned = [p for p in cells if cell_ok(p)]
    if not pruned:
      self._clear_path()
      return False
    pruned_keys = {(round(p["x"], 3), round(p["z"], 3)) for p in pruned}

    # Quantization helper
    g = self._grid

    def q(x):
      return round(round(x / g) * g, 3)

    # 4. Define start cell (quantized to nearest pruned cell if needed)
    agent = self.env.controller.last_event.metadata["agent"]
    start = {"x": q(agent["position"]["x"]), "z": q(agent["position"]["z"])}
    if (start["x"], start["z"]) not in pruned_keys:
      # snap to closest pruned cell
      sc = min(pruned,
               key=lambda p: (p["x"] - agent["position"]["x"])**2 +
               (p["z"] - agent["position"]["z"])**2)
      start = {"x": round(sc["x"], 3), "z": round(sc["z"], 3)}

    # 5. Goal = reachable cell closest to target
    goal_cell = self._nearest_reachable_to(target_xyz, pruned)
    if goal_cell is None:
      self._clear_path()
      return False
    goal = {"x": round(goal_cell["x"], 3), "z": round(goal_cell["z"], 3)}

    # 6. Build 4-connected neighbor graph
    def neighbors(x, z):
      for dx, dz in ((g, 0), (-g, 0), (0, g), (0, -g)):
        nx, nz = round(x + dx, 3), round(z + dz, 3)
        if (nx, nz) in pruned_keys:
          yield (nx, nz), (int(dx / g), int(dz / g)
                           )  # also return the unit direction (+/-1, 0)

    def h(x, z):  # Euclidean heuristic
      return sqrt((x - goal["x"])**2 + (z - goal["z"])**2)

    # 7. Standard A* search
    # heap items: (f, g_here, (x,z), parent, dir_from_parent)
    # dir is a tuple in {(1,0), (-1,0), (0,1), (0,-1)} or None for start
    penalty_turn_any = g
    openq = []
    heappush(openq, (h(start["x"], start["z"]), 0.0,
                     (start["x"], start["z"]), None, None))
    came_from = {}  # node -> parent node
    came_dir = {}  # node -> direction (unit) used to reach 'node'
    gscore = {(start["x"], start["z"]): 0.0}

    while openq:
      _, g_here, node, parent, node_dir = heappop(openq)
      if node not in came_from:
        came_from[node] = parent
        came_dir[node] = node_dir
      if node == (goal["x"], goal["z"]):
        break
      x, z = node
      for (nbx, nbz), step_dir in neighbors(x, z):
        # base step cost (grid distance)
        step_cost = g

        # add a small turn penalty if changing direction
        prev_dir = came_dir.get(node, None)
        if prev_dir is not None and step_dir != prev_dir:
          # base penalty for any turn
          step_cost += penalty_turn_any

        tentative = g_here + step_cost
        nb = (nbx, nbz)
        if tentative < gscore.get(nb, float("inf")):
          gscore[nb] = tentative
          f = tentative + h(nbx, nbz)
          heappush(openq, (f, tentative, nb, node, step_dir))

    # 8. Reconstruct path (goal may not be reached; fallback to closest)
    cur = (goal["x"], goal["z"])
    if cur not in came_from:
      if not came_from:
        self._clear_path()
        return False
      # fallback: pick explored node closest to goal
      cur = min(gscore.keys(),
                key=lambda k: (k[0] - goal["x"])**2 + (k[1] - goal["z"])**2)
      if cur not in came_from:
        self._clear_path()
        return False

    # Reconstruct
    path = []
    while cur is not None:
      path.append(cur)
      cur = came_from.get(cur)
    path.reverse()

    # 9. Cache path for step-by-step execution
    self._path = path
    self._path_i = 0
    self._last_goal_q = self._q2(target_xyz["x"], target_xyz["z"])
    return True

  def _clear_path(self):
    '''
    Reset path-related state (e.g., after a failed planning attempt).
    '''
    self._path = []
    self._path_i = 0
    self._last_goal_q = None

  # ========== Read-only geometry/env helpers ==========
  def _reachable_cells(self):
    '''
    Query AI2-THOR for all grid cells that the agent can reach from its current
    location.

    Returns:
        list of dicts: Each dict is a cell with keys {"x", "y", "z"}.
    '''
    return self.env.controller.step(
        action="GetReachablePositions").metadata["actionReturn"]

  def _nearest_reachable_to(self, target_xyz, cells):
    '''
    Find the reachable cell closest to a target location.

    Args:
        target_xyz (dict): {"x", "z"} target position in world coordinates.
        cells (list): list of reachable cells (as returned by _reachable_cells).

    Returns:
        dict or None: the closest reachable cell to the target, or None if n
        cells.
    '''

    def d2(p):
      dx = p["x"] - target_xyz["x"]
      dz = p["z"] - target_xyz["z"]
      return dx * dx + dz * dz

    return min(cells, key=d2) if cells else None

  def _object_footprint_rect(self, o: Dict, margin: float = 0.02):
    '''
    Compute an axis-aligned footprint rectangle for an object on the floor
    plane (XZ).

    The footprint is described as a tuple (cx, cz, hx, hz):
      - center = (cx, cz)
      - half-widths = (hx, hz) along X and Z
      - margin inflates the footprint to give obstacles extra buffer

    Priority of sources:
      1. axisAlignedBoundingBox (most reliable).
      2. objectOrientedBoundingBox projected into XZ (if available).
      3. fallback small disk (~15 cm radius) around object position.

    Args:
        o (dict): object metadata from THOR.
        margin (float): safety padding in meters.

    Returns:
        (cx, cz, hx, hz): center and half-extents of object footprint.
    '''
    aabb = o.get("axisAlignedBoundingBox", None)
    if aabb and "center" in aabb and "size" in aabb:
      cx = float(aabb["center"]["x"])
      cz = float(aabb["center"]["z"])
      sx = float(aabb["size"]["x"])
      sz = float(aabb["size"]["z"])
      hx = max(0.0, sx * 0.5 + margin)
      hz = max(0.0, sz * 0.5 + margin)
      return (cx, cz, hx, hz)

    oobb = o.get("objectOrientedBoundingBox", None)
    if oobb and "center" in oobb and "size" in oobb:
      cx = float(oobb["center"]["x"])
      cz = float(oobb["center"]["z"])
      sx = float(oobb["size"]["x"])
      sz = float(oobb["size"]["z"])
      hx = max(0.0, sx * 0.5 + margin)
      hz = max(0.0, sz * 0.5 + margin)
      return (cx, cz, hx, hz)

    cx = float(o["position"]["x"])
    cz = float(o["position"]["z"])
    r = 0.15 + margin
    return (cx, cz, r, r)

  def _agent_pos(self) -> Dict[str, float]:
    '''
    Get the agent's current position (XZ plane only).
    '''
    a = self.env.controller.last_event.metadata["agent"]
    return {"x": a["position"]["x"], "z": a["position"]["z"]}

  def _agent_yaw(self) -> float:
    '''
    Get the agent's current facing direction (yaw angle in degrees).
    Normalized to [0, 360).
    '''
    return self.env.controller.last_event.metadata["agent"]["rotation"][
        "y"] % 360.0

  def _dist_xz(self, p, q) -> float:
    '''
    Euclidean distance in the ground plane (XZ).
    '''
    dx = p["x"] - q["x"]
    dz = p["z"] - q["z"]
    return sqrt(dx * dx + dz * dz)

  def _desired_yaw_cardinal(self, target_xyz: Dict[str, float]) -> int:
    '''
    Compute the cardinal yaw (0, 90, 180, 270) that most directly faces a
    target.

    AI2-THOR convention:
      - 0° yaw = facing +Z axis
      - 90° yaw = facing +X axis
      - 180° yaw = facing -Z axis
      - 270° yaw = facing -X axis

    Logic:
      - Compare absolute deltas in X and Z.
      - If Z dominates => face along Z axis (0 or 180).
      - If X dominates => face along X axis (90 or 270).
    '''
    a = self.env.controller.last_event.metadata["agent"]
    vx = target_xyz["x"] - a["position"]["x"]
    vz = target_xyz["z"] - a["position"]["z"]
    if abs(vz) >= abs(vx):
      return 0 if vz >= 0 else 180
    else:
      return 90 if vx >= 0 else 270

  def _turn_action_towards(self, target_xyz: Dict[str, float]) -> Optional[int]:
    '''
    Decide whether the agent should rotate left or right to face a target.

    - Uses the agent's current yaw and desired cardinal yaw.
    - If already aligned => return None (no turn needed).
    - Otherwise, choose the shorter rotation direction.
    - Returns the index of RotateLeft or RotateRight action.
    '''
    yaw = self._agent_yaw()
    desired = self._desired_yaw_cardinal(target_xyz)
    if yaw == desired:
      return None
    cw = (desired - yaw) % 360
    ccw = (yaw - desired) % 360
    return self._aidx["RotateRight"] if cw <= ccw else self._aidx["RotateLeft"]

  def _held_object(self):
    '''
    Return the currently held object (if any), from the agent's inventory.
    If inventory is empty, return None.
    '''
    inv = self.env.controller.last_event.metadata.get("inventoryObjects", [])
    return inv[0] if inv else None

  def _visible_objects(self):
    '''
    Return all objects that are currently marked as visible to the agent.
    '''
    return [
        o for o in self.env.controller.last_event.metadata["objects"]
        if o.get("visible")
    ]

  def _objs_to_avoid(self) -> List[str]:
    if self.task is None:
      return []

    # print('Current task: ', self.task)
    mode_name = self.task["mode"]
    obj_type_name = self.task["obj_type"][0].lower() + self.task["obj_type"][1:]
    cur_option = PnP_HL_Actions[f'{mode_name}_{obj_type_name}'].value
    return [k for k, vals in self.env.pref_dict.items() if cur_option in vals]

  # Quantization helpers
  def _q(self, x: float) -> float:
    '''
    Quantize a continuous coordinate to the nearest grid-aligned value.

    Steps:
      1. Divide the input coordinate x by the grid size g.
         - This converts the coordinate into "grid cell units".
      2. Round the result to the nearest integer cell index.
         - Effectively snaps the coordinate to the closest grid cell.
      3. Multiply the cell index back by g.
         - Converts the snapped index back into world units (meters).
      4. Round the final result to 3 decimal places.
         - Avoids floating-point drift when storing positions as keys.
         - Ensures consistent lookups in sets/dicts.
    '''
    g = self._grid
    return round(round(x / g) * g, 3)

  def _q2(self, x: float, z: float) -> Tuple[float, float]:
    '''
    Quantize a 2D coordinate (x, z) onto the navigation grid.
    '''
    return (self._q(x), self._q(z))
