import random
from copy import deepcopy


def _dist(item1, item2):
  pos1 = item1["position"]
  pos2 = item2["position"]
  return (pos1["x"] - pos2["x"])**2 + (pos1["y"] - pos2["y"])**2 + (
      pos1["z"] - pos2["z"])**2


def _dist_xz(item1, item2):
  pos1 = item1["position"]
  pos2 = item2["position"]
  return (pos1["x"] - pos2["x"])**2 + (pos1["z"] - pos2["z"])**2


def _get_obj_by_id(controller, oid):
  for o in controller.last_event.metadata["objects"]:
    if o["objectId"] == oid:
      return o
  return None


def _find_supporting_counter(controller, obj):
  """Return the receptacle object (CounterTop/Table-like) that supports the object.
        If none in the chain, fall back to a random receptacle."""
  RECEPTACLE_TYPES = {
      "CounterTop",
      "Table",
      "DiningTable",
      "CoffeeTable",  # "SinkBasin"
  }
  parents = obj.get("parentReceptacles") or []
  for rid in parents:
    par = _get_obj_by_id(controller, rid)
    if par and par.get("receptacle") and par["objectType"] in RECEPTACLE_TYPES:
      return par
  # Fallback: Random supporting counter in scene
  objs = controller.last_event.metadata["objects"]
  counters = [o for o in objs
              if o["objectType"] == "SinkBasin"]  # in RECEPTACLE_TYPES]
  if not counters:
    return None
  # print("found", [o["objectType"] for o in counters])
  counter = random.choice(counters)
  # print("placing on", counter["objectType"])
  return counter
  # # nearest by Euclidean distance to the object
  # ax, ay, az = obj["position"]["x"], obj["position"]["y"], obj["position"]["z"]

  # def d2(o):
  #   p = o["position"]
  #   return (p["x"] - ax)**2 + (p["y"] - ay)**2 + (p["z"] - az)**2


def _spawn_points_above_receptacle(controller, receptacle_id):
  ev = controller.step(action="GetSpawnCoordinatesAboveReceptacle",
                       objectId=receptacle_id,
                       anywhere=True)
  pts = ev.metadata.get("actionReturn") or []
  return pts  # list of {"x","y","z"} dicts


def _place_object_at_point(controller,
                           object_id,
                           point,
                           use_no_rot=True,
                           rot=None,
                           use_physics=True):
  """
  Try PlaceObjectAtPoint; fall back to TeleportObject + physics settle.
  """
  if use_no_rot:
    # No rotation
    rot = {"x": 0.0, "y": 0.0, "z": 0.0}
  else:
    if rot:
      # Try to use the provided rotation first
      rot = rot
    else:
      # Keep current rotation if we teleport
      cur = next(o for o in controller.last_event.metadata["objects"]
                 if o["objectId"] == object_id)
      rot = cur["rotation"]
    # Prefer PlaceObjectAtPoint if available in your build
  try:
    ev = controller.step(action="PlaceObjectAtPoint",
                         objectId=object_id,
                         rotation=rot,
                         position=point)
    if ev.metadata.get("lastActionSuccess"):
      return True
  except Exception:
    pass
  # Fallback: precise teleport then short settle
  print("Place failed, try teleport")
  ev = controller.step(action="TeleportObject",
                       objectId=object_id,
                       position=point,
                       rotation=rot,
                       forceAction=True)
  if use_physics:
    controller.step(action="AdvancePhysicsStep", simSeconds=0.2)
  return ev.metadata.get("lastActionSuccess", False)


def _move_object_to_point(controller, object, point):
  # Teleport stool to new position
  ev = controller.step(action="TeleportObject",
                       objectId=object["objectId"],
                       position=point,
                       rotation=object["rotation"],
                       forceAction=True)

  # print(f"Moved {object['name']}:", ev.metadata["lastActionSuccess"])


def _disable_object(controller, object_id):
  ev = controller.step(action="DisableObject", objectId=object_id)
  success = ev.metadata.get("lastActionSuccess", False)
  if success:
    # print(f"Disabled object {object_id}.")
    return True
  else:
    msg = ev.metadata.get("errorMessage", "Unknown error")
    # print(f"Failed to disable object {object_id}: {msg}")
    return False


def _disable_all_objects_of_type(controller, object_type):
  objects = [
      o for o in controller.last_event.metadata["objects"]
      if o["objectType"] == object_type
  ]
  for obj in objects:
    _disable_object(controller, obj["objectId"])


# If len(spawn_points) is less than num_to_spawn, random spawnpoints will
# be generated for the extra objects (on the DiningTable)
def _spawn_pickable_object_of_type(controller,
                                   object_type,
                                   num_to_spawn=1,
                                   spawn_points=None,
                                   use_physics=True):
  # 1) Read current scene objects
  objs = controller.last_event.metadata["objects"]

  # 2) Pick a source object (must be pickupable)
  try:
    src_obj = next(
        o for o in objs
        if o["objectType"] == object_type and o.get("pickupable", False))
  except StopIteration:
    raise RuntimeError(
        f"No pickupable object of type {object_type} found in this scene.")
  src_name = src_obj["name"]
  src_height = src_obj["position"]["y"]

  # 3) Find the target counter/table the first object is on (or nearest CounterTop)
  target_counter = _find_supporting_counter(controller, src_obj)
  if target_counter is None:
    print("No supporting counter found.")
  else:
    counter_id = target_counter["objectId"]

  # 4) Build the full 'poses' set:
  #    - Keep ALL existing movables/pickables as-is EXCEPT the object of interest (we’ll replace them)
  poses = []
  existing_ooi = [
      o for o in objs if o["objectType"] == object_type and (
          o.get("pickupable") or o.get("moveable"))
  ]
  other_movables = [
      o for o in objs if (o.get("pickupable") or o.get("moveable"))
      and o["objectType"] != object_type
  ]

  for o in other_movables:
    poses.append({
        "objectName": o["name"],
        "position": deepcopy(o["position"]),
        "rotation": deepcopy(o["rotation"]),
    })

  # 5) Add duplicates of the source objects (counts only; we will place them via spawn points later)
  add_duplicates = num_to_spawn - 1
  # Layout grid of offsets on the counter (expand as needed)
  offsets = [
      (+0.00, +0.00),
      (+0.06, +0.00),
      (-0.06, +0.00),
      (+0.00, +0.06),
      (+0.06, +0.06),
      (-0.06, +0.06),
      (+0.00, -0.06),
      (+0.06, -0.06),
      (-0.06, -0.06),
  ]

  surface_y = src_height

  def obj_pose_at(dx, dz, name_for_this_entry):
    return {
        "objectName":
        name_for_this_entry,  # existing instance name -> keep/move; reuse src name → duplicate
        "position": {
            "x": dx,
            "y": surface_y,
            "z": dz
        },
        "rotation": {
            "x": 0,
            "y": 0,
            "z": 0
        }
    }

  # Place each existing object onto the counter at successive offsets
  if len(existing_ooi) > len(offsets):
    # extend offsets when we need to spawn many objects
    extra = len(existing_ooi) - len(offsets)
    offsets.extend([(0.12 + 0.06 * i, 0.00) for i in range(extra)])

  for j in range(add_duplicates):
    dx, dz = offsets[len(existing_ooi) + j]
    poses.append(obj_pose_at(dx, dz,
                             src_name))  # duplicate source by reusing its name

  # 6) Also keep each existing object (temporary position — we’ll re-place via spawn points)
  for o in existing_ooi:
    poses.append({
        "objectName": o["name"],
        "position": deepcopy(o["position"]),
        "rotation": deepcopy(o["rotation"]),
    })

  # Apply curated poses (this replaces the movable set)
  ev = controller.step(action="SetObjectPoses", objectPoses=poses)
  ok = ev.metadata.get("lastActionSuccess", False)
  if not ok:
    print("SetObjectPoses error:", ev.metadata.get("errorMessage", ""))

  # --- precise placement using receptacle spawn points --------------------

  # Re-query objects after SetObjectPoses to get fresh IDs (duplicates now exist)
  objs = controller.last_event.metadata["objects"]
  ooi_ids = [
      o["objectId"] for o in objs if o["objectType"] == object_type and (
          o.get("pickupable") or o.get("moveable"))
  ]
  # print(f"Will place {len(ooi_ids)} objects on a receptacle.")

  # Set points for objects
  num_spawn_random = num_to_spawn
  pts_iter = []
  if spawn_points is not None:
    if len(spawn_points) < len(ooi_ids):
      # Generate random spawn points for extra objects
      num_spawn_random -= len(spawn_points)

      # raise ValueError(
      #     f"Not enough spawn points ({len(spawn_points)}) for {len(ooi_ids)} objects."
      # )
    pts_iter = spawn_points

  assert counter_id is not None
  points = _spawn_points_above_receptacle(controller, counter_id)
  # print(f"Found {len(points)} spawn points above counter")
  if not points:
    raise RuntimeError(
        "No spawn coordinates returned for counter; try a different receptacle."
    )
  random.shuffle(points)
  pts_iter.extend([points[i % len(points)] for i in range(num_spawn_random)])

  # Place each object on the counter surface
  all_ok = True
  for ooi_id, pt in zip(ooi_ids, pts_iter):
    # print('Placing object ', ooi_id, ' at ', pt)
    ok = _place_object_at_point(controller, ooi_id, pt, use_no_rot=True)
    all_ok = all_ok and ok

  # Final small settle for safety
  if use_physics:
    controller.step(action="AdvancePhysicsStep", simSeconds=0.2)
  # print(f"Placed {len(ooi_ids)} objects on the same counter. success={all_ok}")
