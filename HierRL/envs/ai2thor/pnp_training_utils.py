from enum import Enum

PnP_LL_Actions = [
    "MoveAhead",
    # "MoveBack",
    # "MoveLeft",
    # "MoveRight",
    "RotateLeft",
    "RotateRight",
    # "LookUp",
    # "LookDown",
    # high-level interactions handled by helper methods below:
    "PickupNearestTarget",
    "PutHeldOnReceptacle",
    # "DropHeld"
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