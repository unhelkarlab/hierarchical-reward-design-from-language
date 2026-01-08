import time
from queue import Queue, Empty
from tqdm import tqdm

from HierRL.envs.ai2thor.pnp_env import ThorPickPlaceEnv
from HierRL.envs.ai2thor.pnp_training_utils import (PnP_HL_Actions,
                                                    PnP_LL_Actions)

env = ThorPickPlaceEnv(low_level=True,
                       option=PnP_HL_Actions.drop_egg.value,
                       render=False)

print("Starting resets...")

for _ in tqdm(range(100)):
  env.reset(options={"option": PnP_HL_Actions.drop_egg.value})
  env.step(0, PnP_HL_Actions.drop_egg.value)
  # print(env._held_object())
  # time.sleep(1)
