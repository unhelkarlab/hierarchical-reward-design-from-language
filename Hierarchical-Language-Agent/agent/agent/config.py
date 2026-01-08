class OvercookedConfig():
  COOKING_TIME_SECONDS = 10  # time required to cook sth
  COOKED_BEFORE_FIRE_TIME_SECONDS = 25  # time before a cooked soup turning into fire
  FIRE_PUTOUT_TIME_SECONDS = 5  # time required to put out the fire
  FIRE_RECOVER_GAP_TIME_SECONDS = 1  # time gap before the fire starts to grow again
  CHOPPING_NUM_STEPS = 5  # steps required to chop some ingredient, e.g. tomato/lettuce
  MAX_ORDER_LENGTH_SECONDS = 100
  ORDER_EXPIRE_PUNISH = 5


class OvercookedPractice1(OvercookedConfig):
  game_map = 'ring'
  user_recipy: bool = True  # whether user can see the recipy
  ai_recipy: bool = True  # whether ai can see the recipy
  max_num_timesteps: int = 150  # max number of timesteps
  max_num_orders: int = 3  # max number of orders
  num_agents: int = 1
  agent_types: list = ['ai']
  is_practice: bool = True


class OvercookedPractice2(OvercookedConfig):
  game_map = 'ring'
  user_recipy: bool = True  # whether user can see the recipy
  ai_recipy: bool = True  # whether ai can see the recipy
  max_num_timesteps: int = 180  # max number of timesteps
  max_num_orders: int = 3  # max number of orders
  num_agents: int = 1
  agent_types: list = ['ai']
  is_practice: bool = True


class OvercookedExp1(OvercookedConfig):
  game_map = 'ring'
  user_recipy: bool = True  # whether user can see the recipy
  ai_recipy: bool = True  # whether ai can see the recipy
  max_num_timesteps: int = 300  # max number of timesteps
  max_num_orders: int = 3  # max number of orders
  num_agents: int = 1
  agent_types: list = ['ai']
  is_practice: bool = True


class OvercookedExp2(OvercookedConfig):
  game_map = 'ring'
  user_recipy: bool = True  # whether user can see the recipy
  ai_recipy: bool = True  # whether ai can see the recipy
  max_num_timesteps: int = 300  # max number of timesteps
  max_num_orders: int = 3  # max number of orders
  num_agents: int = 1
  agent_types: list = ['h']
  is_practice: bool = True
