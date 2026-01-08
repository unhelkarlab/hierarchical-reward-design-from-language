# from gym_cooking.utils.config import *
from typing import List
from collections import defaultdict
from collections import Counter
from gym_cooking.envs.env_settings import *
from agent.executor.low import EnvState, bfs_reachable, fname
from agent.executor.high import \
    HighTask, HTChop, HTAssemble, HTPutout, HTCook, HTPick, HTServe, HTDrop, HTWait, \
    OBJ_TO_GOODS_GS, OBJ_TO_GOODS_POT, ALL_FRESH_FOOD, ALL_ASSEMBLE, ALL_SOUP, HT_MAP
# from minigrid.core.constants import IDX_TO_OBJECT, IDX_TO_COLOR, IDX_TO_STATE

ORDER_NAMES = {
    "CookedLettuce-CookedOnion-Plate": "Alice Soup",
    "CookedLettuce-CookedTomato-Plate": "Bob Soup",
    "CookedOnion-CookedTomato-Plate": "Cathy Soup",
    "CookedLettuce-CookedOnion-CookedTomato-Plate": "David Soup",
}

ORDER_TO_ALL_NAMES = {
    'David Soup': [
        'ChoppedLettuce-ChoppedOnion-ChoppedTomato',
        'CookingLettuce-CookingOnion-CookingTomato',
        'CookedLettuce-CookedOnion-CookedTomato',
        'CookedLettuce-CookedOnion-CookedTomato-Plate'
    ],
    'Cathy Soup': [
        'ChoppedOnion-ChoppedTomato', 'CookingOnion-CookingTomato',
        'CookedOnion-CookedTomato', 'CookedOnion-CookedTomato-Plate'
    ],
    'Bob Soup': [
        'ChoppedLettuce-ChoppedTomato', 'CookingLettuce-CookingTomato',
        'CookedLettuce-CookedTomato', 'CookedLettuce-CookedTomato-Plate'
    ],
    'Alice Soup': [
        'ChoppedLettuce-ChoppedOnion', 'CookingLettuce-CookingOnion',
        'CookedLettuce-CookedOnion', 'CookedLettuce-CookedOnion-Plate'
    ]
}

MOVE_TO_HT = \
    {**{f"{HT_MAP['Chop']} {x}": HTChop(x) for x in ALL_FRESH_FOOD},
     **{f"{HT_MAP['Assemble']} {x}": HTAssemble(x) for x in ALL_ASSEMBLE},
     **{f"{HT_MAP['Putout']}": HTPutout()},
     **{f"{HT_MAP['Cook']} {x}": HTCook(x) for x in ALL_SOUP},
     **{f"{HT_MAP['Pick']} {x}": HTPick(x) for x in ALL_SOUP},
     **{f"{HT_MAP['Serve']} {x}": HTServe(x) for x in ALL_SOUP},
     **{f"{HT_MAP['Drop']}": HTDrop()},
     **{f"{HT_MAP['Wait']}": HTWait()}}
print('Move to ht: ', MOVE_TO_HT)
ALL_MOVES = list(MOVE_TO_HT.keys())
print(f'All {len(ALL_MOVES)} moves: {ALL_MOVES}')


def prep_chk_moves(env: EnvState, user_reward=False) -> list:
  # print('user reward: ', user_reward)
  # moves
  all_moves = []
  for obj in ALL_FRESH_FOOD:
    can_begin = HTChop(obj).can_begin(env, user_reward)
    all_moves.append([f"{HT_MAP['Chop']} {obj}", *can_begin])
  for obj in ALL_ASSEMBLE:
    can_begin = HTAssemble(obj).can_begin(env, user_reward)
    all_moves.append([f"{HT_MAP['Assemble']} {obj}", *can_begin])
  can_begin = HTPutout().can_begin(env)
  all_moves.append([f"{HT_MAP['Putout']}", *can_begin])
  for obj in ALL_SOUP:
    can_begin = HTCook(obj).can_begin(env, user_reward)
    all_moves.append([f"{HT_MAP['Cook']} {obj}", *can_begin])
  for obj in ALL_SOUP:
    can_begin = HTPick(obj).can_begin(env, user_reward)
    all_moves.append([f"{HT_MAP['Pick']} {obj}", *can_begin])
  for obj in ALL_SOUP:
    can_begin = HTServe(obj).can_begin(env, user_reward)
    all_moves.append([f"{HT_MAP['Serve']} {obj}", *can_begin])
  can_begin = HTDrop().can_begin(env)
  all_moves.append([f"{HT_MAP['Drop']}", *can_begin])

  for x in all_moves:
    assert x[0] in ALL_MOVES, f'Invalid move {x[0]}'

  return all_moves


def prep_prompt_order(env: EnvState) -> list:
  ret = []
  for rec in env.order.current_orders:
    soup_name = OBJ_TO_GOODS_GS[rec[0].full_name]
    rate = rec[1] / rec[2]
    ret.append({'name': soup_name, 'rate': rate})

  return ret


def prep_prompt_map(env: EnvState) -> str:
  # current map

  prompt = '''
Items on the map:
'''
  # obj gs reachable info
  res = env.get_all_grid_info()

  obj_count = {}
  for a in res:
    if (a["gs"].name == "Counter" or a["gs"].name == "Cutboard") and a["rch"] and a["obj"] is not None \
            or a["obj"] is not None and a["rch"] and a["obj"].is_held:
      obj = OBJ_TO_GOODS_GS[a["obj"].full_name]
      obj_count[obj] = obj_count.get(obj, 0) + 1
  for key, value in obj_count.items():
    if 'Ingredients' in key:
      key = ('Assembled ' + key)
    prompt += f'- {value} {key}\n'

  # Pot
  num_empty_pot = len([
      a for a in res if a["gs"].name == "Pot" and a["obj"] is None and a["rch"]
  ])
  prompt += f'- {num_empty_pot} empty {"pots" if num_empty_pot > 1 else "pot"}\n'
  for a in res:
    if a["gs"].name == "Pot" and a["rch"] and a["obj"] is not None:
      obj = OBJ_TO_GOODS_POT[a["obj"].full_name]
      if 'Cooking' in a["obj"].full_name:
        remain_time = a["obj"].rest_turn_time()
        rate = remain_time / COOKING_TIME_SECONDS
        prompt += f'- 1 pot still cooking {obj}: '
        # progress
        if rate > 0.75:
          prompt += 'just started and far from finished'
        elif rate > 0.5:
          prompt += 'has been cooking for a while and needs some time to finish'
        elif rate > 0.25:
          prompt += 'has been cooking for a long time and will finish soon'
        else:
          prompt += 'will finish in no time'
      elif 'Cooked' in a["obj"].full_name:
        remain_time = a["obj"].rest_turn_time()
        rate = remain_time / COOKED_BEFORE_FIRE_TIME_SECONDS
        prompt += f'- 1 pot with cooked {obj}: '
        if rate > 0.75:
          prompt += 'just cooked and far from charred'
        elif rate > 0.5:
          prompt += 'has been cooked for a while and needs some time to get charred'
        elif rate > 0.25:
          prompt += 'has been cooked for a long time and will get charred soon'
        else:
          prompt += 'will get charred in no time'
      elif 'Fire' in a["obj"].full_name:
        remain_time = a["obj"].rest_turn_time()
        rate = remain_time / FIRE_PUTOUT_TIME_SECONDS
        prompt += f' - 1 pot on fire with charred {obj}'
        if rate > 0.5:
          prompt += 'The fire is big and will take some time to put out'
        else:
          prompt += 'The fire is small and will get put out soon'
      else:
        prompt += f' - 1 pot with charred {obj}'

      prompt += '\n'

  return prompt


def prep_prompt_map_s(env: EnvState) -> str:
  # current map

  prompt = '''
Items on the map:

'''

  res = env.get_all_grid_info()

  for a in res:
    if not a['rch']:
      continue
    if a['gs'].name == "Floor":
      continue
    prompt += f"{a['gs'].name} at {a['gs'].location}"
    if a['gs'].name in ["Counter", "Cutboard"] and a['obj'] is not None:
      obj_name = OBJ_TO_GOODS_GS[a["obj"].full_name]
      prompt += f" with {obj_name} on it"
    elif a['gs'].name == "Pot" and a['obj'] is not None:
      obj = OBJ_TO_GOODS_POT[a["obj"].full_name]
      if 'Cooking' in a["obj"].full_name:
        remain_time = a["obj"].rest_turn_time()
        rate = remain_time / COOKING_TIME_SECONDS
        prompt += f' cooking {obj}: '
        # progress
        if rate > 0.75:
          prompt += 'just started and far from finished'
        elif rate > 0.5:
          prompt += 'has been cooking for a while and needs some time to finish'
        elif rate > 0.25:
          prompt += 'has been cooking for a long time and will finish soon'
        else:
          prompt += 'will finish in no time'
      elif 'Cooked' in a["obj"].full_name:
        remain_time = a["obj"].rest_turn_time()
        rate = remain_time / COOKED_BEFORE_FIRE_TIME_SECONDS
        prompt += f' with cooked {obj}: '
        if rate > 0.75:
          prompt += 'just cooked and far from charred'
        elif rate > 0.5:
          prompt += 'has been cooked for a while and needs some time to get charred'
        elif rate > 0.25:
          prompt += 'has been cooked for a long time and will get charred soon'
        else:
          prompt += 'will get charred in no time'
      elif 'Fire' in a["obj"].full_name:
        remain_time = a["obj"].rest_turn_time()
        rate = remain_time / FIRE_PUTOUT_TIME_SECONDS
        prompt += f' on fire with charred {obj}'
        if rate > 0.5:
          prompt += 'The fire is big and will take some time to put out'
        else:
          prompt += 'The fire is small and will get put out soon'
      else:
        prompt += f' - 1 pot with charred {obj}'

    prompt += ".\n"

  # self pos and hold
  prompt += f"Currently you are at {env.self_pos}"
  if env.agents[env.agent_idx].holding is not None:
    hold_name = OBJ_TO_GOODS_GS[env.agents[env.agent_idx].holding.full_name]
    prompt += f" holding {hold_name}"
  prompt += '.\n'

  # other pos and hold
  prompt += f"The human player is at {env.agents[1 - env.agent_idx].location}"
  if env.agents[1 - env.agent_idx].holding is not None:
    hold_name = OBJ_TO_GOODS_GS[env.agents[1 - env.agent_idx].holding.full_name]
    prompt += f" holding {hold_name}"
  prompt += ".\n"

  return prompt


def prep_prompt(env,
                int_hist: list,
                llm_his: list,
                mov_his: list,
                chat: str,
                pref: str = '',
                operation: str = '',
                instr_reasoning=None,
                consider_instr_reasoning=False,
                gen_mode: str = '',
                env_type: str = 'overcooked',
                objects_seen: set = None,
                door_blocked: bool = False,
                domain: str = 'BlockedUnlockPickup',
                sit_pref: list = [],
                available_actions: list = [],
                user_reward: bool = False) -> dict:
  ret = {}
  if env_type == 'overcooked':
    ret['chk_moves'] = prep_chk_moves(env, user_reward)
    ret['order'] = prep_prompt_order(env)
    ret['map'] = prep_prompt_map(env)
    # if sit_pref == []:
    #   ret['sit_pref'] = prep_prompt_sit_pref(env)
    # else:
    #   ret['sit_pref'] = sit_pref
    ret['sit_pref'] = sit_pref
    ret['available_actions'] = available_actions
  elif env_type == 'minigrid':
    assert NotImplementedError
    # ret['chk_moves'] = None
    # ret['order'] = None
    # if domain == 'BlockedUnlockPickup':
    #   ret['map'] = env['language']
    # elif domain == 'PickupMultigoals':
    #   ret['map'] = prep_prompt_map_minigrid(env, door_blocked)
    # ret['objects_seen'] = objects_seen
    # ret['blocked'] = door_blocked
    # ret['domain'] = domain
  ret['int_hist'] = int_hist[-3:]
  ret['llm_hist'] = llm_his[-3:]
  ret['mov_hist'] = mov_his[-100:]
  ret['chatin'] = chat

  # The following are only used by GuidedAgent
  # A string indicating the human's preference
  ret['pref'] = pref
  # The operation performed when aggragating actions from the LM and those from
  # an IL model. Only used if consider_instr_reasoning is set to true
  ret['operation'] = operation
  # An imitation learning model's output
  ret['instr_reasoning'] = instr_reasoning
  # Whether the LM considers the imitation learning model's output
  ret['consider_instr_reasoning'] = consider_instr_reasoning
  # How the top actions are generated
  ret['gen_mode'] = gen_mode
  # Type of the environment
  ret['env_type'] = env_type

  return ret


def prep_prompt_s(env: EnvState, int_hist: list, llm_his: list, mov_his: list,
                  chat: str) -> dict:
  ret = {}
  ret['chk_moves'] = prep_chk_moves(env)
  ret['order'] = prep_prompt_order(env)
  ret['map'] = prep_prompt_map_s(env)
  ret['int_hist'] = int_hist[-3:]
  ret['llm_hist'] = llm_his[-3:]
  ret['mov_hist'] = mov_his[-100:]
  ret['chatin'] = chat

  return ret


# def prep_prompt_map_minigrid(env, blocked):
#   obj_desc = defaultdict(int)
#   in_front = ''
#   holding = ''

#   if blocked:
#     middle_col = 2
#     last_row = 4
#   else:
#     middle_col = 1
#     last_row = 2

#   obs_img = env['image']
#   # print(obs_img)
#   for col_idx in range(len(obs_img)):
#     col = obs_img[col_idx]
#     # print('column: ', col)
#     for row_idx in range(len(col)):
#       obj = col[row_idx]
#       if obj.tolist() != [1, 0, 0] and obj.tolist() != [0, 0, 0]:
#         object = IDX_TO_OBJECT[obj[0]]
#         color = IDX_TO_COLOR[obj[1]]
#         state = IDX_TO_STATE[obj[2]]
#         if object == 'door':
#           desc = f'{color} {state} door'
#         else:
#           desc = f'{color} {object}'
#         if not (col_idx == middle_col and row_idx == last_row):
#           obj_desc[desc] += 1
#         if col_idx == middle_col and row_idx == (last_row - 1):
#           in_front = desc
#         if col_idx == middle_col and row_idx == last_row:
#           holding = desc

#   return obj_desc, in_front, holding


def prep_prompt_sit_pref(env):
  """
  Prepare the prompt for GROUND-TRUTH context depedent preferences.
  This function is only used for proof of concept testing.
  """
  current_orders = env.order.current_orders
  order_names = [order.full_name for order, _, _, _ in current_orders]
  order_names = [ORDER_NAMES[name] for name in order_names]
  print('order names: ', order_names)
  num_david_soups = get_num_priority_orders(env, 'David Soup', order_names)
  if num_david_soups > 0:
    print('here 1')
    return ['David Soup']

  num_alice_soups = get_num_priority_orders(env, 'Alice Soup', order_names)
  if num_alice_soups > 0:
    print('here 2')
    return ['Alice Soup']

  print('here 3')
  return ['David Soup']


def get_num_priority_orders(env: EnvState, priority_order: str,
                            order_names: List[str]):
  """
  For the order of interest, calculate how many of that order we need to make.
  This number is essentially calculated as the number of orders - how many 
  we are currently cooking. 
  """
  if priority_order in order_names:
    order_counts = Counter(order_names)
    cooking_count = 0
    pots = env.world.get_all_gridsquares('Pot')
    for pot in pots:
      if (pot.holding is not None
          and pot.holding.full_name in ORDER_TO_ALL_NAMES[priority_order]):
        cooking_count += 1
    for agent in env.agents:
      if (agent.holding is not None
          and agent.holding.full_name in ORDER_TO_ALL_NAMES[priority_order]):
        cooking_count += 1
    return order_counts[priority_order] - cooking_count
  else:
    return 0
