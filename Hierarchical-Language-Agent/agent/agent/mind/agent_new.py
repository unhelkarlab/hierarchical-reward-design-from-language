from __future__ import annotations
import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field
import torch
import dill
import ast
import random
import heapq
from collections import Counter
from typing import Dict
from openai import OpenAI
import numpy as np
import traceback

from agent.executor.low import EnvState
from agent.executor.high import (HighTask, OBJ_TO_GOODS_GS, OBJ_TO_GOODS_POT,
                                 HTCook, HTAssemble, HTChop)
from agent.mind.prompt_local import (MOVE_TO_HT, prep_prompt, prep_chk_moves,
                                     prep_prompt_map, prep_prompt_order)
from agent.mind.call import (low, high, mix_L, mix_L_new, instr_reasoning,
                             guided_hl)
from gym_cooking.utils.replay import Replay
from agent.mind.lsh import LSHTable
from agent.mind.prompt_local import ALL_MOVES

from rw4t.utils import RW4T_HL_Actions

ORDER_NAMES = {
    "CookedLettuce-CookedOnion-Plate": "Alice Soup",
    "CookedLettuce-CookedTomato-Plate": "Bob Soup",
    "CookedOnion-CookedTomato-Plate": "Cathy Soup",
    "CookedLettuce-CookedOnion-CookedTomato-Plate": "David Soup",
}

ORDER_TO_INGRE = {
    "Alice Soup": "Alice Ingredients",
    "Bob Soup": "Bob Ingredients",
    "Cathy Soup": "Cathy Ingredients",
    "David Soup": "David Ingredients",
}

INTENTS_TO_MOVES = {
    ('chop', "Lettuce"): ['Chop Lettuce'],
    ('chop', "Tomato"): ['Chop Tomato'],
    ('chop', "Onion"): ['Chop Onion'],
    ('prepare', "Alice Soup"): ['Prepare Alice Ingredients'],
    ('prepare', "Bob Soup"): ['Prepare Bob Ingredients'],
    ('prepare', "Cathy Soup"): ['Prepare Cathy Ingredients'],
    ('prepare', "David Soup"): ['Prepare David Ingredients'],
    ('cook', "Alice Soup"): ['Cook Alice Soup'],
    ('cook', "Bob Soup"): ['Cook Bob Soup'],
    ('cook', "Cathy Soup"): ['Cook Cathy Soup'],
    ('cook', "David Soup"): ['Cook David Soup'],
    ('plate', "Alice Soup"): ['Plate Alice Soup'],
    ('plate', "Bob Soup"): ['Plate Bob Soup'],
    ('plate', "Cathy Soup"): ['Plate Cathy Soup'],
    ('plate', "David Soup"): ['Plate David Soup'],
    ('serve', "Alice Soup"): ['Plate Alice Soup', 'Serve Alice Soup'],
    ('serve', "Bob Soup"): ['Plate Bob Soup', 'Serve Bob Soup'],
    ('serve', "Cathy Soup"): ['Plate Cathy Soup', 'Serve Cathy Soup'],
    ('serve', "David Soup"): ['Plate David Soup', 'Serve David Soup'],
    ('put out', 'fire'): ['Putout', 'Drop'],
    ('Wait'): ['Wait']
}

INGRE_OF_INTEREST = [
    'Alice Ingredients', 'Bob Ingredients', 'Cathy Ingredients',
    'David Ingredients'
]

INGRE_TO_SOUP = {
    'Alice Ingredients': "Alice Soup",
    'Bob Ingredients': "Bob Soup",
    'Cathy Ingredients': "Cathy Soup",
    'David Ingredients': "David Soup"
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


def get_order_names(env_orders):
  order_names = [order.full_name for order, _, _, _ in env_orders]
  return [ORDER_NAMES[name] for name in order_names]


def get_ingredient_list_for_order(order_name: str):
  order_to_ingre = {name: ingre for ingre, name in ORDER_NAMES.items()}
  ingre = order_to_ingre[order_name]
  ingre_list = []
  if 'Lettuce' in ingre:
    ingre_list.append('Lettuce')
  if 'Onion' in ingre:
    ingre_list.append('Onion')
  if 'Tomato' in ingre:
    ingre_list.append('Tomato')
  return ingre_list


def request_client(mode, llm, data):
  if mode in ['L1l']:
    return mix_L(mode, data)
  elif mode in ['L1l_new']:
    return mix_L_new(data)
  elif mode in ['Ei', 'El', 'Hl', 'Ei_h', 'El_h', 'Hl_h']:
    return high(mode, data)
  elif mode in ['Em', 'Sm', 'Em_h']:
    return low(mode, data)
  elif mode in ['instr_reasoning']:
    return instr_reasoning(data)
  elif mode in ['guided_hl', 'guided_hl_minigrid', 'guided_hl_rw4t']:
    return guided_hl(data)
  else:
    raise NotImplementedError


@dataclass
class AgentSetting:
  mode: str
  hl_mode: str
  ll_mode: str
  prompt_style: str
  top_k_llm: int = 2
  no_hl_rec: bool = False
  text_desc: bool = False
  prev_skill: bool = True
  high_llm: str = 'gpt-4o-mini'  # gpt-4o
  low_llm: str = 'llama'
  speed: float = 3
  pref: str = ''
  operation: str = 'multiply'
  fast_il: bool = False
  gen_mode: str = ''
  auto: bool = True  # Whether the agent is autonomous (not following instructions)
  interpolation: bool = True  # Whether we will interpolate with IL's output
  q_eval: bool = True  # Whether we use q evaluation
  e_greedy: bool = False  # Whether we choose a random action with e probability
  user_reward: bool = False  # Whether we calculate a hand crated Q based on user reward


class LLMAgent:

  def __init__(self) -> None:
    self.cur_intent = None
    self.intent_hist = []

    # High level action name/object
    self._task = None
    # High level action index
    self.prev_intent_idx = 0
    self.prep = {}
    self.verifier_error_rate = 0.0
    # Situational preference
    self.sit_pref = []
    self.sit_pref_counter = 0  # Number of llm queries made before sit_pref is reset
    self.computing_sit_pref = False

    # All high level actions
    self.all_moves = []
    self.use_prev_intent = True

    self.num_valid_actions = 0  # Number of valid actions executed by the slow mind
    self.num_total_actions = 0  # Total number of actions executed by the slow mind

    self.env_type = ''

  def process_action_dict(self,
                          ht_dict,
                          env_tensor,
                          lm_temp=3,
                          lm_default=0.001,
                          il_temp=0.5,
                          il_default=0.001,
                          il_k=3):
    if len(ht_dict) > 0:
      ht_dict = self.adjust_temperature(ht_dict, temperature=lm_temp)
    print('ht dict: ',
          dict(sorted(ht_dict.items(), key=lambda item: item[1], reverse=True)))

    il_dict = {}
    # Should we combine the policies?
    if self.model_type != 'none' and self.interpolation:
      if self.use_prev_intent:
        top_actions = self.model.choose_top_actions(torch.cat(
            (env_tensor, torch.tensor([self.prev_intent_idx]))),
                                                    k=il_k)
      else:
        top_actions = self.model.choose_top_actions(env_tensor, k=il_k)
      for action_idx in top_actions:
        il_dict[self.all_moves[action_idx]] = top_actions[action_idx]
      il_dict = self.adjust_temperature(il_dict, temperature=il_temp)
    print('il dict: ',
          dict(sorted(il_dict.items(), key=lambda item: item[1], reverse=True)))

    default_val_il = il_default
    default_val_ht = lm_default
    # Get the set of all possible keys (union of keys from both dictionaries)
    all_keys = set(il_dict.keys()).union(ht_dict.keys())
    # Initialize the result dictionary
    merged_dict = {}
    # Iterate through all possible keys
    for key in all_keys:
      # Get values from both dictionaries, using the default value for missing keys
      val1 = il_dict.get(key, default_val_il)
      val2 = ht_dict.get(key, default_val_ht)
      if self.operation == 'add':
        # Add the values and store in the result dictionary
        merged_dict[key] = val1 + val2
      else:
        # Multiply the values and store in the result dictionary
        merged_dict[key] = val1 * val2

    normalized_dict = {
        key: value / sum(merged_dict.values())
        for key, value in merged_dict.items()
    }

    # Do we assume an available affordance function?
    # normalized_dict = {
    #     key: normalized_dict[key]
    #     for key in normalized_dict if self.move_verifier(key)
    # }
    print(
        'Processed dict: ',
        dict(
            sorted(normalized_dict.items(),
                   key=lambda item: item[1],
                   reverse=True)))
    if len(normalized_dict) == 0:
      return 'Wait'

    hl_action = max(normalized_dict, key=lambda k: normalized_dict[k])
    print('Output action: ', hl_action)
    # Calculate the proportion of valid slow mind actions
    if self.move_verifier(hl_action):
      self.num_valid_actions += 1
    self.num_total_actions += 1
    print('valid percentage: ', self.num_valid_actions / self.num_total_actions)
    if self.env_type == 'rw4t' and hl_action == 'pick':
      # In rw4t, if the agent performs the pick action, then reset its goal to
      # None
      self.goal = None
    if self.env_type == 'rw4t' and self.e_greedy:
      rand_f = np.random.rand()
      if rand_f <= 0.1:
        hl_action = np.random.choice(list(RW4T_HL_Actions)).name
        print('random action: ', hl_action)
    return hl_action
    # sampled_key = random.choices(list(normalized_dict.keys()),
    #                              list(normalized_dict.values()),
    #                              k=1)[0]
    # print('sampled key: ', sampled_key)
    # return sampled_key

  def adjust_temperature(self, logit_dict, temperature):
    # Extract keys and logits from the dictionary
    keys = list(logit_dict.keys())
    logits = np.array(list(logit_dict.values()))

    # Adjust the logits by the temperature
    adjusted_logits = logits / temperature

    # Convert adjusted logits to probabilities
    probabilities = self.softmax(adjusted_logits)

    # Create a new dictionary mapping strings to the new probabilities
    adjusted_dict = {keys[i]: probabilities[i] for i in range(len(keys))}
    return adjusted_dict

  def softmax(self, logits):
    # Subtract max logit for numerical stability
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

  def move_verifier(self, move: str):
    # Every move is valid if we don't know which moves are validcl
    if 'chk_moves' not in self.prep or self.prep['chk_moves'] is None:
      return True

    if move == 'Wait':
      return True
    valid_moves = [move[0] for move in self.prep['chk_moves'] if move[1]]
    is_valid = move in valid_moves
    return is_valid if random.random(
    ) > self.verifier_error_rate else 1 - is_valid

  def set_prev_intent(self):
    if self._task is not None:
      self.prev_intent_idx = self.all_moves.index(str(self._task))


class GuidedAgent(LLMAgent):
  INT_HIST_MAX_LEN = 5 * 10000
  LLM_HIST_MAX_LEN = 5 * 10000
  MAX_LLM_UNG_MOV = 9
  MAX_LLM_FIN_MOV = 6

  def __init__(self,
               setting: AgentSetting,
               replay: Replay,
               use_composite: bool = True,
               request_type='guided_hl'):
    super().__init__()
    self.setting = setting
    self.replay = replay
    self.pref = setting.pref
    self.operation = setting.operation
    self.gen_mode = setting.gen_mode
    self.interpolation = setting.interpolation

    self._lock = threading.Lock()
    self._last_env = None
    self._last_tensor = None
    self._new_order = True
    # high level
    self._int_hist = []  # submit_time, finish_time, chat, ret (intention)
    # int_time(finish_time), submit_time, finish_time, chat, ret {Chat, Demand, Finished}
    self._llm_hist = []
    self._num_high_threads = 0
    # low level
    self._mov_hist: list = []  # task, status, submit_time, finish_time

    # thread safe
    self._it_time = 0
    self._lt_time = 0  # last time
    self._mt_time = 0
    self._tasks = []
    self.env_tensor = None

    self.cur_intent = None
    self.intent_hist = []

    # Not a good name, but it means what the AI thinks the human wants it to do
    self.reasoning = ''

    # All human instructions when the agent is learning the human's preference
    self.all_instructions = []

    # Whether the agent has processed the information that the game has ended
    self.game_end_processed = False

    self.client = OpenAI()
    self.gpt_version = 'gpt-4o'  # gpt-4o

    # Name of LLM request
    self.request_type = request_type
    self.env_type = 'overcooked'
    self.all_moves = ALL_MOVES
    # Objects that the agent has previously seen. Only used in Minigrid
    self.objects_seen = None
    # Can the door be blocked. Only used in Minigrid - PickupMultigoals
    self.door_blocked = False
    # The domain we are using for Minigrid
    self.domain = None
    # The type of IL model we are using with the LLM
    self.model_type = 'none'

    # Whether the agent acts autonomously or follows decisions
    self.auto = self.setting.auto
    # The agent's plan based on human instructions
    self.plan = []
    # Whether the agent needs a new plan
    self.need_plan = True
    # Whether the agent uses composite skills when reasoning about instructions
    self.use_composite = use_composite
    # Prev composite skill
    self.prev_sit_pref = []
    # Whether it is the first time calling llm infer
    self.first_it = True
    # Whether to use the text desription of the agent's preference
    self.text_desc = self.setting.text_desc
    # Whether to use the user reward
    self.user_reward = self.setting.user_reward

    self.num_gpt_calls = 0
    self.num_steps = 0

  @property
  def _is_finished(self):
    if not self._int_hist:
      return True
    intent = self._int_hist[-1]

    if intent['finish_time'] is None:
      return False
    intent_time = intent['finish_time']

    llms = self._llm_hist[-100:]
    llms = [l for l in llms if l['submit_time'] > intent_time]

    return any([l['ret']['Finished'] for l in llms])

  def _get_high_mov_hist(self):
    if self._is_finished:
      mov_his = self._mov_hist[-100:]
      mov_his = [
          m for m in mov_his if m['status'].startswith('Success')
          or m['status'].startswith('Ongoing')
      ]
      mov_his = mov_his[-self.MAX_LLM_FIN_MOV:]
    else:
      intent_time = self._int_hist[-1]['submit_time']
      mov_his = self._mov_hist[-100:]
      mov_his = [m for m in mov_his if m['finish_time'] > intent_time]
      mov_his = [
          m for m in mov_his if m['status'].startswith('Success')
          or m['status'].startswith('Ongoing')
      ]
      mov_his = mov_his[-self.MAX_LLM_UNG_MOV:]

    return mov_his

  def _int_infer(self, chat: str = ''):
    # submit_time = time.time()
    # with self._lock:
    #   self._int_hist.append({
    #       'submit_time': submit_time,
    #       'finish_time': None,
    #       'chat': chat,
    #       'ret': None
    #   })

    #   int_his = self._int_hist[-2:]

    #   prep = prep_prompt(self._last_env, int_his, [], [], chat)
    #   prep = deepcopy(prep)

    #   self._num_high_threads += 1

    # try:
    #   js = request_client("Ei_h", self.setting.high_llm, prep)
    # except:
    #   with self._lock:
    #     self._num_high_threads -= 1
    #   return
    # finish_time = time.time()

    # # log
    # self.replay.log("ai.int_infer", {
    #     "prep": prep,
    #     "ret": js,
    #     "time_start": submit_time,
    #     "time_end": finish_time
    # })

    # with self._lock:
    #   self._num_high_threads -= 1
    #   if js is None:
    #     js = "None"
    #   # no cross-time period
    #   intent = [i for i in self._int_hist if i['submit_time'] == submit_time]
    #   if not intent:
    #     return
    #   intent = intent[0]
    #   intent['finish_time'] = finish_time
    #   intent['ret'] = js
    #   self._int_hist = self._int_hist[-self.INT_HIST_MAX_LEN:]

    self._llm_infer()

  def _llm_infer(self, env_tensor):
    with self._lock:
      self.env_tensor = env_tensor
      submit_time = time.time()
      # if int is inferring, return
      if self._int_hist and self._int_hist[-1]['finish_time'] is None:
        return
      int_time = self._int_hist[-1]['submit_time'] if self._int_hist else 0

      assert self.env_type == 'overcooked' or self.env_type == 'rw4t'
      if self.env_type == 'overcooked' and self.first_it:
        self.set_sit_pref()
        self.first_it = False
      if self.env_type == 'overcooked' and self.sit_pref == [] and self.q_eval:
        return
      if self.env_type == 'rw4t':
        if (not self.computing_sit_pref
            and self.sit_pref == []) or self.sit_pref_counter == 0:
          if self._new_order:
            self._new_order = False
          # Get context dependent preferences
          self.computing_sit_pref = True
          self.set_sit_pref()
          self.sit_pref_counter = 0
        elif not self.computing_sit_pref:
          self.sit_pref_counter += 1
      # prep = prep_prompt(self._last_env, [], [],
      #                    mov_his,
      #                    intention,
      #                    pref=self.pref,
      #                    operation=self.operation,
      #                    instr_reasoning=macro_actions,
      #                    consider_instr_reasoning=consider_instr_reasoning,
      #                    gen_mode=self.gen_mode,
      #                    env_type=self.env_type,
      #                    objects_seen=objects_seen,
      #                    door_blocked=self.door_blocked,
      #                    domain=self.domain,
      #                    sit_pref=self.sit_pref,
      #                    available_actions=available_actions)
      prep = self.prep_prompt()
      if self.model_type != 'none' and self.setting.fast_il:
        macro_actions = self.get_macro_list(env_tensor)
        self.choose_il_macro_action(macro_actions, prep)
      prep = deepcopy(prep)
      self.prep = prep

      self._num_high_threads += 1

    try:
      with self._lock:
        self.num_gpt_calls += 1
        print('Number of GPT calls: ', self.num_gpt_calls)
        js = request_client(self.request_type, None, prep)
    except Exception as e:
      with self._lock:
        self._num_high_threads -= 1
      print(f'Exception occurred when calling guided HL {e}')
      traceback.print_exc()  # Prints the full traceback
      print(f"Type of exception: {type(e).__name__}")
      return

    # log
    # self.replay.log("ai.llm_infer", {
    #     "prep": prep,
    #     "ret": js,
    #     "time_start": submit_time,
    #     "time_end": finish_time
    # })

    with self._lock:
      finish_time = time.time()
      self._num_high_threads -= 1

      if js is None:
        return
      if self._int_hist and self._int_hist[-1]['finish_time'] is None:
        return

      # check intention out date
      int_time2 = self._int_hist[-1]['submit_time'] if self._int_hist else 0
      if int_time != int_time2:
        return
      # check llm out date
      if self._llm_hist:
        llm2 = self._llm_hist[-1]
        if llm2['submit_time'] > submit_time:
          return

      self._llm_hist.append({
          'submit_time': submit_time,
          'finish_time': finish_time,
          'chat': '',
          'ret': js
      })
      self._llm_hist = self._llm_hist[-self.LLM_HIST_MAX_LEN:]

  def high_level_infer(self, env: EnvState = None, chat: str = ''):
    if env is not None:
      with self._lock:
        self._last_env = env

    thread = threading.Thread(target=self._int_infer, args=(chat, ))
    thread.daemon = True
    thread.start()

  def low_level_infer(self):
    submit_time = time.time()

    if not self._tasks:
      return

    ht = self._tasks[0]
    if ht is None:
      self._tasks.pop(0)
      return
    task = deepcopy(MOVE_TO_HT[ht])
    can_begin = task.can_begin(self._last_env)

    if can_begin[0]:
      self._tasks.pop(0)
      self._task = task
      finish_time = time.time()
      self._mov_hist.append({
          'task': str(self._task),
          'status': 'Ongoing. Initiated.',
          'submit_time': submit_time,
          'finish_time': finish_time
      })
      self.replay.log(
          "ai.mov_infer", {
              'prep': None,
              'ret': str(self._task),
              'time_start': submit_time,
              'time_end': submit_time
          })
      return
    else:
      self._tasks.pop(0)
      self._mov_hist.append({
          'task': str(task),
          'status': 'Failed. ' + can_begin[1],
          'submit_time': submit_time,
          'finish_time': None
      })

    # Do not expand actions if an action cannot be currently performed.
    # while not can_begin[0] and len(can_begin[2]) > 0:
    #   task = deepcopy(MOVE_TO_HT[can_begin[2][0]])
    #   can_begin = task.can_begin(self._last_env)
    # if can_begin[0]:
    #   self._task = task
    #   finish_time = time.time()
    #   self._mov_hist.append({
    #       'task': str(self._task),
    #       'status': 'Ongoing. Initiated.',
    #       'submit_time': submit_time,
    #       'finish_time': finish_time
    #   })
    #   self.replay.log(
    #       "ai.mov_infer", {
    #           'prep': None,
    #           'ret': str(self._task),
    #           'time_start': submit_time,
    #           'time_end': submit_time
    #       })
    #   return
    # else:
    #   self._mov_hist.append({
    #       'task': str(task),
    #       'status': 'Failed. ' + can_begin[1],
    #       'submit_time': submit_time,
    #       'finish_time': None
    #   })
    #   self._tasks.pop(0)
    #   return

  def _check_interrupt(self):
    if self._llm_hist and self._lt_time < self._llm_hist[-1]['finish_time']:
      # print('in check interrupt')
      # obtain chat
      chat = self._llm_hist[-1]['ret']['Chat']
      # self.set_prev_intent()
      self._task = None
      if self._mov_hist and self._mov_hist[-1]['status'].startswith('Ongoing'):
        self._mov_hist[-1]['status'] = 'Interrupted. '
      ht_dict = self._llm_hist[-1]['ret']['Action']
      self.sit_pref = self._llm_hist[-1]['ret']['Reasoning']
      self.prev_sit_pref = self.sit_pref

      il_temp = 0.5
      il_default = 0.001
      if self.env_type == 'overcooked':
        lm_default = 0.00099 if self.setting.pref != '' else 0.00099  # 0.0015
      else:
        lm_default = 0.001 if self.setting.pref != '' else 0.001
      if self.setting.gen_mode == '5_unranked':
        if self.env_type == 'overcooked':
          lm_temp = 0.05 if self.setting.pref != '' else 0.05  # 0.3
          il_temp = 1
        else:
          lm_temp = 0.05
          il_temp = 1
      elif self.setting.gen_mode == 'all_yes_no' or self.setting.gen_mode == 'all_yes_no_include_false':
        if self.env_type == 'overcooked':
          lm_temp = 0.3 if self.setting.pref != '' else 1
        else:
          lm_temp = 1  # 0.05
          il_temp = 0.5  # 1
      else:
        lm_temp = 1
      top_action = self.process_action_dict(ht_dict,
                                            self.env_tensor,
                                            lm_temp=lm_temp,
                                            lm_default=lm_default,
                                            il_temp=il_temp,
                                            il_default=il_default)
      self._tasks = [top_action]
      if self.env_type == 'overcooked' and self.setting.q_eval:
        thread = threading.Thread(target=self.set_sit_pref, args=(top_action, ))
        thread.start()
      # if self._llm_hist[-1]['ret']['Action_backup'] != '':
      #   self._tasks.append(self._llm_hist[-1]['ret']['Action_backup'])
      self._lt_time = self._llm_hist[-1]['finish_time']
    else:
      chat = ''

    # check immediate action
    if self._int_hist and self._it_time < self._int_hist[-1]['submit_time']:
      self._it_time = self._int_hist[-1]['submit_time']
      if self._it_time > self._lt_time:
        if self._mov_hist and self._mov_hist[-1]['status'].startswith(
            'Ongoing'):
          self._mov_hist[-1]['status'] = 'Interrupted. '
        # self.set_prev_intent()
        self._task = None
        self._tasks = []

    return chat

  def step(self, env, env_tensor):
    return self.__call__(env, env_tensor)

  def __call__(self, env, env_tensor, instr: str = ''):
    self._lock.acquire()
    if self.auto:
      if self.env_type == 'overcooked':
        prev_orders = [
            order_name for (order_name, _restTime, _timeLimit,
                            _bonus) in self._last_env.order.current_orders
        ] if self._last_env is not None else []
        cur_orders = [
            order_name for (order_name, _restTime, _timeLimit,
                            _bonus) in env.order.current_orders
        ]
        # print('cur orders: ', cur_orders)
        if prev_orders != cur_orders:
          self._new_order = True
      # update env
      self._last_env = env
      self._last_tensor = env_tensor

      # check high level incoming
      chat = self._check_interrupt()
      # submit count check
      start_llm_infer = False
      if self._num_high_threads <= 0:
        if not self._llm_hist:
          start_llm_infer = True
        if self._task is None and not self._tasks:  # additional
          start_llm_infer = True

      if start_llm_infer:
        self._lock.release()
        thread = threading.Thread(target=self._llm_infer,
                                  args=(env_tensor, ),
                                  daemon=True)
        thread.daemon = True
        thread.start()
        self._lock.acquire()

    if not self.auto:
      # update env
      self._last_env = env
      chat = ''

    while True:
      for _ in range(10):
        if self._task is not None:
          break
        self.low_level_infer()

      if self._task is None:
        self._lock.release()
        return (0, 0), chat

      state, move, msg = self._task(env)
      if state == HighTask.Working:  # working
        self._mov_hist[-1]['status'] = 'Ongoing. ' + msg
        # print('current macro action: ', str(self._task))
        self.set_prev_intent()
        self._lock.release()
        self.num_steps += 1
        # print('num steps: ', self.num_steps)
        return move, chat
      elif state == HighTask.Failed:  # reassign task
        self._mov_hist[-1]['status'] = 'Failed. ' + msg
        print(f"Move Failed: {self._mov_hist[-1]['task']}")
        # self.set_prev_intent()
        self._task = None
        self._lock.release()
        self.num_steps += 1
        # print('num steps: ', self.num_steps)
        return (0, 0), chat
      else:
        self._mov_hist[-1]['status'] = 'Success.'
        # if 'Cook' in self._mov_hist[-1]['task']:
        #   self.sit_pref = []
        # self.set_prev_intent()
        self._task = None

  def reason_about_instruction(self, instr: str):
    """
    Reason about an instruction.
    """
    # print('Instr: ', instr)
    self.all_instructions.append(instr)
    if instr not in list(self.all_moves):
      actions = self.get_action_name_gpt(instr)
      actions = [action for action in actions if action in self.all_moves]
    else:
      actions = [instr]
    self.all_instructions.append(instr)
    self.plan = actions
    return actions

  def add_plan_to_tasks(self):
    self._tasks.extend(self.plan)

  def empty_plan(self):
    self.plan = []

  def modify_plan(self, indices):
    new_plan = []
    for index in indices:
      new_plan.append(self.plan[index - 1])
    self.plan = new_plan

  def get_plan_length(self):
    return len(self.plan)

  def process_state_dict(self, state_dict: dict):
    state = torch.cat((torch.from_numpy(state_dict['map']).view(-1),
                       torch.from_numpy(state_dict['current_orders']).view(-1)))
    for _agent_name, holdings in state_dict['current_holdings'].items():
      state = torch.cat((state, torch.from_numpy(holdings).view(-1)))
    return state

  def reason_about_instruction_lsh(self, instr: str, state_dict: dict):
    state = self.process_state_dict(state_dict)
    self.lsh_table.insert(state, instr)

  def query_lsh_table(self, state_dict: dict):
    actions_list = self.lsh_table.query(self.process_state_dict(state_dict))
    print('LSH actions list: ', actions_list)
    if len(actions_list) == 0:
      print('No hit!')

    return list(set(actions_list))

  def get_cur_intent(self):
    return str(self._task)
    # return self.cur_intent

  def add_intent_to_hist(self):
    self.intent_hist.append(self.cur_intent)

  def get_intent_hist(self):
    return self.intent_hist[-5:]

  def game_end(self):
    if self.game_end_processed:
      return

  def get_action_name_gpt(self, instruction):
    avaialble_actions = ', '.join(self.all_moves)
    # print('Available actions: ', avaialble_actions)
    system_prompt = f'''Game Scenario:
You are a helpful and compliant assistant in a simplified Overcooked game and you are working with a human player to complete soup orders.
The human player will give you an instruction that specifies what you should do periodically.
You will reason about the human's instruction and convert the instruction to one or more available actions that you can perform.

Steps to make a soup:
    a. First, if there are no chopped vegetables on the map, you need to chop vegetables.
       Available vegetables to be chopped include Tomato, Lettuce, Onion.
       Chop lettuce and onion to make Alice soup.
       Chop lettuce and tomato to make Bob soup.
       Chop onion and lettuce to make Cathy soup.
       Chop lettuce, onion, and tomato to make David Soup.
    b. Second, assemble the chopped vegetables.
       Once all required vegetables are chopped, you need to assemble them for a corresponding soup.
       You CANNOT begin assembling soup ingredients until all required ingredients for that soup are chopped.
    c. Third, bring the assembled ingredients to a pot to start cooking the soup.
       You CANNOT begin cooking a soup until the ingredients for that soup are assembled.
    d. Fourth, transfer the cooked soup to a plate.
       It takes a while for the soup to finish cooking.
       If the soup is still cooking, work on some other order before plating the soup.
    e. Fifth, serve the soup to customers gain points.
       Once the soup is transferred to a plate, if there is an order for the soup, you should immediately serve it to the customers before taking any other action.
       Serving a soup might take a while, as you need to go to a delivery location to serve the soup, so it is imperative to serve a plated soup as soon as possible.

If a soup stays in the pot too long, it gets charred.
    a. Putout: If the pot catches fire, extinguish it.
    b. Drop: Discard charred soup. Put out the fire in the pot if needed.
    Make sure to plate the soup as soon as it is cooked because putting out a fire takes a long time and will slow you down.

Assuming that you have been playing the game for a while.
The list of available actions is: [''' + avaialble_actions + '''].
In the list above, "Assemble ... Ingredients" means assembling the chopped vegetables for a soup;
"Cook ... Soup" means bringing the assembled ingredients to a pot and start cooking the soup;
"Plate ... Soup" means putting the soup on a plate ready to be served;
"Serve ... Soup" means serving the soup to the customers at the delivery location.
'''

    if self.use_composite:
      skill_str = ''
      for skill_idx in range(len(self.composite_skills)):
        skill_str += f'{self.skill_idx_to_name[skill_idx]}: {self.composite_skills[skill_idx]}\n'
      system_prompt += f'''
You and the human have together come up with the following list of composite skills.
Each composite skill is a sequence of actions to achieve a long-horizon behavior.
{skill_str}
'''

    user_prompt = f'''
Now you need to translate the human's instruction into a list of one or more actions from the list of available actions.
The human's instruction to you is: {instruction}.
If the human refers to a composite skill by its name, then you should output the action sequence that corresponds to the composite skill.
Otherwise, interpret the human's instruction based on the rules of the game to generate your output.
Please provide your response in a list format: ['action 1', ..., 'action n'].
Do not output anything else.
'''

    completion = self.client.chat.completions.create(model=self.gpt_version,
                                                     messages=[{
                                                         "role":
                                                         "system",
                                                         "content":
                                                         system_prompt
                                                     }, {
                                                         "role":
                                                         "user",
                                                         "content":
                                                         user_prompt
                                                     }])
    response = completion.choices[0].message.content
    print('GPT repsponse: ', response)
    return ast.literal_eval(response.strip())

  def get_macro_list(self, env_tensor):
    # submit_time = time.time()
    # print('Prev intent: ', ALL_MOVES[self.prev_intent_idx])
    # action_idx = self.model.choose_action(
    #     torch.cat((env_tensor, torch.tensor([self.prev_intent_idx]))))
    # if self._task is None:
    #   self._task = deepcopy(MOVE_TO_HT[ALL_MOVES[action_idx]])
    #   finish_time = time.time()
    #   self._mov_hist.append({
    #       'task': str(self._task),
    #       'status': 'Ongoing. Initiated.',
    #       'submit_time': submit_time,
    #       'finish_time': finish_time
    #   })

    top_actions_w_names = {}
    top_actions = self.model.choose_top_actions(torch.cat(
        (env_tensor, torch.tensor([self.prev_intent_idx]))),
                                                k=len(self.all_moves))
    for action_idx in top_actions:
      top_actions_w_names[self.all_moves[action_idx]] = top_actions[action_idx]
    return top_actions_w_names

  def choose_il_macro_action(self, macro_actions: Dict[int, str], prep: Dict):
    if self._task is None:
      valid_moves = [move[0] for move in prep['chk_moves'] if move[1]]
      for action, _val in macro_actions.items():
        if action in valid_moves:
          self._tasks = [action]
          break

  def get_available_actions(self, env_tensor):
    return []
    # return list(self.get_macro_list(env_tensor).keys())[:15]

  def set_il_model(self, model_type, model):
    if model_type == 'none':
      self.model_type = 'none'
      return
    if model_type not in ['bc', 'bci', 'lsh', 'iql']:
      raise NotImplementedError
    self.model_type = model_type
    self.model = model

  def set_sit_pref(self, next_action=None):
    pass

  def prep_prompt(self):
    ret = {}

    ret['chk_moves'] = prep_chk_moves(self._last_env, self.user_reward)
    ret['order'] = prep_prompt_order(self._last_env)
    ret['map'] = prep_prompt_map(self._last_env)
    ret['sit_pref'] = self.sit_pref
    # Get the available actions that can be currently performed.
    # This can be a ground-truth action verfier or some sort of heuristics.
    ret['available_actions'] = self.get_available_actions(self.env_tensor)

    ret['int_hist'] = []
    ret['llm_hist'] = []
    ret['mov_hist'] = self._mov_hist[-5:]

    if self._is_finished:
      chat = "None"
    else:
      chat = self._int_hist[-1]['ret'] if self._int_hist else "None"
    ret['chatin'] = chat

    # The following are only used by GuidedAgent
    # A string indicating the human's preference
    ret['pref'] = self.pref
    # Whether we include a text description of the human's preference in the prompt
    ret['text_desc'] = self.text_desc
    # The operation performed when aggragating actions from the LM and those from
    # an IL model. Only used if consider_instr_reasoning is set to true
    ret['operation'] = self.operation
    # An imitation learning model's output
    ret['instr_reasoning'] = {}
    # Whether the LM considers the imitation learning model's output
    ret['consider_instr_reasoning'] = False
    # How the top actions are generated
    ret['gen_mode'] = self.gen_mode
    # Type of the environment
    ret['env_type'] = self.env_type
    # What the agent is currently holding
    if self._last_env.hold is not None:
      ret['holding'] = self._last_env.hold.full_name
    else:
      ret['holding'] = ''
    # Prev composite skill
    ret['prev_sit_pref'] = self.prev_sit_pref
    # The best action sequences for each composite skill
    if self.sit_pref_actions is not None:
      ret['sit_pref_actions'] = self.sit_pref_actions
    # How many top k composite skills the LLM uses
    ret['top_k_llm'] = self.setting.top_k_llm
    # Whether the LLM gives high level actions recommendations
    ret['no_hl_rec'] = self.setting.no_hl_rec
    # The type of prompts to use
    ret['prompt_style'] = self.setting.prompt_style
    # Input features for the IQL model
    ret['features'] = self._last_tensor
    # The high level prompt mode
    ret['hl_mode'] = self.setting.hl_mode

    return ret


class HierGuidedAgent(LLMAgent):
  '''
  The decison making processes are exactly the same as the HLAgent in agent.py.
  This is just subclassed under LLMAgent for to make it easier to run experiments.
  '''
  INT_HIST_MAX_LEN = 5 * 10000
  LLM_HIST_MAX_LEN = 5 * 10000
  MAX_LLM_UNG_MOV = 15
  MAX_LLM_FIN_MOV = 6
  MAX_MOV_UNG_MOV = 15
  MAX_MOV_FIX_MOV = 3

  def __init__(self, setting: AgentSetting, replay: Replay):
    super().__init__()
    self.setting = setting
    self.replay = replay

    self._lock = threading.Lock()
    self._last_env = None
    # high level
    self._int_hist = []  # submit_time, finish_time, chat, ret (intention)
    # int_time(finish_time), submit_time, finish_time, chat, ret {Chat, Demand, Finished}
    self._llm_hist = []
    self._num_high_threads = 0
    # low level
    self._mov_hist: list = []  # task, status, submit_time, finish_time

    # thread safe
    self._it_time = 0
    self._lt_time = 0  # last time
    self._mt_time = 0
    self._task = None

    self.cur_intent = None
    self.intent_hist = []

    self.operation = setting.operation
    self.pref = setting.pref
    self.interpolation = setting.interpolation

    self.env_type = 'overcooked'
    self.all_moves = ALL_MOVES

    # Whether to use the user reward
    self.user_reward = self.setting.user_reward

  @property
  def _is_finished(self):
    if not self._int_hist:
      return True
    intent = self._int_hist[-1]

    if intent['finish_time'] is None:
      return False
    intent_time = intent['finish_time']

    llms = self._llm_hist[-100:]
    llms = [l for l in llms if l['submit_time'] > intent_time]

    return any([l['ret']['Finished']
                for l in llms]) or len(llms) > self.MAX_LLM_UNG_MOV

  def _get_high_mov_hist(self):
    if self._is_finished:
      mov_his = self._mov_hist[-100:]
      mov_his = [
          m for m in mov_his if m['status'].startswith('Success')
          or m['status'].startswith('Ongoing')
      ]
      mov_his = mov_his[-self.MAX_LLM_FIN_MOV:]
    else:
      intent_time = self._int_hist[-1]['submit_time']
      mov_his = self._mov_hist[-100:]
      mov_his = [m for m in mov_his if m['finish_time'] > intent_time]
      mov_his = [
          m for m in mov_his if m['status'].startswith('Success')
          or m['status'].startswith('Ongoing')
      ]
      mov_his = mov_his[-self.MAX_LLM_UNG_MOV:]

    return mov_his

  def _get_low_mov_hist(self):
    if self._is_finished:
      mov_his = self._mov_hist[-100:]
      mov_his = [m for m in mov_his if m['status'].startswith('Success')]
      mov_his = mov_his[-self.MAX_MOV_FIX_MOV:]
    else:
      intent_time = self._int_hist[-1]['submit_time']
      mov_his = self._mov_hist[-100:]
      mov_his = [m for m in mov_his if m['finish_time'] > intent_time]
      mov_his = [m for m in mov_his if m['status'].startswith('Success')]
      mov_his = mov_his[-self.MAX_MOV_UNG_MOV:]

    return mov_his

  def _get_mov_infer_prep(self):
    if self._is_finished:
      ret = {'ret': {'Demand': '', 'Chat': ''}}
      if self._llm_hist:
        ret = {'ret': {'Demand': '', 'Chat': self._llm_hist[-1]['ret']['Chat']}}
    else:
      intent = self._int_hist[-1]
      if intent['finish_time'] is None:
        ret = {'ret': {'Demand': intent['chat'], 'Chat': ''}}
      else:
        ret = {'ret': {'Demand': intent['ret'], 'Chat': ''}}
    return [ret]

  def _int_infer(self, chat: str = ''):
    submit_time = time.time()
    with self._lock:
      self._int_hist.append({
          'submit_time': submit_time,
          'finish_time': None,
          'chat': chat,
          'ret': None
      })

      int_his = self._int_hist[-2:]

      prep = prep_prompt(self._last_env,
                         int_his, [], [],
                         '',
                         user_reward=self.user_reward)
      prep = deepcopy(prep)

      self._num_high_threads += 1

    try:
      js = request_client("Ei", self.setting.high_llm, prep)
    except:
      with self._lock:
        self._num_high_threads -= 1
      return
    # print(js)
    finish_time = time.time()

    # log
    self.replay.log("ai.int_infer", {
        "prep": prep,
        "ret": js,
        "time_start": submit_time,
        "time_end": finish_time
    })

    with self._lock:
      self._num_high_threads -= 1
      if js is None:
        js = "None"
      # no cross-time period
      intent = [i for i in self._int_hist if i['submit_time'] == submit_time]
      if not intent:
        return
      intent = intent[0]
      intent['finish_time'] = finish_time
      intent['ret'] = js
      self._int_hist = self._int_hist[-self.INT_HIST_MAX_LEN:]

    self._llm_infer()

  def _llm_infer(self):
    submit_time = time.time()
    with self._lock:
      # if int is inferring, return
      if self._int_hist and self._int_hist[-1]['finish_time'] is None:
        return
      # no new chat checker
      if self._is_finished:
        intention = "None"
      else:
        i = self._int_hist[-1]
        # (or in other words: {i['ret']})
        intention = f"{i['ret']}" if self._int_hist else "None"
      int_time = self._int_hist[-1]['submit_time'] if self._int_hist else 0

      # prepare moves history
      mov_his = self._get_high_mov_hist()

      prep = prep_prompt(self._last_env, [], [],
                         mov_his,
                         intention,
                         user_reward=self.user_reward)
      prep = deepcopy(prep)

      self._num_high_threads += 1

    try:
      js = request_client("El", self.setting.high_llm, prep)
    except:
      with self._lock:
        self._num_high_threads -= 1
      return
    finish_time = time.time()

    # log
    self.replay.log("ai.llm_infer", {
        "prep": prep,
        "ret": js,
        "time_start": submit_time,
        "time_end": finish_time
    })

    with self._lock:
      self._num_high_threads -= 1

      if js is None:
        return
      if self._int_hist and self._int_hist[-1]['finish_time'] is None:
        return

      # check intention out date
      int_time2 = self._int_hist[-1]['submit_time'] if self._int_hist else 0
      if int_time != int_time2:
        return
      # check llm out date
      if self._llm_hist:
        llm2 = self._llm_hist[-1]
        if llm2['submit_time'] > submit_time:
          return

      self._llm_hist.append({
          'submit_time': submit_time,
          'finish_time': finish_time,
          'chat': '',
          'ret': js
      })
      self._llm_hist = self._llm_hist[-self.LLM_HIST_MAX_LEN:]

  def high_level_infer(self, env: EnvState = None, chat: str = ''):
    if env is not None:
      with self._lock:
        self._last_env = env

    thread = threading.Thread(target=self._int_infer,
                              args=(chat, ),
                              daemon=True)
    thread.daemon = True
    thread.start()

  def low_level_infer(self, env_tensor):
    submit_time = time.time()

    llm_his = self._get_mov_infer_prep()
    mov_his = self._get_low_mov_hist()

    prep = prep_prompt(self._last_env, [],
                       llm_his,
                       mov_his,
                       '',
                       user_reward=self.user_reward)
    try:
      ht_dict = request_client("Em", self.setting.low_llm, prep)
    except:
      return

    ht = self.process_action_dict(ht_dict, env_tensor)
    finish_time = time.time()
    self.replay.log("ai.mov_infer", {
        "prep": prep,
        "ret": ht,
        "time_start": submit_time,
        "time_end": finish_time
    })
    self.cur_intent = ht
    self.add_intent_to_hist()
    self._task = deepcopy(MOVE_TO_HT[ht])
    self._mov_hist.append({
        'task': str(self._task),
        'status': 'Ongoing. Initiated.',
        'submit_time': submit_time,
        'finish_time': finish_time
    })

  def _check_interrupt(self):
    # check llm incoming
    if self._llm_hist and self._lt_time < self._llm_hist[-1]['finish_time']:
      # obtain chat
      chat = self._llm_hist[-1]['ret']['Chat']
      self._lt_time = self._llm_hist[-1]['finish_time']
    else:
      chat = ''

    # check int incoming
    if self._int_hist and self._it_time < self._int_hist[-1]['submit_time']:
      self._it_time = self._int_hist[-1]['submit_time']
      if self._it_time > self._lt_time:
        if self._mov_hist and self._mov_hist[-1]['status'].startswith(
            'Ongoing'):
          self._mov_hist[-1]['status'] = 'Interrupted. '
        self.set_prev_intent()
        self._task = None
    if self._int_hist and self._int_hist[-1]['finish_time'] is not None \
            and self._it_time < self._int_hist[-1]['finish_time']:
      self._it_time = self._int_hist[-1]['finish_time']
      if self._it_time > self._lt_time:
        if self._mov_hist and self._mov_hist[-1]['status'].startswith(
            'Ongoing'):
          self._mov_hist[-1]['status'] = 'Interrupted. '
        self.set_prev_intent()
        self._task = None

    return chat

  def step(self, env: EnvState, env_tensor):
    return self(env, env_tensor)

  def __call__(self, env: EnvState, env_tensor):
    self._lock.acquire()
    # update env
    self._last_env = env

    # check high level incoming
    chat = self._check_interrupt()

    # submit count check
    start_llm_infer = False
    if self._num_high_threads <= 0:
      if not self._llm_hist:
        start_llm_infer = True
      elif self._llm_hist[-1]['finish_time'] < time.time() - 5:
        start_llm_infer = True
      if not self._is_finished and self._mov_hist and self._mov_hist[-1][
          'submit_time'] != self._mt_time:
        start_llm_infer = True
        self._mt_time = self._mov_hist[-1]['submit_time']

    if start_llm_infer:
      self._lock.release()
      thread = threading.Thread(target=self._llm_infer, daemon=True)
      thread.daemon = True
      thread.start()
      self._lock.acquire()

    while True:
      if self._task is None:
        self.low_level_infer(env_tensor)
      if self._task is None:
        continue

      state, move, msg = self._task(env)
      if state == HighTask.Working:  # working
        self._mov_hist[-1]['status'] = 'Ongoing. ' + msg
        self._lock.release()
        return move, chat
      elif state == HighTask.Failed:  # reassign task
        self._mov_hist[-1]['status'] = 'Failed. ' + msg
        print(f"Move Failed: {self._mov_hist[-1]['task']}")
        self.set_prev_intent()
        self._task = None
        self._lock.release()
        return (0, 0), chat
      else:
        self._mov_hist[-1]['status'] = f'Success.'
        self.set_prev_intent()
        self._task = None

  def get_cur_intent(self):
    return self.cur_intent

  def add_intent_to_hist(self):
    self.intent_hist.append(self.cur_intent)

  def get_intent_hist(self):
    return self.intent_hist[-5:]

  def set_il_model(self, model_type, model):
    if model_type == 'none':
      self.model_type = 'none'
      return
    if model_type not in ['bc', 'bci', 'lsh', 'iql']:
      raise NotImplementedError
    self.model_type = model_type
    self.model = model


class QAgent_Overcooked:

  def __init__(self, user_reward):
    self.user_reward = user_reward
    self._last_env = None

    self._tasks = []
    self._task = None

  def low_level_infer(self):
    if not self._tasks:
      return

    ht = self._tasks[0]
    if ht is None:
      self._tasks.pop(0)
      return
    task = deepcopy(MOVE_TO_HT[ht])
    can_begin = task.can_begin(self._last_env)
    self._tasks.pop(0)

    if can_begin[0]:
      self._task = task

  def step(self, env, env_tensor):
    self._last_env = env
    while True:
      if self._task is None and self._tasks == []:
        all_moves = prep_chk_moves(self._last_env, self.user_reward)
        max_score = max(x[4] for x in all_moves)
        tied_moves = [m for m in all_moves if m[4] == max_score]
        chosen_move = random.choice(tied_moves)
        self._tasks.append(chosen_move[0])
        self.new_task = True
      else:
        self.new_task = False
      self.low_level_infer()

      if self._task is None:
        return (0, 0), ''

      state, move, msg = self._task(env)
      if state == HighTask.Working:  # working
        return move, ''
      elif state == HighTask.Failed:  # reassign task
        self._task = None
        return (0, 0), ''
      else:
        self._task = None

  def get_cur_intent(self):
    return str(self._task)


class SimHumanParent:

  def __init__(self,
               seed,
               do_other_orders,
               priority=None,
               plate_prob=1,
               cook_prob=1,
               combine_prob=1,
               chop_prob=1) -> None:
    """
    Priority should in the format of a list of lists of order names. 
    Soup orders in the same inner list has the same priorit, but orders in 
    the inner list at index 0 has higher priority than orders at index 1. 
    Orders not present in any inner list does not need to be completed.
    """
    random.seed(seed)
    self.cur_intent_idx = 0
    self.cur_intent = None
    self.intent_hist = []
    self.cur_move_idx = 0
    self._task = None
    self.priority = priority
    self.priority_flat = []
    for inner_list in self.priority:
      self.priority_flat.extend(inner_list)
    if do_other_orders:
      # Work on other orders not in the priority list as well if there is no
      # order from the priority list to work on
      all_orders = list(ORDER_NAMES.values())
      if len(self.priority_flat) < len(all_orders):
        for order in all_orders:
          if order not in self.priority_flat:
            self.priority_flat.append(order)
    print('Demonstrator priority list: ', self.priority_flat)

    self._lock = threading.Lock()

    self.plate_prob = plate_prob
    self.cook_prob = cook_prob
    self.combine_prob = combine_prob
    self.chop_prob = chop_prob

  def plate_and_serve(self, env: EnvState):
    order_names = self.get_order_names(env)
    pots = env.world.get_all_gridsquares('Pot')
    for pot in pots:
      if pot.holding is not None and pot.holding.is_cooked(
      ) and 'Fire' not in pot.holding.full_name and env.rch_map[
          pot.location[0]][pot.location[1]]:
        if random.random() < self.plate_prob:
          dish_name = OBJ_TO_GOODS_POT[pot.holding.full_name]
          if dish_name in order_names and dish_name in self.priority_flat:
            self.cur_intent = ('serve', dish_name)
            break
          else:
            if ('plate', dish_name) in INTENTS_TO_MOVES:
              self.cur_intent = ('plate', dish_name)
              break
        else:
          continue

  def cook_soup(self, env: EnvState):
    order_names = self.get_order_names(env)
    counters = env.world.get_all_gridsquares('Counter')
    for counter in counters:
      if counter.holding is not None:
        ingre_name = OBJ_TO_GOODS_GS[counter.holding.full_name]
        if ingre_name in INGRE_OF_INTEREST:
          if INGRE_TO_SOUP[ingre_name] in order_names and INGRE_TO_SOUP[
              ingre_name] in self.priority_flat:
            cook_task = HTCook(INGRE_TO_SOUP[ingre_name])
            if cook_task.can_begin(env)[0]:
              self.cur_intent = ('cook', INGRE_TO_SOUP[ingre_name])
              break

  def combine_ingredients(self, env: EnvState):
    order_names = self.get_order_names(env)

    for p in self.priority_flat:
      num_priority = self.get_num_priority_orders(env, p, order_names)
      if num_priority > 0:
        assemble_task = HTAssemble(ORDER_TO_INGRE[p])
        if assemble_task.can_begin(env)[0]:
          self.cur_intent = ('prepare', p)
        return

  def chop_ingredients(self, env: EnvState):
    order_names = self.get_order_names(env)

    for p in self.priority_flat:
      num_priority = self.get_num_priority_orders(env, p, order_names)
      if num_priority > 0:
        assemble_task = HTAssemble(ORDER_TO_INGRE[p])
        if not assemble_task.can_begin(env)[0]:
          missing_ingres = assemble_task.can_begin(env)[2]
          ingre = random.choice(missing_ingres).split(' ')[1]
          chop_task = HTChop(ingre)
          if chop_task.can_begin(env)[0]:
            self.cur_intent = ('chop', ingre)
            return

  def get_order_names(self, env: EnvState):
    current_orders = env.order.current_orders
    order_names = [order.full_name for order, _, _, _ in current_orders]
    return [ORDER_NAMES[name] for name in order_names]

  def step(self, env: EnvState, state=None):
    raise NotImplementedError

  def get_cur_intent(self):
    if self.cur_intent is not None:
      moves_list = INTENTS_TO_MOVES[self.cur_intent]
      if self.cur_move_idx < len(moves_list):
        return moves_list[self.cur_move_idx]

    return None

    # return self.cur_intent

  def add_intent_to_hist(self):
    if self.cur_intent is not None:
      moves_list = INTENTS_TO_MOVES[self.cur_intent]
      if self.cur_move_idx < len(moves_list):
        self.intent_hist.append(moves_list[self.cur_move_idx])

  def get_intent_hist(self):
    return self.intent_hist[-5:]

  def get_num_priority_orders(self, env: EnvState, priority_order: str,
                              order_names: list[str]):
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


class SimHumanPref(SimHumanParent):

  def __init__(self,
               seed,
               do_other_orders,
               priority=[['David Soup', 'Cathy Soup', 'Bob Soup',
                          'Alice Soup']],
               chop_prob=0.99):
    super().__init__(seed,
                     do_other_orders=do_other_orders,
                     priority=priority,
                     chop_prob=chop_prob)
    self.new_task = False

  def step(self, env: EnvState, state=None):
    self.new_task = False
    while True:
      while self._task is None:
        if self.cur_intent is None:
          self.plate_and_serve(env)

          if self.cur_intent is None:
            self.cook_soup(env)

          if self.cur_intent is None:
            self.combine_ingredients(env)

          if self.cur_intent is None:
            if random.random() < self.chop_prob:
              self.chop_ingredients(env)

          if self.cur_intent is None:
            self.cur_intent = ('Wait')

          moves_list = INTENTS_TO_MOVES[self.cur_intent]
          self._task = deepcopy(MOVE_TO_HT[moves_list[self.cur_move_idx]])
          self.new_task = True
          self.add_intent_to_hist()
        else:
          self.cur_move_idx += 1
          moves_list = INTENTS_TO_MOVES[self.cur_intent]
          if self.cur_move_idx == len(moves_list):
            self.cur_intent = None
            self.cur_move_idx = 0
          else:
            self._task = deepcopy(MOVE_TO_HT[moves_list[self.cur_move_idx]])
            self.new_task = True
            self.add_intent_to_hist()

      state, move, msg = self._task(env)
      if state == HighTask.Working:  # working
        return move, None
      elif state == HighTask.Failed:  # reassign task
        print(f"Move Failed: {move}")
        self._task = None
        return (0, 0), None
      else:
        self._task = None


def get_agent(sett: AgentSetting, replay: Replay):
  mapping = {'GuidedAgent': GuidedAgent, 'HierGuidedAgent': HierGuidedAgent}
  return mapping[sett.mode](sett, replay)
