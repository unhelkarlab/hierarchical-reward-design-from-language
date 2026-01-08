from pathlib import Path
import threading
import requests
import time
from copy import copy, deepcopy
import random
from dataclasses import dataclass, field
from typing import List
from collections.abc import Iterable

from agent.executor.low import EnvState
from agent.executor.high import HighTask, OBJ_TO_GOODS_GS, OBJ_TO_GOODS_POT
from agent.mind.prompt_local import MOVE_TO_HT, prep_prompt, prep_prompt_s
from agent.mind.call import low, high, mix_L, mix_L_new
from gym_cooking.utils.replay import Replay

import gym
from agent.mind.dqn import DeepQLearning


def request_client(mode, llm, data):
  if mode in ['L1l']:
    return mix_L(mode, data)
  elif mode in ['L1l_new']:
    return mix_L_new(data)
  elif mode in ['Ei', 'El', 'Hl', 'Ei_h', 'El_h', 'Hl_h']:
    return high(mode, data)
  elif mode in ['Em', 'Sm', 'Em_h']:
    return low(mode, data)
  else:
    raise NotImplementedError


AVAILABLE_ACTIONS = ['prepare', 'cook', 'serve', 'put out']
AVAILABLE_TARGETS = [
    'Alice Soup', 'Bob Soup', 'Cathy Soup', 'David Soup', 'fire'
]

AVAILABLE_INTENTS = []
for action in AVAILABLE_ACTIONS:
  for target in AVAILABLE_TARGETS:
    if action != 'put out' and target != 'fire':
      AVAILABLE_INTENTS.append((action, target))
# AVAILABLE_INTENTS.append(('put out', 'fire'))
# print(AVAILABLE_INTENTS)

INTENTS_TO_MOVES = {
    ('prepare', "Alice Soup"):
    ['Chop Lettuce', 'Chop Onion', 'Prepare Alice Ingredients'],
    ('prepare', "Bob Soup"):
    ['Chop Lettuce', 'Chop Tomato', 'Prepare Bob Ingredients'],
    ('prepare', "Cathy Soup"):
    ['Chop Tomato', 'Chop Onion', 'Prepare Cathy Ingredients'],
    ('prepare', "David Soup"):
    ['Chop Lettuce', 'Chop Onion', 'Chop Tomato', 'Prepare David Ingredients'],
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
    ('put out', 'fire'): ['Putout', 'Drop']
}

ORDER_NAMES = {
    "CookedLettuce-CookedOnion-Plate": "Alice Soup",
    "CookedLettuce-CookedTomato-Plate": "Bob Soup",
    "CookedOnion-CookedTomato-Plate": "Cathy Soup",
    "CookedLettuce-CookedOnion-CookedTomato-Plate": "David Soup",
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


@dataclass
class AgentSetting:
  mode: str
  high_llm: str = 'gpt-3.5'
  low_llm: str = 'llama'
  speed: float = 2.5
  other_agents_w_intents: list = field(default_factory=list)


class HLAagent2:
  INT_HIST_MAX_LEN = 5 * 10000
  LLM_HIST_MAX_LEN = 5 * 10000
  MAX_LLM_UNG_MOV = 15
  MAX_LLM_FIN_MOV = 6
  MAX_MOV_UNG_MOV = 15
  MAX_MOV_FIX_MOV = 3

  def __init__(self, setting: AgentSetting, replay: Replay):
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

    self.other_agents_w_intents = []

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

  def _get_own_intent_reasoning(self):
    if self._is_finished and self._llm_hist and self._llm_hist[-1]['ret'][
        'Reasoning'] is not None:
      return {'Reasoning': self._llm_hist[-1]['ret']['Reasoning']}
    else:
      return {'Reasoning': ''}

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
                         other_agents_w_intents=self.other_agents_w_intents)
      prep = deepcopy(prep)

      self._num_high_threads += 1

    try:
      js = request_client("Ei_h", self.setting.high_llm, prep)
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
                         other_agents_w_intents=self.other_agents_w_intents)
      prep = deepcopy(prep)

      self._num_high_threads += 1

    try:
      js = request_client("El_h", self.setting.high_llm, prep)
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

    thread = threading.Thread(target=self._int_infer, args=(chat, ))
    thread.daemon = True
    thread.start()

  def low_level_infer(self):
    submit_time = time.time()

    llm_his = self._get_mov_infer_prep()
    mov_his = self._get_low_mov_hist()
    ai_intent = self._get_own_intent_reasoning()['Reasoning']

    prep = prep_prompt(self._last_env, [],
                       llm_his,
                       mov_his,
                       '',
                       ai_intent=ai_intent,
                       other_agents_w_intents=self.other_agents_w_intents)
    try:
      ht = request_client("Em_h", self.setting.low_llm, prep)
    except:
      return
    finish_time = time.time()

    self.replay.log("ai.mov_infer", {
        "prep": prep,
        "ret": ht,
        "time_start": submit_time,
        "time_end": finish_time
    })
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
        self._task = None
    if self._int_hist and self._int_hist[-1]['finish_time'] is not None \
            and self._it_time < self._int_hist[-1]['finish_time']:
      self._it_time = self._int_hist[-1]['finish_time']
      if self._it_time > self._lt_time:
        if self._mov_hist and self._mov_hist[-1]['status'].startswith(
            'Ongoing'):
          self._mov_hist[-1]['status'] = 'Interrupted. '
        self._task = None

    return chat

  def __call__(self, env: EnvState):
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
      thread = threading.Thread(target=self._llm_infer)
      thread.daemon = True
      thread.start()
      self._lock.acquire()

    while True:
      if self._task is None:
        self.low_level_infer()
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
        self._task = None
        self._lock.release()
        return (0, 0), chat
      else:
        self._mov_hist[-1]['status'] = f'Success.'
        self._task = None


class HLAagent:
  INT_HIST_MAX_LEN = 5 * 10000
  LLM_HIST_MAX_LEN = 5 * 10000
  MAX_LLM_UNG_MOV = 15
  MAX_LLM_FIN_MOV = 6
  MAX_MOV_UNG_MOV = 15
  MAX_MOV_FIX_MOV = 3

  def __init__(self, setting: AgentSetting, replay: Replay):
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
    self.other_agents_w_intents = []

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

      prep = prep_prompt(self._last_env, int_his, [], [], '')
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

      prep = prep_prompt(self._last_env, [], [], mov_his, intention)
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

    thread = threading.Thread(target=self._int_infer, args=(chat, ))
    thread.daemon = True
    thread.start()

  def low_level_infer(self):
    submit_time = time.time()

    llm_his = self._get_mov_infer_prep()
    mov_his = self._get_low_mov_hist()

    prep = prep_prompt(self._last_env, [], llm_his, mov_his, '')
    try:
      ht = request_client("Em", self.setting.low_llm, prep)
    except:
      return
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
        self._task = None
    if self._int_hist and self._int_hist[-1]['finish_time'] is not None \
            and self._it_time < self._int_hist[-1]['finish_time']:
      self._it_time = self._int_hist[-1]['finish_time']
      if self._it_time > self._lt_time:
        if self._mov_hist and self._mov_hist[-1]['status'].startswith(
            'Ongoing'):
          self._mov_hist[-1]['status'] = 'Interrupted. '
        self._task = None

    return chat

  def __call__(self, env: EnvState):
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
      thread = threading.Thread(target=self._llm_infer)
      thread.daemon = True
      thread.start()
      self._lock.acquire()

    while True:
      if self._task is None:
        self.low_level_infer()
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
        self._task = None
        self._lock.release()
        return (0, 0), chat
      else:
        self._mov_hist[-1]['status'] = f'Success.'
        self._task = None

  def get_cur_intent(self):
    return self.cur_intent

  def add_intent_to_hist(self):
    self.intent_hist.append(self.cur_intent)

  def get_intent_hist(self):
    return self.intent_hist[-5:]


class SMOAagent2:
  """
  SMOA2 and SMOA3 are both language agents, but they don't take communication
  into account.
  """
  INT_HIST_MAX_LEN = 5 * 10000
  LLM_HIST_MAX_LEN = 5 * 10000
  MAX_LLM_UNG_MOV = 9
  MAX_LLM_FIN_MOV = 6

  def __init__(self, setting: AgentSetting, replay: Replay):
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
    self._tasks = []
    self._task = None

    self.cur_intent = None
    self.intent_hist = []
    self.other_agents_w_intents = []

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
        intention = self._int_hist[-1]['ret'] if self._int_hist else "None"
      int_time = self._int_hist[-1]['submit_time'] if self._int_hist else 0

      # prepare moves history
      # mov_his = self._get_high_mov_hist()
      mov_his = self._mov_hist[-5:]
      # mov_his = [m for m in mov_his
      #            if not m['status'].startswith('Interrupt')][-5:]

      prep = prep_prompt(self._last_env, [], [],
                         mov_his,
                         intention,
                         other_agents_w_intents=self.other_agents_w_intents)
      prep = deepcopy(prep)

      self._num_high_threads += 1

    try:
      js = request_client("Hl_h", self.setting.high_llm, prep)
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
      self._mov_hist.append({
          'task': str(task),
          'status': 'Failed. ' + can_begin[1],
          'submit_time': submit_time,
          'finish_time': None
      })

    while not can_begin[0] and len(can_begin[2]) > 0:
      task = deepcopy(MOVE_TO_HT[can_begin[2][0]])
      can_begin = task.can_begin(self._last_env)
    if can_begin[0]:
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
      self._mov_hist.append({
          'task': str(task),
          'status': 'Failed. ' + can_begin[1],
          'submit_time': submit_time,
          'finish_time': None
      })
      self._tasks.pop(0)
      return

  def _check_interrupt(self):
    if self._llm_hist and self._lt_time < self._llm_hist[-1]['finish_time']:
      # obtain chat
      chat = self._llm_hist[-1]['ret']['Chat']
      self._task = None
      if self._mov_hist and self._mov_hist[-1]['status'].startswith('Ongoing'):
        self._mov_hist[-1]['status'] = 'Interrupted. '
      self._tasks = [
          self._llm_hist[-1]['ret']['Action'],
      ]
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
        self._task = None
        self._tasks = []

    return chat

  def __call__(self, env: EnvState, _instr: str, _state_dict: dict):
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
      if self._task is None and not self._tasks:  # additional
        start_llm_infer = True

    if start_llm_infer:
      self._lock.release()
      thread = threading.Thread(target=self._llm_infer)
      thread.daemon = True
      thread.start()
      self._lock.acquire()

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
        self._lock.release()
        return move, chat
      elif state == HighTask.Failed:  # reassign task
        self._mov_hist[-1]['status'] = 'Failed. ' + msg
        print(f"Move Failed: {self._mov_hist[-1]['task']}")
        self._task = None
        self._lock.release()
        return (0, 0), chat
      else:
        self._mov_hist[-1]['status'] = f'Success.'
        self._task = None

  def get_cur_intent(self):
    return self.cur_intent

  def add_intent_to_hist(self):
    self.intent_hist.append(self.cur_intent)

  def get_intent_hist(self):
    return self.intent_hist[-5:]


class SMOAagent3:
  INT_HIST_MAX_LEN = 5 * 10000
  LLM_HIST_MAX_LEN = 5 * 10000
  MAX_LLM_UNG_MOV = 9
  MAX_LLM_FIN_MOV = 6

  def __init__(self, setting: AgentSetting, replay: Replay):
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
    self._tasks = []
    self._task = None

    self.cur_intent = None
    self.intent_hist = []

    self.first_call = True
    self.other_agents_w_intents = []

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
    submit_time = time.time()
    with self._lock:
      self._int_hist.append({
          'submit_time': submit_time,
          'finish_time': None,
          'chat': chat,
          'ret': None
      })

      int_his = self._int_hist[-2:]

      prep = prep_prompt(self._last_env, int_his, [], [], chat)
      prep = deepcopy(prep)

      self._num_high_threads += 1

    try:
      js = request_client("Ei", self.setting.high_llm, prep)
    except:
      with self._lock:
        self._num_high_threads -= 1
      return
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
        intention = self._int_hist[-1]['ret'] if self._int_hist else "None"
      int_time = self._int_hist[-1]['submit_time'] if self._int_hist else 0

      # prepare moves history
      mov_his = self._get_high_mov_hist()

      prep = prep_prompt(self._last_env, [], [],
                         mov_his,
                         intention,
                         ai_intent=self.get_cur_intent(),
                         other_agents_w_intents=self.other_agents_w_intents)
      prep = deepcopy(prep)

      self._num_high_threads += 1

    try:
      js = request_client("Hl_h", self.setting.high_llm, prep)
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
      self._check_interrupt()

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
    self.cur_intent = ht
    task = deepcopy(MOVE_TO_HT[ht])
    can_begin = task.can_begin(self._last_env)

    if can_begin[0]:
      self._tasks.pop(0)
      print('popppp')
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

    while not can_begin[0] and len(can_begin[2]) > 0:
      task = deepcopy(MOVE_TO_HT[can_begin[2][0]])
      can_begin = task.can_begin(self._last_env)
    if can_begin[0]:
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
      return

  def _check_interrupt(self):
    if self._llm_hist and self._lt_time < self._llm_hist[-1]['finish_time']:
      # obtain chat
      chat = self._llm_hist[-1]['ret']['Chat']
      # self._task = None
      if self._mov_hist and self._mov_hist[-1]['status'].startswith('Ongoing'):
        self._mov_hist[-1]['status'] = 'Interrupted. '
      self._tasks.append(self._llm_hist[-1]['ret']['Action'])
      if self.cur_intent is None:
        self.cur_intent = self._llm_hist[-1]['ret']['Action']
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
        self._task = None
        self._tasks = []

    return chat

  def __call__(self, env: EnvState):
    self._lock.acquire()
    print('current self tasks: ', self._tasks)
    # update env
    self._last_env = env

    # check high level incoming
    chat = self._check_interrupt()

    # submit count check
    start_llm_infer = False
    if self._num_high_threads <= 0:
      if not self.first_call and len(self._tasks) < 1:
        start_llm_infer = True
      if not self._llm_hist:
        start_llm_infer = True
        self.first_call = False
      # if self._task is None and not self._tasks:  # additional
      #   start_llm_infer = True

    if start_llm_infer:
      self._lock.release()
      thread = threading.Thread(target=self._llm_infer)
      thread.daemon = True
      thread.start()
      self._lock.acquire()

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
        self._lock.release()
        return move, chat
      elif state == HighTask.Failed:  # reassign task
        self._mov_hist[-1]['status'] = 'Failed. ' + msg
        print(f"Move Failed: {self._mov_hist[-1]['task']}")
        self._task = None
        self._lock.release()
        return (0, 0), chat
      else:
        self._mov_hist[-1]['status'] = f'Success.'
        self._task = None

  def get_cur_intent(self):
    return self.cur_intent

  def add_intent_to_hist(self):
    self.intent_hist.append(self.cur_intent)

  def get_intent_hist(self):
    return self.intent_hist[-5:]


class SMOAagent:
  INT_HIST_MAX_LEN = 5 * 10000
  LLM_HIST_MAX_LEN = 5 * 10000
  MAX_LLM_UNG_MOV = 9
  MAX_LLM_FIN_MOV = 6

  def __init__(self, setting: AgentSetting, replay: Replay):
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
    self._tasks = []
    self._task = None

    self.cur_intent = None
    self.intent_hist = []

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
    submit_time = time.time()
    with self._lock:
      self._int_hist.append({
          'submit_time': submit_time,
          'finish_time': None,
          'chat': chat,
          'ret': None
      })

      int_his = self._int_hist[-2:]

      prep = prep_prompt(self._last_env, int_his, [], [], chat)
      prep = deepcopy(prep)

      self._num_high_threads += 1

    try:
      js = request_client("Ei", self.setting.high_llm, prep)
    except:
      with self._lock:
        self._num_high_threads -= 1
      return
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
        intention = self._int_hist[-1]['ret'] if self._int_hist else "None"
      int_time = self._int_hist[-1]['submit_time'] if self._int_hist else 0

      # prepare moves history
      mov_his = self._get_high_mov_hist()

      prep = prep_prompt(self._last_env, [], [], mov_his, intention)
      prep = deepcopy(prep)

      self._num_high_threads += 1

    try:
      js = request_client("Hl", self.setting.high_llm, prep)
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

    while not can_begin[0] and len(can_begin[2]) > 0:
      task = deepcopy(MOVE_TO_HT[can_begin[2][0]])
      can_begin = task.can_begin(self._last_env)
    if can_begin[0]:
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
      return

  def _check_interrupt(self):
    if self._llm_hist and self._lt_time < self._llm_hist[-1]['finish_time']:
      # obtain chat
      chat = self._llm_hist[-1]['ret']['Chat']
      self._task = None
      if self._mov_hist and self._mov_hist[-1]['status'].startswith('Ongoing'):
        self._mov_hist[-1]['status'] = 'Interrupted. '
      self._tasks = [
          self._llm_hist[-1]['ret']['Action'],
      ]
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
        self._task = None
        self._tasks = []

    return chat

  def __call__(self, env: EnvState, _instr, _state_dict):
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
      if self._task is None and not self._tasks:  # additional
        start_llm_infer = True

    if start_llm_infer:
      self._lock.release()
      thread = threading.Thread(target=self._llm_infer)
      thread.daemon = True
      thread.start()
      self._lock.acquire()

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
        self._lock.release()
        return move, chat
      elif state == HighTask.Failed:  # reassign task
        self._mov_hist[-1]['status'] = 'Failed. ' + msg
        print(f"Move Failed: {self._mov_hist[-1]['task']}")
        self._task = None
        self._lock.release()
        return (0, 0), chat
      else:
        self._mov_hist[-1]['status'] = f'Success.'
        self._task = None

  def get_cur_intent(self):
    return self.cur_intent

  def add_intent_to_hist(self):
    self.intent_hist.append(self.cur_intent)

  def get_intent_hist(self):
    return self.intent_hist[-5:]


class FMOAagent:
  LLM_HIST_MAX_LEN = 5 * 10000
  MAX_MOV_UNG_MOV = 9
  MAX_MOV_FIX_MOV = 3

  def __init__(self, setting: AgentSetting, replay: Replay):
    self.setting = setting
    self.replay = replay

    self._last_env = None

    self._int_hist: list = []  # submit_time, finish_time, chat
    # submit_time, finish_time, chat, ret (Action, Chat)
    self._llm_hist: list = []
    self._mov_hist: list = []  # submit_time, finish_time, task, status

    # atom level
    self._task = None
    self._chat = ''

    self.cur_intent = None
    self.intent_hist = []

  @property
  def _is_finished(self):
    if not self._int_hist:
      return True
    intent = self._int_hist[-1]
    intent_time = intent['finish_time']

    llms = self._llm_hist[-100:]
    llms = [l for l in llms if l['submit_time'] > intent_time]

    return len(llms) > self.MAX_MOV_UNG_MOV

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

  def _infer(self):
    submit_time = time.time()

    chat = self._int_hist[-1][
        'chat'] if not self._is_finished and self._int_hist else ""
    mov_his = self._get_low_mov_hist()

    prep = prep_prompt(self._last_env, self._int_hist[-2:], [], mov_his, chat)

    try:
      js = request_client("L1l", self.setting.low_llm, prep)
    except:
      return
    finish_time = time.time()

    self.replay.log("ai.llm_infer", {
        "prep": prep,
        "ret": js,
        "time_start": submit_time,
        "time_end": finish_time
    })

    # drop long ago history
    self._llm_hist.append({
        'submit_time': submit_time,
        'finish_time': finish_time,
        'chat': chat,
        'ret': js
    })
    self._llm_hist = self._llm_hist[-self.LLM_HIST_MAX_LEN:]

    # update mov hist
    if self._mov_hist and self._mov_hist[-1]['status'].startswith('Ongoing'):
      self._mov_hist[-1]['status'] = 'Interrupted. '
    self._task = deepcopy(MOVE_TO_HT[js['Action']])
    self._chat = js['Chat']
    self._mov_hist.append({
        'task': str(self._task),
        'status': 'Ongoing. Initiated.',
        'submit_time': finish_time,
        'finish_time': finish_time
    })
    self.replay.log(
        "ai.mov_infer", {
            "prep": None,
            "ret": str(self._task),
            "time_start": finish_time,
            "time_end": finish_time
        })

  def high_level_infer(self, env: EnvState = None, chat: str = ''):
    if env is not None:
      self._last_env = env

    # update int
    submit_time = time.time()
    self._int_hist.append({
        'submit_time': submit_time,
        'finish_time': submit_time,
        'chat': chat
    })
    self.replay.log(
        "ai.int_infer", {
            "prep": {
                "chatin": chat
            },
            "ret": None,
            "time_start": submit_time,
            "time_end": submit_time
        })

    self._infer()

  def __call__(self, env: EnvState):
    # update env
    self._last_env = env

    while True:
      if self._task is None:
        self._infer()
      if self._task is None:
        continue

      state, move, msg = self._task(env)
      if state == HighTask.Working:  # working
        self._mov_hist[-1]['status'] = 'Ongoing. ' + msg
        chat = self._chat
        self._chat = ''
        return move, chat
      elif state == HighTask.Failed:  # reassign task
        self._mov_hist[-1]['status'] = 'Failed. ' + msg
        print(f"Move Failed: {self._mov_hist[-1]['task']}")
        self._task = None
        chat = self._chat
        self._chat = ''
        return (0, 0), chat
      else:
        self._mov_hist[-1]['status'] = f'Success.'
        self._task = None

  def get_cur_intent(self):
    return self.cur_intent

  def get_intent_hist(self):
    return self.intent_hist[-5:]


class FMOAagent2:
  LLM_HIST_MAX_LEN = 5 * 10000
  MAX_MOV_UNG_MOV = 9
  MAX_MOV_FIX_MOV = 3

  def __init__(self, setting: AgentSetting, replay: Replay):
    self.setting = setting
    self.replay = replay

    self._last_env = None

    self._int_hist: list = []  # submit_time, finish_time, chat
    # submit_time, finish_time, chat, ret (Action, Chat)
    self._llm_hist: list = []
    self._mov_hist: list = []  # submit_time, finish_time, task, status

    # atom level
    self._task = None
    self._chat = ''

    self.cur_intent = None
    self.intent_hist = []
    self.other_agents_w_intents = []

  @property
  def _is_finished(self):
    if not self._int_hist:
      return True
    intent = self._int_hist[-1]
    intent_time = intent['finish_time']

    llms = self._llm_hist[-100:]
    llms = [l for l in llms if l['submit_time'] > intent_time]

    return len(llms) > self.MAX_MOV_UNG_MOV

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

  def _infer(self):
    submit_time = time.time()

    chat = self._int_hist[-1][
        'chat'] if not self._is_finished and self._int_hist else "None"
    mov_his = self._get_low_mov_hist()

    prep = prep_prompt(self._last_env,
                       self._int_hist[-2:], [],
                       mov_his,
                       chat,
                       other_agents_w_intents=self.other_agents_w_intents)

    try:
      js = request_client("L1l_new", self.setting.low_llm, prep)
    except:
      return
    finish_time = time.time()

    self.replay.log("ai.llm_infer", {
        "prep": prep,
        "ret": js,
        "time_start": submit_time,
        "time_end": finish_time
    })

    # drop long ago history
    self._llm_hist.append({
        'submit_time': submit_time,
        'finish_time': finish_time,
        'chat': chat,
        'ret': js
    })
    self._llm_hist = self._llm_hist[-self.LLM_HIST_MAX_LEN:]

    # update mov hist
    if self._mov_hist and self._mov_hist[-1]['status'].startswith('Ongoing'):
      self._mov_hist[-1]['status'] = 'Interrupted. '
    self._task = deepcopy(MOVE_TO_HT[js['Action']])
    self._chat = js['Chat']
    self._mov_hist.append({
        'task': str(self._task),
        'status': 'Ongoing. Initiated.',
        'submit_time': finish_time,
        'finish_time': finish_time
    })
    self.replay.log(
        "ai.mov_infer", {
            "prep": None,
            "ret": str(self._task),
            "time_start": finish_time,
            "time_end": finish_time
        })

  def high_level_infer(self, env: EnvState = None, chat: str = ''):
    if env is not None:
      self._last_env = env

    # update int
    submit_time = time.time()
    self._int_hist.append({
        'submit_time': submit_time,
        'finish_time': submit_time,
        'chat': chat
    })
    self.replay.log(
        "ai.int_infer", {
            "prep": {
                "chatin": chat
            },
            "ret": None,
            "time_start": submit_time,
            "time_end": submit_time
        })

    self._infer()

  def __call__(self, env: EnvState):
    # update env
    self._last_env = env

    while True:
      if self._task is None:
        self._infer()
      if self._task is None:
        continue

      state, move, msg = self._task(env)
      if state == HighTask.Working:  # working
        self._mov_hist[-1]['status'] = 'Ongoing. ' + msg
        chat = self._chat
        self._chat = ''
        return move, chat
      elif state == HighTask.Failed:  # reassign task
        self._mov_hist[-1]['status'] = 'Failed. ' + msg
        print(f"Move Failed: {self._mov_hist[-1]['task']}")
        self._task = None
        chat = self._chat
        self._chat = ''
        return (0, 0), chat
      else:
        self._mov_hist[-1]['status'] = f'Success.'
        self._task = None

  def get_cur_intent(self):
    return self.cur_intent

  def get_intent_hist(self):
    return self.intent_hist[-5:]


class NEAagent:
  INT_HIST_MAX_LEN = 5 * 10000
  LLM_HIST_MAX_LEN = 5 * 10000
  MAX_LLM_UNG_MOV = 9
  MAX_LLM_FIN_MOV = 6
  MAX_MOV_UNG_MOV = 9
  MAX_MOV_FIX_MOV = 3

  def __init__(self, setting: AgentSetting, replay: Replay):
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

      prep = prep_prompt(self._last_env, int_his, [], [], '')
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

      prep = prep_prompt(self._last_env, [], [], mov_his, intention)
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

    thread = threading.Thread(target=self._int_infer, args=(chat, ))
    thread.daemon = True
    thread.start()

  def low_level_infer(self):
    submit_time = time.time()

    llm_his = self._get_mov_infer_prep()
    mov_his = self._get_low_mov_hist()

    prep = prep_prompt_s(self._last_env, [], llm_his, mov_his, '')
    # try:
    ht = request_client("Sm", self.setting.low_llm, prep)
    # except:
    #     return
    finish_time = time.time()

    self.replay.log("ai.mov_infer", {
        "prep": prep,
        "ret": ht,
        "time_start": submit_time,
        "time_end": finish_time
    })
    self._task = ht
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
        self._task = None
    if self._int_hist and self._int_hist[-1]['finish_time'] is not None \
            and self._it_time < self._int_hist[-1]['finish_time']:
      self._it_time = self._int_hist[-1]['finish_time']
      if self._it_time > self._lt_time:
        if self._mov_hist and self._mov_hist[-1]['status'].startswith(
            'Ongoing'):
          self._mov_hist[-1]['status'] = 'Interrupted. '
        self._task = None

    return chat

  def __call__(self, env: EnvState):
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
      thread = threading.Thread(target=self._llm_infer)
      thread.daemon = True
      thread.start()
      self._lock.acquire()

    while True:
      if self._task is None:
        self.low_level_infer()
      if self._task is None:
        continue

      move_map = dict(left=(-1, 0), right=(1, 0), up=(0, 1), down=(0, -1))
      move = move_map[self._task]

      self._mov_hist[-1]['status'] = f'Success.'
      self._task = None
      self._lock.release()
      return move, chat


class SimHuman:

  def __init__(self, seed) -> None:
    random.seed(seed)
    self.cur_intent_idx = 0
    self.cur_intent = None
    self.intent_hist = []
    self.cur_move_idx = 0
    self._task = None

    self._lock = threading.Lock()

    self.plate_prob = 0.2

  def step(self, env: EnvState):
    while True:
      while self._task is None:
        if self.cur_intent is None:
          current_orders = env.order.current_orders
          order_names = [order.full_name for order, _, _, _ in current_orders]
          order_names = [ORDER_NAMES[name] for name in order_names]

          pots = env.world.get_all_gridsquares('Pot')
          for pot in pots:
            if pot.holding is not None and pot.holding.is_cooked(
            ) and 'Fire' not in pot.holding.full_name:
              if random.random() < self.plate_prob:
                dish_name = OBJ_TO_GOODS_POT[pot.holding.full_name]
                if dish_name in order_names:
                  self.cur_intent = ('serve', dish_name)
                  break
                else:
                  if ('plate', dish_name) in INTENTS_TO_MOVES:
                    self.cur_intent = ('plate', dish_name)
                    break
              else:
                continue

          if self.cur_intent is None:
            counters = env.world.get_all_gridsquares('Counter')
            for counter in counters:
              if counter.holding is not None:
                ingre_name = OBJ_TO_GOODS_GS[counter.holding.full_name]
                if ingre_name in INGRE_OF_INTEREST:
                  if INGRE_TO_SOUP[ingre_name] in order_names:
                    self.cur_intent = ('cook', INGRE_TO_SOUP[ingre_name])
                    break

          if self.cur_intent is None:
            rand_order = random.choice(order_names)
            self.cur_intent = ('prepare', rand_order)

          moves_list = INTENTS_TO_MOVES[self.cur_intent]
          self._task = deepcopy(MOVE_TO_HT[moves_list[self.cur_move_idx]])

          self.add_intent_to_hist()
        else:
          self.cur_move_idx += 1
          moves_list = INTENTS_TO_MOVES[self.cur_intent]
          if self.cur_move_idx == len(moves_list):
            self.cur_intent = None
            self.cur_move_idx = 0
          else:
            self._task = deepcopy(MOVE_TO_HT[moves_list[self.cur_move_idx]])
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

  def get_cur_intent(self):
    if self.cur_intent is not None:
      moves_list = INTENTS_TO_MOVES[self.cur_intent]
      if self.cur_move_idx < len(moves_list):
        return moves_list[self.cur_move_idx] + ' to ' + self.cur_intent[
            0] + ' ' + self.cur_intent[1]

    return None

    # return self.cur_intent

  def add_intent_to_hist(self):
    if self.cur_intent is not None:
      moves_list = INTENTS_TO_MOVES[self.cur_intent]
      if self.cur_move_idx < len(moves_list):
        self.intent_hist.append(moves_list[self.cur_move_idx])

  def get_intent_hist(self):
    return self.intent_hist[-5:]


class LearningSMOA:
  """
  Incorporates results and a DQN, but still need to modify _int_infer to
  take communication into account.
  """
  INT_HIST_MAX_LEN = 5 * 10000
  LLM_HIST_MAX_LEN = 5 * 10000
  MAX_LLM_UNG_MOV = 9
  MAX_LLM_FIN_MOV = 6

  def __init__(self, setting: AgentSetting, replay: Replay):
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
    self._tasks = []
    self._task = None

    self.cur_intent = None
    self.intent_hist = []
    self.other_agents_w_intents = []

    self.env = gym.make('OvercookedCommunication-v0')
    self.dqn = DeepQLearning(self.env)
    self.state = None
    self.action = None
    self.reward = 0
    self.next_state = None
    self.new_transition = False
    self.comm = False

    self.action_failure_penalty = -2
    self.communication_penalty = -3

    self.cur_reward = 0
    self.cumulative_reward = 0

    self.num_infers = 0

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

  def _llm_infer(self):
    submit_time = time.time()
    with self._lock:
      latent = self.get_action()
      self.action = latent
      if self.action != 0:
        self.comm = True
      else:
        self.comm = False

      # if int is inferring, return
      if self._int_hist and self._int_hist[-1]['finish_time'] is None:
        return
      # no new chat checker
      if self._is_finished:
        intention = "None"
      else:
        intention = self._int_hist[-1]['ret'] if self._int_hist else "None"
      int_time = self._int_hist[-1]['submit_time'] if self._int_hist else 0

      # prepare moves history
      # mov_his = self._get_high_mov_hist()
      mov_his = self._mov_hist[-5:]
      # mov_his = [m for m in mov_his
      #            if not m['status'].startswith('Interrupt')][-5:]

      prep = prep_prompt(self._last_env, [], [],
                         mov_his,
                         intention,
                         other_agents_w_intents=self.other_agents_w_intents,
                         latent=latent)
      prep = deepcopy(prep)

      self._num_high_threads += 1

    try:
      print('start llm infer')
      self.num_infers += 1
      js = request_client("Hl_h", self.setting.high_llm, prep)
      self.new_transition = True
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

  def get_state(self):
    mov_his = self._mov_hist
    mov_his = [m for m in mov_his
               if not m['status'].startswith('Interrupt')][-5:]
    mov_his_start = 5 - len(mov_his)
    mov_his_idx = 0
    state = [0, 0, 0, 0, 0, 0]
    for i in range(5):
      if i >= mov_his_start:
        if mov_his[mov_his_idx]['status'].startswith('Failed'):
          state[i] = 1
        mov_his_idx += 1

    if self._is_finished:
      chat = None
    else:
      chat = self._int_hist[-1]['ret'] if self._int_hist else None

    if chat is not None:
      state[5] = 1

    return state

  def get_action(self):
    # Generate action with behavior policy
    self.state = self.get_state()
    return self.dqn.behavior_policy(self.state)

  def get_reward(self, comm=False, action_failed=False):
    game_reward = self.cumulative_reward - self.cur_reward
    self.cur_reward = self.cumulative_reward
    if comm:
      game_reward += self.communication_penalty
    if action_failed:
      game_reward += self.action_failure_penalty
    return game_reward

  def add_to_replay(self):
    self.dqn.replay_buffer.add(self.state, self.action, self.next_state,
                               self.reward, False)
    print('##### Added: ')
    print('##### state: ', self.state)
    print('##### action: ', self.action)
    print('##### next state: ', self.next_state)
    print('##### reward: ', self.reward)
    self.update_agent()

  def update_agent(self):
    self.dqn.current_step += 1
    self.dqn.update()

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
      self._mov_hist.append({
          'task': str(task),
          'status': 'Failed. ' + can_begin[1],
          'submit_time': submit_time,
          'finish_time': None
      })
      if self.new_transition:
        self.next_state = self.get_state()
        self.reward = self.get_reward(comm=self.comm, action_failed=True)
        self.add_to_replay()
        self.new_transition = False

    while not can_begin[0] and len(can_begin[2]) > 0:
      task = deepcopy(MOVE_TO_HT[can_begin[2][0]])
      can_begin = task.can_begin(self._last_env)
    if can_begin[0]:
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
      self._mov_hist.append({
          'task': str(task),
          'status': 'Failed. ' + can_begin[1],
          'submit_time': submit_time,
          'finish_time': None
      })
      if self.new_transition:
        self.next_state = self.get_state()
        self.reward = self.get_reward(comm=self.comm, action_failed=True)
        self.add_to_replay()
        self.new_transition = False
      self._tasks.pop(0)
      return

  def _check_interrupt(self):
    if self._llm_hist and self._lt_time < self._llm_hist[-1]['finish_time']:
      # obtain chat
      chat = self._llm_hist[-1]['ret']['Chat']
      self._task = None
      if self._mov_hist and self._mov_hist[-1]['status'].startswith('Ongoing'):
        self._mov_hist[-1]['status'] = 'Interrupted. '
      self._tasks = [
          self._llm_hist[-1]['ret']['Action'],
      ]
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
        self._task = None
        self._tasks = []

    return chat

  def __call__(self, env: EnvState):
    self._lock.acquire()
    print('num infers: ', self.num_infers)
    # update env
    self._last_env = env

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
      thread = threading.Thread(target=self._llm_infer)
      thread.daemon = True
      thread.start()
      self._lock.acquire()

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
        if self.new_transition:
          self.next_state = self.get_state()
          self.reward = self.get_reward(comm=self.comm)
          self.add_to_replay()
          self.new_transition = False
        self._lock.release()
        return move, chat
      elif state == HighTask.Failed:  # reassign task
        self._mov_hist[-1]['status'] = 'Failed. ' + msg
        print(f"Move Failed: {self._mov_hist[-1]['task']}")
        self._task = None
        if self.new_transition:
          self.new_transition = False
        self._lock.release()
        return (0, 0), chat
      else:
        self._mov_hist[-1]['status'] = f'Success.'
        self._task = None

  def get_cur_intent(self):
    return self.cur_intent

  def add_intent_to_hist(self):
    self.intent_hist.append(self.cur_intent)

  def get_intent_hist(self):
    return self.intent_hist[-5:]


def get_agent(sett: AgentSetting, replay: Replay):
  mapping = {
      'HLA2': HLAagent2,
      "HLA": HLAagent,
      "SMOA3": SMOAagent3,
      "SMOA2": SMOAagent2,
      "SMOA": SMOAagent,
      "FMOA": FMOAagent,
      "FMOA2": FMOAagent2,
      "NEA": NEAagent,
      'LearningSMOA': LearningSMOA
  }
  return mapping[sett.mode](sett, replay)
