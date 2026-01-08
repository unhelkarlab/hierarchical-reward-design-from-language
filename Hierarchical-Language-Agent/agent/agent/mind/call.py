import json
import os
import re
import html
from collections import defaultdict
from typing import List, Tuple
import ast
from collections import Counter
import random
import heapq
import math
from copy import deepcopy

import numpy as np

from agent.mind.prompt_local import ALL_MOVES
from agent.mind.llm_api import LLM_LLAMA_LOCAL, LLM_GPT_API
from agent.mind.prompt import (
    prep_mov_hist, prompt_base_Ei, prompt_base_Ei_w_human_intent,
    prompt_order_int, prompt_order, prompt_map, prompt_reason_Ei,
    prompt_reason_Ei_w_human_intent, prompt_base_El_s, prompt_base_El_s2,
    prompt_base_El_s2_w_human_intent, prompt_base_El_1,
    prompt_base_El_1_w_human_intent, prompt_base_El_2, prompt_base_El_22,
    prompt_base_El_22_w_human_intent, prompt_base_El_3, prompt_base_Hl_s,
    prompt_base_El_5, prompt_instr_reasoning, prompt_guided_reasoning,
    prompt_reasoning, prompt_composite_reasoning, prompt_reasoning_minigrid,
    prompt_reasoning_minigrid_bup, prompt_reasoning_rw4t,
    prompt_composite_reasoning_rw4t, progprompt_reasoning,
    progprompt_composite_reasoning)
from agent.mind.prompt_new import (Lang_Skill_Prompt_Overcooked,
                                   Lang_Composite_Prompt_Overcooked,
                                   Prog_Skill_Prompt_Overcooked,
                                   Prog_Composite_Prompt_Overcooked)
from rw4t.il_agents.llm.prompt_rw4t import (Lang_Skill_Prompt_RW4T,
                                            Lang_Composite_Prompt_RW4T,
                                            Prog_Skill_Prompt_RW4T,
                                            Prog_Composite_Prompt_RW4T,
                                            Lang_Action_Prompt_RW4T)

from pickup_multigoals import NAMES_TO_ACTIONS
from minigrid.envs.babyai.unlock import MACRO_ACTION_SPACE
import rw4t.utils as rw4t_utils


def Ei_prompt(prep: dict) -> List[List[str]]:
  order_prep = prep['order']
  int_hist = prep['int_hist']

  base = prompt_base_Ei()
  order = prompt_order_int(order_prep) + '\n'
  reason = prompt_reason_Ei(int_hist) + '\n'

  base[-1][0] += order
  base[-1][0] += reason

  base[-1][0] += "\n\nNow, your answer is:"

  # print('high level prompt: ', base)
  return base


def Ei_prompt_w_human_intent(prep: dict) -> List[List[str]]:
  base = prompt_base_Ei_w_human_intent()
  reason = prompt_reason_Ei_w_human_intent(prep) + '\n'
  base[-1][0] += reason
  return base


class chatter:
  MAX_RETRY_TIMES = 6

  def __init__(self, prompt: callable, prep: dict):
    self.init_prompt = prompt
    self.prep = prep
    self.hist = []

    self._res = None
    self._retry = 0

  def __call__(self, text=None):
    if text is not None:
      text = text.choices[0].message.content
      self.hist[-1].append(text)

    # first round
    if len(self.hist) == 0:
      chat = self.init_prompt(self.prep)
      self.hist = chat
      return None, self.hist
    else:
      self._res = text
      return self._res, None


class El_chatter(chatter):

  def __init__(self, prompt: callable, prep: dict):
    super().__init__(prompt, prep)
    self.has_request = self.prep['chatin'] != "None"

  def check_reasoning(self, text):
    self._res["Reasoning"] = text
    return False, prompt_base_El_2(self.prep)

  def check_chat(self, text):
    self._res["Chat"] = text

    if self.has_request:
      return False, prompt_base_El_3(self.prep)
    else:
      return True, ""

  def check_finished(self, text):
    if "yes" in text.lower():
      self._res["Finished"] = True
    elif "no" in text.lower():
      self._res["Finished"] = False
    else:
      return False, prompt_base_El_3(self.prep)
    return True, None

  def __call__(self, text=None):
    # print(text)
    if text is not None:
      text = text.choices[0].message.content
      self.hist[-1].append(text)

    # first round
    if len(self.hist) == 0:
      if self.has_request:
        chat = [[prompt_base_El_s(self.prep), "Ok"],
                [prompt_base_El_1(self.prep)]]
        self._res = {
            "Reasoning": None,
            "Chat": None,
            "Finished": None,
            # "Demand": None
        }
      else:
        chat = [[prompt_base_El_s2(self.prep), "Ok"],
                [prompt_base_El_22(self.prep)]]
        self._res = {
            "Reasoning": "",
            "Chat": None,
            "Finished": True,
            # "Demand": None
        }
      self.hist = chat
      return None, self.hist
    elif self._retry >= self.MAX_RETRY_TIMES:
      self._res = {
          "Reasoning": "ERROR",
          "Chat": "ERROR",
          "Finished": True,
          # "Demand": "ERROR"
      }
      return self._res, None
    else:  # proceed
      if self._res["Reasoning"] is None:
        ok, hint = self.check_reasoning(text)
      elif self._res["Chat"] is None:
        ok, hint = self.check_chat(text)
      elif self._res["Finished"] is None:
        ok, hint = self.check_finished(text)
      else:
        ok, hint = True, None
      if not ok:
        self.hist.append([hint])
        self._retry += 1
        return None, self.hist
      else:
        return self._res, None


class El_chatter_w_human_intent(chatter):

  def __init__(self, prompt: callable, prep: dict):
    super().__init__(prompt, prep)
    self.has_request = self.prep['chatin'] != "None"

  def check_reasoning(self, text):
    self._res["Reasoning"] = text
    return False, prompt_base_El_2(self.prep)

  def check_chat(self, text: str):
    added_text = False
    if ';' in text:
      sub_texts = text.split(';')
      if len(sub_texts) == 2:
        action, msg = sub_texts
        self._res['Chat'] = msg
        self._res['Reasoning'] = action
        added_text = True

    if not added_text:
      self._res["Chat"] = text

    if self.has_request:
      return False, prompt_base_El_3(self.prep)
    else:
      return True, ""

  def check_finished(self, text):
    if "yes" in text.lower():
      self._res["Finished"] = True
    elif "no" in text.lower():
      self._res["Finished"] = False
    else:
      return False, prompt_base_El_3(self.prep)
    return True, None

  def __call__(self, text=None):
    if text is not None:
      self.hist[-1].append(text)

    # first round
    if len(self.hist) == 0:
      if self.has_request:
        chat = [[prompt_base_El_s(self.prep), "Ok"],
                [prompt_base_El_1_w_human_intent(self.prep)]]
        self._res = {
            "Reasoning": None,
            "Chat": None,
            "Finished": None,
            # "Demand": None
        }
      else:
        chat = [[prompt_base_El_s2_w_human_intent(self.prep), "Ok"],
                [prompt_base_El_22_w_human_intent(self.prep)]]
        self._res = {
            "Reasoning": "",
            "Chat": None,
            "Finished": True,
            # "Demand": None
        }
      self.hist = chat
      return None, self.hist
    elif self._retry >= self.MAX_RETRY_TIMES:
      self._res = {
          "Reasoning": "ERROR",
          "Chat": "ERROR",
          "Finished": True,
          # "Demand": "ERROR"
      }
      return self._res, None
    else:  # proceed
      if self._res["Reasoning"] is None:
        ok, hint = self.check_reasoning(text)
      elif self._res["Chat"] is None:
        ok, hint = self.check_chat(text)
      elif self._res["Finished"] is None:
        ok, hint = self.check_finished(text)
      else:
        ok, hint = True, None
      if not ok:
        self.hist.append([hint])
        self._retry += 1
        return None, self.hist
      else:
        return self._res, None


class Hl_chatter(chatter):

  def check_reasoning(self, text):
    self._res["Reasoning"] = text
    return False, prompt_base_El_2(self.prep)

  def check_chat(self, text):
    self._res["Chat"] = text
    return False, prompt_base_El_3(self.prep)

  def check_finished(self, text):
    if "yes" in text.lower():
      self._res["Finished"] = True
    elif "no" in text.lower():
      self._res["Finished"] = False
    else:
      return False, prompt_base_El_3(self.prep)
    return False, prompt_base_El_5(self.prep)

  def check_action(self, text):
    text = text.replace('"', '').replace("`",
                                         "").replace("'",
                                                     "").replace(".",
                                                                 "").strip()
    # print('Action: ', text)
    if text not in ALL_MOVES:
      p = f"Action \"{text}\" is not available.\n"
      p += prompt_base_El_5(self.prep)
      print(p)
      return False, p
    self._res["Action"] = text
    return True, None

  def __call__(self, text=None):
    # print(text)
    if text is not None:
      self.hist[-1].append(text)

    # first round
    if len(self.hist) == 0:
      chat = [[prompt_base_Hl_s(self.prep), "Ok"],
              [prompt_base_El_1(self.prep)]]
      self._res = {
          "Reasoning": None,
          "Chat": None,
          "Finished": None,
          "Action": None
      }
      self.hist = chat
      return None, self.hist
    elif self._retry >= self.MAX_RETRY_TIMES:
      self._res = {
          "Reasoning": "ERROR",
          "Chat": "ERROR",
          "Finished": True,
          "Action": ALL_MOVES[0]
      }
      return self._res, None
    else:  # proceed
      if self._res["Reasoning"] is None:
        ok, hint = self.check_reasoning(text)
      elif self._res["Chat"] is None:
        ok, hint = self.check_chat(text)
      elif self._res["Finished"] is None:
        ok, hint = self.check_finished(text)
      elif self._res["Action"] is None:
        ok, hint = self.check_action(text)
      else:
        ok, hint = True, None
      if not ok:
        self.hist.append([hint])
        self._retry += 1
        return None, self.hist
      else:
        return self._res, None


class Hl_chatter_w_human_intent(chatter):

  def __init__(self, prompt: callable, prep: dict):
    super().__init__(prompt, prep)
    self.has_request = self.prep['chatin'] != "None"

  def check_reasoning(self, text):
    self._res["Reasoning"] = text
    return False, prompt_base_El_2(self.prep)

  def check_chat(self, text: str):
    added_text = False
    if ';' in text:
      sub_texts = text.split(';', 1)
      if len(sub_texts) == 2:
        action, msg = sub_texts
        action = action.replace('"',
                                '').replace("`",
                                            "").replace("'",
                                                        "").replace(".",
                                                                    "").strip()
        if action not in ALL_MOVES:
          p = f"Action \"{action}\" is not a valid action.\n"
          p += prompt_base_El_5(self.prep)
          print(p)
          return False, p
        self._res['Chat'] = msg
        self._res['Action'] = action
        added_text = True

    if not added_text:
      p = f"Response \"{text}\" is not in the right format.\n"
      p += prompt_base_El_5(self.prep)
      return False, p

    if self.has_request:
      return False, prompt_base_El_3(self.prep)
    else:
      return True, ""

  def check_finished(self, text):
    if "yes" in text.lower():
      self._res["Finished"] = True
    elif "no" in text.lower():
      self._res["Finished"] = False
    else:
      return False, prompt_base_El_3(self.prep)
    return True, None

  def __call__(self, text=None):
    print('Returned text: ', text)
    if text is not None:
      self.hist[-1].append(text)

    # first round
    if len(self.hist) == 0:
      if self.has_request:
        # chat = [[prompt_base_El_s(self.prep), "Ok"],
        #         [prompt_base_El_1_w_human_intent(self.prep)]]
        # self._res = {
        #     "Reasoning": None,
        #     "Chat": None,
        #     "Finished": None,
        #     "Action": None
        #     # "Demand": None
        # }
        chat = [[prompt_base_El_s2_w_human_intent(self.prep), "Ok"],
                [prompt_reason_Ei_w_human_intent(self.prep)]]
        self._res = {
            "Reasoning": "",
            "Chat": None,
            "Finished": True,
            "Action": None
            # "Demand": None
        }
      else:
        chat = [[prompt_base_El_s2_w_human_intent(self.prep), "Ok"],
                [prompt_base_El_22_w_human_intent(self.prep)]]
        self._res = {
            "Reasoning": "",
            "Chat": None,
            "Finished": True,
            "Action": None
            # "Demand": None
        }
      self.hist = chat
      return None, self.hist
    elif self._retry >= self.MAX_RETRY_TIMES:
      self._res = {
          "Reasoning": "ERROR",
          "Chat": "ERROR",
          "Finished": True,
          "Action": "ERROR"
          # "Demand": "ERROR"
      }
      return self._res, None
    else:  # proceed
      if self._res["Reasoning"] is None:
        ok, hint = self.check_reasoning(text)
      elif self._res["Chat"] is None:
        ok, hint = self.check_chat(text)
      elif self._res["Finished"] is None:
        ok, hint = self.check_finished(text)
      else:
        ok, hint = True, None
      if not ok:
        self.hist.append([hint])
        self._retry += 1
        return None, self.hist
      else:
        return self._res, None


def Em_prompt_ep(
    prep: dict) -> Tuple[List[List[str]], List[str], List[str], dict]:
  chk_moves = prep['chk_moves']
  llm_hist = prep['llm_hist']
  mov_hist = prep['mov_hist']

  available_moves = [m[0] for m in chk_moves if m[1]]
  # available_moves = [m[0] for m in chk_moves]

  prompts = []
  q0 = '''Game Situation:
You and another human player are playing a simplified version of the video game Overcooked. Your goal is to cooperatively finish a dynamically changing list of soup orders as fast as possible. The game has different orders from the original video game. There are two players: you (an AI assistant) and another human player. Your primary goal is to cooperate and make the human player feel engaged, happy, and satisfied while also earning more points.

Game Rules:
1. All available actions are: ''' + ', '.join(ALL_MOVES) + '''.
2. There is a changing list of soup orders, each with a time limit for completion. Completing an order on time earns a bonus, while failing to do so results in losing the bonus.
3. The inverse action sequence to finish soup orders:
    To finish Alice Soup order, you need to Serve Alice Soup, which needs Plate Alice Soup and Cook Alice Soup. Alice Soup can be done after you Prepare Alice Ingredients, which needs Chop Lettuce and Chop Onion.
    To finish Bob Soup order, you need to Serve Bob Soup, which needs Plate Bob Soup and Cook Bob Soup. Bob Soup can be done after you Prepare Bob Ingredients, which needs Chop Lettuce and Chop Tomato.
    To finish Cathy Soup order, you need to Serve Cathy Soup, which needs Plate Cathy Soup and Cook Cathy Soup. Cathy Soup can be done after you Prepare Cathy Ingredients, which needs Chop Onion and Chop Tomato.
    To finish David Soup order, you need to Serve David Soup, which needs Plate David Soup and Cook David Soup. David Soup can be done after you Prepare David Ingredients, which needs Chop Lettuce, Chop Onion, and Chop Tomato.
4. If a cooked soup remains in the pot for a long time, it becomes charred, and the pot catches fire.
    a. Putout: To regain the pot, you must extinguish the fire.
    b. Drop: If a soup becomes charred, you must discard it.

Let's say you're playing this game and it's been a while. The human may specify his demand, and maybe you have some planning. Now you need to give your actions based on them.
Please note that when you carry out an action, you just do it once. If you want to do it multiple times, you need to repeat it multiple times. If there is many subtasks in the human's demand, you need to finish them in order. 
If the demand contains "Stop xxx" or "Avoid xxx", you should never do it. 
If the demand contains "Focus xxx", "Keep xxx" or "Always xxx", then you should always do it, and never doing other actions.

'''
  prompts.append([q0, "Ok."])

  qf = ''
  llm = llm_hist[-1]['ret']
  if llm['Demand'] != '':
    qf += f"The human's demand is:\n"
    qf += f"{llm['Demand']}\n"
    qf += '\n'
  if llm['Chat'] != '':
    qf += f"Your planning:\n"
    qf += f"{llm['Chat']}\n"
    qf += '\n'

  print('llama prompt: ', qf)

  chosen_actions = [a['task'] for a in mov_hist]
  chosen_actions = ', '.join(chosen_actions)

  if len(chosen_actions) > 0:
    af = "My actions are: " + chosen_actions
    choices = [f", {m}" for m in available_moves]
  else:
    af = "My actions are: "
    choices = [f"{m}" for m in available_moves]

  prompts.append([qf, af])

  prob_base = defaultdict(lambda: 0)

  ratio = 0.2 if llm['Demand'] != '' else 100
  # ratio = 0.0 if llm['Demand'] != '' else 0.0

  for m in chk_moves:
    prob_base[m[0]] -= m[4] * ratio

  return prompts, choices, available_moves, prob_base


def Em_prompt_ep_w_human_intent(
    prep: dict) -> Tuple[List[List[str]], List[str], List[str], dict]:
  chk_moves = prep['chk_moves']
  llm_hist = prep['llm_hist']
  mov_hist = prep['mov_hist']
  human_intents = prep['human_intents']
  ai_intent = prep['ai_intent']

  available_moves = [m[0] for m in chk_moves if m[1]]

  prompts = []
  q0 = '''Game Situation:
You and another human player are playing a simplified version of the video game Overcooked. 
Your goal is to cooperatively finish a dynamically changing list of soup orders as fast as possible.

Game Rules:
1. All available actions are: [''' + ', '.join(ALL_MOVES) + '''].

Let's say you're playing this game and it's been a while.
You received the following command to be executed.
Based on the command, please map it to one the available actions from the list above.
'''

  # prompts.append([q0, "Ok."])

  qf = ''
  llm = llm_hist[-1]['ret']
  if llm['Demand'] != '':
    qf += f"The human's demand is:\n"
    qf += f"{llm['Demand']}\n"
    qf += '\n'
  # if llm['Chat'] != '':
  #   qf += f"Your message:\n"
  #   qf += f"{llm['Chat']}\n"
  #   qf += '\n'
  if ai_intent != '':
    q0 += f"Command:\n"
    q0 += f"{ai_intent}\n"
    q0 += '\n'

  prompts.append([q0, "Ok."])

  chosen_actions = [a['task'] for a in mov_hist]
  chosen_actions = ', '.join(chosen_actions)

  if len(chosen_actions) > 0:
    # af = "My actions are: " + chosen_actions
    af = ''
    choices = [f", {m}" for m in available_moves]
  else:
    # af = "My actions are: "
    af = ''
    choices = [f"{m}" for m in available_moves]

  prompts.append([qf, af])

  prob_base = defaultdict(lambda: 0)

  ratio = 0.2 if llm['Demand'] != '' else 100.0

  for m in chk_moves:
    prob_base[m[0]] -= m[4] * ratio

  return prompts, choices, available_moves, prob_base


def L1_prompt_ep(prep: dict):
  chk_moves = prep['chk_moves']
  order_prep = prep['order']
  env = prep['map']
  int_hist = prep['int_hist']
  llm_hist = prep['llm_hist']
  mov_hist = prep['mov_hist']
  chat = prep['chatin']

  prompts = []
  q0 = '''Game Situation:
You and another human player are playing a simplified version of the video game Overcooked. Your goal is to cooperatively finish a dynamically changing list of soup orders as fast as possible. The game has different orders from the original video game. There are two players: you (an AI assistant) and another human player. Your primary goal is to cooperate and make the human player feel engaged, happy, and satisfied while also earning more points.

Game Rules:
1. All available actions are: ''' + ', '.join(ALL_MOVES) + '''.
2. There is a changing list of soup orders, each with a time limit for completion. Completing an order on time earns a bonus, while failing to do so results in losing the bonus.
3. The inverse action sequence to finish soup orders:
    To finish Alice Soup order, you need to Serve Alice Soup, which needs Plate Alice Soup and Cook Alice Soup. Alice Soup can be done after you Prepare Alice Ingredients, which needs Chop Lettuce and Chop Onion.
    To finish Bob Soup order, you need to Serve Bob Soup, which needs Plate Bob Soup and Cook Bob Soup. Bob Soup can be done after you Prepare Bob Ingredients, which needs Chop Lettuce and Chop Tomato.
    To finish Cathy Soup order, you need to Serve Cathy Soup, which needs Plate Cathy Soup and Cook Cathy Soup. Cathy Soup can be done after you Prepare Cathy Ingredients, which needs Chop Onion and Chop Tomato.
    To finish David Soup order, you need to Serve David Soup, which needs Plate David Soup and Cook David Soup. David Soup can be done after you Prepare David Ingredients, which needs Chop Lettuce, Chop Onion, and Chop Tomato.
4. If a cooked soup remains in the pot for a long time, it becomes charred, and the pot catches fire.
    a. Putout: To regain the pot, you must extinguish the fire.
    b. Drop: If a soup becomes charred, you must discard it.

Let's say you're playing this game and it's been a while. The human may specify his demand, and maybe you have some planning. Now you need to give your actions based on them.
Please note that when you carry out an action, you just do it once. If you want to do it multiple times, you need to repeat it multiple times. If there is many subtasks in the human's demand, you need to finish them in order. 
If the demand contains "Stop xxx" or "Avoid xxx", you should never do it. 
If the demand contains "Focus xxx", "Keep xxx" or "Always xxx", then you should always do it, and never doing other actions.
'''
  prompts.append([q0, "Ok."])

  prompts.append([""])

  # new round
  qf = ''
  if chat != '':
    if len(int_hist) == 2:
      qf += "The human player's demand in the last round (which has already been satisfied):\n"
      qf += f'"{int_hist[0]["chat"]}"\n\n'
    qf += f"The human player's demand is:\n"
    qf += f"{chat}\n"
    qf += '\n'

  qf += prompt_order(order_prep)
  qf += prompt_map(env)

  # answer
  chosen_actions = [a['task'] for a in mov_hist]
  chosen_actions = ', '.join(chosen_actions)
  if len(chosen_actions) > 0:
    af = "My actions are: " + chosen_actions + ','
  else:
    af = "My actions are:"

  prompts[-1][0] += qf
  prompts[-1].append(af)

  available_moves = [m[0] for m in chk_moves if m[1]]
  choices = [f" {m}" for m in available_moves]

  prob_base = defaultdict(lambda: 0)

  ratio = 0.2 if chat != '' else 100.0

  for m in chk_moves:
    prob_base[m[0]] -= m[4] * ratio

  return prompts, choices, available_moves, prob_base


def L1_prompt_chat(prep: dict, next_action: str):
  chk_moves = prep['chk_moves']
  order_prep = prep['order']
  env = prep['map']
  int_hist = prep['int_hist']
  llm_hist = prep['llm_hist']
  mov_hist = prep['mov_hist']
  chat = prep['chatin']

  prompts = []

  q0 = '''Game Scenario:
As an AI assistant in a simplified Overcooked game, work with a human player to complete soup orders. Focus on cooperation, player engagement, fulfillment, and point accrual.
Game Guidelines:
Current orders for soup vary, each with a time limit. Earn a bonus for on-time completion.
To make a soup: 
    a. Chop fresh vegetables - Tomato, Lettuce, Onion to obtain chopped vegetables. 
    b. Prepare soup ingredients with chopped vegetables once all required types are ready.
        Alice: Chopped Lettuce, Onion.
        Bob: Chopped Lettuce, Tomato.
        Cathy: Chopped Onion, Tomato.
        David: Chopped Lettuce, Onion, Tomato. 
    c. Cook the soup. Cooking starts once the required ingredients are ready.
        Alice Soup: Alice Ingredients.
        Bob Soup: Bob Ingredients.
        Cathy Soup: Cathy Ingredients.
        David Soup: David Ingredients.
    d. Plate the cooked soup.
    e. Serve the plated soup in the serving area for a shared bonus.
If a soup stays in the pot too long, it gets charred. 
    a. Putout: If the pot catches fire, extinguish it. 
    b. Drop: Discard charred soup. Put out the fire in the pot if needed.

Assuming that you have been playing the game for a while. Now you will be informed of the current situation, and need to generate your chat message to be sent to the human player.
If the human player raises a question, you must answer it. If not, you are recommended to give your future plan, information about current orders and their time limit.
You answer must be concrete and informative with no more than 10 words. Just give your chat message with no explanation, no comments, no quotation marks and no emojis.
'''
  prompts.append([q0, "Ok."])
  prompts.append([""])

  # new round
  chosen_actions = [a['task'] for a in mov_hist] + [next_action]
  chosen_actions = prep_mov_hist(chosen_actions)
  chosen_actions = ', '.join(chosen_actions)

  qf = ''
  qf += prompt_order(order_prep) + '\n'
  qf += env + '\n'
  if chat != '':
    if len(int_hist) == 2:
      qf += "The human player's demand in the last round (which has already been satisfied):\n"
      qf += f'"{int_hist[0]["chat"]}"\n\n'
    qf += f"The human player's incoming message:\n"
    qf += f"{chat}\n"
    qf += '\n'
    qf += "Actions you've done since the human gave the message: " + chosen_actions + '\n'
    qf += "\n"
    qf += '''Generate your chat message to be send to the human. Your communication should be polite, helpful. Aim to demonstrate your enthusiasm and friendliness while assisting the player. 
If the human player asks a question, ensure to provide an appropriate response. For example, if he asks "What are the current orders?", you should respond with the current orders and their time remaining.
You also have the opportunity to inform the player of your current and planned actions. 
Just give your message, with no quotation marks or emojis.
'''
  else:
    qf += "Actions you've done recently: \n" + chosen_actions + '\n'
    qf += "\n"
    qf += "Now give your chat message to be sent to the human.\n"

  prompts[-1][0] += qf

  return prompts


def Sm_prompt_ep(
    prep: dict) -> Tuple[List[List[str]], List[str], List[str], dict]:
  chk_moves = prep['chk_moves']
  order_prep = prep['order']
  env = prep['map']
  int_hist = prep['int_hist']
  llm_hist = prep['llm_hist']
  mov_hist = prep['mov_hist']
  chat = prep['chatin']

  prompts = []
  q0 = '''Game Situation:
You and another human player are playing a simplified version of the video game Overcooked. Your goal is to cooperatively finish a dynamically changing list of soup orders as fast as possible. The game has different orders from the original video game. There are two players: you (an AI assistant) and another human player. Your primary goal is to cooperate and make the human player feel engaged, happy, and satisfied while also earning more points.

Game Rules:
1. All available actions are: left, right, up, down, which will change your location by (-1, 0), (1, 0), (0, 1) and (0, -1) respectively. When you stand next to a grid, you can move towards it to interactive with it, for example, pick up things from table or cook a soup.
2. There is a changing list of soup orders, each with a time limit for completion. Completing an order on time earns a bonus, while failing to do so results in losing the bonus.
3. The inverse action sequence to finish soup orders:
    To finish Alice Soup order, you need to Serve Alice Soup, which needs Plate Alice Soup and Cook Alice Soup. Alice Soup can be done after you Prepare Alice Ingredients, which needs Chop Lettuce and Chop Onion.
    To finish Bob Soup order, you need to Serve Bob Soup, which needs Plate Bob Soup and Cook Bob Soup. Bob Soup can be done after you Prepare Bob Ingredients, which needs Chop Lettuce and Chop Tomato.
    To finish Cathy Soup order, you need to Serve Cathy Soup, which needs Plate Cathy Soup and Cook Cathy Soup. Cathy Soup can be done after you Prepare Cathy Ingredients, which needs Chop Onion and Chop Tomato.
    To finish David Soup order, you need to Serve David Soup, which needs Plate David Soup and Cook David Soup. David Soup can be done after you Prepare David Ingredients, which needs Chop Lettuce, Chop Onion, and Chop Tomato.
4. If a cooked soup remains in the pot for a long time, it becomes charred, and the pot catches fire.
    a. Putout: To regain the pot, you must extinguish the fire.
    b. Drop: If a soup becomes charred, you must discard it.

Let's say you're playing this game and it's been a while. The human may specify his demand, and maybe you have some planning. Now you need to give your actions based on them.
Please note that when you carry out an action, you just do it once. If you want to do it multiple times, you need to repeat it multiple times. If there is many subtasks in the human's demand, you need to finish them in order. 
If the demand contains "Stop xxx" or "Avoid xxx", you should never do it. 
If the demand contains "Focus xxx", "Keep xxx" or "Always xxx", then you should always do it, and never doing other actions.

'''
  prompts.append([q0, "Ok."])

  qf = ''

  # map info
  qf += env + "\n"

  # chat/command info
  llm = llm_hist[-1]['ret']
  if llm['Demand'] != '':
    qf += f"The human's demand is:\n"
    qf += f"{llm['Demand']}\n"
    qf += '\n'
  if llm['Chat'] != '':
    qf += f"Your planning:\n"
    qf += f"{llm['Chat']}\n"
    qf += '\n'

  chosen_actions = ["left", "right", "up", "down"]
  choices = [f" {m}" for m in chosen_actions]

  af = "My action is to move towards"

  prompts.append([qf, af])
  prob_base = defaultdict(lambda: 0)

  ratio = 0.2 if llm['Demand'] != '' else 100.0

  for m in chk_moves:
    prob_base[m[0]] -= m[4] * ratio

  return prompts, choices, chosen_actions, prob_base


nodes = [
    {
        'chat': f'http://{os.environ["LLAMA_ADDRESS"]}/api/v1/chat',
        'chateval': f'http://{os.environ["LLAMA_ADDRESS"]}/api/v1/chateval'
    },
]
LLM_LOCAL = LLM_LLAMA_LOCAL(nodes)
LLM_HIGH_3 = LLM_GPT_API(
    ['gpt-4o-2024-08-06'],  # 'gpt-4o', 'gpt-4o-2024-08-06'
    os.environ['OPENAI_API_KEY'],
    getattr(os.environ, 'OPENAI_ORGANIZATION', ""))


def low(mode, prep):
  LOW_LLM_PRESENTS = {
      "Em": Em_prompt_ep,
      "Sm": Sm_prompt_ep,
      'Em_h': Em_prompt_ep_w_human_intent
  }
  prompts, choices, am, pb = LOW_LLM_PRESENTS[mode](prep)
  score = LLM_LOCAL.eval_prob(prompts, choices)
  # print('Score: ', score)

  for idx in range(len(choices)):
    # print(
    #     f'{am[idx]}: {score[idx]} - {pb[am[idx]]} = {score[idx] - pb[am[idx]]}')
    score[idx] = score[idx] - pb[am[idx]]
    # score[idx] = score[idx] - 0

  ret_dict = {}
  print("############## LOW level infer ##############")
  for idx in range(len(choices)):
    ret_dict[am[idx]] = score[idx]
    print(f"      {am[idx]}: {score[idx]:.2f}")
  print("MAX: ", am[np.argmax(score)])
  print("################# LOW end  ##################\n\n")

  # # argmax
  # ret = am[np.argmax(score)]

  # return ret

  # Return the score dictionary instead of just the top action
  return ret_dict


def high(mode, prep):
  global chatter, El_chatter, Hl_chatter
  HIGH_LLM_PRESENTS = {
      "Ei": (chatter, Ei_prompt),
      "El": (El_chatter, None),
      "Hl": (Hl_chatter, None),
      "Ei_h": (chatter, Ei_prompt_w_human_intent),
      "El_h": (El_chatter_w_human_intent, None),
      "Hl_h": (Hl_chatter_w_human_intent, None),
  }
  cha, init_prompt = HIGH_LLM_PRESENTS[mode]
  js = LLM_HIGH_3(cha(init_prompt, prep))

  print("%%%%%%%%%%%%%% HIGH level infer %%%%%%%%%%%%%%")
  print(mode, " :", js)
  print("%%%%%%%%%%%%%%%%%% HIGH end %%%%%%%%%%%%%%%%%%\n\n")

  return js


def mix_L(mode, prep):
  prompts, choices, am, pb = L1_prompt_ep(prep)
  score = LLM_LOCAL.eval_prob(prompts, choices)

  for idx in range(len(choices)):
    score[idx] = score[idx] - pb[am[idx]]

  print("############## LOW level infer ##############")
  for idx in range(len(choices)):
    print(f"      {am[idx]}: {score[idx]:.2f}")
  print("################## LOW end  #################\n\n")

  action = am[np.argmax(score)]

  prompts = L1_prompt_chat(prep, action)
  chato = LLM_LOCAL._chat(prompts[-1][0], prompts[:-1])

  chato = html.unescape(chato).replace('"', '')

  chato_rep = chato.replace('\U0001f44d', '').replace('\U0001f44b', '')
  print(f"Chat: {chato_rep}")

  return {"Action": action, "Chat": chato}


def mix_L_new(prep):
  js = LLM_LOCAL(Hl_chatter_w_human_intent(None, prep))

  print("%%%%%%%%%%%%%% LlaMa infer %%%%%%%%%%%%%%")
  print("LlaMa :", js)
  print("%%%%%%%%%%%%%%%%%% HIGH end %%%%%%%%%%%%%%%%%%\n\n")

  return js


def instr_reasoning(prep):
  js = LLM_HIGH_3(chatter(prompt_instr_reasoning, prep))

  print("%%%%%%%%%%%%%% Instruction reasoning %%%%%%%%%%%%%%")
  print("Instruction reasoning:", js)
  print("%%%%%%%%%%%%%%%%%% Reasoning end %%%%%%%%%%%%%%%%%%\n\n")

  return js


def guided_hl(prep):
  if prep['gen_mode'] == 'top':
    LLM_HIGH_3.n = 5
    LLM_HIGH_3.temp = 1.9
  js = LLM_HIGH_3(guided_Hl_chatter(None, prep))

  print("%%%%%%%%%%%%%% Guided high level infer %%%%%%%%%%%%%%")
  print("Instruction reasoning:", js)
  print("%%%%%%%%%%%%%%%%%%% Reasoning end %%%%%%%%%%%%%%%%%%%\n\n")

  return js


class guided_Hl_chatter(chatter):

  def __init__(self, prompt: callable, prep: dict):
    super().__init__(prompt, prep)
    self.addition_alpha = 2  # 1.5
    self.top_k = 2
    self.verifier_error_rate = 0.0

    self.operation = prep['operation']
    self.consider_il_output = prep['consider_instr_reasoning']
    if self.consider_il_output:
      self.il_output = prep['instr_reasoning']
    else:
      self.il_output = {}

    self.input_hist = []

  def check_reasoning(self, gpt_ret):
    '''
    Save the llm's choice of the composite skill to execute.

    When self._res(ponse) is returned to the agent, the agent then updates its
    composite skill record to reflect the llm's choice.
    '''
    reasoning = gpt_ret.choices[0].message.content.strip()
    print('Reasoning: ', reasoning)
    self._res['Reasoning'] = [reasoning]  # Will be returned to the agent
    prep = deepcopy(self.prep)
    prep['sit_pref'] = [reasoning]
    if self.prep['env_type'] == 'rw4t':
      if self.prep['prompt_style'] == 'lang':
        return False, Lang_Skill_Prompt_RW4T(prep).return_prompt()
      else:
        return False, Prog_Skill_Prompt_RW4T(prep).return_prompt()
    else:
      if self.prep['prompt_style'] == 'lang':
        return False, Lang_Skill_Prompt_Overcooked(prep).return_prompt()
      else:
        return False, Prog_Skill_Prompt_Overcooked(prep).return_prompt()

  def check_chat_prog(self, gpt_ret):
    if self.prep['gen_mode'] != '5_unranked':
      print('Not Implemented check chat 1')
      raise NotImplementedError

    gpt_ret_str = gpt_ret.choices[0].message.content.strip()
    print('GPT ret str: ', gpt_ret_str)
    if self.prep['env_type'] == 'overcooked':
      action_name = check_chat_oc(gpt_ret_str)
    else:
      action_name = check_chat_rw4t(gpt_ret_str)

    llm_action_dict = {}
    llm_action_dict[action_name] = 1.0
    print('Action dict: ', llm_action_dict)
    self._res['Action'] = llm_action_dict
    return True, ""

  def check_chat_lang(self, gpt_ret):
    if self.prep['gen_mode'] == '5_ranked':
      text = gpt_ret.choices[0].message.content
      llm_action_dict = ast.literal_eval(text.strip())
      self._res['Action'] = llm_action_dict
      return True, ""
    elif self.prep['gen_mode'] == '5_unranked':
      pattern = r'[a-zA-Z]+'
      prob_list = []
      cur_prob = 0
      new_action = True
      # Get probability list
      for logprob in gpt_ret.choices[0].logprobs.content:
        is_word_token = re.fullmatch(pattern, logprob.token.strip())
        if is_word_token and new_action:
          cur_prob += logprob.logprob
          new_action = False
        elif is_word_token and not new_action:
          cur_prob += logprob.logprob
        elif not is_word_token and new_action:
          pass
        else:
          prob_list.append(cur_prob)
          cur_prob = 0
          new_action = True
        # print(f'token: {logprob.token}, logprob: {logprob.logprob}')

      # assert len(prob_list) == 5
      # print('Prob list: ', prob_list)

      # Get action list and list of repeat indices
      print('ret: ', gpt_ret.choices[0].message.content.strip())
      llm_action_list_temp = ast.literal_eval(
          gpt_ret.choices[0].message.content.strip())
      llm_action_list = []
      repeat_indices = []
      if self.prep['env_type'] == 'overcooked':
        all_actions = ALL_MOVES
      elif self.prep['env_type'] == 'minigrid':
        all_actions = list(NAMES_TO_ACTIONS.keys())
      elif self.prep['env_type'] == 'rw4t':
        if 'cur_level' in self.prep and self.prep['cur_level'] == 'll':
          action_enums = rw4t_utils.RW4T_LL_Actions
        else:
          action_enums = rw4t_utils.RW4T_HL_Actions
          llm_action_list_temp_new = []
          for ret_action in llm_action_list_temp:
            llm_action_list_temp_new.append(
                rw4t_utils.HL_Name_2_HL_Action[ret_action])
          llm_action_list_temp = llm_action_list_temp_new
        all_actions = [action.name for action in action_enums]
      else:
        raise NotImplementedError
      for idx in range(len(llm_action_list_temp)):
        action = llm_action_list_temp[idx]
        if self.prep['env_type'] == 'overcooked' and 'Assemble' in action:
          action = action.replace('Assemble', 'Prepare')
        if self.prep['env_type'] == 'rw4t' and ' ' in action:
          action = action.replace(' ', '_')
        if action not in llm_action_list and action in all_actions:
          # Add action if it's not already in there and if it's in the action list
          llm_action_list.append(action)
        else:
          repeat_indices.append(idx)
      if self.prep['env_type'] == 'minigrid':
        print('llm action list: ', llm_action_list)
        llm_action_list = [NAMES_TO_ACTIONS[name] for name in llm_action_list]
        # print('here')
      # print('repeats: ', repeat_indices)
      # Filter out repeats in prob list
      filtered_prob_list = [
          string for i, string in enumerate(prob_list)
          if i not in repeat_indices
      ]
      # print('filtered prob: ', filtered_prob_list)
      assert len(llm_action_list) == len(filtered_prob_list)
      llm_action_dict = dict(zip(llm_action_list, filtered_prob_list))
      llm_action_dict = dict(
          sorted(llm_action_dict.items(),
                 key=lambda item: item[1],
                 reverse=True))
      if len(llm_action_dict) == 0:
        if self.prep['env_type'] == 'overcooked':
          llm_action_dict['Wait'] = 1.0
        # elif self.prep['env_type'] == 'rw4t':
        #   llm_action_dict[rw4t_utils.RW4T_LL_Actions.idle.name] = 1.0
      print('Action dict: ', llm_action_dict)
      self._res['Action'] = llm_action_dict
      return True, ""
    elif self.prep['gen_mode'] == 'all_yes_no':
      print('LLM raw output: ', gpt_ret.choices[0].message.content.strip())
      llm_action_dict = ast.literal_eval(
          gpt_ret.choices[0].message.content.strip())
      filtered_llm_action_dict = {
          key.replace('Assemble', 'Prepare'): value
          for key, value in llm_action_dict.items() if value
      }
      if self.prep['env_type'] == 'minigrid':
        new_filtered_dict = {}
        for name, val in filtered_llm_action_dict.items():
          if self.prep['domain'] == 'PickupMultigoals':
            if name in NAMES_TO_ACTIONS:
              new_filtered_dict[NAMES_TO_ACTIONS[name]] = val
          elif self.prep['domain'] == 'BlockedUnlockPickup':
            if name in MACRO_ACTION_SPACE:
              new_filtered_dict[name] = val
          else:
            raise NotImplementedError
        filtered_llm_action_dict = new_filtered_dict
      print('Filtered llm dict: ', filtered_llm_action_dict)

      prob_list = []
      for logprob in gpt_ret.choices[0].logprobs.content:
        if logprob.token.strip() == 'True':
          prob_list.append(logprob.logprob)
      print('Prob list: ', prob_list)

      assert len(filtered_llm_action_dict) == len(prob_list)
      if len(filtered_llm_action_dict) != 0:
        final_llm_action_dict = dict(
            zip(list(filtered_llm_action_dict.keys()), prob_list))
        final_llm_action_dict = dict(
            sorted(final_llm_action_dict.items(),
                   key=lambda item: item[1],
                   reverse=True))
      else:
        final_llm_action_dict = {'Wait': 1}
      print('Final llm dict: ', final_llm_action_dict)
      self._res['Action'] = final_llm_action_dict
      return True, ""
    elif self.prep['gen_mode'] == 'all_yes_no_include_false':
      print('LLM raw output: ', gpt_ret.choices[0].message.content.strip())
      llm_action_dict = ast.literal_eval(
          gpt_ret.choices[0].message.content.strip())
      if self.prep['env_type'] == 'minigrid':
        new_dict = {}
        for name, val in llm_action_dict.items():
          if name in NAMES_TO_ACTIONS:
            new_dict[NAMES_TO_ACTIONS[name]] = val
        llm_action_dict = new_dict
      print('llm return: ', llm_action_dict)

      prob_list = []
      for logprob in gpt_ret.choices[0].logprobs.content:
        if logprob.token.strip() == 'True':
          prob_list.append(logprob.logprob)
        elif logprob.token.strip() == 'False':
          if logprob.logprob == 0.0:
            prob_list.append(float('-inf'))
          else:
            prob_list.append(math.log(1 - math.exp(logprob.logprob)))
      print('Prob list: ', prob_list)

      assert len(llm_action_dict) == len(prob_list)
      final_llm_action_dict = dict(zip(list(llm_action_dict.keys()), prob_list))
      sorted_items = sorted(final_llm_action_dict.items(),
                            key=lambda item: item[1],
                            reverse=True)
      top_three_items = sorted_items[:3]
      final_llm_action_dict = {k: v for k, v in top_three_items}
      print('Final llm dict (include false): ', final_llm_action_dict)
      self._res['Action'] = final_llm_action_dict
      return True, ""
    elif self.prep['gen_mode'] == 'top':
      pattern = r'[a-zA-Z]+'
      llm_action_dict = {}
      for choice in gpt_ret.choices:
        text = choice.message.content.strip()
        # print('Text: ', text)
        prob = 0
        for logprob in choice.logprobs.content:
          if re.fullmatch(pattern, logprob.token.strip()):
            prob += logprob.logprob
        if text not in llm_action_dict or llm_action_dict[text] > prob:
          llm_action_dict[text] = prob
      llm_action_dict = dict(
          sorted(llm_action_dict.items(),
                 key=lambda item: item[1],
                 reverse=True))
      print('Action dict: ', llm_action_dict)
      self._res['Action'] = llm_action_dict
      return True, ""
    else:
      print('Not implemented')
      raise NotImplementedError

  def move_verifier(self, move: str):
    valid_moves = [move[0] for move in self.prep['chk_moves'] if move[1]]
    is_valid = move in valid_moves
    return is_valid if random.random(
    ) > self.verifier_error_rate else 1 - is_valid

  def __call__(self, text=None):
    # print('Returned text: ', text)
    if text is not None:
      self.hist[-1].append(text)

    # first round
    reasoning = None
    if len(self.hist) == 0:
      if self.prep['env_type'] == 'overcooked':
        if self.prep['prompt_style'] == 'lang':
          if len(self.prep['sit_pref']) > 1 and (
              self.prep['hl_mode'] == 'prompt'
              or self.prep['hl_mode'] == 'prompt+Qlearned-comp'):
            comp_prompt = Lang_Composite_Prompt_Overcooked(self.prep)
            chat = comp_prompt.return_prompt()
          else:
            prompt = Lang_Skill_Prompt_Overcooked(self.prep)
            chat = prompt.return_prompt()
            reasoning = self.prep['sit_pref']
        else:
          if len(self.prep['sit_pref']) > 1 and (
              self.prep['hl_mode'] == 'prompt'
              or self.prep['hl_mode'] == 'prompt+Qlearned-comp'):
            comp_prompt = Prog_Composite_Prompt_Overcooked(self.prep)
            chat = comp_prompt.return_prompt()
          else:
            prompt = Prog_Skill_Prompt_Overcooked(self.prep)
            chat = prompt.return_prompt()
            reasoning = self.prep['sit_pref']
      elif self.prep['env_type'] == 'minigrid':
        if self.prep['domain'] == 'PickupMultigoals':
          chat = prompt_reasoning_minigrid(self.prep)
        elif self.prep['domain'] == 'BlockedUnlockPickup':
          chat = prompt_reasoning_minigrid_bup(self.prep)
        else:
          raise NotImplementedError
      elif self.prep['env_type'] == 'rw4t':
        if self.prep['prompt_style'] == 'lang':
          if 'cur_level' in self.prep and self.prep['cur_level'] == 'll':
            comp_prompt = Lang_Action_Prompt_RW4T(self.prep)
            chat = comp_prompt.return_prompt()
            reasoning = self.prep['sit_pref']
          else:
            if (len(self.prep['sit_pref']) > 1
                and self.prep['hl_mode'] == 'prompt+Qlearned-comp'):
              comp_prompt = Lang_Composite_Prompt_RW4T(self.prep)
              chat = comp_prompt.return_prompt()
            else:
              prompt = Lang_Skill_Prompt_RW4T(self.prep)
              chat = prompt.return_prompt()
              reasoning = self.prep['sit_pref']
        else:
          if 'cur_level' in self.prep and self.prep['cur_level'] == 'll':
            raise NotImplementedError
          else:
            if (len(self.prep['sit_pref']) > 1
                and self.prep['hl_mode'] == 'prompt+Qlearned-comp'):
              comp_prompt = Prog_Composite_Prompt_RW4T(self.prep)
              chat = comp_prompt.return_prompt()
            else:
              prompt = Prog_Skill_Prompt_RW4T(self.prep)
              chat = prompt.return_prompt()
              reasoning = self.prep['sit_pref']
      else:
        print('Not implemented')
        raise NotImplementedError
      # print('Prompt generated: ', chat)
      self._res = {
          "Reasoning": reasoning,
          "Action": "",
          "Action_backup": "",
          "Chat": None,
      }
      self.hist = chat
      return None, self.hist
    elif self._retry >= self.MAX_RETRY_TIMES:
      self._res = {"Chat": "ERROR", "Action": "ERROR"}
      return self._res, None
    else:  # proceed
      if self._res["Reasoning"] is None:
        ok, hint = self.check_reasoning(text)
      elif self._res["Chat"] is None:
        if self.prep['prompt_style'] == 'lang':
          ok, hint = self.check_chat_lang(text)
        else:
          ok, hint = self.check_chat_prog(text)
      else:
        ok, hint = True, None
      if not ok:
        self.input_hist = hint
        self._retry += 1
        return None, self.input_hist
      else:
        return self._res, None


def parse_argument(arg_str):
  arg_str = arg_str.strip()
  # Check if it's a string (i.e., enclosed in quotes)
  if arg_str.startswith('"') and arg_str.endswith('"'):
    return arg_str[1:-1]
  if arg_str.startswith("'") and arg_str.endswith("'"):
    return arg_str[1:-1]
  # Otherwise, return the argument as a string
  return arg_str


def parse_function_call(call_str):
  # Regular expression to match the function name and argument list
  pattern = r'([a-zA-Z]+)\((.*)\)'
  match = re.match(pattern, call_str.strip())

  if not match:
    raise ValueError(f"Invalid function call: {call_str}")

  func_name = match.group(1)  # Extract the function name
  args_str = match.group(2)  # Extract the argument list as a string

  # Handle empty argument lists
  if not args_str:
    return (func_name, [])

  # Split the arguments by commas
  args = [arg.strip() for arg in args_str.split(',')]

  # Parse each argument individually
  parsed_args = [parse_argument(arg) for arg in args]

  return (func_name, parsed_args)


def check_chat_oc(gpt_ret_str):
  func_name, func_args = parse_function_call(gpt_ret_str)
  if func_name == 'chop':
    if len(func_args) == 1:
      if func_args[0] == 'whole lettuce':
        return 'Chop Lettuce'
      if func_args[0] == 'whole onion':
        return 'Chop Onion'
      if func_args[0] == 'whole tomato':
        return 'Chop Tomato'
  if func_name == 'combine':
    if len(func_args) == 2:
      if any('onion' in a for a in func_args) and any(
          'lettuce' in a for a in func_args) and not any('combined' in a
                                                         for a in func_args):
        return 'Prepare Alice Ingredients'
      if any('tomato' in a for a in func_args) and any(
          'lettuce' in a for a in func_args) and not any('combined' in a
                                                         for a in func_args):
        return 'Prepare Bob Ingredients'
      if any('tomato' in a for a in func_args) and any(
          'onion' in a for a in func_args) and not any('combined' in a
                                                       for a in func_args):
        return 'Prepare Cathy Ingredients'
      if (any('combined onion and lettuce' in a
              for a in func_args) and any('tomato' in a for a in func_args)
          ) or (any('combined tomato and lettuce' in a for a in func_args)
                and any('onion' in a for a in func_args)) or any(
                    'combined onion and tomato' in a
                    for a in func_args) and any('lettuce' in a
                                                for a in func_args):
        return 'Prepare David Ingredients'
    if len(func_args) == 3:
      if (any('onion' in a for a in func_args)
          and any('lettuce' in a for a in func_args)
          and any('tomato' in a for a in func_args)):
        return 'Prepare David Ingredients'
  if func_name == 'putin':
    if len(func_args) == 2:
      if ((func_args[0] == 'combined onion and lettuce'
           or func_args[0] == 'combined lettuce and onion')
          and func_args[1] == 'pot'):
        return 'Cook Alice Soup'
      if ((func_args[0] == 'combined tomato and lettuce'
           or func_args[0] == 'combined lettuce and tomato')
          and func_args[1] == 'pot'):
        return 'Cook Bob Soup'
      if ((func_args[0] == 'combined onion and tomato'
           or func_args[0] == 'combined tomato and onion')
          and func_args[1] == 'pot'):
        return 'Cook Cathy Soup'
      if ((func_args[0] == 'combined onion and tomato and lettuce'
           or func_args[0] == 'combined onion and lettuce and tomato'
           or func_args[0] == 'combined tomato and onion and lettuce'
           or func_args[0] == 'combined tomato and lettuce and onion'
           or func_args[0] == 'combined lettuce and tomato and onion'
           or func_args[0] == 'combined lettuce and onion and tomato')
          and func_args[1] == 'pot'):
        return 'Cook David Soup'
      if (func_args[0] == 'cooked Alice soup' and func_args[1] == 'plate'):
        return 'Plate Alice Soup'
      if (func_args[0] == 'cooked Bob soup' and func_args[1] == 'plate'):
        return 'Plate Bob Soup'
      if (func_args[0] == 'cooked Cathy soup' and func_args[1] == 'plate'):
        return 'Plate Cathy Soup'
      if (func_args[0] == 'cooked David soup' and func_args[1] == 'plate'):
        return 'Plate David Soup'
  if func_name == 'serve':
    if len(func_args) == 1:
      if 'Alice soup' in func_args[0]:
        return 'Serve Alice Soup'
      if 'Bob soup' in func_args[0]:
        return 'Serve Bob Soup'
      if 'Cathy soup' in func_args[0]:
        return 'Serve Cathy Soup'
      if 'David soup' in func_args[0]:
        return 'Serve David Soup'
  if func_name == 'putout':
    return 'Putout'
  if func_name == 'discard':
    return 'Drop'
  return 'Wait'


# def check_chat_oc(gpt_ret_str):
#   func_name, func_args = parse_function_call(gpt_ret_str)
#   if func_name == 'chop':
#     if len(func_args) == 1:
#       if func_args[0] == 'whole lettuce':
#         return 'Chop Lettuce'
#       if func_args[0] == 'whole onion':
#         return 'Chop Onion'
#       if func_args[0] == 'whole tomato':
#         return 'Chop Tomato'
#   if func_name == 'combine':
#     if len(func_args) == 1:
#       if func_args[0] == 'Alice ingredients':
#         return 'Prepare Alice Ingredients'
#       if func_args[0] == 'Bob ingredients':
#         return 'Prepare Bob Ingredients'
#       if func_args[0] == 'Cathy ingredients':
#         return 'Prepare Cathy Ingredients'
#       if func_args[0] == 'David ingredients':
#         return 'Prepare David Ingredients'
#   if func_name == 'putin':
#     if len(func_args) == 2:
#       if (func_args[0] == 'uncooked Alice soup' and func_args[1] == 'pot'):
#         return 'Cook Alice Soup'
#       if (func_args[0] == 'uncooked Bob soup' and func_args[1] == 'pot'):
#         return 'Cook Bob Soup'
#       if (func_args[0] == 'uncooked Cathy soup' and func_args[1] == 'pot'):
#         return 'Cook Cathy Soup'
#       if (func_args[0] == 'uncooked David soup' and func_args[1] == 'pot'):
#         return 'Cook David Soup'
#       if (func_args[0] == 'cooked Alice soup' and func_args[1] == 'plate'):
#         return 'Plate Alice Soup'
#       if (func_args[0] == 'cooked Bob soup' and func_args[1] == 'plate'):
#         return 'Plate Bob Soup'
#       if (func_args[0] == 'cooked Cathy soup' and func_args[1] == 'plate'):
#         return 'Plate Cathy Soup'
#       if (func_args[0] == 'cooked David soup' and func_args[1] == 'plate'):
#         return 'Plate David Soup'
#   if func_name == 'serve':
#     if len(func_args) == 1:
#       if 'Alice soup' in func_args[0]:
#         return 'Serve Alice Soup'
#       if 'Bob soup' in func_args[0]:
#         return 'Serve Bob Soup'
#       if 'Cathy soup' in func_args[0]:
#         return 'Serve Cathy Soup'
#       if 'David soup' in func_args[0]:
#         return 'Serve David Soup'
#   if func_name == 'putout':
#     return 'Putout'
#   if func_name == 'discard':
#     return 'Drop'
#   return 'Wait'


def check_chat_rw4t(gpt_ret_str):
  func_name, func_args = parse_function_call(gpt_ret_str)
  if func_name == 'pick':
    if len(func_args) == 1:
      if func_args[0] == 'circle':
        return rw4t_utils.RW4T_HL_Actions.go_to_circle.name
      elif func_args[0] == 'square':
        return rw4t_utils.RW4T_HL_Actions.go_to_square.name
      elif func_args[0] == 'triangle':
        return rw4t_utils.RW4T_HL_Actions.go_to_triangle.name
  if func_name == 'drop':
    if len(func_args) == 1:
      if func_args[0] == 'school':
        return rw4t_utils.RW4T_HL_Actions.go_to_school.name
      elif func_args[0] == 'hospital':
        return rw4t_utils.RW4T_HL_Actions.go_to_hospital.name
      elif func_args[0] == 'park':
        return rw4t_utils.RW4T_HL_Actions.go_to_park.name
  return ''
