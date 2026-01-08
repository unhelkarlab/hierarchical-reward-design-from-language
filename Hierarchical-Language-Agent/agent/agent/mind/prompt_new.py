from abc import ABC, abstractmethod
from copy import deepcopy
import random

from agent.executor.high import OBJ_TO_GOODS_GS
from agent.mind.prompt_local import ALL_MOVES
from agent.mind.prompt import prompt_order
from gym_cooking.utils.core import *


class Prompt(ABC):

  def __init__(self, prep) -> None:
    self.prep = prep

  @abstractmethod
  def task_description(self):
    pass

  @abstractmethod
  def action_description(self):
    pass

  @abstractmethod
  def preference_description(self):
    pass

  @abstractmethod
  def env_description(self):
    pass

  @abstractmethod
  def human_suggestion(self):
    pass

  @abstractmethod
  def move_history_description(self):
    pass

  @abstractmethod
  def request_description(self):
    pass

  def in_context_examples(self):
    return ''

  def return_prompt(self):
    p1 = self.task_description() + self.action_description(
    ) + self.preference_description()
    p2 = 'Ok.'
    p3 = self.env_description() + self.human_suggestion(
    ) + self.move_history_description() + self.request_description(
    ) + self.in_context_examples()
    # print('p1...: ', p1)
    # print('p3...: ', p3)
    return [[p1, p2], [p3]]


# Overcooked entity to a most descriptive name for the LLM agent
ENTITY_2_DESCRIPTION = {
    'ChoppedTomato': 'chopped tomato',
    'ChoppedOnion': 'chopped onion',
    'ChoppedLettuce': 'chopped lettuce',
    'ChoppedLettuce-ChoppedOnion': 'combined onion and lettuce',
    'ChoppedLettuce-ChoppedTomato': 'combined tomato and lettuce',
    'ChoppedOnion-ChoppedTomato': 'combined onion and tomato',
    'ChoppedLettuce-ChoppedOnion-ChoppedTomato':
    'combined onion and tomato and lettuce',
    'CookedLettuce-CookedOnion': 'cooked Alice soup',
    'CookedLettuce-CookedTomato': 'cooked Bob soup',
    'CookedOnion-CookedTomato': 'cooked Cathy soup',
    'CookedLettuce-CookedOnion-CookedTomato': 'cooked David soup',
    'CookedLettuce-CookedOnion-Plate': 'plated Alice soup',
    'CookedLettuce-CookedTomato-Plate': 'plated Bob soup',
    'CookedOnion-CookedTomato-Plate': 'plated Cathy soup',
    'CookedLettuce-CookedOnion-CookedTomato-Plate': 'plated David soup',
    'CharredLettuce-CharredOnion': 'burned Alice soup',
    'CharredLettuce-CharredTomato': 'burned Bob soup',
    'CharredOnion-CharredTomato': 'burned Cathy soup',
    'CharredLettuce-CharredOnion-CharredTomato': 'burned David soup',
}

# ENTITY_2_DESCRIPTION = {
#     'ChoppedTomato': 'chopped tomato',
#     'ChoppedOnion': 'chopped onion',
#     'ChoppedLettuce': 'chopped lettuce',
#     'ChoppedLettuce-ChoppedOnion': 'uncooked Alice soup',
#     'ChoppedLettuce-ChoppedTomato': 'uncooked Bob soup',
#     'ChoppedOnion-ChoppedTomato': 'uncooked Cathy soup',
#     'ChoppedLettuce-ChoppedOnion-ChoppedTomato': 'uncooked David soup',
#     'CookedLettuce-CookedOnion': 'cooked Alice soup',
#     'CookedLettuce-CookedTomato': 'cooked Bob soup',
#     'CookedOnion-CookedTomato': 'cooked Cathy soup',
#     'CookedLettuce-CookedOnion-CookedTomato': 'cooked David soup',
#     'CookedLettuce-CookedOnion-Plate': 'plated Alice soup',
#     'CookedLettuce-CookedTomato-Plate': 'plated Bob soup',
#     'CookedOnion-CookedTomato-Plate': 'plated Cathy soup',
#     'CookedLettuce-CookedOnion-CookedTomato-Plate': 'plated David soup',
#     'CharredLettuce-CharredOnion': 'burned Alice soup',
#     'CharredLettuce-CharredTomato': 'burned Bob soup',
#     'CharredOnion-CharredTomato': 'burned Cathy soup',
#     'CharredLettuce-CharredOnion-CharredTomato': 'burned David soup',
# }

# Action description to ProgPrompt-like actions
sit_prefs_to_prog_actions = {
    "Chop Tomato": "chop('whole tomato')",
    "Chop Onion": "chop('whole onion')",
    "Chop Lettuce": "chop('whole lettuce')",
    "Prepare Alice Ingredients": "combine('chopped onion', 'chopped lettuce')",
    "Prepare Bob Ingredients": "combine('chopped tomato', 'chopped lettuce')",
    "Prepare Cathy Ingredients": "combine('chopped onion', 'chopped tomato')",
    "Prepare David Ingredients":
    "combine('chopped onion', 'chopped tomato', 'chopped lettuce')",
    "Cook Alice Soup": "putin('combined onion and lettuce', 'pot')",
    "Cook Bob Soup": "putin('combined tomato and lettuce', 'pot')",
    "Cook Cathy Soup": "putin('combined onion and tomato', 'pot')",
    "Cook David Soup": "putin('combined onion and tomato and lettuce', 'pot')",
    "Plate Alice Soup": "putin('cooked Alice soup', 'plate')",
    "Plate Bob Soup": "putin('cooked Bob soup', 'plate')",
    "Plate Cathy Soup": "putin('cooked Cathy soup', 'plate')",
    "Plate David Soup": "putin('cooked David soup', 'plate')",
    "Serve Alice Soup": "serve('plated Alice soup')",
    "Serve Bob Soup": "serve('plated Bob soup')",
    "Serve Cathy Soup": "serve('plated Cathy soup')",
    "Serve David Soup": "serve('plated David soup')",
    "Putout": "putout",
    "Drop": "discard",
    "Wait": "wait"
}

# sit_prefs_to_prog_actions = {
#     "Chop Tomato": "chop('whole tomato')",
#     "Chop Onion": "chop('whole onion')",
#     "Chop Lettuce": "chop('whole lettuce')",
#     "Prepare Alice Ingredients": "combine('Alice ingredients')",
#     "Prepare Bob Ingredients": "combine('Bob ingredients')",
#     "Prepare Cathy Ingredients": "combine('Cathy ingredients')",
#     "Prepare David Ingredients": "combine('David ingredients')",
#     "Cook Alice Soup": "putin('uncooked Alice Soup', 'pot')",
#     "Cook Bob Soup": "putin('uncooked Bob Soup', 'pot')",
#     "Cook Cathy Soup": "putin('uncooked Cathy Soup', 'pot')",
#     "Cook David Soup": "putin('uncooked David Soup', 'pot')",
#     "Plate Alice Soup": "putin('cooked Alice soup', 'plate')",
#     "Plate Bob Soup": "putin('cooked Bob soup', 'plate')",
#     "Plate Cathy Soup": "putin('cooked Cathy soup', 'plate')",
#     "Plate David Soup": "putin('cooked David soup', 'plate')",
#     "Serve Alice Soup": "serve('plated Alice soup')",
#     "Serve Bob Soup": "serve('plated Bob soup')",
#     "Serve Cathy Soup": "serve('plated Cathy soup')",
#     "Serve David Soup": "serve('plated David soup')",
#     "Putout": "putout",
#     "Drop": "discard",
#     "Wait": "wait"
# }


def get_available_objs_overcooked(prep):
  '''
  Helper function for getting a list of available objects for the current
  Overcooked environment.
  '''
  available_objs = [
      'pot', 'plate', 'whole onion', 'whole lettuce', 'whole tomato'
  ]
  features = prep['features']
  FOOD_ENTITIES = ASSEMBLE_CHOPPED_FOOD + ASSEMBLE_COOKING_FOOD + \
      ASSEMBLE_COOKED_FOOD + ASSEMBLE_COOKED_PLATE_FOOD + \
      ASSEMBLE_CHARRED_FOOD + ASSEMBLE_CHARRED_PLATE_FOOD
  for idx in range(len(FOOD_ENTITIES)):
    if features[idx] != 0:
      entity = FOOD_ENTITIES[idx]
      if entity in ENTITY_2_DESCRIPTION:
        available_objs.insert(0, ENTITY_2_DESCRIPTION[entity])
  return available_objs


def preference_description_overcooked_helper(prep):
  p_2_order_name = {
      'A': 'Alice Soup',
      'B': 'Bob Soup',
      'C': 'Cathy Soup',
      'D': 'David Soup'
  }

  p_list = prep['pref'].split('_')
  processed_p = 'The human prefers you to work on the following soups: \n'
  for i in range(len(p_list)):
    line = ''
    letters_list = list(p_list[i])
    for letter in letters_list:
      soup_name = p_2_order_name[letter]
      if i < len(p_list) - 1:
        line += (soup_name + ', ')
      else:
        line += (soup_name + '.')
    processed_p += line

  return processed_p


def game_action_to_prog_action(action_name):
  return sit_prefs_to_prog_actions[action_name]


def get_pot_info_overcooked(prep):
  '''
  Helper function for getting the statuses of the pots in the overcooked
  domain.
  '''
  pot_info = ''
  map_info = prep['map'].split('-')
  for line in map_info:
    if 'pot' in line:
      pot_info += (line.strip() + '\n')
  return pot_info


class Prompt_Overcooked(Prompt):

  def preference_description(self):
    super().preference_description()
    if self.prep['text_desc']:
      return preference_description_overcooked_helper(self.prep)
    else:
      return ''


class Lang_Prompt_Overcooked(Prompt_Overcooked):

  def task_description(self):
    super().task_description()
    p = '''Game Scenario:
You are a compliant and helpful AI assistant in a simplified version of the Overcooked game.
Your goal is to follow the human's suggestions and achieve a high score in the game.

Game Rules:
There are four different types of soup: Alice soup, Bob soup, Cathy soup, and David soup.
At any given time, you will see three soup orders.
Your task is to prepare soups according to these current orders.
Each soup order has a time limit.
Points are awarded for completing only some of the soup orders.
You do not know in advance which soups will yield points, but the human does, and they will provide suggestions on the order to prioritize.
Therefore, always follow the human's suggestions.

Making a soup:
1.  If there are no chopped vegetables on the map, you need to chop them.
    Chop lettuce and onion to make Alice soup.
    Chop lettuce and tomato to make Bob soup.
    Chop onion and lettuce to make Cathy soup.
    Chop lettuce, onion, and tomato to make David Soup.
2.  Assemble the chopped vegetables.
    Once all the required vegetables are chopped, assemble them for the corresponding soup.
    You cannot begin assembling soup ingredients until all required ingredients are chopped.
3.  Cook the soup.
    You cannot start cooking a soup until all the ingredients for that soup are assembled.
4.  Plate the soup.
    Once the soup is cooked, transfer it to a plate.
    While the soup is cooking, work on other orders before plating.
5.  Serve the soup.
    After you plate a soup, if there is a matching soup order, serve the soup.
    Serving soups can take a while and each order has a time limit, so prioritize serving soups over other actions when there is a plated soup.

Managing burned soup:
If a soup stays in the pot too long, it will get charred.
1.  Putout: If the pot catches fire, extinguish it.
2.  Drop: Discard charred soup. Put out the fire in the pot if needed.
Note: Putting out fires takes a long time, so plate a soup as soon as it's done to avoid delays.

Assuming that you have been playing the game for a while.
You will be presented with the current orders, available items on the map, and potentially other information like past actions.
'''
    return p

  def env_description(self):
    super().env_description()
    order_info = prompt_order(self.prep['order'])
    map_info = self.prep['map']
    if self.prep['holding'] != '':
      holding_info = f'You are holding: {OBJ_TO_GOODS_GS[self.prep["holding"]]}.'
    else:
      holding_info = ''
    guide = 'Please examine the environment information carefully when you decide your action.\n'
    return order_info + map_info + holding_info + guide


class Lang_Skill_Prompt_Overcooked(Lang_Prompt_Overcooked):

  def action_description(self):
    super().action_description()

    all_moves_copy = deepcopy(ALL_MOVES)
    all_moves_copy.remove('Wait')
    avaialble_actions = ', '.join(all_moves_copy)

    p = f'''The list of available actions is: [{avaialble_actions}].
"Assemble ... Ingredients": assemble the chopped vegetables for a soup;
"Cook ... Soup": bring the assembled ingredients to a pot and start cooking;
"Plate ... Soup": transfer the soup to a plate, ready to be served;
"Serve ... Soup": serve the plated soup to the customers at the delivery location.
'''
    return p

  def human_suggestion(self):
    super().human_suggestion()
    if len(self.prep['sit_pref']) == 1 and (self.prep['hl_mode'] == 'prompt'
                                            or self.prep['hl_mode']
                                            == 'prompt+Qlearned-comp'):
      if self.prep['no_hl_rec']:
        hl_rec = ''
      else:
        if self.prep['sit_pref'][0] in self.prep['sit_pref_actions']:
          cur_skill = self.prep['sit_pref_actions'][self.prep['sit_pref'][0]]
        else:
          cur_skill = []
        hl_rec = f'\nActions to perform the task: {cur_skill}.'
      p = f'''
The human suggests you to work on the following task: {self.prep['sit_pref'][0]}{hl_rec}
- Choose an action for this task, unless there is a more urgent action.
- More urgent actions can include plating a soup or serving a soup.
- The action you take should utilize the existing ingredients on the map.
- For example, if a vegetable is already chopped, avoid chopping it again.
- Do not prepare for a soup order that differs from the one suggested by the human.
'''
    elif self.prep['hl_mode'] == 'prompt+Qlearned-skill':
      top_k = self.prep['top_k_llm']
      p = f'''
The human suggests you to output one of the following actions: {self.prep['sit_pref'][:top_k]}
- Choose an action from this list, unless there is a more urgent action.
- More urgent actions can include plating a soup or serving a soup.
- The action you take should utilize the existing ingredients on the map.
- For example, if a vegetable is already chopped, avoid chopping it again.
'''
    else:
      p = ''
    print('Human suggestion: ', p)
    return p

  def move_history_description(self):
    super().move_history_description()

    mov_hist = self.prep['mov_hist']
    mh = [m['task'] for m in mov_hist][::-1][:1]
    ms = [m['status'] for m in mov_hist][::-1][:1]

    prompt = "Your previous action and its corresponding status: "
    history_status_pairs = [f"{a} ({b})" for a, b in zip(mh, ms)]
    prompt += (";\n".join(history_status_pairs) +
               '\n') if len(history_status_pairs) > 0 else "None\n"
    prompt += 'If an action fails, please generate a different action, as this means that this action might not be available.\n'
    return prompt

  def request_description(self):
    super().request_description()

    p = '''
Now please respond to the following request:
Output one action that is the most suitable for the current game state.
The action should be feasible to perform and also contribute toward completing soup orders.
Do not output the same action twice.

Please format your output like a Python list.
Your final output should look like this: "['action 1', 'action 2', 'action 3']"
Do not output anything else.
'''
    return p


class Lang_Composite_Prompt_Overcooked(Lang_Prompt_Overcooked):

  def action_description(self):
    super().action_description()
    return ''

  def human_suggestion(self):
    super().human_suggestion()
    if self.prep['top_k_llm'] == -1:
      sit_pref_copy = deepcopy(self.prep['sit_pref'])
      random.shuffle(sit_pref_copy)
      cur_sit_pref = str(sit_pref_copy)
    elif self.prep['top_k_llm'] == 1:
      cur_sit_pref = f'1. {self.prep["sit_pref"][0]}'
    elif self.prep['top_k_llm'] == 2:
      cur_sit_pref = f'1. {self.prep["sit_pref"][0]}; 2. {self.prep["sit_pref"][1]}'
    elif self.prep['top_k_llm'] == 3:
      cur_sit_pref = f'1. {self.prep["sit_pref"][0]}; 2. {self.prep["sit_pref"][1]}; 3. {self.prep["sit_pref"][2]}'
    p = f'''
Now, the human recommends you to work on one of the following actions:
{cur_sit_pref}
- Choose one action from the list above to perform next.
- Your selection should help complete one of the current soup orders.
'''
    return p

  def move_history_description(self):
    super().move_history_description()
    if len(self.prep['prev_sit_pref']) > 0 and self.prep['top_k_llm'] != -1:
      mov_hist = self.prep['mov_hist']
      mh = [m['task'] for m in mov_hist][::-1][:1]
      ms = [m['status'] for m in mov_hist][::-1][:1]
      history_status_pairs = [f"{a} ({b})" for a, b in zip(mh, ms)]
      last_action = (";\n".join(history_status_pairs) +
                     '\n') if len(history_status_pairs) > 0 else "None\n"

      prev_sit_pref = f'''
Currently, your action is to {self.prep["prev_sit_pref"][0]}.
You just performed: {last_action}
If this action is not complete and still recommended by the human, you are discouraged to start a new action.'''
    else:
      prev_sit_pref = '\nYou can continue your previous action or switch to a new action.'
    return prev_sit_pref

  def request_description(self):
    super().request_description()

    p = '''
Now please output the most suitable action verbatim from the list of human's suggestions based on the current state.
Do not output anything else. Do not output square brackets.
'''
    return p


class Prog_Prompt_Overcooked(Prompt_Overcooked):

  def task_description(self):
    super().task_description()

    p = '''Game Scenario:
You are a compliant and helpful AI assistant in a simplified version of the Overcooked game.
Your goal is to follow the human's suggestions and achieve a high score in the game.

Game Rules:
There are four different types of soup: Alice soup, Bob soup, Cathy soup, and David soup.
At any given time, you will see three soup orders.
Your task is to prepare soups according to these current orders.
Each soup order has a time limit.
Points are awarded for completing only some of the soup orders.
You do not know in advance which soups will yield points, but the human does, and they will provide suggestions on the order to prioritize.
Therefore, always follow the human's suggestions.
'''

    p += '''Here are the recipes for making different soups.
def make_Alice_soup():
  """
  Recipe for Alice soup
  """
  # Chop Alice ingredients
  chop('whole onion')
  chop('whole lettuce')
  # Assemble Alice ingredients
  combine('chopped onion', 'chopped lettuce')
  # Cook Alice soup in pot
  putin('combined onion and lettuce', 'pot')
  # Plate Alice soup
  putin('cooked Alice soup', 'plate')
  # Serve Alice soup
  serve('plated Alice soup')

def make_Bob_soup():
  """
  Recipe for Bob soup
  """
  # Chop Bob ingredients
  chop('whole tomato')
  chop('whole lettuce')
  # Assemble Bob ingredients
  combine('chopped tomato', 'chopped lettuce')
  # Cook Bob soup in pot
  putin('combined tomato and lettuce', 'pot')
  # Plate Bob soup
  putin('cooked Bob soup', 'late')
  # Serve Bob soup
  serve('plated Bob soup')

def make_Cathy_soup():
  """
  Recipe for Cathy soup
  """
  # Chop Cathy ingredients
  chop('whole onion')
  chop('whole tomato')
  # Assemble Cathy ingredients
  combine('chopped onion', 'chopped tomato')
  # Cook Cathy soup in pot
  putin('combined onion and tomato', 'pot')
  # Plate Cathy soup
  putin('cooked Cathy soup', 'plate')
  # Serve Cathy soup
  serve('plated Cathy soup')

def make_David_soup():
  """
  Recipe for David soup
  """
  # Chop David ingredients
  chop('whole onion')
  chop('whole tomato')
  chop('whole lettuce')
  # Assemble David ingredients
  # You can assemble David ingredients by doing one of the following:
  # combine('combined onion and lettuce', 'chopped tomato')
  # combine('combined tomato and lettuce', 'chopped onion')
  # combine('combined onion and tomato', 'chopped lettuce')
  # in addition to assembling 3 ingredients together.
  combine('chopped onion', 'chopped tomato', 'chopped lettuce')
  # Cook David soup in pot
  putin('combined onion and tomato and lettuce', 'pot')
  # Plate David soup
  putin('cooked David soup', 'plate')
  # Serve David soup
  serve('plated David soup')

def manage_burned_soup():
  """
  Recipe for managing a burned soup.
  If a soup stays in the pot too long, it will catch on fire. Putting out fires
  takes a long time, so make sure to plate a soup as soon as it's done to avoid
  delays by calling putin('... soup', 'plate').
  """
  putout()
  discard()

Note:
if there is a plated soup and the soup is among one of the three current soup orders,
you should prioritize serving it by returning serve('plated ... soup').
However, if the soup is not among the current orders, you should NOT serve it.
'''
    #     p += '''Here are the recipes for making different soups.
    # def make_Alice_soup():
    #   """
    #   Recipe for Alice soup
    #   """
    #   # Chop Alice ingredients
    #   chop('whole onion')
    #   chop('whole lettuce')
    #   # Assemble Alice ingredients
    #   # You can perform combine('Alice ingredients') when the available objects include choppoed onions and lettuce.
    #   combine('Alice ingredients')
    #   # Cook Alice soup in pot
    #   putin('uncooked Alice soup', 'pot')
    #   # Plate Alice soup
    #   putin('cooked Alice soup', 'plate')
    #   # Serve Alice soup
    #   serve('plated Alice soup')

    # def make_Bob_soup():
    #   """
    #   Recipe for Bob soup
    #   """
    #   # Chop Bob ingredients
    #   chop('whole tomato')
    #   chop('whole lettuce')
    #   # Assemble Bob ingredients
    #   # You can perform combine('Bob ingredients') when the available objects include choppoed tomato and lettuce.
    #   combine('Bob ingredients')
    #   # Cook Bob soup in pot
    #   putin('uncooked Bob soup', 'pot')
    #   # Plate Bob soup
    #   putin('cooked Bob soup', 'late')
    #   # Serve Bob soup
    #   serve('plated Bob soup')

    # def make_Cathy_soup():
    #   """
    #   Recipe for Cathy soup
    #   """
    #   # Chop Cathy ingredients
    #   chop('whole onion')
    #   chop('whole tomato')
    #   # Assemble Cathy ingredients
    #   # You can perform combine('Cathy ingredients') when the available objects include choppoed onion and tomato.
    #   combine('Cathy ingredients')
    #   # Cook Cathy soup in pot
    #   putin('uncooked Cathy soup', 'pot')
    #   # Plate Cathy soup
    #   putin('cooked Cathy soup', 'plate')
    #   # Serve Cathy soup
    #   serve('plated Cathy soup')

    # def make_David_soup():
    #   """
    #   Recipe for David soup
    #   """
    #   # Chop David ingredients
    #   chop('whole onion')
    #   chop('whole tomato')
    #   chop('whole lettuce')
    #   # Assemble David ingredients
    #   # You can perform combine('David ingredients') when the available objects include choppoed onion, tomato, and lettuce.
    #   combine('David ingredients')
    #   # Cook David soup in pot
    #   putin('uncooked David soup', 'pot')
    #   # Plate David soup
    #   putin('cooked David soup', 'plate')
    #   # Serve David soup
    #   serve('plated David soup')

    # def manage_burned_soup():
    #   """
    #   Recipe for managing a burned soup.
    #   If a soup stays in the pot too long, it will catch on fire. Putting out fires
    #   takes a long time, so make sure to plate a soup as soon as it's done to avoid
    #   delays by calling putin('... soup', 'plate').
    #   """
    #   putout()
    #   discard()

    # Note:
    # if there is a plated soup and the soup is among one of the three current soup orders,
    # you should prioritize serving it by returning serve('plated ... soup').
    # However, if the soup is not among the current orders, you should NOT serve it.
    # '''
    # print('task: ', p)
    return p

  def env_description(self):
    super().env_description()
    order_info = prompt_order(self.prep['order'])
    # map_info = self.prep['map']
    pots_info = f'Current pots status: \n {get_pot_info_overcooked(self.prep)}'
    if self.prep['holding'] != '':
      holding_info = f'You are holding: {OBJ_TO_GOODS_GS[self.prep["holding"]]}.'
    else:
      holding_info = ''
    guide = '''Please examine the environment information carefully when you decide your action.
You should utilize the available objects on the map.
For instance, if you already have a chopped ingredient, avoid chopping it again.'''
    available_objects = f'available_objects = {get_available_objs_overcooked(self.prep)}'
    # print('env: ',
    #       order_info + pots_info + holding_info + available_objects + guide)
    return order_info + pots_info + holding_info + available_objects + guide


class Prog_Skill_Prompt_Overcooked(Prog_Prompt_Overcooked):

  def action_description(self):
    super().action_description()

    p = '''Your available actions include:
chop(<obj>)
combine(<obj>, <obj>) # Combine two ingredients
combine(<obj>, <obj>, <obj>) # Combine three ingredients for David Soup
putin(<obj>, <obj>)
serve(<obj>)
putout()
discard()
'''
    #     p = '''Your available actions include:
    # chop(<obj>)
    # combine(<obj>, <obj>)
    # putin(<obj>, <obj>)
    # serve(<obj>)
    # putout()
    # discard()
    # '''
    return p

  def human_suggestion(self):
    super().human_suggestion()
    if len(self.prep['sit_pref']) == 1 and (self.prep['hl_mode'] == 'prompt'
                                            or self.prep['hl_mode']
                                            == 'prompt+Qlearned-comp'):
      if self.prep['no_hl_rec']:
        hl_rec = ''
      else:
        if self.prep['sit_pref'][0] in self.prep['sit_pref_actions']:
          cur_skill = self.prep['sit_pref_actions'][self.prep['sit_pref'][0]]
        else:
          cur_skill = []
        hl_rec = f'\nActions to perform the task: {cur_skill}.'
      p = f'''
The human suggests you to work on the following task: {self.prep['sit_pref'][0]}{hl_rec}
- Choose an action for this task, unless there is a more urgent action.
- More urgent actions can include plating a soup or serving a soup.
- The action you take should utilize the existing ingredients on the map.
- For example, if a vegetable is already chopped, avoid chopping it again.
- Do not prepare for a soup order that differs from the one suggested by the human.
'''
    elif self.prep['hl_mode'] == 'prompt+Qlearned-skill':
      top_k = self.prep['top_k_llm']
      actions_wo_wait = [a for a in self.prep['sit_pref'] if a != 'wait']
      top_actions = [
          game_action_to_prog_action(a) for a in actions_wo_wait[:top_k]
      ]
      p = f'''
The human suggests you to output one of the following actions: {top_actions}
- Choose an action from this list, unless there is a more urgent action.
- More urgent actions can include plating a soup or serving a soup.
- The action you take should utilize the existing ingredients on the map.
- For example, if a vegetable is already chopped, avoid chopping it again.
'''
    else:
      p = ''
    print('Human suggestion: ', p)
    return p

  def move_history_description(self):
    super().move_history_description()

    mov_hist = self.prep['mov_hist']
    mh = [m['task'] for m in mov_hist][::-1][:1]
    ms = [m['status'] for m in mov_hist][::-1][:1]

    prompt = "Your previous action and its corresponding status: "
    history_status_pairs = [
        f"{game_action_to_prog_action(a)} ({b})" for a, b in zip(mh, ms)
    ]
    prompt += (";\n".join(history_status_pairs) +
               '\n') if len(history_status_pairs) > 0 else "None\n"
    prompt += 'If an action fails, please generate a different action, as this means that this action might not be available.\n'
    # print('Move hist: ', prompt)
    return prompt

  def request_description(self):
    super().request_description()

    p = '''
Now please respond to the following request:
Output one action that is the most suitable for the current game state.
The action should be feasible to perform and also contribute toward completing soup orders.
The action you select should have the same name as a function call in one of the recipes.
Please output only the action, and nothing else.
'''
    return p


class Prog_Composite_Prompt_Overcooked(Prog_Prompt_Overcooked):

  def action_description(self):
    super().action_description()
    return ''

  def human_suggestion(self):
    super().human_suggestion()
    if self.prep['top_k_llm'] == -1:
      sit_pref_copy = deepcopy(self.prep['sit_pref'])
      random.shuffle(sit_pref_copy)
      cur_sit_pref = str(sit_pref_copy)
    elif self.prep['top_k_llm'] == 1:
      cur_sit_pref = f'1. {self.prep["sit_pref"][0]}'
    elif self.prep['top_k_llm'] == 2:
      cur_sit_pref = f'1. {self.prep["sit_pref"][0]}; 2. {self.prep["sit_pref"][1]}'
    elif self.prep['top_k_llm'] == 3:
      cur_sit_pref = f'1. {self.prep["sit_pref"][0]}; 2. {self.prep["sit_pref"][1]}; 3. {self.prep["sit_pref"][2]}'
    p = f'''
Now, the human recommends you to work on one of the following actions:
{cur_sit_pref}
- Choose one action from the list above to perform next.
- Your selection should help complete one of the current soup orders.
'''
    return p

  def move_history_description(self):
    super().move_history_description()
    if len(self.prep['prev_sit_pref']) > 0 and self.prep['top_k_llm'] != -1:
      mov_hist = self.prep['mov_hist']
      mh = [m['task'] for m in mov_hist][::-1][:1]
      ms = [m['status'] for m in mov_hist][::-1][:1]
      history_status_pairs = [
          f"{game_action_to_prog_action(a)} ({b})" for a, b in zip(mh, ms)
      ]
      last_action = (";\n".join(history_status_pairs) +
                     '\n') if len(history_status_pairs) > 0 else "None\n"

      prev_sit_pref = f'''
Currently, your action is to {self.prep["prev_sit_pref"][0]}.
You just performed: {last_action}
If this action is not complete and still recommended by the human, you are discouraged to start a new action.'''
    else:
      prev_sit_pref = '\nYou can continue your previous action or switch to a new action.'
    return prev_sit_pref

  def request_description(self):
    super().request_description()

    p = '''
Now please output the most suitable action verbatim from the list of human's suggestions based on the current state.
Do not output anything else. Do not output square brackets.
'''
    return p
