from typing import List
from copy import deepcopy
from agent.executor.high import OBJ_TO_GOODS_GS
from agent.mind.prompt_local import ALL_MOVES
from pickup_multigoals import MACRO_ACTIONS, NAMES_TO_ACTIONS, ACTIONS_TO_NAMES
from minigrid.envs.babyai.unlock import MACRO_ACTION_SPACE
from gym_cooking.utils.core import *
import rw4t.utils as rw4t_utils
import random
import numpy as np

# Action sequences of completing each soup
make_soup_a = [
    'Chop Lettuce', 'Chop Onion', 'Assemble Alice Ingredients',
    'Cook Alice Soup', 'Plate Alice Soup', 'Serve Alice Soup'
]
make_soup_b = [
    'Chop Tomato', 'Chop Lettuce', 'Assemble Bob Ingredients', 'Cook Bob Soup',
    'Plate Bob Soup', 'Serve Bob Soup'
]
make_soup_c = [
    'Chop Tomato', 'Chop Onion', 'Assemble Cathy Ingredients',
    'Cook Cathy Soup', 'Plate Cathy Soup', 'Serve Cathy Soup'
]
make_soup_d = [
    'Chop Tomato', 'Chop Lettuce', 'Chop Onion', 'Assemble David Ingredients',
    'Cook David Soup', 'Plate David Soup', 'Serve David Soup'
]
prep_soup_a = ['Chop Lettuce', 'Chop Onion', 'Assemble Alice Ingredients']
prep_soup_b = ['Chop Tomato', 'Chop Lettuce', 'Assemble Bob Ingredients']
prep_soup_c = ['Chop Tomato', 'Chop Onion', 'Assemble Cathy Ingredients']
prep_soup_d = [
    'Chop Tomato', 'Chop Lettuce', 'Chop Onion', 'Assemble David Ingredients'
]
composite_skills = [
    make_soup_a, make_soup_b, make_soup_c, make_soup_d, prep_soup_a,
    prep_soup_b, prep_soup_c, prep_soup_d
]

skill_idx_to_name = {
    0: 'Make Alice Soup',
    1: 'Make Bob Soup',
    2: 'Make Cathy Soup',
    3: 'Make David Soup',
    4: 'Prepare Alice Soup',
    5: 'Prepare Bob Soup',
    6: 'Prepare Cathy Soup',
    7: 'Prepare David Soup'
}


def prep_mov_hist(mov_hist):
  moves = {}
  for m in mov_hist:
    if m not in moves:
      moves[m] = 1
    else:
      moves[m] += 1

  def times(i):
    if i == 1:
      return " once"
    elif i == 2:
      return " twice"
    else:
      return f" {i} times"

  moves = [f"{k}{times(v)}" for k, v in moves.items()]

  return moves


latent_num_to_str = {0: 'Act', 1: 'Ask', 2: 'Reply', 3: 'Inform'}

# PROMPT


def prompt_base_Ei() -> List[List[str]]:
  # Intention Inference
  p = '''Game Scenario:
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

In-game Decision:
You need to interpret the human player's message into a simpler form, which will be sent to a downstream AI without access to human message history. Your answer must be clear and succinct.

The human's message can be:
1. Useless message: Message that has no specific demand such as "Enough", "Never mind", "You are free to do anything else" or "Try your best to earn more points." translates to "None."
2. Short-term request: "Chop 4 more" means "Chop xxx 4 times.", where "xxx" should be the vegetable in past intention. Keep you answer concise and make sure the numbers are corrent. "Plate the soup now" should be "Plate Soup once."
3. Intention needs to be inferred: Sometimes you need to infer about the hidden meaning of messages. For instance, "I will cook the first order. Can you take charge of the rest?" implies "Cook xxx once and Cook xxx once." where "xxx" are the subsequent soup orders. Similarly, "xxx is handled by me." implies "Cook xxx." where the two "xxx" are different soup in the orders. Emotional and cryptic message like "The David Soup is about to timeout!" suggest "Serve David Soup once."
4. Long-term request: Messages such as "Keep chopping tomatoes" become "Always keep chopping tomatoes, and don't stop." 
5. Questions: Like "What are the orders", "What is xxx Soup" or any question-like queries. You must repeat the original question completely in your output. You must leave the question to the downstream AI intactly who will answer it.
6. Special case: Messages related to asking for orders, like "Tell me the orders", "Keep telling me the orders" or "I want to know the orders" should be translated to "What are the orders now?".

If the human's intention conflicts with soup orders, you should follow the human's intention even if it is not on the orders. Always prioritize the human's message.

Any explanations, comments, tips or modal particles must not be included.
'''
  return [[p, "Ok."], [""]]


def prompt_base_Ei_w_human_intent() -> List[List[str]]:
  # Intention Inference
  avaialble_actions = ', '.join(ALL_MOVES)
  p = f'''Game Scenario:
As an AI assistant in a simplified Overcooked game, work with a human player to complete soup orders. Focus on cooperation, player engagement, fulfillment, and point accrual.

Game Guidelines:
Current orders for soup vary, each with a time limit. Earn a bonus for on-time completion.
To make a soup:
    a. Chop fresh vegetables - Tomato, Lettuce, Onion to obtain chopped vegetables. 
    b. Prepare soup ingredients. 
       Once all required ingredients are chopped, you need to assemble chopped vegetables.
       Here is a list of required ingredients for each soup.
        Alice: Chopped Lettuce, Chopped Onion.
        Bob: Chopped Lettuce, Chopped Tomato.
        Cathy: Chopped Onion, Chopped Tomato.
        David: Chopped Lettuce, Chopped Onion, Chopped Tomato.
    c. Cook the soup. 
       Once the ingredients are prepared, you need to bring the assembled ingredients to cook in the pot.
        Alice Soup: Alice Ingredients.
        Bob Soup: Bob Ingredients.
        Cathy Soup: Cathy Ingredients.
        David Soup: David Ingredients.
    d. Plate the cooked soup. 
       It takes 15 seconds to cook a soup. As you cannot plate the soup immediately after it starts cooking, work on something else before going to the pot to plate the soup.
    e. Serve the plated soup in the serving area to gain points.
If a soup stays in the pot too long, it gets charred. 
    a. Putout: If the pot catches fire, extinguish it. 
    b. Drop: Discard charred soup. Put out the fire in the pot if needed.

Assuming that you have been playing the game for a while. 
Now you will be informed of what the other human player(s) has/have been doing, what you've done, the current state of the game, and messages from other players. 
Based on this information, you need to generate an action for yourself.
The list of available actions is: [{avaialble_actions}].
In the list above, the action 'Prepare <name> Ingredients' means assembling the ingredients that have been chopped; 
the action 'Serve <name> Soup' means serving a plated soup - you need to plate a soup before serving it.
The action you choose from the list above should be the most useful based on the current game state and other players' actions.

You will find your and other players' past actions below. 
Your and other players' past actions are ordered chronologically, meaning that the actions are listed in the sequence they occurred over time.
The first item represents the earliest action, and each subsequent item represents the next action in order.
The most recent action is at the end of the sequence.
Knowing other players' action sequences informs you what they might do in the future, so you can plan accordingly and divide tasks effectively, ensuring that all necessary roles are covered without duplication of effort.
For example, if Alice soup and Cathy soup are among the current orders and another player is preparing the ingredients for Cathy soup, you can work on Alice soup to prevent duplicate efforts.

You will also be presented with the current order and the items that are available on the map.
If a current order does not have much time left and has not started cooking, focus on the other orders.
Finally, take the other players' messages into account.
If the message is a request, you should fulfill the request if you think the request is reasonable given the current game state.
Be sure to consider the your actions, other players' past actions, other players' message, the current orders, and the items on the map when deciding what you will do.
'''
  return [[p, "Ok."], [""]]


def prompt_order_int(order_prep: list) -> str:
  if len(order_prep) == 0:
    prompt = 'Soup orders are not visible to you.\n\n'
    return prompt

  orders = [rec['name'].replace("Plated ", "") for rec in order_prep]
  prompt = f"Current soup orders: {', '.join(orders)}\n\n"

  return prompt


def prompt_order(order_prep: list) -> str:
  # current orders
  prompt = '''
Current soup orders:
'''
  if len(order_prep) == 0:
    prompt += 'Soup orders are currently not visible for you.\n'
    return prompt

  for rec in order_prep:
    soup_name = rec['name'].replace("Plated ", "")
    rate = rec['rate']
    prompt += f'- {soup_name}: '
    if rate > 0.75:
      prompt += 'plenty of time\n'
    elif rate > 0.5:
      prompt += 'still some time\n'
    elif rate > 0.25:
      prompt += 'not much time\n'
    else:
      prompt += 'will expire in no time\n'

  return prompt


def prompt_map(inp: str) -> str:
  return inp


def prompt_reason_Ei(int_hist: list) -> str:
  prompt = ""
  # task history
  if len(int_hist) == 2 and int_hist[0]['ret'] is not None:
    prompt += "The human player's intention in the last round (which has already been satisfied):\n"
    prompt += f'"{int_hist[0]["ret"]}"\n\n'

  prompt += "The human player's message now:\n"
  prompt += f'"{int_hist[-1]["chat"]}"\n\n'

  prompt += "\n"
  prompt += '''Be very careful that if the message is a question, you must repeat it completely in your answer. DO NOT ANSWER IT. You need to interpret the human player's message only if it is not a question.'''

  return prompt


def prompt_reason_Ei_w_human_intent(prep, msgs=True) -> str:
  # Chat & Completion Assessment: chat message (without ongoing intention)
  def prompt_reason_El2(mov_hist: list) -> str:
    # print('before human intent')
    prompt = 'What other human player(s) has/have been doing:\n'
    human_intents = prep['human_intents']
    # print(str(human_intents[0]))
    human_intents_str = ''
    if len(human_intents) == 0:
      human_intents_str += 'Unknown\n'
    for player_idx in range(len(human_intents)):
      human_intents_str += ('Human player ' + str(player_idx + 1) + ': ')
      human_intents_str += str(human_intents[player_idx])
      human_intents_str += '\n'
    prompt += f'{human_intents_str}\n'
    # print('after human intent')

    mh = [m['task'] for m in mov_hist]
    moves = prep_mov_hist(mh)

    prompt += "Actions you've done recently:\n"
    prompt += f'{", ".join(moves)}\n\n' if len(moves) > 0 else "None\n\n"

    if msgs:
      int_hist = prep['int_hist']
      prompt += "The other player's message:\n"
      prompt += f'"{int_hist[-1]["chat"]}"\n\n'

    if 'latent' in prep:
      print('Latent :', latent_num_to_str[prep['latent']])

    return prompt

  order_prep = prep['order']
  env = prep['map']
  mov_hist = prep['mov_hist']

  order = prompt_order(order_prep) + '\n'
  reason = prompt_reason_El2(mov_hist)
  reason += '\n'

  p = order
  p += env + '\n\n'
  p += reason

  last_move = prep['ai_intent']

  if last_move == '':
    p += '''
Now respond with an action that you will execute.
End your response in a semicolon.
'''
  else:
    p += f'''
Now respond with an action that you will execute after your current action: {last_move}.
End your response in a semicolon.
'''

  return p


def prompt_base_El_s(prep):
  # Chat & Completion Assessment: init (with ongoing intention)
  p = '''Game Scenario:
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

Gameplay Rounds:
Round One - Action Summary: In this stage, your task is to summarize the actions you've made that are directly beneficial to the human player's request. 
Round Two - Communication: Here, you generate your chat message to be sent to the human player.
Round Three - Satisfaction Evaluation: In this round, it's your responsibility to judge whether the player's request has been fully met based on your actions.

Note that there are multiple types of human's incoming message:
1. Short term request: Like "Chop 4 times", "Chop once", "Cook 2 Soup" or "Plate once". If you have done ALL actions he requests, then it is satisfied. It is OK if you've done more than he asks. If there are still actions to be done, then it is not satisfied.
2. Long term request: Like "Always prepare", "Keep chopping", "Plating continuously", "Cook don't stop" or "Avoid serving". In these cases, the requests will never be satisfied because they need to be done continuously, even if your actions conflict with them,
3. Question: Like "What are the current orders?" or "What is xxx Soup?" You need to answer to the question in the chat message. And you must give "Yes" in the Satisfaction Evaluation round.
4. Useless message: Like "None", "Free to do anything", "No specific intention", or statement of fact like "The orders are xxx". You must "Yes" in the Satisfaction Evaluation round.
'''
  return p


def prompt_base_El_s2(prep):
  # Chat & Completion Assessment: init init (without ongoing intention)
  p = '''Game Scenario:
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
You are recommended to give your future plan. Giving information about current orders and their time limit is also a good idea. You shouldn't focus on the Fire Extinguisher.
You answer must be concrete and informative with no more than 10 words. Just give your chat message with no explanation, no comments, no quotation marks and no emojis.
'''

  return p


def prompt_base_El_s2_w_human_intent(prep):
  if 'latent' in prep:
    latent = latent_num_to_str[prep['latent']]
    if latent == 'Act':
      instruction = '''
Based on this information, you need to generate an action to be executed yourself.
'''
    elif latent == 'Ask':
      instruction = '''
Based on this information, you need to generate an action to be executed yourself and a chat message to ask a question to ask the other agents.
'''
    elif latent == 'Respond':
      instruction = '''
Based on this information, you need to generate an action to be executed yourself and a chat message to response to a message from the other agents.
'''
    else:
      instruction = '''
Based on this information, you need to generate an action to be executed yourself and a chat message to provide information on what you are currently doing.
'''
  else:
    instruction = '''
Based on this information, you need to generate an action for yourself as well as a chat message to be sent to the human player(s).
'''

  # Chat & Completion Assessment: init init (without ongoing intention)
  avaialble_actions = ', '.join(ALL_MOVES)
  p = '''Game Scenario:
As an AI assistant in a simplified Overcooked game, work with a human player to complete soup orders. Focus on cooperation, player engagement, fulfillment, and point accrual.

Game Guidelines:
Current orders for soup vary, each with a time limit. Earn a bonus for on-time completion.
To make a soup:
    a. Chop fresh vegetables - Tomato, Lettuce, Onion to obtain chopped vegetables.
    b. Prepare soup ingredients.
       Once all required ingredients are chopped, you need to assemble chopped vegetables.
       Here is a list of required ingredients for each soup.
        Alice: Chopped Lettuce, Chopped Onion.
        Bob: Chopped Lettuce, Chopped Tomato.
        Cathy: Chopped Onion, Chopped Tomato.
        David: Chopped Lettuce, Chopped Onion, Chopped Tomato.
    c. Cook the soup.
       Once the ingredients are prepared, you need to bring the assembled ingredients to cook in the pot.
        Alice Soup: Alice Ingredients.
        Bob Soup: Bob Ingredients.
        Cathy Soup: Cathy Ingredients.
        David Soup: David Ingredients.
    d. Plate the cooked soup.
       It takes 15 seconds to cook a soup. As you cannot plate the soup immediately after it starts cooking, work on something else before going to the pot to plate the soup.
    e. Serve the plated soup in the serving area to gain points.
If a soup stays in the pot too long, it gets charred.
    a. Putout: If the pot catches fire, extinguish it.
    b. Drop: Discard charred soup. Put out the fire in the pot if needed.

Assuming that you have been playing the game for a while.
Now you will be informed of what the other human player(s) has/have been doing, what you've done, the current state of the game, and messages from other players. ''' + \
instruction + '''
The list of available actions is: [''' + avaialble_actions + '''].
In the list above, the action 'Prepare <name> Ingredients' means assembling the ingredients that have been chopped;
the action 'Serve <name> Soup' means serving a plated soup - you need to plate a soup before serving it.
The action you choose from the list above should be the most useful based on the current game state and other players' actions.

You will find your and other players' past actions in below.
Your and other players' past actions are ordered chronologically, meaning that the actions are listed in the sequence they occurred over time.
The first item represents the earliest action, and each subsequent item represents the next action in order.
The most recent action is at the end of the sequence.
Knowing other players' action sequences informs you what they might do in the future, so you can plan accordingly and divide tasks effectively, ensuring that all necessary roles are covered without duplication of effort.
For example, if Alice soup and Cathy soup are among the current orders and another player is preparing the ingredients for Cathy soup, you can work on Alice soup to prevent duplicate efforts.
Knowing your past actions and their corresponding statuses can inform you about the feasibility of various actions.
If an action has failed multiple times, this means that you cannot perform this action and need to ask ohter players to perform this action instead.

You will also be presented with the current order and the items that are available on the map.
If a current order does not have much time left and has not started cooking, focus on the other orders.
Be sure to consider the your and the other players' past actions as well as the current orders and the items are the map when deciding what you will do.
'''

  if ('latent' not in prep) or ('latent' in prep and latent != 'Act'):
    p += """
The chat message must be specific and informative with no more than 15 words. 
Just give your chat message with no explanation, no comments, no quotation marks and no emojis.
"""

  # print(p)
  return p


def prompt_base_El_1(prep):
  # Chat & Completion Assessment: reasoning (with ongoing intention)
  def prompt_reason_El(mov_hist: list, chat: str) -> str:
    prompt = ""
    prompt += "The human player's incoming message:\n"
    prompt += f'"{chat}"\n\n'

    mh = [m['task'] for m in mov_hist]
    moves = prep_mov_hist(mh)

    prompt += "Actions you've done since the human gave the message:\n"
    prompt += f'{", ".join(moves)}\n\n' if len(moves) > 0 else "None\n\n"

    return prompt

  # chk_moves = prep['chk_moves']
  order_prep = prep['order']
  env = prep['map']
  # int_hist = prep['int_hist']
  # llm_hist = prep['llm_hist']
  mov_hist = prep['mov_hist']
  chat = prep['chatin']

  order = prompt_order(order_prep) + '\n'
  reason = prompt_reason_El(mov_hist, chat) + '\n'

  p = order
  p += env
  p += reason
  p += "\n"
  p += '''You need to examine the current state of the game environment, the human player's message, and actions you've taken so far. Now summarize the actions you've done that are directly beneficial to the human player's request. Any action not related to their request can be ignored.\n'''
  p += '''If the human player's request is "None" or a question, just briefly summarize your current actions.\n'''
  p += '''You must be honest and give actions that is surely done by yourself. Do not make up!\n'''
  p += '''Keep your answer short and concise. No more than 20 words.\n'''

  # print(reason)

  return p


def prompt_base_El_1_w_human_intent(prep):
  # Chat & Completion Assessment: reasoning (with ongoing intention)
  def prompt_reason_El(mov_hist: list, chat: str) -> str:
    prompt = ""
    prompt += "The human player's incoming message:\n"
    prompt += f'"{chat}"\n\n'

    prompt += 'What other human player(s) has/have been doing:\n'
    human_intents = prep['human_intents']
    # print(str(human_intents[0]))
    human_intents_str = ''
    for player_idx in range(len(human_intents)):
      human_intents_str += ('Human player ' + str(player_idx + 1) + ': ')
      human_intents_str += str(human_intents[player_idx])
      human_intents_str += '\n'
    prompt += f'{human_intents_str}\n'

    mh = [m['task'] for m in mov_hist]
    moves = prep_mov_hist(mh)

    prompt += "Actions you've done since the human gave the message:\n"
    prompt += f'{", ".join(moves)}\n\n' if len(moves) > 0 else "None\n\n"

    return prompt

  # chk_moves = prep['chk_moves']
  order_prep = prep['order']
  env = prep['map']
  # int_hist = prep['int_hist']
  # llm_hist = prep['llm_hist']
  mov_hist = prep['mov_hist']
  chat = prep['chatin']

  order = prompt_order(order_prep) + '\n'
  reason = prompt_reason_El(mov_hist, chat) + '\n'

  p = order
  p += env
  p += reason
  p += "\n"
  p += '''You need to examine the current state of the game environment, the human player's current action, the human player's message, and actions you've taken so far.\n'''
  p += '''Now summarize the actions you've done that are directly beneficial to the human player's request. Any action not related to their request can be ignored.\n'''
  p += '''If the human player's request is "None" or a question, just briefly summarize your current actions.\n'''
  p += '''You must be honest and give actions that is surely done by yourself. Do not make up!\n'''
  p += '''Keep your answer short and concise. No more than 20 words.\n'''

  # print(reason)

  return p


def prompt_base_El_2(prep):
  # Chat & Completion Assessment: chat message (without ongoing intention)
  p = '''Generate your chat message to be send to the human. Your communication should be polite, helpful, and limited to 20 words max. Aim to demonstrate your enthusiasm and friendliness while assisting the player. 
If the human player asks a question, ensure to provide an appropriate response. For example, if he asks "What are the current orders?", you should respond with the current orders and their time remaining.
You also have the opportunity to inform the player of your current and planned actions. 
Just give your message, with no quotation marks or emojis.'''

  return p


def prompt_base_El_22(prep):
  # Chat & Completion Assessment: chat message (without ongoing intention)
  def prompt_reason_El2(mov_hist: list, chat: str) -> str:
    mh = [m['task'] for m in mov_hist]
    moves = prep_mov_hist(mh)

    prompt = "Actions you've done recently:\n"
    prompt += f'{", ".join(moves)}\n\n' if len(moves) > 0 else "None\n\n"

    return prompt

  order_prep = prep['order']
  env = prep['map']
  # int_hist = prep['int_hist']
  # llm_hist = prep['llm_hist']
  mov_hist = prep['mov_hist']
  chat = prep['chatin']

  order = prompt_order(order_prep) + '\n'
  reason = prompt_reason_El2(mov_hist, chat) + '\n'

  p = order
  p += env + '\n\n'
  p += reason

  p += '''
Now give your chat message to be sent to the human. 
'''

  return p


def prompt_base_El_22_w_human_intent(prep, msgs=False):
  # Chat & Completion Assessment: chat message (without ongoing intention)
  def prompt_reason_El2(mov_hist: list) -> str:
    # print('before human intent')
    prompt = 'What other human player(s) has/have been doing:\n'
    human_intents = prep['human_intents']
    # print(str(human_intents[0]))
    human_intents_str = ''
    if len(human_intents) == 0:
      human_intents_str += 'Unknown\n'
    for player_idx in range(len(human_intents)):
      human_intents_str += ('Human player ' + str(player_idx + 1) + ': ')
      human_intents_str += str(human_intents[player_idx])
      human_intents_str += '\n'
    prompt += f'{human_intents_str}\n'
    # print('after human intent')

    mh = [m['task'] for m in mov_hist]
    # moves = prep_mov_hist(mh)

    prompt += "Actions you've done recently:\n"
    # prompt += f'{", ".join(moves)}\n\n' if len(moves) > 0 else "None\n\n"
    prompt += f'{", ".join(mh[-5:])}\n\n' if len(mh) > 0 else "None\n\n"

    ms = [m['status'] for m in mov_hist]
    prompt += 'Corresponding status of the actions you\'ve done recently:\n'
    prompt += f'{", ".join(ms[-5:])}\n\n' if len(mh) > 0 else "None\n\n"

    if msgs:
      int_hist = prep['int_hist']
      prompt += "The other player's message:\n"
      prompt += f'"{int_hist[-1]["chat"]}"\n\n'

    return prompt

  order_prep = prep['order']
  env = prep['map']
  mov_hist = prep['mov_hist']

  order = prompt_order(order_prep) + '\n'
  reason = prompt_reason_El2(mov_hist)
  reason += '\n'

  p = order
  p += env + '\n\n'
  p += reason

  last_move = prep['ai_intent']
  if 'latent' in prep:
    print(latent_num_to_str[prep['latent']])
    act_only = (latent_num_to_str[prep['latent']] == 'Act')
  else:
    act_only = False

  p += 'Based on your past interactions with the human player, you observe that: '
  p += prep['prev_reasoning']
  p += ' Please decide an action that matches the human\'s preference.'

  if last_move == '':
    p += '''
Now respond with an action that you will execute.'''
  else:
    p += f'''
Now respond with an action that you will execute after your current action: {last_move}.
'''

  if act_only:
    p += """
End your response with a semicolon.
"""
  else:
    p += """
Also generate a chat message to be sent to the other player(s).
Separate the action from the chat message with a semicolon.
Just output one action and one chat message. Do not output anything else.
"""
  # print(p)
  return p


def prompt_base_El_3(prep):
  # Chat & Completion Assessment: completion assessment (with ongoing intention)
  p = '''Judge whether the player's request has been fulfilled by your actions. The possible responses are "Yes" or "No". 
If the human's incoming message is a question or a useless message, give "Yes". '''
  return p


def prompt_base_Hl_s(prep):
  # SMOA: init
  p = '''Game Scenario:
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

Gameplay Rounds:
Round One - Action Summary: In this stage, your task is to summarize the actions you've made that are directly beneficial to the human player's request. 
Round Two - Communication: Here, you generate your chat message to be sent to the human player.
Round Three - Satisfaction Evaluation: In this round, it's your responsibility to judge whether the player's request has been fully met based on your actions.
Round Four - Action Execution: You are to give your action to be carried out next.

Note that there are multiple types of human's incoming message:
1. Short term request: Like "Chop 4 times", "Chop once", "Cook 2 Soup" or "Plate once". If you have done ALL actions he requests, then it is satisfied. It is OK if you've done more than he asks. If there are still actions to be done, then it is not satisfied.
2. Long term request: Like "Always prepare", "Keep chopping", "Plating continuously", "Cook don't stop" or "Avoid serving". In these cases, the requests will never be satisfied because they need to be done continuously, even if your actions conflict with them,
3. Question: Like "What are the current orders?" or "What is xxx Soup?" You need to answer to the question in the chat message. And you must give "Yes" in the Satisfaction Evaluation round.
4. Useless message: Like "None", "Free to do anything", "No specific intention", or statement of fact like "The orders are xxx". You must "Yes" in the Satisfaction Evaluation round.
'''
  return p


def prompt_base_El_5(prep):
  # SMOA: act
  chk_moves = prep['chk_moves']
  all_moves = [m[0] for m in chk_moves if m[1]]
  # all_moves = [m[0] for m in chk_moves]
  p = "Give your action to be carried out next. You should try to serve, plate and cook soup when possible. Select it from " + ", ".join(
      all_moves
  ) + ". You can only choose one from it and not allowed to make up new action. Explanation or comment is not needed."
  return p


def prompt_instr_reasoning(prep):
  instr = prep['chatin']
  prev_reasoning = prep['prev_reasoning']
  order_prep = prep['order']
  env = prep['map']
  mh = [m['task'] for m in prep['mov_hist']]
  print(mh)

  order = prompt_order(order_prep)

  p1 = '''Game Scenario:
You are an assistant in a simplified Overcooked game and you are working with a human player to complete soup orders.
The human player will give you an instruction that specifies what you should do periodically.
You will follow the human's instruction and reason about what the human wants you to do in different game situations.

Game rules:
Current soup orders can vary, and each order has a time limit.
To make a soup:
    a. Chop fresh vegetables - Tomato, Lettuce, Onion to obtain chopped vegetables.
    b. Prepare soup ingredients.
       Once all required ingredients are chopped, you need to assemble chopped vegetables.
       Here is a list of required ingredients for each soup.
        Alice: Chopped Lettuce, Chopped Onion.
        Bob: Chopped Lettuce, Chopped Tomato.
        Cathy: Chopped Onion, Chopped Tomato.
        David: Chopped Lettuce, Chopped Onion, Chopped Tomato.
    c. Cook the soup.
       Once the ingredients are prepared, you need to bring the assembled ingredients to cook in the pot.
        Alice Soup: Alice Ingredients.
        Bob Soup: Bob Ingredients.
        Cathy Soup: Cathy Ingredients.
        David Soup: David Ingredients.
    d. Plate the cooked soup.
    e. Serve the plated soup in the serving area to gain points.
If a soup stays in the pot too long, it gets charred.
    a. Putout: If the pot catches fire, extinguish it.
    b. Drop: Discard charred soup. Put out the fire in the pot if needed.
'''

  p2 = f"""
Assume you have been playing the game for a while.
Here is the current state of the game:
{order}
{env}

Here is the human's instruction: {instr}

Here is a history of your actions: {", ".join(mh)}

Based on the game state, human instruction, and your action history, 
please provide specific and concise answers to the following questions:
1. Summarize your action history.
2. Are there any patterns in the actions you take?
3. What is the task division in this game based on your observations?
4. More concretely, what tasks are you responsible for?

The purpose of these questions is for you to learn and infer what the human prefers you to work on, 
so you can collaborate with the human fluently without human instructions.

Your previous answers to these questions are shown below.
{prev_reasoning}.

Your answers will change as you gather more information, so make necessary modifications to your previous response.
Do not output anything else.
"""

  p3 = "My response is: "

  return [[p1, 'Ok.'], [p2, p3]]


def prompt_reasoning(prep, no_hl_rec=False):
  if 'no_hl_rec' in prep:
    # print('no hl rec: ', prep['no_hl_rec'])
    no_hl_rec = prep['no_hl_rec']
  all_moves_copy = deepcopy(ALL_MOVES)
  for move_idx in range(len(all_moves_copy)):
    if 'Prepare' in all_moves_copy[move_idx]:
      all_moves_copy[move_idx] = all_moves_copy[move_idx].replace(
          'Prepare', 'Assemble')
  all_moves_copy.remove('Wait')
  avaialble_actions = ', '.join(all_moves_copy)
  p1 = '''Game Scenario:
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
The list of available actions is: [''' + avaialble_actions + '''].
"Assemble ... Ingredients": assemble the chopped vegetables for a soup;
"Cook ... Soup": bring the assembled ingredients to a pot and start cooking;
"Plate ... Soup": transfer the soup to a plate, ready to be served;
"Serve ... Soup": serve the plated soup to the customers at the delivery location.
'''

  if prep['pref'] != '':
    p1 += oc_pref_helper(prep['pref'])

  if len(prep['sit_pref']) > 1:
    raise NotImplementedError

  if len(prep['sit_pref']) == 1:
    if prep['sit_pref'][0] in prep['sit_pref_actions']:
      cur_skill = prep['sit_pref_actions'][prep['sit_pref'][0]]
    else:
      cur_skill = []

  p2 = 'Ok.'

  if no_hl_rec:
    #     hl_rec = '''Explanantion of possible human suggestions:
    # 'Make ... Soup': prepare the ingredients and cook the specific soup;
    # 'Get ... Soup': plate and serve the cooked soup.'''
    hl_rec = ''
  else:
    hl_rec = f'\nActions to perform the task: {cur_skill}.'

  if len(prep['sit_pref']) == 1:
    p3 = f'''
The human suggests you to work on the following task: {prep['sit_pref'][0]}{hl_rec}
- Choose an action for this task, unless there is an urgent action not from this list.
- More urgent actions can include plating a soup or serving a soup.
- The action you take should utilize the existing ingredients on the map as much as possible.
- For example, if a vegetable is already chopped, avoid chopping it again; if you already have all the ingredients for a soup, proceed directly to assembling or cooking.
- Do not work on a soup order that is not suggested by the human.
'''
    print('Composite: ', prep['sit_pref'][0])

  if len(prep['sit_pref']) == 0:
    p3 = ''

  def move_history_helper(mov_hist: list) -> str:
    mh = [m['task'] for m in mov_hist][::-1][:1]
    ms = [m['status'] for m in mov_hist][::-1][:1]
    # moves = prep_mov_hist(mh)

    prompt = "Your previous action and its corresponding status: "
    history_status_pairs = [f"{a} ({b})" for a, b in zip(mh, ms)]
    # print('History status pairs: ', history_status_pairs)
    prompt += (";\n".join(history_status_pairs[-7:]) +
               '\n') if len(history_status_pairs) > 0 else "None\n"
    # print('Prompt so far: ', prompt)
    prompt += 'If an action fails, please generate a different action, as this means that this action might not be available.\n'

    return prompt

  order_prep = prep['order']
  order = prompt_order(order_prep)
  env = prep['map']
  p3 += order
  p3 += env
  if prep['holding'] != '':
    p3 += f'You are holding: {OBJ_TO_GOODS_GS[prep["holding"]]}.'
  p3 += '''
Please examine the environment information carefully when you decide your action.
'''

  mov_hist_p = move_history_helper(prep['mov_hist'])
  p3 += (mov_hist_p + '\n')

  if prep['gen_mode'] == '5_ranked':
    p3 += '''
Now please also respond to the following request:
Output the top 5 actions that you would perform and give them a confidence score from 1 to 5. 
The scores should be proportional to how likely you will perform this action.

Please format your output like a Python dictionary, mapping the name of the action, to a number between 1 and 5.
Your final output should look like this: "{'action 1': score_1, 'action 2': score_2, 'action 3': score_3, 'action 4': score_4, 'action 5': score_5}"
Do not output anything else.
'''
  elif prep['gen_mode'] == '5_unranked':
    p3 += '''
Now please respond to the following request:
Output one action that is the most suitable for the current game state.
The action should be feasible to perform and also contribute toward completing soup orders.
Do not output the same action twice.

Please format your output like a Python list.
Your final output should look like this: "['action 1', 'action 2', 'action 3']"
Do not output anything else.
'''
  elif prep['gen_mode'] == 'all_yes_no' or prep[
      'gen_mode'] == 'all_yes_no_include_false':
    # executable_actions = [m[0] for m in prep['chk_moves'] if m[1]]
    # executable_actions.append('Wait')
    # print('executable actions: ', executable_actions)
    #     p3 += f'''
    # Now please also respond to the following request:
    # Among all the actions in the list of available actions, you can only execute the following executable actions: {executable_actions}.
    # For each action in the list of executable actions, output True if you think it is a productive action to be performed at this moment and False otherwise.
    # You can output True for multiple actions.

    # Please format your output like a dictionary, mapping the name of the action, to True or False.
    # Your final output should look like this: "{{'action 1': True, 'action 2': False, 'action 3': True, ...}}"
    # Do not output anything else.
    # '''

    # recommended_actions = prep['available_actions']
    # plate_actions = [
    #     'Plate Alice Soup', 'Plate Bob Soup', 'Plate Cathy Soup',
    #     'Plate David Soup'
    # ]
    # for p_a in plate_actions:
    #   if p_a not in recommended_actions:
    #     recommended_actions.append(p_a)
    p3 += f'''Please also respond to the following request:
The list of available actions is: {avaialble_actions}.
However, not every action in the list above is productive and feasible to be performed at the moment.
For each action in the list of, output True if you think it is both productive and feasible and output False otherwise.

Please format your output like a dictionary, mapping the name of the action, to True or False.
Your final output should look like this: "{{'action 1': True, 'action 2': False, 'action 3': True, ...}}"
Do not output anything else.
'''
  elif prep['gen_mode'] == 'top':
    executable_actions = [m[0] for m in prep['chk_moves'] if m[1]]
    executable_actions.append('Wait')
    p3 += f'''
Among all the actions in the list of available actions, you can only execute the following executable actions: {executable_actions}.
Now please also respond to the following request:
Select an action that you would like to perform at this moment from the list of executable actions. Just output the action name. Do not output anything else.
'''
  else:
    raise NotImplementedError

  # print(p1)
  # print(p3)
  # print('holding: ', prep['holding'])
  # print('done with prompt reasoning')
  prompt = [[p1, p2], [p3]]
  return prompt


def prompt_composite_reasoning(prep, top_k=2):
  if 'top_k_llm' in prep:
    # print('top k llm: ', prep['top_k_llm'])
    top_k = prep['top_k_llm']
  p1 = '''Game Scenario:
You are a compliant and helpful AI assistant in a simplified version of the Overcooked game.
Your goal is to follow the human's suggestion (if provided) and achieve a high score in the game.

Game Rules:
You need to make soups according to the current soup orders, and each order has a time limit.

Steps to make a soup:
1.  Chop vegetables.
    Chop lettuce and onion to make Alice soup.
    Chop lettuce and tomato to make Bob soup.
    Chop onion and lettuce to make Cathy soup.
    Chop lettuce, onion, and tomato to make David Soup.
2.  Assemble the chopped vegetables.
3.  Cook the assembled ingredients.
4.  Plate the soup.
5.  Serve the soup.

Managing burned soup:
If a soup stays in the pot too long, it will get charred.
1.  Putout: If the pot catches fire, extinguish it.
2.  Drop: Discard charred soup.
Note: Putting out fires takes time, so ensure to plate the soup as soon as it is cooked to avoid delays.

Assuming that you have been playing the game for a while.
You will be presented with the current orders, available items on the map, and potentially other information like past actions.
'''

  if prep['pref'] != '':
    p1 += oc_pref_helper(prep['pref'])

  if prep['sit_pref'] == [] or len(prep['sit_pref']) <= 1:
    raise NotImplementedError

  cur_skill = ''
  skill_str = ''
  for skill_idx in range(len(composite_skills)):
    skill_str += f'{skill_idx_to_name[skill_idx]}: {composite_skills[skill_idx]}\n'
    if cur_skill == skill_idx_to_name[skill_idx]:
      cur_skill = composite_skills[skill_idx]

  p2 = 'Ok.'

  if top_k == -1:
    top_k = len(prep['sit_pref'])
    cur_sit_pref = str(prep['sit_pref'])
    # suggestion = ''
  elif top_k == 2:
    cur_sit_pref = f'1. {prep["sit_pref"][0]}; 2. {prep["sit_pref"][1]}'
    # suggestion = 'As each order has a time limit, prioritize getting soups. '
  elif top_k == 3:
    cur_sit_pref = f'1. {prep["sit_pref"][0]}; 2. {prep["sit_pref"][1]}; 3. {prep["sit_pref"][2]}'
    # suggestion = 'As each order has a time limit, prioritize plating and serving soups. '
  else:
    raise NotImplementedError

  if len(prep['prev_sit_pref']) > 0:
    prev_sit_pref = f'Currently, your action is to {prep["prev_sit_pref"][0]}.\n'
    prev_sit_pref += f'You just performed: {oc_move_history_helper(prep, False)}.'
    if top_k == 2 or top_k == 3:
      prev_sit_pref += f'''
Reason about whether you have completed '{prep["prev_sit_pref"][0]}'.
If the action is not complete and still recommended by the human, you are discouraged to start a new action.'''
    else:
      prev_sit_pref += '\nYou can continue your previous action or switch to a new action.'
  else:
    prev_sit_pref = ''

  p3 = f'''
Now, the human recommends you to work on one of the following actions:
{cur_sit_pref}
- Choose one action from the list above to perform next.
- Your selection should help complete one of the current soup orders.
{prev_sit_pref}
'''
  # print(f'Options: {prep["sit_pref"][:3]}')

  order_prep = prep['order']
  order = prompt_order(order_prep)
  env = prep['map']
  p3 += order
  p3 += env
  p3 += '''
Please examine the current orders and items on the map carefully when you decide.
'''

  # mov_hist_p = move_history_helper(prep['mov_hist'])
  # p3 += (mov_hist_p + '\n')

  p3 += '''
Now please output the most suitable action verbatim from the list of human's suggestions based on the current state.
Do not output anything else. Do not output square brackets.
'''

  # print(p1)
  # print(p3)
  # print('done with prompt composite reasoning')
  prompt = [[p1, 'Ok.'], [p3]]
  return prompt


def progprompt_reasoning(prep):
  p1 = '''Game Scenario:
You are a compliant and helpful AI assistant in a simplified version of the Overcooked game.
Your goal is to follow the human's suggestions and achieve a high score in the game.

Game rules:
There are four different types of soup: Alice soup, Bob soup, Cathy soup, and David soup.
At any given time, you will see three soup orders.
Your task is to prepare soups according to these current orders.
Each soup order has a time limit.
Points are awarded for completing only some of the soup orders.
You do not know in advance which soups will yield points, but the human does, and they will provide suggestions on the order to prioritize.
Therefore, always follow the human's suggestions.

Your available actions include:
chop(<obj>)
combine(<obj>, <obj>)
combine(< obj>, <obj>, <obj>)
putin(<obj>, <obj>)
serve(<obj>)
putout()
discard()
'''

  p2 = '''Here are the recipes for making different soups.
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
'''

  p3 = '''def make_Cathy_soup():
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

  order_prep = prep['order']
  order = prompt_order(order_prep)
  #   p4 = f'''Current game information:
  # {order}
  # Current pots status:
  # {oc_get_pot_info(prep)}
  # available_objects = {oc_get_available_objs(prep)}

  # The human suggests you to choose from the following actions:
  # {oc_sit_pref_to_actions(prep)}

  # Your previous action and its status: {oc_move_history_helper(prep)}
  # Please output an action to perform based on the suggestions, the current environment, and your previous action.
  # In general, you should not always start from the beginning of a recipe;
  # instead, you should utilize the available objects on the map.
  # For example, avoid chopping an ingredient if already chopped ones are available.
  # You should follow the human's suggestions as much as possible.
  # If you think that none of the actions suggested by the human is suitable, you can output a different action.
  # Ensure that any action involving an object uses one from the available_objects list.
  # Do not select an action that is not the exact same as a function call in one of the recipes.
  # Please output only the action, and nothing else.
  # '''

  p4 = f'''Current game information:
{order}
Current pots status:
{oc_get_pot_info(prep)}
available_objects = {oc_get_available_objs(prep)}

The human suggests you to choose an action that is in this function: {prep["sit_pref"][0]}.
You should follow the human's suggestions as much as possible, unless you think there is a more urgent action.
Your previous action and its status: {oc_move_history_helper(prep)}
Please output an action to perform based on the suggestion, the current environment, and your previous action.
In general, you should not always start from the beginning of a recipe;
instead, you should utilize the available objects on the map.
For example, avoid chopping an ingredient if already chopped ones are available.
Ensure that any action involving an object uses one from the available_objects list.
Your selected action has to correspond to a function call in one of the recipes.
Please output only the action, and nothing else.
'''

  #   p4 = f'''Current game information:
  # {order}
  # Current pots status:
  # {oc_get_pot_info(prep)}
  # available_objects = {oc_get_available_objs(prep)}

  # {oc_pref_helper(prep['pref'])}

  # Your previous action and its status: {oc_move_history_helper(prep)}
  # Please output an action to perform based on the suggestions, the current environment, and your previous action.
  # In general, you should not always start from the beginning of a recipe;
  # instead, you should utilize the available objects on the map.
  # For example, avoid chopping an ingredient if already chopped ones are available.
  # Ensure that any action involving an object uses one from the available_objects list.
  # Do not select an action that is not the exact same as a function call in one of the recipes.
  # Please output only the action, and nothing else.
  # '''
  print('Done with progprompt reasoning')
  prompt = [[p1 + p2 + p3, 'Ok.'], [p4]]
  return prompt


def progprompt_composite_reasoning(prep):
  p1 = '''You are a compliant and helpful AI assistant in a simplified version of the Overcooked game.
Your goal is to follow the human's suggestions and achieve a high score in the game.

Game rules:
There are four different types of soup: Alice soup, Bob soup, Cathy soup, and David soup.
At any given time, you will see three soup orders.
Your task is to prepare soups according to these current orders.
Each soup order has a time limit.
Points are awarded for completing only some of the soup orders.
You do not know in advance which soups will yield points, but the human does, and they will provide suggestions on the order to prioritize.
Therefore, always follow the human's suggestions.
'''

  order_prep = prep['order']
  order = prompt_order(order_prep)
  if len(prep['prev_sit_pref']) > 0:
    prev_sit_pref = f'Currently, your action is to {prep["prev_sit_pref"][0]}\n'
    prev_sit_pref += 'If this action is not complete and still recommended by the human, you are discouraged to start a new action.'
  else:
    prev_sit_pref = ''
  p4 = f'''Current game information:
{order}
Current pots status:
{oc_get_pot_info(prep)}

The human suggests you to choose one of the following actions:
1. {prep["sit_pref"][0]} (more preferred) 2. {prep["sit_pref"][1]}

{prev_sit_pref}
Please output an action to perform based on the suggestion and the current environment.
You should select the more preferred action unless this action is not productive for the current state.
Please output only the action, and nothing else.
'''
  print('Done with progprompt composite reasoning')
  prompt = [[p1, 'Ok.'], [p4]]
  return prompt


def prompt_guided_reasoning(prep):
  avaialble_actions = ', '.join(ALL_MOVES)
  p1 = '''Game Scenario:
As an AI assistant in a simplified Overcooked game, work with a human player to complete soup orders. 
Focus on cooperation, player engagement, fulfillment, and point accrual.

Game Guidelines:
Current orders for soup vary, each with a time limit. Earn a bonus for on-time completion.
To make a soup:
    a. Chop fresh vegetables - Tomato, Lettuce, Onion to obtain chopped vegetables.
    b. Prepare soup ingredients.
       Once all required ingredients are chopped, you need to assemble chopped vegetables.
       Here is a list of required ingredients for each soup.
        Alice: Chopped Lettuce, Chopped Onion.
        Bob: Chopped Lettuce, Chopped Tomato.
        Cathy: Chopped Onion, Chopped Tomato.
        David: Chopped Lettuce, Chopped Onion, Chopped Tomato.
    c. Cook the soup.
       Once the ingredients are prepared, you need to bring the assembled ingredients to cook in the pot.
        Alice Soup: Alice Ingredients.
        Bob Soup: Bob Ingredients.
        Cathy Soup: Cathy Ingredients.
        David Soup: David Ingredients. (Please prioritize preparing David Soup as it is worth more points than other soups.)
    d. Plate the cooked soup.
       It takes 15 seconds to cook a soup. As you cannot plate the soup immediately after it starts cooking, work on something else before going to the pot to plate the soup.
    e. Serve the plated soup in the serving area to gain points.
If a soup stays in the pot too long, it gets charred.
    a. Putout: If the pot catches fire, extinguish it.
    b. Drop: Discard charred soup. Put out the fire in the pot if needed.

Assuming that you have been playing the game for a while.
You will be presented with the current order, and the items that are available on the map, your past actions and your past actions' status,.
Based on this information, you need to generate an action to take next.
The list of available actions is: [''' + avaialble_actions + '''].
In the list above, the action 'Prepare <name> Ingredients' means assembling the ingredients that have been chopped;
the action 'Serve <name> Soup' means serving a plated soup.
Make sure to plate a soup before serving it and cook a soup before plating it.
The action you choose from the list above should be the most useful based on the current game state.
'''

  p2 = 'Ok.'

  def prompt_helper(mov_hist: list) -> str:
    mh = [m['task'] for m in mov_hist]
    ms = [m['status'] for m in mov_hist]
    # moves = prep_mov_hist(mh)

    prompt = "Actions you've done recently and their corresponding statuses:\n"
    history_status_pairs = [f"{a}: {b}" for a, b in zip(mh, ms)]
    # print('History status pairs: ', history_status_pairs)
    prompt += (";\n".join(history_status_pairs[-7:]) +
               '\n') if len(history_status_pairs) > 0 else "None\n"
    # print('Prompt so far: ', prompt)
    prompt += 'If an action keeps failing, please generate a different action, as this means that this action might not be available.\n'

    return prompt

  order_prep = prep['order']
  env = prep['map']
  mov_hist = prep['mov_hist']

  order = prompt_order(order_prep) + '\n'
  reason = prompt_helper(mov_hist)
  reason += '\n'

  p3 = order
  p3 += env + '\n'
  p3 += '''
Please examine the current orders and items on the map carefully when you decide your action.
Please utilize chopped or combined ingredients as much as possible before producing new ones. 
Remember your goal is to complete as many orders as possible.
'''
  p3 += reason

  p3 += 'The human prefers you to choose from the following list of actions: '
  p3 += str(prep['prev_reasoning'])
  print('Human preference: ', str(prep['prev_reasoning']))
  p3 += '.\n'

  p3 += '''
Now please also respond to the following requests:
1. Output the top 3 actions that you would perform and give them a score from 1 to 10. 
   The scores should be proportional to how likely you will perform this action.
   In your response, you must generate actions that match the human's preference, even if it conflicts with your intention or reasoning.
2. Still output the top 3 actions that you would perform and give them a score from 1 to 10.
   But now, in your response to request 2, you don't need to consider the human's preference and can generate actions based on your reasoning.

For both requests 1 and 2, please format your output like a Python dictionary, mapping the name of the action, to a number between 1 and 10.
Separate the two dictionaries using a semi-colon.
Your final output should look like this: {'action 1': score_1, 'action 2': score_2, 'action 3': score_3}; {'action a': score_a, 'action b': score_b, 'action c': score_c}
Do not output anything else.
'''

  # print(p3)
  prompt = [[p1, p2], [p3]]
  return prompt


def prompt_reasoning_minigrid(prep):
  # print('In prompt_reasoning_minigrid')
  obj_desc, in_front, holding = prep['map']
  objects_seen = prep['objects_seen']
  avaialble_actions = ', '.join(list(NAMES_TO_ACTIONS.keys()))
  # print('Available actions: ', avaialble_actions)

  if prep['blocked']:
    blocked_msg = '''
If you have seen a yellow ball, then the yellow ball is blocking you from the door.
Because of that, you should prioritize moving the yellow ball away from the door by doing the following: 
release any object you are holding, go to the yellow ball, pick up the yellow ball (do not pick up the previous object you were holding), bring it to a drop location, and release the yellow ball.
After you move the yellow ball, you can go back to the door and use the key to unlock the door.
'''
  else:
    blocked_msg = ''

  if prep['pref'] != '':
    goal = 'a red ball'
  else:
    goal = 'either a green ball or a red ball'

  p1 = f'''You are an AI agent in a BabyAI environment called "Unlock Pickup".
You will find a description of the game and the current game state below.
You need to decide which actions to perform based on the current game state.

In this environment, your goal is to pick up {goal}.
To achieve this goal, you may need to unlock a door using a key and and enter the new room before you can pick up a ball.
Because of that, it is a good idea to go to and pick up a key if you see one in case you encoutner a door later on.
{blocked_msg}
Your available actions are as follows: [{avaialble_actions}].
'''

  p1 += '''
IMPORTANT:
- You should explore the environment if you are very unsure about what you should do.
- You can only go to an object (ball/key/door) that you have seen.
- When you perform a 'Go to' action, you will take one step towards the object of interest if there is one.
  So, you may need to repeatedly perform the 'Go to' action to reach the object.
- To pick up a ball or key or to unlock a door, this object (ball/key/door) has to be in front of you.
- If you would like to pick up an object when you are holding something, you have to drop what you are holding before you can pick up the object.
- To drop an object you are carrying, you need to release the object. But before you perform the "Release" action, you have to first perform "Go to drop location".
- If you previous action was Go to drop location, you should output Release as your next action.
- You should drop the key after you successfully unlock a door.
- You will find your previous action in addition to the current state to help you decide your next action.
- You will also find the environment information below. Please analyze this information carefully when deciding which actions to perform.
'''

  def move_history_helper(mov_hist: list) -> str:
    # if prep['blocked']:
    #   prompt = 'Here is a list of your previous actions from most recent to least recent: '
    #   mh = [m['task'] for m in mov_hist][-5:]
    #   prompt += ', '.join([ACTIONS_TO_NAMES[m] for m in reversed(mh)])
    #   prompt += '\n'
    # else:
    #   prompt = 'Your previous action was '
    #   mh = [m['task'] for m in mov_hist]
    #   if len(mh) > 0:
    #     prompt += (ACTIONS_TO_NAMES[mh[-1]] + '\n')
    #   else:
    #     prompt += 'none \n'
    prompt = 'Your previous action was '
    mh = [m['task'] for m in mov_hist]
    if len(mh) > 0:
      prompt += (ACTIONS_TO_NAMES[mh[-1]] + '\n')
    else:
      prompt += 'none \n'
    print('Prev action: ', prompt)
    return prompt


#   p1 += f'''
# Information to help you decide your action:

# {move_history_helper(prep['mov_hist'])}
# IMPORTANT: If your previous action was Release, do not perform Pickup.

# {process_minigrid_map(obj_desc, in_front, holding, objects_seen)}
# '''

  p1 += f'''
Information to help you decide your action:

{move_history_helper(prep['mov_hist'])}

{process_minigrid_map(obj_desc, in_front, holding, objects_seen)}
'''

  p2 = 'Ok.'

  if prep['gen_mode'] == 'all_yes_no' or prep[
      'gen_mode'] == 'all_yes_no_include_false':
    p3 = f'''
Now please also respond to the following request:
For each action in the list of available actions, output True if you think it is a productive action to be performed at this moment and False otherwise.
You can output True for multiple actions.
Remember your goal is to pick up {goal}.

Please format your output like a dictionary, mapping the name of the action, to True or False.
Your final output should look like this: "{{'action 1': True, 'action 2': False, 'action 3': True, ...}}"
Do not output anything else.
'''
  elif prep['gen_mode'] == '5_unranked':
    p3 = '''
Now please also respond to the following request:
Output the top 3 actions that you would most likely perform at this state. Do not output the same action twice.

Please format your output like a Python list.
Your final output should look like this: "['action 1', 'action 2', 'action 3']"
Do not output anything else.
'''
  else:
    print('Not implemented in prompt reasoning minigrid')
    raise NotImplementedError
  prompt = [[p1, p2], [p3]]
  # print('prompts:')
  # print(p1)
  # print(p3)

  return prompt


def prompt_reasoning_minigrid_bup(prep):
  # print('In prompt_reasoning_minigrid')
  map_info = prep['map']
  avaialble_actions = ', '.join(MACRO_ACTION_SPACE)
  # print('Available actions: ', avaialble_actions)

  if prep['pref'] != '':
    goal = 'a red box'
  else:
    goal = 'either a green box or a red box'

  p1 = f'''You are an AI agent in a BabyAI environment called "Blocked Unlock Pickup".
You will find a description of the game and the current game state below.
You need to decide which actions to perform based on the current game state.

In this environment, your goal is to pick up {goal}.

Your available actions are as follows: [{avaialble_actions}].
'''

  p1 += '''
IMPORTANT:
- You cannot unlock the door when the door is blocked.
- If the door is not blocked, you need to be carrying the key to unlock the door.
- You cannot carry two objects at the same time.
  In other words, if you would like to pick up "Object A" when you are carrying "Object B", 
  you need to first drop "Object B" before you can pick up "Object A".
- You will find your previous action in addition to the current state to help you decide your next action.
- You will also find the environment information below. Please analyze this information carefully when deciding which actions to perform.
- The environment information provides implicit feedback on whether your past action was successful.
  For instance, if your most recent action was to unlock a door, if the door is currently locked, that means your action was not successful.
  Or if your most recent action was to pick up an object, if you are not currently carrying the object, that means your action was not successful.
  If an action was not successful, you should perform some other actions before trying this action again.
'''

  # - Your previous actions might not be always succeed. For instance, you may have tried to unlock a door.
  #   But if door is still locked, that means this action is not successful.
  # - If you see many repeated actions in your action history, that means the action might not be currently feasible.
  #   In the case of an infeasible or unsuccessful action, you may need to perform some other actions before you try this action again.

  def move_history_helper(mov_hist: list) -> str:
    prompt = 'Here is a list of your previous actions from most recent to least recent: '
    mh = [m['task'] for m in mov_hist][-5:]
    prompt += ', '.join([m for m in reversed(mh)])
    prompt += '\n'
    # prompt = 'Your previous action was '
    # mh = [m['task'] for m in mov_hist]
    # if len(mh) > 0:
    #   prompt += (mh[-1] + '\n')
    # else:
    #   prompt += 'none \n'
    # print('Prev action: ', prompt)
    return prompt

  p1 += f'''
Information to help you decide your action:

{move_history_helper(prep['mov_hist'])}

Environment information: {map_info}
'''

  p2 = 'Ok.'

  if prep['gen_mode'] == 'all_yes_no' or prep[
      'gen_mode'] == 'all_yes_no_include_false':
    p3 = f'''
Now please also respond to the following request:
For each action in the list of available actions, output True if you think it is a productive and feasible action to be performed at this moment and False otherwise.
You can output True for multiple actions.
Remember your goal is to pick up {goal}.

Please format your output like a dictionary, mapping the name of the action, to True or False.
Your final output should look like this: "{{'action 1': True, 'action 2': False, 'action 3': True, ...}}"
Do not output anything else.
'''
  elif prep['gen_mode'] == '5_unranked':
    p3 = '''
Now please also respond to the following request:
Output the top 3 actions that you would most likely perform at this state. Do not output the same action twice.

Please format your output like a Python list.
Your final output should look like this: "['action 1', 'action 2', 'action 3']"
Do not output anything else.
'''
  else:
    print('Not implemented in prompt reasoning minigrid')
    raise NotImplementedError
  prompt = [[p1, p2], [p3]]
  # print('prompts:')
  # print(p1)
  # print(p3)

  return prompt


def process_minigrid_map(obj_desc, in_front, holding, objects_seen):
  obj_str = 'A list of objects that you see right now (you CANNOT go to an object or door if you have not seen it): \n'
  for obj, count in obj_desc.items():
    if count == 1:
      obj_str += f'- a {obj}\n'
    else:
      obj_str += f'- {count} {obj}s\n'

  seen_str = 'Objects you have seen previously (you CANNOT go to an object or door if you have not seen it): '
  seen_str += f'{list(objects_seen)}\n'

  if in_front != '':
    in_front_str = 'The object in front of you: '
    in_front_str += (in_front + '\n')
    # in_front_str += f'You can interact with the {in_front} but nothing else.\n'
  else:
    in_front_str = 'There is no object in front of you. Do not perform Pickup or Unlock.\n'

  if holding != '':
    holding_str = 'The object you are holding: '
    holding_str += (holding + '\n')
    holding_str += (
        'If you like to pick up another object you need to drop the object you are holding.\n'
    )
  else:
    holding_str = 'You are not holding any object.\n'

  # print(seen_str)

  return obj_str + in_front_str + holding_str
  # return obj_str + seen_str + in_front_str + holding_str


def prompt_reasoning_rw4t(prep, no_hl_rec=False):
  '''
  Generate a prompt for the rw4t domain asking the agent to select a high level
  action.
  '''
  if len(prep['sit_pref']) > 1:
    raise NotImplementedError
  elif len(prep['sit_pref']) == 1:
    rec_cs = prep['sit_pref'][0]
    if rec_cs in prep['sit_pref_actions'] and len(
        prep["sit_pref_actions"][rec_cs]) > 0 and not no_hl_rec:
      print('rec hl: ', prep["sit_pref_actions"][rec_cs][0])
      rec_hl = f'To {rec_cs}, the human recommends you to perform this low level action: {prep["sit_pref_actions"][rec_cs][0]}.'
    else:
      rec_hl = 'The human does not have a low level action recommendation.'
  else:
    rec_cs = None

  if prep['text_desc']:
    text_pref = prep['pref']
  else:
    text_pref = ''

  p1 = f'''Game Scenario:
You are an agent in a grid world environment.
Your goal is to pick up and drop off the objects on the map according to the human's preference.

Game Rules:
There are three different types of objects: circles, squares, and triangles.
There are two different types of dropoff locations: school and hospital.
Your task is to pick up an object and drop it off at a dropoff location.
Points are awarded for picking up specific objects and drop them off at specific locations.
You do not know in advance which objects/dropoff locations will yield points, but the human does, and they will provide suggestions to you.
Therefore, always follow the human's suggestions.

Environment:
{prep['env_desc']}

You have picked up: {prep['holding_desc']}.
If you have picked up an object, you need to drop it off by going to a dropoff location before you can pick up another object.

{text_pref}
'''

  if rec_cs is not None:
    p1 += '''There are two types of actions: composite actions and low level actions.
Composite actions are sequences of low level actions that achieve a particular goal.
The human will tell you what composite action to take.
You need to choose a low level action to satisfy the composite action based on the current state.
'''
    p1 += f'''The human commands you to do the following composite action: {rec_cs}.
{rec_hl}
'''

  p1 += f'''
All available low level actions include: {[a.name.replace('_', ' ') for a in rw4t_utils.RW4T_HL_Actions]}

{rw4t_hl_hist_helper(prep['hl_hist'])}
'''
  # print(p1)

  p2 = 'Ok.'

  if rec_cs is not None:
    p3 = '''
Now please respond to the following request:
Output a low level action that is the most suitable for the current state.
The low level action you choose must contribute to the goal of the composite action.
Most of the time, you should follow the human's recommended low level action because they are knowledgeable and have a good understanding of the environment.
The recommended low level action can appear counterproductive at first, but this could be due to it navigating around an obstacle or danger zone.
Remember that your overall objective is to pick up objects according to the human's preference.

Please format your output like a Python list.
Your final output should look like this: ['action 1']
Do not output anything other than the low level action.
'''
  else:
    p3 = '''
Now please respond to the following request:
Output a low level action that is the most suitable for the current state.
Please format your output like a Python list.
Your final output should look like this: ['action 1']
Do not output anything other than the low level action.
'''
  prompt = [[p1, p2], [p3]]
  return prompt


def prompt_composite_reasoning_rw4t(prep):
  '''
  Generate a prompt for the rw4t domain asking the agent to select a
  composite skill.
  '''

  def prev_comp_skill_helper():
    if len(prep['prev_sit_pref']) == 1:
      p = f'''Currently, your action is to '{prep["prev_sit_pref"][0]}'.
Reason about whether you completed '{prep["prev_sit_pref"][0]}'.
If the action is not complete and is still recommended by the human, you are discouraged to start a new action.
'''
      return p
    return ''

  if len(prep['sit_pref']) >= 1:
    if prep['top_k_llm'] == -1:
      top_k = len(prep['sit_pref'])
      action_choices = prep['sit_pref'][:prep['top_k_llm']]
    # if prep['top_k_llm'] == 2:
    #   action_choices = f'1. {prep["sit_pref"][0]} (more preferred), 2. {prep["sit_pref"][1]} (less preferred)'
    #   print('action choices: ', action_choices)
    else:
      action_choices = prep['sit_pref'][:prep['top_k_llm']]
  else:
    raise NotImplementedError
    # action_choices = rw4t_utils.composite_skills

  if prep['pref'] != '':
    text_pref = prep['pref']
  else:
    text_pref = ''

  p1 = f'''Game Scenario:
You are an agent in a grid world environment.
Your goal is to pick up and drop off the objects on the map according to the human's preference.

Game Rules:
There are three different types of objects: circles, squares, and triangles.
There are two different types of dropoff locations: school and hospital.
Your task is to pick up an object and drop it off at a dropoff location.
Points are awarded for picking up specific objects and drop them off at specific locations.
You do not know in advance which objects/dropoff locations will yield points, but the human does, and they will provide suggestions to you.
Therefore, always follow the human's suggestions.

Environment:
{prep['env_desc']}

You have picked up: {prep['holding_desc']}.
If you have picked up an object, you need to drop it off by going to a dropoff location before you can pick up another object.

{text_pref}

The human will recommend you a list of actions to take at this current state.
You should choose an action that is the most suitable to take.
Here is the list of actions that the human recommends:
{action_choices}

{prev_comp_skill_helper()}
'''
  # print('rw4t comp p1: ', p1)
  p2 = 'Ok.'

  p3 = '''
Now please output the most suitable action verbatim from the list of human's suggestions based on the current state.
Do not output anything else. Do not output square brackets.
'''

  prompt = [[p1, p2], [p3]]
  return prompt


def prompt_goal_reasoning_rw4t(prep):
  '''
  Generate a prompt for the rw4t domain asking the agent to select a goal
  location.
  '''
  p1 = f'''Game Scenario:
You are an agent in a grid world environment.
You need to go to the objects on the map and pick them up.
Your goal is to pick up all the objects as fast as possible.

Here is the map of the environment:
{prep['map']}.
Grids with value 0 are walkable.
Grids with value 1 are obstacles that you cannot go through.
Grids with value 2 contain the object you need to pick up.
You need to be at the grid where the object is to pick it up.

To summarize, here are the objects and their locations in the environment:
{rw4t_all_obj_helper(prep['map'], coord=False)}
You can go to and pick up the objects in any order.

Your current location is: row {prep['pos'][1]}, column {prep['pos'][0]}
Both the row and column indicees start from 0.
'''
  # print(p1)
  p2 = 'Ok.'

  p3 = '''
Now please respond to the following request:
Based on your current location, select one object to pick up and output its row and column index.
In general, you should prioritize selecting locations that you are closest to.
Your output should look like this: <row>, <column>
Do not output anything else.
'''
  prompt = [[p1, p2], [p3]]
  return prompt


def rw4t_map_helper(obs_map, pos):
  '''
  A helper function that converts the 2D array observation (and the agent's
  location) to a combined grid representation.
  '''
  map_str = ''
  row_idx = 0
  for row in obs_map:
    num_white_spaces = 6
    white_spaces = num_white_spaces * ' '
    # Get the string representation of the row
    temp_str = white_spaces.join(map(str, row))
    if pos[1] == row_idx:
      # Add robot location
      temp_str = temp_str[:6 * pos[0] + 1] + '(r)' + temp_str[6 * pos[0] + 4:]
    map_str += (temp_str + '\n')
    row_idx += 1
  return map_str


def rw4t_danger_obj_helper(obs_map, coord=False):
  '''
  a helper function that generates a string description of danger zones and 
  their locations. The locations can either be x, y coordinates or row, column
  indices.
  '''
  loc_str = ''
  num_objs = 0
  for row in range(len(obs_map)):
    for col in range(len(obs_map[row])):
      if obs_map[row][col] == rw4t_utils.RW4T_State.danger.value:
        num_objs += 1
        if coord:
          temp_str = f'Danger zone {num_objs} at ({col}, {row})\n'
        else:
          temp_str = f'Danger zone {num_objs} at row {row}, column {col}\n'
        loc_str += temp_str
  return loc_str


def rw4t_all_obj_helper(obs_map, coord=False):
  '''
  a helper function that generates a string description of objects and their
  locations. The locations can either be x, y coordinates or row, column
  indices.
  '''
  loc_str = ''
  num_objs = 0
  for row in range(len(obs_map)):
    for col in range(len(obs_map[row])):
      if obs_map[row][col] == rw4t_utils.RW4T_State.kit.value:
        num_objs += 1
        if coord:
          temp_str = f'Object {num_objs} at ({col}, {row})\n'
        else:
          temp_str = f'Object {num_objs} at row {row}, column {col}\n'
        loc_str += temp_str
  return loc_str


def rw4t_close_obj_helper(obs_map, pos, top_k=2):
  loc_2_dist = {}
  for row in range(len(obs_map)):
    for col in range(len(obs_map[row])):
      if obs_map[row][col] == rw4t_utils.RW4T_State.kit.value:
        loc_2_dist[(col, row)] = dist_heuristic((col, row), pos)
  sorted_dict = dict(sorted(loc_2_dist.items(), key=lambda item: item[1]))
  ret_str = ''
  num_objs = min(top_k, len(sorted_dict))
  dict_keys = list(sorted_dict.keys())
  for i in range(num_objs):
    ret_str += f'Object at row {dict_keys[i][1]}, column {dict_keys[i][0]}\n'
  return ret_str


def rw4t_goal_helper(goal):
  '''
  A helper function that generates a string description of the agent's current
  goal location if there is one.
  '''
  if goal is None or goal == '' or len(goal) != 2:
    return ''

  print(f'Goal: row {goal[1]}, column {goal[0]}')
  return (
      f'You should go to the object at row {goal[1]}, column {goal[0]} and pick it up. \n'
  )


def rw4t_danger_obj_helper_partial(obs_map, pos, coord=False, error_rate=0):
  loc_str = ''
  num_objs = 0
  for row in range(len(obs_map)):
    for col in range(len(obs_map[row])):
      if obs_map[row][col] == rw4t_utils.RW4T_State.danger.value:
        num_objs += 1
        if dist_heuristic(
            (col, row), pos) <= 1 and random.random() >= error_rate:
          if coord:
            temp_str = f'Danger zone at ({col}, {row})\n'
          else:
            temp_str = f'Danger zone near you at row {row}, column {col}.\n'
          loc_str += temp_str
  return loc_str


def rw4t_danger_obj_helper_partial_far(obs_map, pos, coord=False, error_rate=0):
  loc_str = ''
  num_objs = 0
  for row in range(len(obs_map)):
    for col in range(len(obs_map[row])):
      if obs_map[row][col] == rw4t_utils.RW4T_State.danger.value:
        num_objs += 1
        if dist_heuristic(
            (col, row), pos) > 1 and random.random() >= error_rate:
          if coord:
            temp_str = f'Danger zone at ({col}, {row})\n'
          else:
            temp_str = f'Danger zone at row {row}, column {col}.\n'
          loc_str += temp_str
  return loc_str


def rw4t_hl_hist_helper(prev_hl_action):
  ret_str = f'''You have been doing: {prev_hl_action}.
You can continue this action or switch to a new action.
'''
  # print('ret str: ', ret_str)
  return ret_str


def dist_heuristic(a, b):
  # Manhattan distance heuristic
  return abs(a[0] - b[0]) + abs(a[1] - b[1])


def rwt4_obstacle_helper(obs_map):
  '''
  a helper function that generates a string description of obstacles and their
  locations. The locations can either be x, y coordinates or row, column
  indices.
  '''
  if np.sum(obs_map == rw4t_utils.RW4T_State.obstacle.value) == 0:
    return 'No obstacles found.\n'

  loc_str = ''
  for row in range(len(obs_map)):
    for col in range(len(obs_map[row])):
      if obs_map[row][col] == rw4t_utils.RW4T_State.obstacle.value:
        temp_str = f'Obstacle at row {row}, column {col}\n'
        loc_str += temp_str
  return loc_str


def rwt4_obstacle_helper_partial(obs_map, pos, error_rate=0):
  loc_str = ''
  for row in range(len(obs_map)):
    for col in range(len(obs_map[row])):
      if obs_map[row][col] == rw4t_utils.RW4T_State.danger.value:
        if dist_heuristic(
            (col, row), pos) <= 3 and random.random() >= error_rate:
          temp_str = f'Obstacle near you at row {row}, column {col}.\n'
          loc_str += temp_str
  return loc_str


def oc_pref_helper(pref_str: str) -> str:
  p_2_order_name = {
      'A': 'Alice Soup',
      'B': 'Bob Soup',
      'C': 'Cathy Soup',
      'D': 'David Soup'
  }

  p_list = pref_str.split('_')
  processed_p = '''
The human prefers you to work on the following soups:
'''
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

  # processed_p += 'End of list.\n'
  return processed_p


def oc_get_pot_info(prep):
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


def oc_get_available_objs(prep):
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


# Action description to ProgPrompt-like actions
sit_prefs_to_actions = {
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


def oc_sit_pref_to_actions(prep, top_k=5):
  '''
  Helper function for translating actions descriptions in Overcooked to
  ProgPrompt-like actions.
  '''
  actions = []
  sit_prefs = prep['sit_pref'][:top_k]
  for sit_pref in sit_prefs:
    if sit_pref in sit_prefs_to_actions:
      actions.append(sit_prefs_to_actions[sit_pref])
  return "\n".join(actions)


def oc_move_history_helper(prep, add_rec=True) -> str:
  '''
  Helper function for describing the agent's action history in the Overcooked
  domain.
  '''
  mov_hist = prep['mov_hist']
  mh = [m['task'] for m in mov_hist][::-1][:1]
  ms = [m['status'] for m in mov_hist][::-1][:1]

  prompt = ""
  history_status_pairs = [
      f"{sit_prefs_to_actions[a]} ({b})" for a, b in zip(mh, ms)
  ]
  prompt += (";\n".join(history_status_pairs[-7:]) +
             '\n') if len(history_status_pairs) > 0 else "None\n"
  if add_rec:
    prompt += 'If an action fails, please output a different action, as this means that this action might not be available.\n'

  return prompt
