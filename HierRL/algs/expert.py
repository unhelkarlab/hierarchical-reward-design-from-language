import numpy as np
from copy import deepcopy

from gym_cooking.envs.overcooked_simple import OvercookedSimpleHL, MapSetting
from gym_cooking.envs.overcooked_simple_semi import OvercookedSimpleSemi
from gym_cooking.utils.core import Ingredients, SoupType


class OC_Expert:

  def __init__(self, env, seed=0):
    self.np_rng = np.random.default_rng(seed)
    if hasattr(env, 'venv'):
      # If the environment is wrapped with VecNormalize
      print('Env is VecNormalized')
      env = env.venv
    if hasattr(env, 'envs'):
      # If the environment is vectorized
      print('Env is vectorized')
      env = env.envs[0]
    env = env.unwrapped
    if hasattr(env, 'base_env'):
      # If the environment is a Semi-MDP
      print('Env is a Semi-MDP')
      env = env.base_env
    elif hasattr(env, 'env'):
      print('Env is an MDP')
      env = env.env
    self.prev_action = -1
    self.prev_ingre_chopped = None
    self.prev_ingre_combined = None
    self.prev_soup_plated = None
    self.all_moves_dict = env.all_moves_dict
    self.dish_type = env.dish_type
    self.convenience_features = env.convenience_features
    self.env = env

    self.map_size = 24 * 4 * 5
    self.current_holdings_len = 13

    if self.convenience_features:
      self.ingre_chopped_start = self.map_size + self.current_holdings_len + len(
          Ingredients)
      self.ingre_chopped_end = self.ingre_chopped_start + len(Ingredients) - 1
      self.ingre_combined_start = self.ingre_chopped_end + len(SoupType)
      self.ingre_combined_end = self.ingre_combined_start + len(SoupType) - 1
      self.soup_plated_start = self.ingre_combined_end + len(SoupType)
      self.soup_plated_end = self.soup_plated_start + len(SoupType) - 1
    else:
      self.chopped_ingredients = np.zeros(len(Ingredients) - 1)

  def predict(self, obs, with_pref=False, deterministic=True):
    if self.convenience_features:
      return self.predict_with_conv_features(obs)
    else:
      return self.predict_without_conv_features(obs, with_pref)

  def predict_with_conv_features(self, obs, deterministic=True):
    ingre_chopped = deepcopy(
        obs[self.ingre_chopped_start:self.ingre_chopped_end])
    ingre_combined = deepcopy(
        obs[self.ingre_combined_start:self.ingre_combined_end])
    soup_plated = deepcopy(obs[self.soup_plated_start:self.soup_plated_end])
    # Only change the action if we have finished the current action
    if (self.prev_action == -1
        or not np.array_equal(self.prev_ingre_chopped, ingre_chopped)
        or not np.array_equal(self.prev_ingre_combined, ingre_combined)
        or not np.array_equal(self.prev_soup_plated, soup_plated)):
      # First check if we need to chop any veggies
      zero_indices_chopped = np.where(ingre_chopped == 0)[0]
      if self.dish_type == 'Alice':
        if Ingredients.tomato.value - 1 in zero_indices_chopped:
          zero_indices_chopped = np.delete(
              zero_indices_chopped,
              np.where(zero_indices_chopped == Ingredients.tomato.value - 1))
      elif self.dish_type == 'Bob':
        if Ingredients.onion.value - 1 in zero_indices_chopped:
          zero_indices_chopped = np.delete(
              zero_indices_chopped,
              np.where(zero_indices_chopped == Ingredients.onion.value - 1))
      elif self.dish_type == 'Cathy':
        if Ingredients.lettuce.value - 1 in zero_indices_chopped:
          zero_indices_chopped = np.delete(
              zero_indices_chopped,
              np.where(zero_indices_chopped == Ingredients.lettuce.value - 1))
      if len(zero_indices_chopped) > 0:
        # Randomly sample one index from the indices with value 0
        random_index = self.np_rng.choice(zero_indices_chopped)
        if random_index == 0:
          action_name = 'Chop Tomato'
        elif random_index == 1:
          action_name = 'Chop Onion'
        else:
          action_name = 'Chop Lettuce'
      # Check if we need to combine the ingredients
      elif self.dish_type == 'Alice' and ingre_combined[SoupType.alice.value -
                                                        1] == 0:
        action_name = 'Prepare Alice Ingredients'
      elif self.dish_type == 'Bob' and ingre_combined[SoupType.bob.value -
                                                      1] == 0:
        action_name = 'Prepare Bob Ingredients'
      elif self.dish_type == 'Cathy' and ingre_combined[SoupType.cathy.value -
                                                        1] == 0:
        action_name = 'Prepare Cathy Ingredients'
      elif self.dish_type == 'David' and ingre_combined[SoupType.david.value -
                                                        1] == 0:
        action_name = 'Prepare David Ingredients'
      # Check if we need to plate the salad
      elif self.dish_type == 'Alice' and soup_plated[SoupType.alice.value -
                                                     1] == 0:
        action_name = 'Plate Alice Salad'
      elif self.dish_type == 'Bob' and soup_plated[SoupType.bob.value - 1] == 0:
        action_name = 'Plate Bob Salad'
      elif self.dish_type == 'Cathy' and soup_plated[SoupType.cathy.value -
                                                     1] == 0:
        action_name = 'Plate Cathy Salad'
      elif self.dish_type == 'David' and soup_plated[SoupType.david.value -
                                                     1] == 0:
        action_name = 'Plate David Salad'
      # Check if we need to serve the salad
      elif self.dish_type == 'Alice':
        action_name = 'Serve Alice Salad'
      elif self.dish_type == 'Bob':
        action_name = 'Serve Bob Salad'
      elif self.dish_type == 'Cathy':
        action_name = 'Serve Cathy Salad'
      elif self.dish_type == 'David':
        action_name = 'Serve David Salad'
      action = self.all_moves_dict[action_name]
    else:
      action = self.prev_action

    self.prev_action = action
    self.prev_ingre_chopped = ingre_chopped
    self.prev_ingre_combined = ingre_combined
    self.prev_soup_plated = soup_plated
    return action

  def predict_without_conv_features(self, obs, with_pref, deterministic=True):
    if self.env.prev_option == self.env.all_moves_dict_with_wait['Wait']:
      self.chopped_ingredients = np.zeros(len(Ingredients) - 1)

    assert self.dish_type == 'David'
    if self.env.masked:
      if np.any(self.env.option_mask):
        return np.argmax(self.env.option_mask)

    action_name = ''
    zero_indices_chopped = np.where(self.chopped_ingredients == 0)[0]
    if self.dish_type == 'Alice':
      if Ingredients.tomato.value - 1 in zero_indices_chopped:
        zero_indices_chopped = np.delete(
            zero_indices_chopped,
            np.where(zero_indices_chopped == Ingredients.tomato.value - 1))
    elif self.dish_type == 'Bob':
      if Ingredients.onion.value - 1 in zero_indices_chopped:
        zero_indices_chopped = np.delete(
            zero_indices_chopped,
            np.where(zero_indices_chopped == Ingredients.onion.value - 1))
    elif self.dish_type == 'Cathy':
      if Ingredients.lettuce.value - 1 in zero_indices_chopped:
        zero_indices_chopped = np.delete(
            zero_indices_chopped,
            np.where(zero_indices_chopped == Ingredients.lettuce.value - 1))
    if len(zero_indices_chopped) > 0:
      # Randomly sample one index from the indices with value 0
      if not with_pref:
        random_index = self.np_rng.choice(zero_indices_chopped)
      else:
        if Ingredients.tomato.value - 1 in zero_indices_chopped:
          random_index = 0
        elif Ingredients.onion.value - 1 in zero_indices_chopped:
          random_index = 1
        elif Ingredients.lettuce.value - 1 in zero_indices_chopped:
          random_index = 2
      self.chopped_ingredients[random_index] += 1
      if random_index == 0:
        action_name = 'Chop Tomato'
      elif random_index == 1:
        action_name = 'Chop Onion'
      else:
        action_name = 'Chop Lettuce'
    if action_name == '':
      prev_option_name = next(
          (k for k, v in self.env.all_moves_dict_with_wait.items()
           if v == self.env.prev_option), None)
      if prev_option_name is None:
        raise NotImplementedError
      if 'Chop' in prev_option_name:
        action_name = 'Prepare David Ingredients'
      elif prev_option_name == 'Prepare David Ingredients':
        action_name = 'Plate David Salad'
      else:
        raise NotImplementedError

    action = self.all_moves_dict[action_name]
    return action


def test_expert_demos(gym_env, with_pref=True, **kwargs):
  if gym_env == OvercookedSimpleSemi:
    assert not kwargs['masked']
  # Initialize the environment
  env = gym_env(arglist=kwargs['arglist'],
                ez=kwargs['ez'],
                masked=kwargs['masked'],
                salad=kwargs['salad'],
                serve=kwargs['serve'],
                detailed_hl_pref=kwargs['detailed_hl_pref'],
                convenience_features=kwargs['convenience_features'],
                render=kwargs['render'])

  all_rewards_sum = None
  num_episodes = 100
  expert = OC_Expert(env)
  for i in range(num_episodes):
    done = False
    truncated = False
    obs, info = env.reset()
    while not done and not truncated:
      action = expert.predict(obs, with_pref=with_pref)
      # Take a step in the environment
      obs, reward, done, truncated, info = env.step(action)
      # print(f"Reward: {reward}, Done: {done}, Truncated: {truncated}")
      # Accumulate rewards
      if all_rewards_sum is None:
        all_rewards_sum = tuple(0 for _ in reward)
      all_rewards_sum = tuple([
          acc_sub_r + sub_r for acc_sub_r, sub_r in zip(all_rewards_sum, reward)
      ])
  env.close()

  all_rewards_avg = tuple(
      [rewards_sum / num_episodes for rewards_sum in all_rewards_sum])
  print("Avg cumulative rewards: ", all_rewards_avg)
  print('Avg task reward: ', all_rewards_avg[0])
  print('Avg high-level pref: ', all_rewards_avg[1])
  print('Avg high-level reward (task + high-level pref): ',
        all_rewards_avg[0] + all_rewards_avg[1])


if __name__ == "__main__":
  map_set = MapSetting(**dict(level="new2", ))
  env_kwargs = dict(arglist=map_set,
                    ez=True,
                    masked=True,
                    salad=True,
                    serve=False,
                    detailed_hl_pref=False,
                    convenience_features=False,
                    render=False)
  test_expert_demos(OvercookedSimpleHL, **env_kwargs)
  # test_expert_demos(OvercookedSimpleSemi, **env_kwargs)
