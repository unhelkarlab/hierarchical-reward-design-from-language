import gymnasium as gym

from gym_cooking.envs.overcooked_simple import MapSetting, OvercookedSimpleHL


class OvercookedSimpleSemi(gym.Env):
  """
  A high-level environment where each step:
    1. Takes a high-level (HL) action 'option'.
    2. Runs the low-level policy until it terminates.
    3. Returns the final HL observation, the sum of rewards, and done/truncated.
  """

  def __init__(self,
               arglist,
               hl_pref_r,
               ez,
               salad,
               serve,
               convenience_features,
               base_env=OvercookedSimpleHL,
               detailed_hl_pref=False,
               masked=False,
               render=False,
               oc_game_params=dict()):
    assert not masked, 'There is no need to have an option mask for SMDP'
    super().__init__()

    # 1) Create the base (full) environment. This is the same environment that
    #    the LL agent interacts with (but with low_level=False).
    self.base_env = base_env(arglist,
                             hl_pref_r,
                             ez,
                             masked,
                             salad,
                             serve,
                             convenience_features=convenience_features,
                             detailed_hl_pref=detailed_hl_pref,
                             smdp=True,
                             render=render,
                             oc_game_params=oc_game_params)
    # 2) Define HL action space
    self.action_space = self.base_env.action_space
    # 3) Define HL observation space.
    self.observation_space = self.base_env.observation_space

  def reset(self, seed=None, options=None):
    """
    Reset the base environment, then build and return the HL observation.
    """
    if seed is not None:
      obs, info = self.base_env.reset(seed)
    else:
      obs, info = self.base_env.reset()
    info['num_steps'] = 0
    return obs, info

  def step(self, hl_action):
    '''
    Run the LL policy until it terminates (or until the episode terminates).
    '''
    num_steps = 0
    total_reward = (0, 0)
    done = False
    truncated = False
    ll_done = False
    ll_truncated = False
    while not done and not truncated and not ll_done and not ll_truncated:
      hl_next_obs, reward, done, truncated, info = self.base_env.step(hl_action)
      total_reward = tuple(
          [acc + sub_r for acc, sub_r in zip(total_reward, reward)])
      ll_done = info['ll_done']
      ll_truncated = info['ll_truncated']
      num_steps += 1
    info['num_steps'] = num_steps
    return hl_next_obs, total_reward, done, truncated, info

  def close(self):
    self.base_env.close()


def test_MDP():
  # Initialize the environment
  arglist = MapSetting(**dict(level="new2", ))
  env = OvercookedSimpleSemi(arglist=arglist,
                             ez=True,
                             salad=True,
                             serve=False,
                             masked=False,
                             convenience_features=False,
                             detailed_hl_pref=False,
                             render=True)

  num_episodes = 2
  for i in range(num_episodes):
    total_reward = 0
    done = False
    truncated = False
    obs, info = env.reset()
    while not done and not truncated:
      # Ask user for an option
      user_input = input("Enter option: ").strip()
      # Convert to int (and handle errors / invalid inputs gracefully)
      try:
        action = int(user_input)
      except ValueError:
        print("Invalid input.")
        continue
      # Take a step in the environment
      obs, reward, done, truncated, info = env.step(action)
      total_reward += (reward[0] + reward[1])
      print(f"Reward: {reward}, Done: {done}, Truncated: {truncated}")
    print('total reward: ', total_reward)
  env.close()


if __name__ == "__main__":
  test_MDP()
