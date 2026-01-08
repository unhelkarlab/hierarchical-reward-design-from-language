import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO

from HierRL.envs.ai2thor.pnp_env import ThorPickPlaceEnv
from HierRL.envs.ai2thor.pnp_ll_planner import ThorPickPlacePlanner


class SemiThorPickPlaceEnv(gym.Env):
  """
  A high-level environment where each step:
    1. Takes a high-level (HL) action 'option'.
    2. Runs the low-level policy until it terminates.
    3. Returns the final HL observation, the sum of rewards, and done/truncated.
  """

  def __init__(self,
               hl_pref_r,
               worker_model_path=None,
               use_ll_planner=True,
               env=ThorPickPlaceEnv,
               render: bool = False,
               one_network: bool = False,
               max_ll_steps: int = 60,
               pnp_game_params: dict = {},
               seed=0):
    super().__init__()

    # 1) Create the base (full) environment. This is the same environment that
    #    the LL agent interacts with (but with low_level=False).
    self.base_env = env(hl_pref_r=hl_pref_r,
                        low_level=False,
                        seed=seed,
                        render=render,
                        pnp_game_params=pnp_game_params)

    # 2) Load the trained low-level policy
    self.one_network = one_network
    if worker_model_path is None:
      assert use_ll_planner
      self.worker_policy = ThorPickPlacePlanner(self.base_env, stop_dist=0.30)
    else:
      # Load a low-level policy per option
      assert not self.one_network
      if self.one_network:
        self.worker_policy = PPO.load(worker_model_path)
      else:
        # worker_mode_path should be a list of paths, one per option
        self.worker_policy = [
            PPO.load(p) if p is not None else None for p in worker_model_path
        ]

    # 3) Define HL action space
    self.action_space = self.base_env.action_space
    assert self.action_space == self.base_env.hl_action_space

    # 4) Define HL observation space
    self.observation_space = self._build_obs_space()

    # 5) Other inits
    self.max_ll_steps = max_ll_steps

  def reset(self, seed=None, options=None):
    """
    Reset the base environment, then build and return the HL observation.
    """
    obs_dict, info = self.base_env.reset()
    hl_obs = self._build_hl_observation(obs_dict)
    self.done = False
    info['num_steps'] = 0
    return hl_obs, info

  def _build_obs_space(self) -> spaces.Box:
    # Get lows and highs of the original Dict obs space
    keys = [
        "apple_1_pos", "apple_2_pos", "egg_1_pos", "egg_2_pos", "stool_pos",
        "sink_pos", "agent_pos", "agent_rot", "apple_1_state", "apple_2_state",
        "egg_1_state", "egg_2_state"
    ]
    lows = []
    highs = []
    for k in keys:
      space = self.base_env.observation_space.spaces[k]
      if isinstance(space, spaces.Box):
        lows.append(space.low.reshape(-1))
        highs.append(space.high.reshape(-1))
      elif isinstance(space, spaces.Discrete):
        lows.append(np.zeros(space.n, dtype=np.float32))
        highs.append(np.ones(space.n, dtype=np.float32))
      else:
        raise NotImplementedError

    # Extra one-hot for high-level actions
    extra_len = len(self.base_env.pnp_hl_actions_with_dummy)
    extra_space = spaces.Box(0.0, 1.0, (extra_len, ), dtype=np.float32)

    lows.append(extra_space.low.reshape(-1))
    highs.append(extra_space.high.reshape(-1))

    low = np.concatenate(lows).astype(np.float32)
    high = np.concatenate(highs).astype(np.float32)

    return spaces.Box(low=low, high=high, dtype=np.float32)

  def _build_hl_observation(self, obs_dict):
    # Extract observations from dict
    apple_1_pos = obs_dict["apple_1_pos"]
    apple_2_pos = obs_dict["apple_2_pos"]
    egg_1_pos = obs_dict["egg_1_pos"]
    egg_2_pos = obs_dict["egg_2_pos"]
    stool_pos = obs_dict["stool_pos"]
    sink_pos = obs_dict["sink_pos"]
    agent_pos = obs_dict["agent_pos"]
    agent_rot = obs_dict["agent_rot"]

    # One-hot encode object states
    apple_1_state = obs_dict["apple_1_state"]
    apple_2_state = obs_dict["apple_2_state"]
    egg_1_state = obs_dict["egg_1_state"]
    egg_2_state = obs_dict["egg_2_state"]
    apple_1_state_oh = np.zeros(3, dtype=np.float32)
    apple_1_state_oh[apple_1_state] = 1.0
    apple_2_state_oh = np.zeros(3, dtype=np.float32)
    apple_2_state_oh[apple_2_state] = 1.0
    egg_1_state_oh = np.zeros(3, dtype=np.float32)
    egg_1_state_oh[egg_1_state] = 1.0
    egg_2_state_oh = np.zeros(3, dtype=np.float32)
    egg_2_state_oh[egg_2_state] = 1.0

    # One-hot encode option
    prev_option = np.zeros(len(self.base_env.pnp_hl_actions_with_dummy),
                           dtype=np.float32)
    prev_option[self.base_env.prev_option] = 1.0

    # Concatenate all parts
    return np.concatenate([
        apple_1_pos, apple_2_pos, egg_1_pos, egg_2_pos, stool_pos, sink_pos,
        agent_pos, agent_rot, apple_1_state_oh, apple_2_state_oh,
        egg_1_state_oh, egg_2_state_oh, prev_option
    ],
                          axis=0)

  def _build_ll_observation(self, obs_dict, option):
    # Extract observations from dict
    apple_1_pos = obs_dict["apple_1_pos"]
    apple_2_pos = obs_dict["apple_2_pos"]
    egg_1_pos = obs_dict["egg_1_pos"]
    egg_2_pos = obs_dict["egg_2_pos"]
    stool_pos = obs_dict["stool_pos"]
    sink_pos = obs_dict["sink_pos"]
    agent_pos = obs_dict["agent_pos"]
    agent_rot = obs_dict["agent_rot"]

    # One-hot encode object states
    apple_1_state = obs_dict["apple_1_state"]
    apple_2_state = obs_dict["apple_2_state"]
    egg_1_state = obs_dict["egg_1_state"]
    egg_2_state = obs_dict["egg_2_state"]
    apple_1_state_oh = np.zeros(3, dtype=np.float32)
    apple_1_state_oh[apple_1_state] = 1.0
    apple_2_state_oh = np.zeros(3, dtype=np.float32)
    apple_2_state_oh[apple_2_state] = 1.0
    egg_1_state_oh = np.zeros(3, dtype=np.float32)
    egg_1_state_oh[egg_1_state] = 1.0
    egg_2_state_oh = np.zeros(3, dtype=np.float32)
    egg_2_state_oh[egg_2_state] = 1.0

    # One-hot encode option
    cur_option = np.zeros(len(self.base_env.pnp_hl_actions), dtype=np.float32)
    cur_option[option] = 1.0

    # Concatenate all parts
    if self.one_network:
      return np.concatenate([
          apple_1_pos, apple_2_pos, egg_1_pos, egg_2_pos, stool_pos, sink_pos,
          agent_pos, agent_rot, apple_1_state_oh, apple_2_state_oh,
          egg_1_state_oh, egg_2_state_oh, cur_option
      ],
                            axis=0)
    else:
      return np.concatenate([
          apple_1_pos, apple_2_pos, egg_1_pos, egg_2_pos, stool_pos, sink_pos,
          agent_pos, agent_rot, apple_1_state_oh, apple_2_state_oh,
          egg_1_state_oh, egg_2_state_oh
      ],
                            axis=0)

  def step(self, hl_action):
    """
    1) We pick the HL action (option).
    2) Run the entire LL sub-policy to completion.
    3) Accumulate reward for the HL agent.
    4) Return (HL_observation, HL_reward, done, truncated, info).
    """

    # We run the low-level policy until it terminates or we get truncated
    total_reward = (0, 0, 0, 0)
    done = False
    truncated = False
    ll_done = False
    ll_truncated = False
    num_ll_steps = 0

    # Fallbacks in case we take zero LL steps this call
    next_obs_dict = self.base_env._get_obs()
    info = {}

    # Start planner option if using planner
    if isinstance(self.worker_policy, ThorPickPlacePlanner):
      self.worker_policy.start_option(hl_action)

    while not ll_done and not ll_truncated and not done and not truncated:
      # 1. Convert the base_env's current state to the LL observation
      ll_obs = self._build_ll_observation(self.base_env._get_obs(), hl_action)

      # 2. Predict low-level action
      if isinstance(self.worker_policy, ThorPickPlacePlanner):
        aidx, pi_info = self.worker_policy.predict(ll_obs, deterministic=True)
        info.update(pi_info or {})
        if aidx is None:
          # Option finished; stop without stepping the env
          ll_done = True
          break
        ll_action = aidx
      else:
        # Learned worker path
        if self.one_network:
          ll_action, _ = self.worker_policy.predict(ll_obs, deterministic=False)
        else:
          ll_action, _ = self.worker_policy[hl_action].predict(
              ll_obs, deterministic=False)

      # 3. Step the base environment
      # print('ll action: ', ll_action)
      # print('hl action: ', hl_action)
      next_obs_dict, reward, done, truncated, step_info = self.base_env.step(
          ll_action, hl_action)
      # print('Step info: ', step_info)
      num_ll_steps += 1
      info.update(step_info or {})

      # 4. Accumulate reward from each LL step
      total_reward = tuple(
          [acc + sub_r for acc, sub_r in zip(total_reward, reward)])

      # 5. Check termination of LL option
      if isinstance(self.worker_policy, ThorPickPlacePlanner):
        ll_done = self.worker_policy.is_done()
        ll_truncated = False
      else:
        ll_done = info['ll_done']
        ll_truncated = num_ll_steps >= self.max_ll_steps

    # Now the option has finished. The environment is in a new state =>
    # build HL observation
    hl_next_obs = self._build_hl_observation(next_obs_dict)
    return hl_next_obs, total_reward, done, truncated, {
        'num_steps': num_ll_steps,
        'c_task_reward': info.get('c_task_reward', 0.0),
        'c_pseudo_reward': info.get('c_pseudo_reward', 0.0),
        'c_gt_hl_pref': info.get('c_gt_hl_pref', 0.0),
        'c_gt_ll_pref': info.get('c_gt_ll_pref', 0.0)
    }

  def close(self):
    self.base_env.close()
