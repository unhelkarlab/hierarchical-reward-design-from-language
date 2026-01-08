from HierRL.envs.ai2thor.pnp_env import ThorPickPlaceEnv
from HierRL.envs.ai2thor.pnp_semi import SemiThorPickPlaceEnv
from HierRL.envs.ai2thor.pnp_ll_planner import ThorPickPlacePlanner
from HierRL.envs.ai2thor.pnp_training_utils import PnP_HL_Actions


def make_semi_env(
    *,
    use_ll_planner: bool = True,
    use_pref: bool = True,
    worker_model_path: str = None,
    base_env_cls=ThorPickPlaceEnv,
    render: bool = True,
    seed: int = 0,
):
  """
  Construct the Semi-MDP environment. By default this uses the planner
  (no trained LL model required). If you pass worker_model_path, the semi
  env will load PPO and use that instead.
  """
  # You can pass any hl_pref_r you use in your setup; here a neutral lambda.
  hl_pref_r = lambda rewards_tuple: rewards_tuple[0]  # use task reward only
  env = SemiThorPickPlaceEnv(
      hl_pref_r=hl_pref_r,
      worker_model_path=worker_model_path,
      use_ll_planner=use_ll_planner if worker_model_path is None else False,
      env=base_env_cls,
      render=render,
      seed=seed,
  )
  if use_ll_planner and use_pref:
    assert isinstance(env.worker_policy, ThorPickPlacePlanner)
    env.worker_policy.use_pref = True
  return env


def run_semi_keyboard(env: SemiThorPickPlaceEnv, *, auto_reset: bool = True):
  """
  Interactive loop to run HL options via keyboard *through the Semi-MDP*.

  Controls:
    - 0/1/2/3 : execute option index (one semi step = run LL to completion)
    - '0312'  : run a sequence of HL options in order
    - r       : reset
    - h       : help
    - q       : quit
  """
  # Sanity checks
  try:
    n_opts = env.action_space.n
  except Exception as e:
    raise AssertionError(
        "Semi env must expose a discrete HL action_space.n") from e

  # Option names (best-effort)
  opt_names = list(PnP_HL_Actions.__members__.keys())

  def print_help():
    print("\n=== Controls ===")
    print(f"  0..{n_opts-1} : execute option by index")
    print(
        f"  e.g., '0312' executes {', '.join(opt_names[i] for i in [0,3,1,2])}")
    print("  r : reset   |  h : help   |  q : quit")
    print("=== Options ===")
    for i, name in enumerate(opt_names):
      print(f"  {i}: {name}")
    print("==============\n")

  # Reset once
  obs, info = env.reset()
  print_help()

  total_num_steps = 0

  while True:
    try:
      line = input(
          f"Enter option(s) [0..{n_opts-1}] or (r/h/q): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
      print("\nExiting.")
      break

    if not line:
      continue
    if line in ("q", "quit", "exit"):
      print("Bye.")
      break
    if line in ("h", "help"):
      print_help()
      continue
    if line in ("r", "reset"):
      obs, info = env.reset()
      print("Environment reset.")
      continue

    # Allow multi-option sequences, like "0123"
    for ch in line:
      if not ch.isdigit():
        print(f"Ignoring non-digit '{ch}'. Use 0..{n_opts-1}, r, h, q.")
        continue

      idx = int(ch)
      if not (0 <= idx < n_opts):
        print(f"Invalid option '{idx}'. Must be 0..{n_opts-1}.")
        continue

      name = opt_names[idx]
      print(f"\n>>> HL step: {idx} — {name}")

      # One Semi-MDP step = run LL until option terminates; returns accumulated
      # rewards
      next_obs, reward_tuple, terminated, truncated, step_info = env.step(idx)
      total_num_steps += step_info.get("num_steps", 0)
      env.base_env.controller.step(action="Pass")

      # Pretty-print rewards (expecting a 4-tuple from your env)
      # (task_reward, pseudo, ll_pref, hl_pref)
      r_task = reward_tuple[0] if len(reward_tuple) > 0 else 0.0
      r_ps = reward_tuple[1] if len(reward_tuple) > 1 else 0.0
      r_ll = reward_tuple[2] if len(reward_tuple) > 2 else 0.0
      r_hl = reward_tuple[3] if len(reward_tuple) > 3 else 0.0

      num_steps = step_info.get("num_steps", 0)
      print(f"Accumulated reward: task={r_task:.3f}, pseudo={r_ps:.3f}, "
            f"ll_pref={r_ll:.3f}, hl_pref={r_hl:.3f}  |  LL steps: {num_steps}")
      print(f"terminated={terminated}, truncated={truncated}")

      # Prepare for next loop
      obs = next_obs
      info = step_info

      if terminated or truncated:
        print("Episode ended.", end=" ")
        if auto_reset:
          obs, info = env.reset()
          print('Total number of steps taken: ', total_num_steps)
          total_num_steps = 0
          print("Auto-reset done.")
        else:
          print("Use 'r' to reset or 'q' to quit.")


# ------------- Run -------------
if __name__ == "__main__":
  # path_prefix = '/home/bill-qian/hierarchical_reward_design/HierRL/results/' + \
  #   'pnp/ai2thor-pnp_sceneFloorPlan20/ll_models/'
  path_prefix = '/home/bill-qian/hierarchical_reward_design/Eureka/eureka/' + \
    'outputs/eureka/2025-10-03_17-57-41_thor_pnp_ll_0/' + \
    'policy-2025-10-03_17-57-53_iter0_response0/runs/' + \
    'ThorPickPlaceEnvLLGPT-2025-10-03_17-57-53'
  worker_model_path = [
      f'{path_prefix}/ll_model_w_llpref_option0_655/best_model.zip',
      f'{path_prefix}/ll_model_w_llpref_option1_655/best_model.zip',
      f'{path_prefix}/ll_model_w_llpref_option2_655/best_model.zip',
      f'{path_prefix}/ll_model_w_llpref_option3_655/best_model.zip'
  ]

  semi_env = make_semi_env(use_ll_planner=False,
                           worker_model_path=worker_model_path,
                           base_env_cls=ThorPickPlaceEnv,
                           render=True,
                           seed=0)
  run_semi_keyboard(semi_env, auto_reset=True)
