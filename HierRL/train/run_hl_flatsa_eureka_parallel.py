import os
import ast
import argparse
import subprocess
from pathlib import Path

from rw4t.utils import rw4t_seeds


def get_args():
  parser = argparse.ArgumentParser(
      description=
      "Run high-level FlatSA experiments from rewards generated from natural" +
      " language.")

  parser.add_argument(
      "--env_name",
      type=str,
      required=True,
      help="Name of the environment (e.g., 'pnp', 'rw4t', 'oc').")

  parser.add_argument("--seed_idx",
                      type=int,
                      required=True,
                      help="Random seed index for the experiment.")

  parser.add_argument(
      "--model_type",
      type=str,
      default='DQN',
      help="The type of model to use (e.g., DQN MaskableDQN VariableStepDQN)")

  args = parser.parse_args()
  return args


if __name__ == "__main__":
  '''
  python train/run_hl_flatsa_eureka_parallel.py
  --env_name pnp --seed_idx 0 --model_type VariableStepDQN
  '''
  args = get_args()
  env_name = args.env_name
  seed_idx = args.seed_idx
  model_type = args.model_type

  if env_name == 'pnp':
    class_name = 'ThorPickPlaceEnvFlatSAGPT'
  else:
    raise NotImplementedError

  run_hl_full_path = Path(__file__).parent / 'run_hl.py'
  eureka_dir_full_path = Path(
      __file__).parent.parent.parent / 'Eureka/eureka/outputs/eureka'

  eureka_dir_and_module_name = []
  trial_dir_pattern = f"{env_name}_flatsa_{seed_idx}"
  # Check each trial directory
  for trial_dir in os.listdir(eureka_dir_full_path):
    if trial_dir_pattern in trial_dir:
      print('==============================')
      print('Found trial dir: ', trial_dir)
      # Find the successful ll models in the trial directory
      trial_dir_full_path = os.path.join(eureka_dir_full_path, trial_dir)
      success_log_file = 'successful_ll_model_paths.txt'
      success_log_full_path = os.path.join(trial_dir_full_path,
                                           success_log_file)
      if not Path(success_log_full_path).exists():
        # !!! Only gather the run info (eureka_dir and module_name) if there
        # is a log file containing the successful ll paths
        print(f'Did not find \"{success_log_file}\" in \"{trial_dir}\", ' +
              f'skipping \"{trial_dir}\"!')
      else:
        successful_ll_path_groups = []
        with open(success_log_full_path, 'r') as f:
          content = f.read()
          successful_ll_path_groups = ast.literal_eval(content)
          assert isinstance(successful_ll_path_groups, list)

          if len(successful_ll_path_groups) == 0:
            # !!! Only gather the run info if one of the runs in the trials is
            # successful
            print(f'Skipping \"{trial_dir}\" as no run is successful!')

          for path_group in successful_ll_path_groups:
            assert len(path_group) > 0
            assert all('best_model.zip' in a_path for a_path in path_group)
            # Find the response idx of this path group
            response_idx = -1
            cur_idx = 0
            while True:
              if f'iter0_response{cur_idx}' in path_group[0]:
                response_idx = cur_idx
                break
              cur_idx += 1
              if cur_idx > 100:
                raise ValueError('Could not find response idx')

            eureka_dir_arg = str(Path(path_group[0]).parent.parent)
            # print('Eureka dir: ', eureka_dir_arg)
            module_name_arg = os.path.join(
                trial_dir_full_path, f'env_iter0_response{response_idx}.py')
            # print('Module name: ', module_name_arg)

            # !!! Only gather the run info if we have yet to learn a high
            # level policy for this one
            policies_and_others = os.listdir(eureka_dir_arg)
            hl_policy_dir = [
                dir_name for dir_name in policies_and_others
                if 'hl_model' in dir_name
            ]
            assert len(hl_policy_dir) <= 1
            if len(hl_policy_dir) == 1 and 'best_model.zip' in os.listdir(
                os.path.join(eureka_dir_arg, hl_policy_dir[0])):
              print(f'Skipping \"iter0_response{response_idx}\" as there is ' +
                    'already a high level policy!')
            else:
              eureka_dir_and_module_name.append(
                  (eureka_dir_arg, module_name_arg))

  # Start the HL training runs
  all_processes = []
  for hl_run_idx in range(len(eureka_dir_and_module_name)):
    eureka_dir = eureka_dir_and_module_name[hl_run_idx][0]
    module_name = eureka_dir_and_module_name[hl_run_idx][1]
    print('Running FlatSA HL expertiment for: ' +
          f'{Path(eureka_dir).parent.parent.parent.name}' +
          f'/{Path(eureka_dir).parent.parent.name}')

    rl_filepath = 'hl_output.txt'
    with open(os.path.join(eureka_dir, rl_filepath), 'w') as f:
      process = subprocess.Popen([
          'python', '-u',
          str(run_hl_full_path), '--env_name', env_name, '--class_name',
          class_name, '--module_name', module_name, '--pref_type', 'flatsa',
          '--eureka_dir', eureka_dir, '--model_type', model_type, '--seed_idx',
          str(seed_idx)
      ],
                                 stdout=f,
                                 stderr=f)
    all_processes.append(process)

  try:
    for proc in all_processes:
      proc.wait()
  except KeyboardInterrupt:
    print("\nCaught Ctrl+C — terminating all subprocesses...")
    for proc in all_processes:
      proc.terminate()  # send SIGTERM to each

    # Give them a moment to exit cleanly
    for proc in all_processes:
      try:
        proc.wait(timeout=3)
      except subprocess.TimeoutExpired:
        print(f"Forcing kill on PID {proc.pid}")
        proc.kill()

    print("All subprocesses terminated.")
