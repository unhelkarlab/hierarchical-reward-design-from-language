import os
import ast
import argparse
import subprocess
from pathlib import Path

from rw4t.utils import rw4t_seeds
from HierRL.train.run_hl_flatsa_eureka_parallel import get_args
from HierRL.eval.eval_helper import get_all_compilable_model_path_groups

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
  all_path_groups_groupedby_trial = get_all_compilable_model_path_groups(
      env_type=env_name, pref_type='flatsa', seed_idx=seed_idx)
  for path_groups_per_trial in all_path_groups_groupedby_trial:
    print('=========================')
    # Check if there is at least one set of compilable policies in the trial
    if 'best_model.zip' in path_groups_per_trial[0][0]:
      trial_dir = Path(
          path_groups_per_trial[0][0]).parent.parent.parent.parent.parent
      print(f'In trial {trial_dir}...')
      for path_group in path_groups_per_trial:
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
        module_name_arg = os.path.join(trial_dir,
                                       f'env_iter0_response{response_idx}.py')
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
          print('Adding the following run info...')
          print('Eureka dir: ', eureka_dir_arg)
          print('Module name: ', module_name_arg)
          eureka_dir_and_module_name.append((eureka_dir_arg, module_name_arg))
    else:
      # !!! Only gather the run info if one of the policies in the trial is
      # successful
      print(f'Skipping {path_groups_per_trial[0][0]}, as it does not ' +
            'have any successful runs!')

  # Start the HL training runs
  print("\n")
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
