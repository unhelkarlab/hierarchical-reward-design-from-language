import subprocess
import os
import json
import logging
from pathlib import Path

from utils.extract_task_code import file_to_string


def set_freest_gpu():
  freest_gpu = get_freest_gpu()
  os.environ['CUDA_VISIBLE_DEVICES'] = str(freest_gpu)


def get_freest_gpu():
  sp = subprocess.Popen(['gpustat', '--json'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
  out_str, _ = sp.communicate()
  gpustats = json.loads(out_str.decode('utf-8'))
  # Find GPU with most free memory
  freest_gpu = min(gpustats['gpus'], key=lambda x: x['memory.used'])

  return freest_gpu['index']


def filter_traceback(s):
  lines = s.split('\n')
  filtered_lines = []
  for i, line in enumerate(lines):
    if line.startswith('Traceback'):
      for j in range(i, len(lines)):
        if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
          break
        filtered_lines.append(lines[j])
      return '\n'.join(filtered_lines)
  return ''  # Return an empty string if no Traceback is found


def block_until_training(rl_filepath,
                         log_status=False,
                         iter_num=-1,
                         response_id=-1):
  # Ensure that the RL training has started before moving on
  while True:
    rl_log = file_to_string(rl_filepath)
    if 'KitchenHLGPT' in rl_log or 'KitchenFlatSAGPT' in rl_log:
      if ("Traceback" in rl_log or 'Using cuda device' in rl_log
          or 'Using cpu device' in rl_log):
        if "Traceback" in rl_log:
          logging.info(
              f"Iteration {iter_num}: Code Run {response_id} execution error!")
        else:
          logging.info(f"Iteration {iter_num}: Code Run {response_id} started!")
        break
    else:
      if "fps" in rl_log or "Traceback" in rl_log:
        if log_status and "fps step:" in rl_log:
          logging.info(
              f"Iteration {iter_num}: Code Run {response_id} successfully training!"
          )
        if log_status and "Traceback" in rl_log:
          logging.info(
              f"Iteration {iter_num}: Code Run {response_id} execution error!")
        break


def replace_str_in_file(file_path, old_str, new_str):
  # Read the file
  with open(file_path, "r") as f:
    content = f.read()
  # Replace all occurrences
  content = content.replace(old_str, new_str)
  # Write back to the file
  with open(file_path, "w") as f:
    f.write(content)


def copy_env_file(env_name):
  if 'rescue_world' in env_name:
    env_file = f'{Path(__file__).parent.parent.parent.parent}/rw4t/rw4t_env.py'
    dst_path = f'{Path(__file__).parent.parent}/envs/rescue_world.py'
    with open(env_file, 'r') as src_file:
      lines = src_file.readlines()
    with open(dst_path, 'w') as dst_file:
      skip = False
      for line in lines:
        if not skip and line.strip().startswith(
            "if __name__") and "__main__" in line:
          skip = True
        if not skip:
          dst_file.write(line)
  elif 'kitchen' in env_name:
    env_file = f'{Path(__file__).parent.parent.parent.parent}/' + \
      'Hierarchical-Language-Agent/testbed-cooking/gym_cooking/' + \
      'envs/overcooked_simple.py'
    dst_path = f'{Path(__file__).parent.parent}/envs/kitchen.py'
    with open(env_file, 'r') as src_file:
      lines = src_file.readlines()
    with open(dst_path, 'w') as dst_file:
      skip = False
      for line in lines:
        if not skip and "def test_MDP():" in line:
          skip = True
        if not skip:
          dst_file.write(line)
  elif 'thor_pnp' in env_name:
    env_file = f'{Path(__file__).parent.parent.parent.parent}/' + \
      'HierRL/envs/ai2thor/pnp_env.py'
    dst_path = f'{Path(__file__).parent.parent}/envs/thor_pnp.py'
    with open(env_file, 'r') as src_file:
      lines = src_file.readlines()
    with open(dst_path, 'w') as dst_file:
      skip = False
      for line in lines:
        if not skip:
          dst_file.write(line)
  else:
    raise NotImplementedError


if __name__ == "__main__":
  print(get_freest_gpu())
