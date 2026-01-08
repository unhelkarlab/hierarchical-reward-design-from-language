import hydra
import numpy as np
import json
import logging
import matplotlib.pyplot as plt
import os
from openai import OpenAI
import re
import subprocess
from pathlib import Path
import shutil
import time

from utils.misc import (set_freest_gpu, filter_traceback, block_until_training,
                        replace_str_in_file, copy_env_file)
from utils.file_utils import load_tensorboard_logs
from utils.create_task import create_task
from utils.extract_task_code import file_to_string, get_function_signature
from utils.check_eureka_code import extract_eureka_code, check_for_statefulness

EUREKA_ROOT_DIR = os.getcwd()
TRAINING_ROOT_DIR = f"{EUREKA_ROOT_DIR}/../training"
METRIC = 'gt_reward'
METRIC_FULLNAME = "Success" if METRIC == "success" else "Ground Truth Reward"


@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
  workspace_dir = Path.cwd()
  logging.info(f"Workspace: {workspace_dir}")
  logging.info(f"Project Root: {EUREKA_ROOT_DIR}")

  logging.info(os.getenv("OPENAI_API_KEY"))
  client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

  task = cfg.env.task
  task_description = cfg.env.description
  suffix = cfg.suffix
  model = cfg.model
  logging.info(f"Using LLM: {model}")
  logging.info("Task: " + task)
  logging.info("Task description: " + task_description)

  env_name = cfg.env.env_name.lower()
  task_file = f'{EUREKA_ROOT_DIR}/envs/{env_name}.py'
  # Copy script of environment code into eureka/envs
  copy_env_file(env_name)
  # Make another copy of the environment code with name {env_name}.py
  if 'rescue_world' in env_name:
    shutil.copy(f'{EUREKA_ROOT_DIR}/envs/rescue_world.py', task_file)
    replace_str = 'RW4TEnv'
    replace_str_in_file(task_file, replace_str, cfg.env.task)
  elif 'kitchen' in env_name:
    shutil.copy(f'{EUREKA_ROOT_DIR}/envs/kitchen.py', task_file)
    replace_str_in_file(task_file, 'OvercookedSimple', cfg.env.task)
    replace_str_in_file(task_file, f'{cfg.env.task}HL',
                        'EurekaOvercookedSimpleHL')
  elif 'thor_pnp' in env_name:
    shutil.copy(f'{EUREKA_ROOT_DIR}/envs/thor_pnp.py', task_file)
    replace_str = 'ThorPickPlaceEnv'
    replace_str_in_file(task_file, replace_str, cfg.env.task)

  # Make a copy of the environment code that only contains the observation info,
  # this is what the LLM sees
  task_obs_file = f'{EUREKA_ROOT_DIR}/envs/{env_name}_obs.py'
  if 'rescue_world' in env_name:
    shutil.copy(f'{EUREKA_ROOT_DIR}/envs/rescue_world_obs.py', task_obs_file)
    replace_str = 'RW4TEnv'
    replace_str_in_file(task_obs_file, replace_str, cfg.env.task)
  elif 'kitchen' in env_name:
    shutil.copy(f'{EUREKA_ROOT_DIR}/envs/kitchen_obs.py', task_obs_file)
    replace_str_in_file(task_obs_file, 'OvercookedSimple', cfg.env.task)
  elif 'thor_pnp' in env_name:
    shutil.copy(f'{EUREKA_ROOT_DIR}/envs/thor_pnp_obs.py', task_obs_file)
    replace_str = 'ThorPickPlaceEnv'
    replace_str_in_file(task_obs_file, replace_str, cfg.env.task)
  shutil.copy(task_obs_file, "env_init_obs.py")
  task_code_string = file_to_string(task_file)
  task_obs_code_string = file_to_string(task_obs_file)
  output_file = f"{TRAINING_ROOT_DIR}/tasks/{env_name}{suffix.lower()}.py"

  # Loading all text prompts
  prompt_dir = f'{EUREKA_ROOT_DIR}/utils/prompts'
  initial_system = file_to_string(f'{prompt_dir}/initial_system.txt')
  code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')
  code_feedback = file_to_string(f'{prompt_dir}/code_feedback.txt')
  initial_user = file_to_string(f'{prompt_dir}/initial_user.txt')
  if cfg.env.level == 'high':
    reward_signature = file_to_string(f'{prompt_dir}/reward_signature_hl.txt')
  elif cfg.env.level == 'low':
    reward_signature = file_to_string(f'{prompt_dir}/reward_signature_ll.txt')
  elif cfg.env.level == 'flatsa':
    reward_signature = file_to_string(
        f'{prompt_dir}/reward_signature_flatsa.txt')
  else:
    raise NotImplementedError
  policy_feedback = file_to_string(f'{prompt_dir}/policy_feedback.txt')
  execution_error_feedback = file_to_string(
      f'{prompt_dir}/execution_error_feedback.txt')

  initial_system = initial_system.format(
      task_reward_signature_string=reward_signature) + code_output_tip
  initial_user = initial_user.format(task_obs_code_string=task_obs_code_string,
                                     task_description=task_description)
  #   print('System message: ', initial_system)
  #   print('User message: ', initial_user)
  messages = [{
      "role": "system",
      "content": initial_system
  }, {
      "role": "user",
      "content": initial_user
  }]

  task_code_string = task_code_string.replace(task, task + suffix)
  # Create Task YAML files
  create_task(TRAINING_ROOT_DIR, cfg.env.task, cfg.env.env_name, suffix)

  DUMMY_FAILURE = -10000.
  # The best values in each iteration
  max_eval_metrics = []
  max_eval_metric_freqs = []
  # Reward correlations for the best run in each iteration
  max_eval_metrics_reward_correlation = []
  # execute_rates = []
  best_code_paths = []
  max_eval_metric_overall = DUMMY_FAILURE
  max_eval_metric_reward_correlation_overall = DUMMY_FAILURE
  max_eval_metric_freq_overall = DUMMY_FAILURE
  max_reward_code_path = None

  # Eureka generation loop
  for iter in range(cfg.iteration):
    # Get Eureka response
    responses = []
    response_cur = None
    total_samples = 0
    total_token = 0
    total_completion_token = 0
    chunk_size = cfg.sample if "gpt-3.5" in model else 4

    logging.info(
        f"Iteration {iter}: Generating {cfg.sample} samples with {cfg.model}")

    while True:
      if total_samples >= cfg.sample:
        break
      for attempt in range(1000):
        try:
          response_cur = client.chat.completions.create(
              model=model,
              messages=messages,
              temperature=cfg.temperature,
              n=chunk_size)
          total_samples += chunk_size
          break
        except Exception as e:
          if attempt >= 10:
            chunk_size = max(int(chunk_size / 2), 1)
            print("Current Chunk Size", chunk_size)
          logging.info(f"Attempt {attempt+1} failed with error: {e}")
          time.sleep(1)
      if response_cur is None:
        logging.info("Code terminated due to too many failed attempts!")
        exit()

      responses.extend(response_cur.choices)
      prompt_tokens = response_cur.usage.prompt_tokens
      total_completion_token += response_cur.usage.completion_tokens
      total_token += response_cur.usage.total_tokens

    if cfg.sample == 1:
      logging.info(f"Iteration {iter}: GPT Output:\n " +
                   responses[0].message.content + "\n")

    # Logging Token Information
    logging.info(f"Iteration {iter}: Prompt Tokens: {prompt_tokens}, " +
                 f"Completion Tokens: {total_completion_token}, " +
                 f"Total Tokens: {total_token}")

    code_runs = []
    rl_runs = []
    for response_id in range(cfg.sample):
      response_cur = responses[response_id].message.content
      logging.info(f"Iteration {iter}: Processing Code Run {response_id}")

      # Regex patterns to extract python code enclosed in GPT response
      patterns = [
          r'```python(.*?)```',
          r'```(.*?)```',
          r'"""(.*?)"""',
          r'""(.*?)""',
          r'"(.*?)"',
      ]
      for pattern in patterns:
        code_string = re.search(pattern, response_cur, re.DOTALL)
        if code_string is not None:
          code_string = code_string.group(1).strip()
          break
      code_string = response_cur if not code_string else code_string
      logging.info(f"Iteration {iter} Run {response_id} Code: {code_string}")

      # Remove unnecessary imports
      code_string = extract_eureka_code(code_string)

      # Check if the code is stateful
      statefulness_check = check_for_statefulness(code_string)
      if (statefulness_check['uses_function_attributes']
          or statefulness_check['uses_global_variables']):
        logging.info(
            f"Iteration {iter}: Code Run {response_id} is stateful, skipping..."
        )
        logging.info(
            f"Iteration {iter}: Code Run {response_id} statefulness check: " +
            f"{statefulness_check['details']}")
        continue

      # Add the Eureka Reward Signature to the environment code
      try:
        gpt_reward_signature, input_lst = get_function_signature(code_string)
        gpt_reward_signature = gpt_reward_signature.replace('self.', '')
        logging.info(
            f"Iteration {iter} Run {response_id} Raw Reward Signature: " +
            f"{gpt_reward_signature}")
      except Exception as e:
        logging.info(
            f"Iteration {iter}: Code Run {response_id} cannot parse function signature!"
        )
        continue

      code_runs.append(code_string)
      reward_signature = [
          "if not isinstance(state, dict): state = state.state_to_dict()",
          f"reward, reward_dict = {gpt_reward_signature}",
          "return reward",
      ]
      indent = " " * 4
      reward_signature = "\n".join([indent + line for line in reward_signature])
      logging.info(f"Iteration {iter} Run {response_id} Processed Reward " +
                   f"Signature: {reward_signature}")
      if cfg.env.level == 'high' and (
          "def get_high_level_pref_gpt(self, state, prev_option, option):"
          in task_code_string):
        task_code_string_iter = task_code_string.replace(
            "def get_high_level_pref_gpt(self, state, prev_option, option):" +
            f"\n{indent}pass",
            "def get_high_level_pref_gpt(self, state, prev_option, option):" +
            f"\n{reward_signature}")
      elif cfg.env.level == 'low' and (
          "def get_low_level_pref_gpt(self, state, option, action):"
          in task_code_string):
        task_code_string_iter = task_code_string.replace(
            "def get_low_level_pref_gpt(self, state, option, action):" +
            f"\n{indent}pass",
            "def get_low_level_pref_gpt(self, state, option, action):" +
            f"\n{reward_signature}")
      elif cfg.env.level == 'flatsa' and (
          'def get_flat_sa_pref_gpt(self, state, action):' in task_code_string):
        task_code_string_iter = task_code_string.replace(
            "def get_flat_sa_pref_gpt(self, state, action):" +
            f"\n{indent}pass",
            "def get_flat_sa_pref_gpt(self, state, action):" +
            f"\n{reward_signature}")
      else:
        raise NotImplementedError

      # Save the new environment code when the output contains valid code
      # string!
      with open(output_file, 'w') as file:
        file.writelines(task_code_string_iter + '\n')
        file.writelines("from typing import Dict, Tuple" + '\n')
        file.writelines("import math" + '\n')
        file.writelines(code_string + '\n')

      with open(f"env_iter{iter}_response{response_id}_rewardonly.py",
                'w') as file:
        file.writelines(code_string + '\n')

      # Copy the generated environment code to hydra output directory for
      # bookkeeping
      # shutil.copy(output_file, f"env_iter{iter}_response{response_id}.py")
      with open(f"env_iter{iter}_response{response_id}.py", 'w') as file:
        file.writelines(task_code_string_iter + '\n')
        file.writelines("from typing import Dict, Tuple" + '\n')
        file.writelines("import math" + '\n')
        file.writelines(code_string + '\n')

      # Find the freest GPU to run GPU-accelerated RL
      set_freest_gpu()

      # Execute the python file with flags
      rl_filepath = f"env_iter{iter}_response{response_id}.txt"
      logging.info(f'seed_idx: {cfg.seed_idx}')
      processes = []
      if cfg.num_options == -1:
        with open(rl_filepath, 'w') as f:
          process = subprocess.Popen([
              'python',
              '-u',
              f'{TRAINING_ROOT_DIR}/train.py',
              'hydra/output=subprocess',
              f'task={task}{suffix}',
              f'wandb_activate={cfg.use_wandb}',
              f'wandb_entity={cfg.wandb_username}',
              f'wandb_project={cfg.wandb_project}',
              f'headless={not cfg.capture_video}',
              f'capture_video={cfg.capture_video}',
              'force_render=False',
              f'total_timesteps={cfg.env.timesteps}',
              f'seed_idx={cfg.seed_idx}',
              f'cur_iter={iter}',
              f'cur_response={response_id}',
          ],
                                     stdout=f,
                                     stderr=f)
        processes.append(process)
      else:
        for i in range(cfg.num_options):
          path_name = Path(rl_filepath).stem
          path_name = f'{path_name}_option{i}.txt'
          with open(path_name, 'w') as f:
            process = subprocess.Popen([
                'python',
                '-u',
                f'{TRAINING_ROOT_DIR}/train.py',
                'hydra/output=subprocess',
                f'task={task}{suffix}',
                f'wandb_activate={cfg.use_wandb}',
                f'wandb_entity={cfg.wandb_username}',
                f'wandb_project={cfg.wandb_project}',
                f'headless={not cfg.capture_video}',
                f'capture_video={cfg.capture_video}',
                'force_render=False',
                f'total_timesteps={cfg.env.timesteps}',
                f'seed_idx={cfg.seed_idx}',
                f'cur_iter={iter}',
                f'cur_response={response_id}',
                f'option_to_use={i}',
            ],
                                       stdout=f,
                                       stderr=f)
          processes.append(process)

      if not cfg.blocking:
        block_until_training(rl_filepath,
                             log_status=True,
                             iter_num=iter,
                             response_id=response_id)
      else:
        print('Waiting for process to finish')
        for p in processes:
          p.wait()
      rl_runs.append(process)

    if cfg.num_options != -1:
      continue

    # Gather RL training results and construct reward reflection
    code_feedbacks = []
    contents = []
    successes = []
    gt_rewards = []
    gt_reward_freqs = []
    reward_correlations = []
    code_paths = []

    exec_success = False
    for response_id, (code_run, rl_run) in enumerate(zip(code_runs, rl_runs)):
      rl_run.communicate()
      rl_filepath = f"env_iter{iter}_response{response_id}.txt"
      code_paths.append(f"env_iter{iter}_response{response_id}.py")
      try:
        with open(rl_filepath, 'r') as f:
          stdout_str = f.read()
      except:
        content = execution_error_feedback.format(
            traceback_msg=
            "Code Run cannot be executed due to function signature error! " +
            "Please re-write an entirely new reward function!")
        content += code_output_tip
        contents.append(content)
        successes.append(DUMMY_FAILURE)
        gt_rewards.append(DUMMY_FAILURE)
        gt_reward_freqs.append(DUMMY_FAILURE)
        reward_correlations.append(DUMMY_FAILURE)
        continue

      content = ''
      traceback_msg = filter_traceback(stdout_str)

      if traceback_msg == '':
        # If RL execution has no error, provide policy statistics feedback
        exec_success = True
        lines = stdout_str.split('\n')
        for i, line in enumerate(lines):
          if line.startswith('Tensorboard Directory:'):
            break
        if cfg.env.level == 'flatsa':
          line = line.rsplit('/', 1)[0] + '/nn/summaries_1'
        tensorboard_logdir = line.split(':')[-1].strip()
        tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
        max_iterations = np.array(tensorboard_logs['gt_reward']).shape[0]
        logging.info(
            f'Iteration {iter} Run {response_id} Num Evals: {max_iterations}')
        epoch_freq = max(int(max_iterations // 10), 1)
        logging.info(
            f'Iteration {iter} Run {response_id} Num Epochs: {epoch_freq}')

        content += policy_feedback.format(epoch_freq=epoch_freq)

        # Compute Correlation between Human-Engineered and GPT Rewards
        if "gt_reward" in tensorboard_logs and "gpt_reward" in tensorboard_logs:
          gt_reward = np.array(tensorboard_logs["gt_reward"])
          gpt_reward = np.array(tensorboard_logs["gpt_reward"])
          reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
          reward_correlations.append(reward_correlation)

        # Add reward components log to the feedback
        for metric in tensorboard_logs:
          # logging.info(f'metric: {metric}')
          if "/" not in metric:
            metric_cur = [
                '{:.2f}'.format(x)
                for x in tensorboard_logs[metric][::epoch_freq]
            ]
            metric_cur_max = max(tensorboard_logs[metric])
            metric_cur_mean = sum(tensorboard_logs[metric]) / len(
                tensorboard_logs[metric])
            if "consecutive_successes" == metric:
              successes.append(metric_cur_max)
            if "gt_reward" == metric:
              logging.info(
                  f'Iteration {iter} Run {response_id} best GT reward: {metric_cur_max}'
              )
              gt_rewards.append(metric_cur_max)
              gt_reward_freqs.append(
                  tensorboard_logs[metric].count(metric_cur_max))
            metric_cur_min = min(tensorboard_logs[metric])
            if metric != "gt_reward" and metric != "gpt_reward":
              if metric != "consecutive_successes":
                metric_name = metric
              else:
                metric_name = "task_score"
              content += (
                  f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, " +
                  f"Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n")
            else:
              # Provide ground-truth score when success rate not applicable
              if "consecutive_successes" not in tensorboard_logs:
                if "gt_reward" == metric:
                  logging.info((
                      f'Iteration {iter} Run {response_id} ' +
                      f"ground-truth score: {metric_cur}, " +
                      f"Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, "
                      + f"Min: {metric_cur_min:.2f} \n"))
                  content += (
                      f"ground-truth score: {metric_cur}, " +
                      f"Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, "
                      + f"Min: {metric_cur_min:.2f} \n")
        code_feedbacks.append(code_feedback)
        content += code_feedback
      else:
        # Otherwise, provide execution traceback error feedback
        successes.append(DUMMY_FAILURE)
        gt_rewards.append(DUMMY_FAILURE)
        gt_reward_freqs.append(DUMMY_FAILURE)
        reward_correlations.append(DUMMY_FAILURE)
        content += execution_error_feedback.format(traceback_msg=traceback_msg)

      content += code_output_tip
      contents.append(content)

    # Repeat the iteration if all code generation failed
    if not exec_success and cfg.sample != 1:
      # execute_rates.append(0.)
      max_eval_metrics.append(DUMMY_FAILURE)
      max_eval_metric_freqs.append(DUMMY_FAILURE)
      max_eval_metrics_reward_correlation.append(DUMMY_FAILURE)
      best_code_paths.append(None)
      logging.info(
          "All code generation failed! Repeat this iteration from the current "
          + "message checkpoint!")
      continue

    # Select the best code sample based on the success rate
    if METRIC == 'gt_reward':
      eval_metric = gt_rewards
    elif METRIC == 'success':
      eval_metric = successes
    else:
      raise NotImplementedError

    # Get the indices of maximum values in eval_metric, and then find the best
    # index using avg rewards
    logging.info(f'Max {METRIC} of each run in iteration {iter}: {eval_metric}')
    max_value = max(eval_metric)
    max_indices = [i for i, num in enumerate(eval_metric) if num == max_value]
    logging.info(
        f'Best runs based on {METRIC} in iteration {iter}: {max_indices}')
    for i in max_indices:
      logging.info(f'Index: {i}, Reward Freq: {gt_reward_freqs[i]}')
    best_sample_idx = max(max_indices, key=lambda i: gt_reward_freqs[i])
    logging.info(
        f'Among the best runs, run {best_sample_idx} has the highest ' +
        f'frequency of achieving optimal {METRIC}.')
    best_content = contents[best_sample_idx]
    max_eval_metric = eval_metric[best_sample_idx]
    max_eval_metric_reward_correlation = reward_correlations[best_sample_idx]
    # execute_rate = np.sum(np.array(successes) >= 0.) / cfg.sample

    # Update the best Eureka Output
    if max_eval_metric > max_eval_metric_overall:
      max_eval_metric_overall = max_eval_metric
      max_eval_metric_reward_correlation_overall = \
        max_eval_metric_reward_correlation
      max_reward_code_path = code_paths[best_sample_idx]
      max_eval_metric_freq_overall = gt_reward_freqs[best_sample_idx]
    elif (max_eval_metric == max_eval_metric_overall
          and gt_reward_freqs[best_sample_idx] > max_eval_metric_freq_overall):
      max_eval_metric_overall = max_eval_metric
      max_eval_metric_reward_correlation_overall = \
        max_eval_metric_reward_correlation
      max_reward_code_path = code_paths[best_sample_idx]
      max_eval_metric_freq_overall = gt_reward_freqs[best_sample_idx]

    # execute_rates.append(execute_rate)
    max_eval_metrics.append(max_eval_metric)
    max_eval_metrics_reward_correlation.append(
        max_eval_metric_reward_correlation)
    max_eval_metric_freqs.append(gt_reward_freqs[best_sample_idx])
    best_code_paths.append(code_paths[best_sample_idx])

    logging.info(
        f"Iteration {iter}: Max {METRIC_FULLNAME}: {max_eval_metric}, " +
        f"Max Reward Correlation: {max_eval_metric_reward_correlation}, " +
        f"Max Reward Freq: {gt_reward_freqs[best_sample_idx]}")
    logging.info(f"Iteration {iter}: Best Generation ID: {best_sample_idx}")
    logging.info(f"Iteration {iter}: GPT Output Content:\n" +
                 responses[best_sample_idx].message.content + "\n")
    logging.info(f"Iteration {iter}: User Content:\n" + best_content + "\n")

    # Plot the success rate
    fig, axs = plt.subplots(1, figsize=(6, 6))
    fig.suptitle(f'{cfg.env.task}')

    x_axis = np.arange(len(max_eval_metrics))

    axs.plot(x_axis, np.array(max_eval_metrics))
    axs.set_title(f"Max {METRIC_FULLNAME}")
    axs.set_xlabel("Iteration")

    # axs[1].plot(x_axis, np.array(execute_rates))
    # axs[1].set_title("Execute Rate")
    # axs[1].set_xlabel("Iteration")

    fig.tight_layout(pad=3.0)
    plt.savefig('summary.png')
    np.savez(
        'summary.npz',
        max_eval_metrics=max_eval_metrics,
        # execute_rates=execute_rates,
        best_code_paths=best_code_paths,
        max_eval_metrics_reward_correlation=max_eval_metrics_reward_correlation)

    if len(messages) == 2:
      messages += [{
          "role": "assistant",
          "content": responses[best_sample_idx].message.content
      }]
      messages += [{"role": "user", "content": best_content}]
    else:
      assert len(messages) == 4
      messages[-2] = {
          "role": "assistant",
          "content": responses[best_sample_idx].message.content
      }
      messages[-1] = {"role": "user", "content": best_content}

    # Save dictionary as JSON file
    with open('messages.json', 'w') as file:
      json.dump(messages, file, indent=4)

  # Evaluate the best reward code many times
  if cfg.num_options != -1:
    return

  if max_reward_code_path is None:
    logging.info("All iterations of code generation failed, aborting...")
    logging.info(
        "Please double check the output env_iter*_response*.txt files for " +
        "repeating errors!")
    exit()
  logging.info(
      f"Task: {task}, Max Training {METRIC_FULLNAME} {max_eval_metric_overall}, "
      + f"Correlation {max_eval_metric_reward_correlation_overall}, " +
      f"Best Reward Code Path: {max_reward_code_path}")
  logging.info(f"Evaluating best reward code {cfg.num_eval} times")
  shutil.copy(max_reward_code_path, output_file)

  return
  eval_runs = []
  for i in range(cfg.num_eval):
    set_freest_gpu()

    # Execute the python file with flags
    rl_filepath = f"reward_code_eval{i}.txt"
    with open(rl_filepath, 'w') as f:
      process = subprocess.Popen([
          'python',
          '-u',
          f'{TRAINING_ROOT_DIR}/train.py',
          'hydra/output=subprocess',
          f'task={task}{suffix}',
          f'wandb_activate={cfg.use_wandb}',
          f'wandb_entity={cfg.wandb_username}',
          f'wandb_project={cfg.wandb_project}',
          f'headless={not cfg.capture_video}',
          f'capture_video={cfg.capture_video}',
          'force_render=False',
          'total_timesteps=50_000',
          f'seed={i}',
      ],
                                 stdout=f,
                                 stderr=f)

    block_until_training(rl_filepath)
    eval_runs.append(process)

  reward_code_final_eval_metrics = []
  reward_code_correlations_final = []
  for i, rl_run in enumerate(eval_runs):
    rl_run.communicate()
    rl_filepath = f"reward_code_eval{i}.txt"
    with open(rl_filepath, 'r') as f:
      stdout_str = f.read()
    lines = stdout_str.split('\n')
    for i, line in enumerate(lines):
      if line.startswith('Tensorboard Directory:'):
        break
    tensorboard_logdir = line.split(':')[-1].strip()
    tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
    max_eval_metric = max(tensorboard_logs['consecutive_successes' if METRIC ==
                                           'success' else 'gt_reward'])
    reward_code_final_eval_metrics.append(max_eval_metric)

    if "gt_reward" in tensorboard_logs and "gpt_reward" in tensorboard_logs:
      gt_reward = np.array(tensorboard_logs["gt_reward"])
      gpt_reward = np.array(tensorboard_logs["gpt_reward"])
      reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
      reward_code_correlations_final.append(reward_correlation)

  logging.info(
      f"Final {METRIC_FULLNAME} Mean: {np.mean(reward_code_final_eval_metrics)}, "
      + f"Std: {np.std(reward_code_final_eval_metrics)}, " +
      f"Raw: {reward_code_final_eval_metrics}")
  logging.info(
      f"Final Correlation Mean: {np.mean(reward_code_correlations_final)}, " +
      f"Std: {np.std(reward_code_correlations_final)}, " +
      f"Raw: {reward_code_correlations_final}")
  np.savez('final_eval.npz',
           reward_code_final_eval_metrics=reward_code_final_eval_metrics,
           reward_code_correlations_final=reward_code_correlations_final)


if __name__ == "__main__":
  main()
