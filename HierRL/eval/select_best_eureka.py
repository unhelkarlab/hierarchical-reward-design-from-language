import os
import re
from datetime import datetime
from collections import defaultdict

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import \
  EventAccumulator


def load_tensorboard_logs(path):
  data = defaultdict(list)
  event_acc = EventAccumulator(path)
  event_acc.Reload()  # Load all data written so far

  for tag in event_acc.Tags()["scalars"]:
    events = event_acc.Scalars(tag)
    for event in events:
      data[tag].append(event.value)

  return data


def get_best_controller_save_path(controller_save_folder, write=False):

  # 1) Get all policy folders
  pattern = r"policy-(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"
  all_policy_folders = []
  for name in os.listdir(controller_save_folder):
    policy_folder = os.path.join(controller_save_folder, name)
    if os.path.isdir(policy_folder):
      match = re.match(pattern, name)
      if match:
        dt_str = match.group(1)
        dt = datetime.strptime(dt_str, "%Y-%m-%d_%H-%M-%S")
        all_policy_folders.append((dt, name))  # Store (datetime, folder name)
  sorted_policy_folders = [name for dt, name in sorted(all_policy_folders)]

  # 2) Find the best policy folder index
  best_max_gt_reward = float('-inf')
  best_gt_reward_freq = float('-inf')
  best_idx = -1
  cur_idx = 0
  for policy_folder in sorted_policy_folders:
    tb_path = os.path.join(controller_save_folder, policy_folder, 'runs')
    policy_subfolders = [
        name for name in os.listdir(tb_path)
        if os.path.isdir(os.path.join(tb_path, name))
    ]
    assert len(policy_subfolders) == 1
    tb_path = os.path.join(tb_path, policy_subfolders[0], 'summaries_1')
    if not os.path.isdir(tb_path):
      cur_idx += 1
      continue
    tb_files = [name for name in os.listdir(tb_path)]
    assert len(tb_files) == 1
    # print(tb_path)
    tb_path = os.path.join(tb_path, tb_files[0])

    tensorboard_logs = load_tensorboard_logs(tb_path)
    metric = 'gt_reward'
    if metric not in tensorboard_logs:
      cur_idx += 1
      continue
    metric_max = max(tensorboard_logs[metric])
    metric_freq = tensorboard_logs[metric].count(metric_max)
    if metric_max > best_max_gt_reward or (metric_max == best_max_gt_reward and
                                           metric_freq > best_gt_reward_freq):
      best_max_gt_reward = metric_max
      best_gt_reward_freq = metric_freq
      best_idx = cur_idx
    cur_idx += 1
  print(f'best folder index: {best_idx}')

  # 3) Write the best folder index to a file if needed
  with open(f'{controller_save_folder}/best_eureka_info.txt', 'w') as f:
    f.write(f'Best Folder Index: {best_idx}')


if __name__ == '__main__':
  #   eureka_parent_folder = 'results/rw4t/rw4t_6by6_4obj_map8_dqn/' + \
  #     'hl_model_w_hlpref_wo_llpref_ll_model_w_llpref_gpt'
  # eureka_parent_folder = 'results/oc/oc_david_dqn_wo_conv/' + \
  #   'hl_model_w_hlpref_gpt'
  # eureka_folders = [
  #     os.path.join(eureka_parent_folder, name)
  #     for name in os.listdir(eureka_parent_folder)
  #     if os.path.isdir(os.path.join(eureka_parent_folder, name))
  # ]
  # for folder in eureka_folders:
  #   print(f"folder: {folder.split('/')[-1]}")
  #   get_best_controller_save_path(folder, write=True)
