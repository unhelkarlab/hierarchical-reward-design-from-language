import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorboard.backend.event_processing.event_accumulator import \
  EventAccumulator


def extract_tb_data(log_dir, tag='eval/mean_reward'):
  '''
  Extracts scalar values from TensorBoard event files in a directory.

  Args:
      log_dir (str): Path to the directory containing TensorBoard log files.
      tag (str): Scalar tag to extract (e.g., 'eval/mean_reward').

  Returns:
      pd.DataFrame: DataFrame containing step and reward values.
  '''
  # Get the log file names from the directory
  event_files = [f for f in os.listdir(log_dir) if 'tfevents' in f]

  all_data = []
  for file in event_files:
    # Set up event accumulator to load all events in the file
    event_path = os.path.join(log_dir, file)
    event_acc = EventAccumulator(event_path)
    event_acc.Reload()

    # Check if the specified tag is in the event accumulator
    if tag not in event_acc.Tags()['scalars']:
      continue

    # Get all the events associated with the tag and make it into a df
    steps = []
    values = []
    for event in event_acc.Scalars(tag):
      steps.append(event.step)
      values.append(event.value)
    df = pd.DataFrame({"step": steps, "reward": values})
    all_data.append(df)

  return all_data


def align_and_average(dataframes):
  '''
  Aligns multiple dataframes based on 'step' and computes the mean reward.

  Args:
      dataframes (list of pd.DataFrame): List of dataframes containing step and
      reward columns.

  Returns:
      pd.DataFrame: DataFrame with mean and standard deviation of rewards.
  '''
  # Concatenate dataframes row wise and get the average for each step
  merged_df = pd.concat(dataframes,
                        axis=0).groupby("step").agg(["mean",
                                                     "std"]).reset_index()
  merged_df.columns = ["step", "reward_mean", "reward_std"]
  return merged_df


def plot_eval_rewards(log_dir_groups, labels):
  '''
  Plot the avergae cumulative rewards for each directory group.
  '''
  assert len(log_dir_groups) == len(labels)
  palette = sns.color_palette('colorblind', len(log_dir_groups))

  plt.figure(figsize=(10, 5))
  for log_dirs, color, label in zip(log_dir_groups, palette, labels):
    # Read data from multiple log files
    all_dfs = []
    for log_dir in log_dirs:
      dfs = extract_tb_data(log_dir)
      all_dfs.extend(dfs)
    # Compute the mean reward over multiple logs
    mean_df = align_and_average(all_dfs)
    # Plot the mean reward with standard deviation
    plt.plot(mean_df["step"], mean_df["reward_mean"], label=label, color=color)
    plt.fill_between(mean_df["step"],
                     mean_df["reward_mean"] - mean_df["reward_std"],
                     mean_df["reward_mean"] + mean_df["reward_std"],
                     alpha=0.2)
  plt.xlabel("Steps")
  plt.ylabel("Return")
  plt.title("Average Episodic Return Over Five Runs")
  plt.legend()
  plt.show()


if __name__ == "__main__":
  # Define log groups and labels
  log_dir_prefix_1 = f'{Path(__file__).parent}/../results/rw4t/rw4t_6by6_4obj_dqn/hl_model_w_hlpref_wo_llpref_ll_model_wo_llpref/tb_logs'
  log_dir_prefix_2 = f'{Path(__file__).parent}/../results/rw4t/rw4t_6by6_4obj_dqn/hl_model_wo_hlpref_wo_llpref_ll_model_wo_llpref/tb_logs'
  log_dir_group_1 = [
      f'{log_dir_prefix_1}/hl_model_w_hlpref_wo_llpref_ll_model_wo_llpref_mdp_no_beta_lr0.0001_exp0.2_pbrs_26_1',
      f'{log_dir_prefix_1}/hl_model_w_hlpref_wo_llpref_ll_model_wo_llpref_mdp_no_beta_lr0.0001_exp0.2_pbrs_115_1',
      f'{log_dir_prefix_1}/hl_model_w_hlpref_wo_llpref_ll_model_wo_llpref_mdp_no_beta_lr0.0001_exp0.2_pbrs_282_1',
      f'{log_dir_prefix_1}/hl_model_w_hlpref_wo_llpref_ll_model_wo_llpref_mdp_no_beta_lr0.0001_exp0.2_pbrs_655_1',
      f'{log_dir_prefix_1}/hl_model_w_hlpref_wo_llpref_ll_model_wo_llpref_mdp_no_beta_lr0.0001_exp0.2_pbrs_760_1',
  ]
  log_dir_group_2 = [
      f'{log_dir_prefix_2}/hl_model_wo_hlpref_wo_llpref_ll_model_wo_llpref_mdp_no_beta_lr0.0001_exp0.2_nopbrs_26_1',
      f'{log_dir_prefix_2}/hl_model_wo_hlpref_wo_llpref_ll_model_wo_llpref_mdp_no_beta_lr0.0001_exp0.2_nopbrs_115_1',
      f'{log_dir_prefix_2}/hl_model_wo_hlpref_wo_llpref_ll_model_wo_llpref_mdp_no_beta_lr0.0001_exp0.2_nopbrs_282_1',
      f'{log_dir_prefix_2}/hl_model_wo_hlpref_wo_llpref_ll_model_wo_llpref_mdp_no_beta_lr0.0001_exp0.2_nopbrs_655_1',
      f'{log_dir_prefix_2}/hl_model_wo_hlpref_wo_llpref_ll_model_wo_llpref_mdp_no_beta_lr0.0001_exp0.2_nopbrs_760_1',
  ]
  log_dir_groups = [log_dir_group_1, log_dir_group_2]
  labels = ["With option-level PBRS", "Without option-level PBRS"]

  plot_eval_rewards(log_dir_groups, labels)
