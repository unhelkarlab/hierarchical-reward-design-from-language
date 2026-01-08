import ast
import pandas as pd
import numpy as np
import utils as rw4t_utils


def read_demonstrations(fname, filter):
  df = pd.read_csv(fname, delimiter='; ', engine='python')
  df['prev_action'] = df['action'].shift(1)
  df['next_action'] = df['action'].shift(-1)
  df.iloc[0, df.columns.get_loc('prev_action')] = 0
  df.iloc[len(df) - 1, df.columns.get_loc('next_action')] = 0

  df['prev_macro_action'] = df['macro_action'].shift(1)
  df['prev_macro_idx'] = df['macro_idx'].shift(1)
  df['next_macro_idx'] = df['macro_idx'].shift(-1)
  df.iloc[0, df.columns.get_loc('prev_macro_idx')] = 0
  df.iloc[0, df.columns.get_loc('prev_macro_action'
                                )] = rw4t_utils.get_enum_name_by_value(
                                    rw4t_utils.RW4T_HL_Actions, 0)
  df.iloc[len(df) - 1, df.columns.get_loc('next_macro_idx')] = 0

  def flatten_state(state):
    state_info = state.split('|')
    state_map = np.array(ast.literal_eval(state_info[0])).flatten()
    state_pos = np.array(ast.literal_eval(state_info[1]))
    state_holding = np.array([ast.literal_eval(state_info[2])])
    state = np.concatenate((state_map, state_pos, state_holding))
    return state

  df['f_state'] = df['state'].apply(flatten_state)
  df['f_next_state'] = df['next_state'].apply(flatten_state)

  if filter:
    # Filter dataframe to skip rows when the agent is exeucting low-level
    # actions
    filtered_df = df[df['macro_idx'] != df['macro_idx'].shift()]
    filtered_df = filtered_df.reset_index(drop=True)
    # Shift the features column one row up
    filtered_df['next_features_temp'] = filtered_df['features'].shift(-1)
    # Set the entry for the last feature
    filtered_df.iloc[
        -1, filtered_df.columns.get_loc('next_features_temp')] = df.iloc[
            -1, df.columns.get_loc('next_features')]
    filtered_df['next_features'] = filtered_df['next_features_temp']
  else:
    filtered_df = df

  def convert_to_nparray(feature):
    return np.array(ast.literal_eval(feature))

  filtered_df['features'] = filtered_df['features'].apply(convert_to_nparray)
  filtered_df['next_features'] = filtered_df['next_features'].apply(
      convert_to_nparray)
  # filtered_df = filtered_df.drop(
  #     ['state', 'f_state', 'next_state', 'f_next_state'], axis=1)
  # filtered_df.to_csv('test.csv')
  return filtered_df


def read_multiple_files(fname_list, filter):
  demos = {}
  for fname in fname_list:
    demos[fname] = read_demonstrations(fname, filter)
  return demos


# read_demonstrations(fname='rw4t_demos_bfs/6by6/v5_train/bfs_control_26.txt',
#                     filter=True)
