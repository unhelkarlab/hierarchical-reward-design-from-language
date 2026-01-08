import ast
import pandas as pd
import numpy as np


def read_demonstrations(fname, filter):
  df = pd.read_csv(fname, delimiter='; ', engine='python')
  df['prev_macro_action'] = df['macro_action'].shift(1)
  df['prev_macro_idx'] = df['macro_idx'].shift(1)
  df.iloc[0, df.columns.get_loc('prev_macro_idx')] = 0

  def flatten_state(state):
    state_info = state.split('|')
    state_map = np.array(ast.literal_eval(state_info[0])).flatten()
    state_orders = np.array(ast.literal_eval(state_info[1]))
    state_holdings = np.array(ast.literal_eval(state_info[2]))
    state = np.concatenate((state_map, state_orders, state_holdings))
    return state

  df['f_state'] = df['state'].apply(flatten_state)
  df['f_next_state'] = df['next_state'].apply(flatten_state)

  if filter:
    # Filter dataframe to skip rows when the agent is exeucting low-level
    # actions
    filtered_df = df[df['new_macro'] == True]
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
  return filtered_df


def read_multiple_files(fname_list, filter):
  demos = {}
  for fname in fname_list:
    demos[fname] = read_demonstrations(fname, filter)
  return demos


# read_demonstrations(fname='demonstrations_new/B_C/B_C_demo_env37_agent0.txt')
