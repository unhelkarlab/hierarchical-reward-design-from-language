import ast
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from natsort import natsorted
from scipy.stats import wilcoxon

from HierRL.eval.eval_helper import get_eureka_run_dir
import rw4t.utils as rw4t_utils
from gym_cooking.utils.core import Ingredients


def read_all_successful_subtask_sequences(eureka_run_dir,
                                          env,
                                          log_path='subtask_sequences.txt'):
  all_successful_subtask_sequences = {}
  with open(f'{eureka_run_dir}/{log_path}', 'r') as f:
    lines = f.readlines()
    for line in lines:
      line = line.strip()
      line_parts = line.split(': ')
      run_name = line_parts[0]
      subtask_sequence = ast.literal_eval(line_parts[1])
      subtask_sequence_str = ''
      subtask_count = 0
      for subtask_obj in subtask_sequence:
        if env == 'rw4t':
          if subtask_obj == rw4t_utils.RW4T_State.circle.value:
            subtask_sequence_str += 'Food'
          elif subtask_obj == rw4t_utils.RW4T_State.square.value:
            subtask_sequence_str += 'Medical Kit'
          else:
            raise NotImplementedError
        elif env == 'oc':
          subtask_sequence_str += Ingredients(subtask_obj).name
        if subtask_count < len(subtask_sequence) - 1:
          subtask_sequence_str += ', '
        subtask_count += 1
      all_successful_subtask_sequences[run_name] = subtask_sequence_str

  return all_successful_subtask_sequences


def post_process_successful_drop_sequences_rw4t(successful_drop_sequences,
                                                prefix):
  # Get a set of unique response names and run names
  run_names = list(successful_drop_sequences.keys())
  response_names = set([])
  seed_names = set([])
  for run_name in run_names:
    response_names.add(run_name.split('_')[1])
    seed_names.add(run_name.split('_')[2])

  # Sort the response names and run names in alphanumeric order
  response_names = natsorted(response_names)
  seed_names = natsorted(seed_names)

  # Create a new dictionary of successful drop sequences with sorted keys
  new_successful_drop_sequences = {}
  for seed_name in seed_names:
    for response_name in response_names:
      run_name = f"iter0_{response_name}_{seed_name}"
      assert run_name in successful_drop_sequences
      drop_sequence = successful_drop_sequences[run_name]
      new_successful_drop_sequences[f"{prefix}_{run_name}"] = drop_sequence

  return new_successful_drop_sequences


def post_process_successful_chop_sequences_oc(successful_chop_sequences,
                                              prefix):
  new_successful_chop_sequences = {}
  for run_name, chop_sequence in successful_chop_sequences.items():
    if len(chop_sequence.split(',')) > 3:
      correct_answer = 'the robot chopped more than 3 ingredients'
    else:
      correct_answer = chop_sequence
    new_successful_chop_sequences[f'{prefix}_{run_name}'] = correct_answer
  return new_successful_chop_sequences


def rename_columns(survey_name, all_successful_subtask_seqs):
  # Rename dataframe columns
  all_model_names = list(all_successful_subtask_seqs.keys())
  # print(all_model_names)
  survey_response_path = f"{Path(__file__).parent.parent}/results/" + \
    f"survey_responses/{survey_name}"
  df = pd.read_csv(survey_response_path)
  df = rename_columns_helper(df, all_model_names)
  # print(df.columns)
  # Drop the first two rows
  df = df.iloc[2:].reset_index(drop=True)
  return df


def rename_columns_helper(df, all_model_names):
  new_col_names = []
  # The test question number begins at 5
  start_question_idx = 5
  end_question_idx = start_question_idx + len(all_model_names) - 1
  for col in df.columns:
    q_name_parts = col.split('.')
    if len(q_name_parts) == 1:
      # If the current column name is not in the format of 'Qx.y'
      new_col_names.append(col)
      continue

    # Get the question number and sub-question number
    q_num = int(q_name_parts[0][1:])
    q_sub_num = int(q_name_parts[1])
    if q_num < start_question_idx or q_num > end_question_idx:
      # If the current question number is not in the range
      new_col_names.append(col)
    else:
      model_idx = q_num - start_question_idx
      assert model_idx < len(all_model_names)
      model_name = all_model_names[model_idx]
      new_col_names.append(f"{model_name}.{q_sub_num}")
  df.columns = new_col_names
  # print('New column names: ', new_col_names)
  return df


def filter_responses(df, all_successful_subtask_seqs, filter_q_num, real_q_num):
  filtered_responses = {}
  for model_key, correct_ans in all_successful_subtask_seqs.items():
    col_filter_q = f"{model_key}.{filter_q_num}"
    col_real_q = f"{model_key}.{real_q_num}"
    assert col_filter_q in df.columns and col_real_q in df.columns
    filtered_responses[model_key] = df.apply(
        lambda row: int(row[col_real_q][0])
        if str(row[col_filter_q]) == str(correct_ans) else None,
        axis=1)
  return filtered_responses


def get_flatsa_and_hrd_avg(model_averages):
  # Group the values by method (i.e. flatsa vs hrd)
  flatsa_scores = [
      v for k, v in model_averages.items() if 'flatsa' in k and not np.isnan(v)
  ]
  hrd_scores = [
      v for k, v in model_averages.items() if 'hrd' in k and not np.isnan(v)
  ]
  # Compute averages for each method
  flatsa_avg = np.mean(flatsa_scores)
  flatsa_max = max(flatsa_scores)
  hrd_avg = np.mean(hrd_scores)
  hrd_max = max(hrd_scores)
  print('FlatSA: ')
  print('FlatSA avg: ', flatsa_avg)
  print('FlatSA max: ', flatsa_max)
  print('HRD: ')
  print('HRD avg: ', hrd_avg)
  print('HRD max: ', hrd_max)

  return {
      'flatsa': {
          'avg': flatsa_avg,
          'max': flatsa_max
      },
      'hrd': {
          'avg': hrd_avg,
          'max': hrd_max
      }
  }


def process_survey_rw4t(survey_name, groupby='participant'):
  assert groupby in ['participant', 'seed', 'sample']

  # Process survey_name
  env_name = survey_name.split('_')[1]
  assert env_name == 'rw'
  env_name = 'rw4t'
  seed_idx = int(survey_name.split('_')[2][4:])

  # Get the successful drop sequences for FlatSA
  eureka_run_dir_flatsa = get_eureka_run_dir(env_name=env_name,
                                             pref_type='flatsa',
                                             seed_idx=seed_idx)
  successful_drop_seqs_flatsa = read_all_successful_subtask_sequences(
      eureka_run_dir=eureka_run_dir_flatsa, env=env_name)
  successful_drop_seqs_flatsa = post_process_successful_drop_sequences_rw4t(
      successful_drop_sequences=successful_drop_seqs_flatsa, prefix='flatsa')

  # Get the successful drop sequences for HRD
  eureka_run_dir_high = get_eureka_run_dir(env_name=env_name,
                                           pref_type='high',
                                           seed_idx=seed_idx)
  successful_drop_seqs_high = read_all_successful_subtask_sequences(
      eureka_run_dir=eureka_run_dir_high, env=env_name)
  successful_drop_seqs_high = post_process_successful_drop_sequences_rw4t(
      successful_drop_sequences=successful_drop_seqs_high, prefix='hrd')

  # Rename dataframe columns
  all_successful_drop_seqs = {
      **successful_drop_seqs_flatsa,
      **successful_drop_seqs_high
  }
  df = rename_columns(survey_name, all_successful_drop_seqs)

  # Map each model (for a specific starting position) to a series of ratings
  # while filtering out responses that have incorrect answers
  real_question_nums = [4, 5, 6]
  flat_and_hrd_results = {}
  for real_question_num in real_question_nums:
    # print('=====================================')
    # print('Question number: ', real_question_num)
    filtered_responses = filter_responses(df,
                                          all_successful_drop_seqs,
                                          filter_q_num=2,
                                          real_q_num=real_question_num)
    # print(filtered_responses)

    if groupby == 'participant':
      filtered_df = pd.DataFrame(filtered_responses)

      # Row-wise average of all 'flatsa' columns
      filtered_df[f'q{real_question_num}_flatsa_mean'] = filtered_df[[
          col for col in filtered_df.columns if 'flatsa' in col
      ]].mean(axis=1)
      flat_and_hrd_results[f'q{real_question_num}_flatsa_mean'] = filtered_df[
          f'q{real_question_num}_flatsa_mean']

      # Row-wise average of all 'hrd' columns
      filtered_df[f'q{real_question_num}_hrd_mean'] = filtered_df[[
          col for col in filtered_df.columns if 'hrd' in col
      ]].mean(axis=1)
      flat_and_hrd_results[f'q{real_question_num}_hrd_mean'] = filtered_df[
          f'q{real_question_num}_hrd_mean']

    elif groupby == 'seed' or groupby == 'sample':
      # Group series by model type (strip _sX)
      model_grouped = defaultdict(list)
      for key, series in filtered_responses.items():
        model_type = "_".join(
            key.split("_")[:-1])  # everything except the last "_sX"
        model_grouped[model_type].append(series)
      # print('Model grouped: ', model_grouped)
      if groupby == 'sample':
        model_grouped_flattened = {}
        for model_type, series_list in model_grouped.items():
          model_grouped_flattened[model_type] = pd.concat(series_list,
                                                          ignore_index=True)
        flat_and_hrd_results[real_question_num] = model_grouped_flattened
        continue

      # Concatenate series within each model type and compute mean
      model_averages = {}
      for model_type, series_list in model_grouped.items():
        all_values = pd.concat(series_list, ignore_index=True)
        numeric_values = all_values.dropna().astype(float)
        model_averages[model_type] = numeric_values.mean()
      print('Model averages: ', model_averages)

      # Group the values by method (i.e. flatsa vs hrd)
      flat_and_hrd_results[real_question_num] = get_flatsa_and_hrd_avg(
          model_averages)
  return flat_and_hrd_results


def process_survey_oc(survey_name, groupby='participant'):
  assert groupby in ['participant', 'seed', 'sample']

  # Process survey_name
  env_name = survey_name.split('_')[1]
  assert env_name == 'oc'
  seed_idx = int(survey_name.split('_')[2][4:])

  # Get the successful drop sequences for FlatSA
  eureka_run_dir_flatsa = get_eureka_run_dir(env_name=env_name,
                                             pref_type='flatsa',
                                             seed_idx=seed_idx)
  successful_chop_seqs_flatsa = read_all_successful_subtask_sequences(
      eureka_run_dir=eureka_run_dir_flatsa, env=env_name)
  successful_chop_seqs_flatsa = post_process_successful_chop_sequences_oc(
      successful_chop_seqs_flatsa, prefix='flatsa')
  # print('Successful chop sequences for FlatSA: ', successful_chop_seqs_flatsa)

  # Get the successful drop sequences for HRD
  eureka_run_dir_high = get_eureka_run_dir(env_name=env_name,
                                           pref_type='high',
                                           seed_idx=seed_idx)
  successful_chop_seqs_high = read_all_successful_subtask_sequences(
      eureka_run_dir=eureka_run_dir_high, env=env_name)
  successful_chop_seqs_high = post_process_successful_chop_sequences_oc(
      successful_chop_seqs_high, prefix='hrd')
  # print('Successful chop sequences for HRD: ', successful_chop_seqs_high)

  # Rename dataframe columns
  all_successful_chop_seqs = {
      **successful_chop_seqs_flatsa,
      **successful_chop_seqs_high
  }
  df = rename_columns(survey_name, all_successful_chop_seqs)

  # Map each model (for a specific starting position) to a series of ratings
  # while filtering out responses that have incorrect answers
  real_question_num = 4
  filtered_responses = filter_responses(df,
                                        all_successful_chop_seqs,
                                        filter_q_num=2,
                                        real_q_num=real_question_num)
  # print(filtered_responses)

  flat_and_hrd_results = {}
  if groupby == 'participant':
    filtered_df = pd.DataFrame(filtered_responses)

    # Row-wise average of all 'flatsa' columns
    filtered_df[f'q{real_question_num}_flatsa_mean'] = filtered_df[[
        col for col in filtered_df.columns if 'flatsa' in col
    ]].mean(axis=1)
    flat_and_hrd_results[f'q{real_question_num}_flatsa_mean'] = filtered_df[
        f'q{real_question_num}_flatsa_mean']

    # Row-wise average of all 'hrd' columns
    filtered_df[f'q{real_question_num}_hrd_mean'] = filtered_df[[
        col for col in filtered_df.columns if 'hrd' in col
    ]].mean(axis=1)
    flat_and_hrd_results[f'q{real_question_num}_hrd_mean'] = filtered_df[
        f'q{real_question_num}_hrd_mean']

  elif groupby == 'seed':
    # Compute mean for each model
    model_averages = {}
    for model_type, series_list in filtered_responses.items():
      numeric_values = series_list.dropna().astype(float)
      model_averages[model_type] = numeric_values.mean()
    print('Model averages: ', model_averages)
    flat_and_hrd_results[real_question_num] = get_flatsa_and_hrd_avg(
        model_averages)
  elif groupby == 'sample':
    flat_and_hrd_results[real_question_num] = filtered_responses

  return flat_and_hrd_results


def process_all_results_groupby_seeds_rw4t(all_results_dicts):
  # Flatten each dictionary
  flattened_rows = []
  for entry in all_results_dicts:
    flat_row = {}
    for q_num, methods in entry.items():
      for method, scores in methods.items():
        for stat, value in scores.items():
          col_name = f"{q_num}_{method}_{stat}"
          flat_row[col_name] = value
    flattened_rows.append(flat_row)

  # Create DataFrame
  df = pd.DataFrame(flattened_rows)

  # Compute mean and std
  mean_series = df.mean()
  std_series = df.std()

  # Combine into one summary DataFrame
  summary_df = pd.DataFrame({'mean': mean_series, 'std': std_series})
  return summary_df


def process_demographic_data(survey_names):
  # Read and concatenate all survey dataframes
  all_df = pd.DataFrame(columns=['Q2.3', 'Q2.4'])
  for survey in survey_names:
    survey_response_path = f"{Path(__file__).parent.parent}/results/" + \
      f"survey_responses/{survey}"
    df = pd.read_csv(survey_response_path)
    df = df.iloc[2:].reset_index(drop=True)
    all_df = pd.concat([all_df, df[['Q2.3', 'Q2.4']]], ignore_index=True)
  all_df = all_df.rename(columns={'Q2.3': 'Age', 'Q2.4': 'Sex'})

  # Get age data
  median_age = all_df['Age'].median()
  min_age = all_df['Age'].min()
  max_age = all_df['Age'].max()
  print(f"median_age: {median_age}")
  print(f"min_age: {min_age}")
  print(f"max_age: {max_age}")

  # Get sex data
  print(all_df['Sex'].value_counts())


if __name__ == '__main__':
  # Process demographic data
  # survey_names = [
  #     'hrd_rw_seed0_May 8, 2025_15.22.csv',
  #     'hrd_rw_seed1_May 8, 2025_16.00.csv',
  #     'hrd_rw_seed2_May 8, 2025_15.17.csv',
  #     'hrd_oc_seed0_May 8, 2025_17.06.csv',
  #     'hrd_oc_seed1_May 8, 2025_20.02.csv', 'hrd_oc_seed2_May 8, 2025_20.37.csv'
  # ]
  # process_demographic_data(survey_names)

  # Get results from one survey
  # survey_name = 'hrd_rw_seed2_May 8, 2025_12.58.csv'
  # assert 'rw' in survey_name or 'oc' in survey_name
  # if 'rw' in survey_name:
  #   process_survey_rw4t(survey_name)
  # else:
  #   process_survey_oc(survey_name)

  # Get results from multiple surveys
  survey_names = [
      'hrd_rw_seed0_May 8, 2025_15.22.csv',
      'hrd_rw_seed1_May 8, 2025_16.00.csv', 'hrd_rw_seed2_May 8, 2025_15.17.csv'
  ]
  # survey_names = [
  #     'hrd_oc_seed0_May 8, 2025_17.06.csv',
  #     'hrd_oc_seed1_May 8, 2025_20.02.csv', 'hrd_oc_seed2_May 8, 2025_20.37.csv'
  # ]
  assert all('rw' in n for n in survey_names) or all(
      'oc' in n for n in survey_names), "All surveys must be either rw or oc"

  groupby = 'sample'
  if groupby == 'participant':
    all_results_list = []
    for survey_name in survey_names:
      print('=====================================')
      print(survey_name)
      if 'rw' in survey_name:
        results_df = pd.DataFrame(process_survey_rw4t(survey_name))
      else:
        results_df = pd.DataFrame(process_survey_oc(survey_name))
      print(results_df)
      all_results_list.append(results_df)

    all_results_df = pd.concat(all_results_list, axis=0, ignore_index=True)
    print('=====================================')
    print(all_results_df)

    print('Statistical significance results: ')
    if all('rw' in n for n in survey_names):
      stat, p = wilcoxon(all_results_df['q4_flatsa_mean'],
                         all_results_df['q4_hrd_mean'],
                         zero_method='zsplit')
      print(f'Persistence. stat: {stat}, p: {p}')

      stat, p = wilcoxon(all_results_df['q5_flatsa_mean'],
                         all_results_df['q5_hrd_mean'],
                         zero_method='zsplit')
      print(f'Safety. stat: {stat}, p: {p}')

      stat, p = wilcoxon(all_results_df['q6_flatsa_mean'],
                         all_results_df['q6_hrd_mean'],
                         zero_method='zsplit')
      print(f'Overall. stat: {stat}, p: {p}')

    elif all('oc' in n for n in survey_names):
      stat, p = wilcoxon(all_results_df['q4_flatsa_mean'],
                         all_results_df['q4_hrd_mean'],
                         zero_method='zsplit')
      print(f'Chopping. stat: {stat}, p: {p}')

    # Show mean and std
    means = all_results_df.mean()
    stds = all_results_df.std()
    summary = pd.DataFrame({
        'mean': all_results_df.mean(),
        'std': all_results_df.std()
    })
    print(summary)
  elif groupby == 'sample':
    all_results_dict = defaultdict(lambda: defaultdict(list))
    count = 0
    for survey_name in survey_names:
      if 'rw' in survey_name:
        results_dict = process_survey_rw4t(survey_name, groupby=groupby)
      else:
        results_dict = process_survey_oc(survey_name, groupby=groupby)

      for question_num in results_dict:
        for response_name, response_ratings in results_dict[question_num].items(
        ):
          all_results_dict[question_num][f"{count}_{response_name}"].append(
              response_ratings)
      count += 1

    final_all_results_dict = defaultdict(dict)
    for question_num in all_results_dict:
      print('=====================================')
      print(question_num)
      for response_name, response_ratings_list in all_results_dict[
          question_num].items():
        final_all_results_dict[question_num][response_name] = pd.concat(
            response_ratings_list, ignore_index=True)
      df = pd.DataFrame(final_all_results_dict[question_num])
      column_means = df.mean()
      print(column_means)

      flatsa_count = (column_means.index.str.contains('flatsa')
                      & (column_means == 5.0)).sum()
      hrd_count = (column_means.index.str.contains('hrd')
                   & (column_means == 5.0)).sum()
      print('Flat policies with perfect user ratings: ', flatsa_count)
      print('HRD policies with perfect user ratings: ', hrd_count)
