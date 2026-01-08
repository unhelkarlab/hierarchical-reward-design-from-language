import os
import ast
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from agent.il_agents.demonstrator_agent import all_env_seeds


def parse_info(s):
  # Replace 'CustomType' with an empty string and convert to a dictionary
  s = re.sub(r"\[.*?\]", "None", s)

  try:
    return ast.literal_eval(s)
  except (ValueError, SyntaxError):
    return {}


algos = [
    'llm_wo_p_cond_iql0_top-1_hl_skip_withprev_noskillrec_prompt_lang0'
]  # 'demo', 'iql', 'llm_wo_p_cond_iql1_top3_hl_skip_withprev_noskillrec_prompt+Qlearned-comp_prog0'
bc_algos = ['iql']
llm_models = ['gpt4o', 'none']  # 'fastmind', 'llama3', 'gpt4o', 'gpt4omini'
p = 'D_A'
map_name = 'ring'
num_soups = 3
gen_mode = '5_unranked'  # '5_unranked', 'hla'
num_demos = [0, 1, 3, 5, 10, 'E', 'LM']  # 1, 3, 5, 10, 'E'
eval_seeds = all_env_seeds[-5:]

data = {'Algo': [], 'Num-demos': [], 'Reward': []}

for algo in algos:
  for llm_model in llm_models:
    for num_demo in num_demos:
      if ('llm_' in algo and llm_model
          != 'none') and ('add' not in algo and 'mul' not in algo
                          and 'cond' not in algo) and (num_demo == 'LM'):
        knows_pref = 'wo' if 'wo' in algo else 'w'
        midfix = gen_mode + '/' if llm_model == 'gpt4o' or llm_model == 'gpt4omini' else ''
        folder_name = f'il_agents/llm/{llm_model}_results/{midfix}{knows_pref}_{p}_{algo.split("_", maxsplit=3)[-1]}'
      elif ('llm_' in algo and llm_model != 'none') and (
          'add' in algo or 'mul' in algo
          or 'cond' in algo) and (algo.split("_")[4][3:] == str(num_demo)):
        if algo.split("_")[3] == 'add' or algo.split("_")[3] == 'cond':
          operation = algo.split("_")[3]
        elif algo.split("_")[3] == 'mul':
          operation = 'multiply'
        else:
          raise NotImplementedError
        if 'fast' in algo and (llm_model == 'gpt4o'
                               or llm_model == 'gpt4omini'):
          suffix = '_fast'
        else:
          suffix = ''
        if llm_model == 'gpt4o' or llm_model == 'gpt4omini':
          midfix = gen_mode + '/'
        else:
          midfix = ''
        folder_name = f'il_agents/llm/{llm_model}_results/{midfix}{map_name}_{num_soups}/{algo.split("_")[1]}_{p}_{operation}_{algo.split("_", maxsplit=4)[-1]}{suffix}'
      elif algo == 'demo' and llm_model == 'none' and num_demo == 'E':
        folder_name = f'demonstrations_new/{p}'
      elif algo in bc_algos and llm_model == 'none' and (num_demo != 'E'
                                                         and num_demo != 'LM'):
        folder_name = f'il_agents/{algo}/{p}/{str(num_demo)}demos'
      else:
        continue
      print('folder name: ', folder_name)

      # List all files and directories
      all_results = os.listdir(folder_name)
      print('all results: ', all_results)

      if 'demonstrations' in folder_name:
        out_dist_results = [
            os.path.join(folder_name, f) for f in all_results
            if '.pth' not in f and int(f.split('_')[-2][3:]) in eval_seeds
        ]
      else:
        out_dist_results = [
            os.path.join(folder_name, f) for f in all_results
            if ('.pth' not in f and '.yaml' not in f and '.pkl' not in f
                and 'best' not in f and int(f.split('_')[-2][3:]) in eval_seeds)
        ]

      cur_reward_list = []
      for log_f in out_dist_results:
        df = pd.read_csv(log_f, delimiter='; ', engine='python')
        reward = df['reward'].sum()
        df['info_dict'] = df['info'].apply(parse_info)
        r_reward = df['info_dict'].apply(lambda x: x['r']).sum()
        p_reward = df['info_dict'].apply(lambda x: x['p']).sum()
        print(f'{log_f}: {reward}, {r_reward}, {p_reward}')
        # print(df['reward'].unique())

        data['Algo'].append(algo)
        data['Num-demos'].append(num_demo)
        data['Reward'].append(reward)
        cur_reward_list.append(reward)

      avg = sum(cur_reward_list) / len(cur_reward_list)
      std = np.std(cur_reward_list)
      print(f'Algo: {algo}, llm: {llm_model}, demo: {num_demo}: {avg} +- {std}')

df = pd.DataFrame(data)
sns.set_theme(style="ticks", palette="pastel")
sns.boxplot(x="Num-demos", y="Reward", hue="Algo", data=df)
plt.show()
