import os

from agent.config import OvercookedExp1
from agent.mind.agent_new import get_agent, AgentSetting
from agent.gameenv_single_concur import GameEnv_Single_Concur, get_env
from gym_cooking.utils.replay import Replay
from agent.il_agents.demonstrator_agent import get_priority_str, all_env_seeds
from agent.il_agents.iql.iql_agent import IQL_Agent, read_datasets

priority = [['Bob Soup'], ['Cathy Soup']]
p = get_priority_str(priority)
p_llm = ''
config = {
    'pref': p_llm,
    'operation': 'multiply',
    'il_model': 'none',
    'fast_il': False,
    'gen_mode':
    '5_unranked',  # '5_ranked', '5_unranked', 'all_yes_no', 'top, 'all_yes_no_include_false',
    'interpolation': False
}
model = 'gpt4omini'


def get_input_size():
  directory = f'demonstrations/{p}'
  env_seed = all_env_seeds[0]
  file_names = [f'{p}_demo_env{env_seed}_agent0.txt']
  file_names = [os.path.join(directory, f) for f in file_names]
  # print('Train files: ', file_names)
  num_demos = config['il_model'].split('_')[1]
  _, input_size = read_datasets(
      fname_list=file_names,
      write=False,
      save_name=
      f'il_agents/iql/{p}/{str(num_demos)}demos/overcooked_ring_{str(num_demos)}demos.pkl'
  )
  return input_size


def main():
  eval_env_seeds = all_env_seeds[-3:]
  suffixes = ['slow0']
  for env_seed in eval_env_seeds:
    for suffix in suffixes:
      overcooked_env = get_env(OvercookedExp1, priority, seed=env_seed)
      agent_set = AgentSetting(mode='GuidedAgent',
                               speed=3,
                               pref=config['pref'],
                               operation=config['operation'],
                               fast_il=config['fast_il'],
                               gen_mode=config['gen_mode'],
                               interpolation=config['interpolation'])
      replay = Replay()
      agent_model = get_agent(agent_set, replay)

      if 'iql' in config['il_model']:
        iql_agent = IQL_Agent()
        num_demos = config['il_model'].split('_')[1]
        iql_agent.load_model(
            f'il_agents/iql/{p}/{num_demos}demos/best_softq_{num_demos}demos',
            f'il_agents/iql/{p}/{num_demos}demos/config_utf.yaml',
            input_size=get_input_size())
        agent_model.set_il_model('iql', iql_agent.model)
      elif config['il_model'] == 'none':
        agent_model.set_il_model('none', None)

      if config['il_model'] == 'none':
        if p_llm == '':
          eval_save_dir = f'il_agents/llm/{model}_results/{config["gen_mode"]}/wo_{p}_{config["il_model"]}'
        else:
          eval_save_dir = f'il_agents/llm/{model}_results/{config["gen_mode"]}/w_{p}_{config["il_model"]}'
      else:
        dir_suffix = '_fast' if config['fast_il'] else ''
        if p_llm == '':
          eval_save_dir = f'il_agents/llm/{model}_results/{config["gen_mode"]}/wo_{p}_{config["operation"]}_{config["il_model"]}{dir_suffix}'
        else:
          eval_save_dir = f'il_agents/llm/{model}_results/{config["gen_mode"]}/w_{p}_{config["operation"]}_{config["il_model"]}{dir_suffix}'
      os.makedirs(eval_save_dir, exist_ok=True)
      game = GameEnv_Single_Concur(
          env=overcooked_env,
          max_timesteps=10000,
          agent_type='ai',
          agent_model=agent_model,
          agent_fps=3,
          game_fps=5,
          play=True,
          write=False,
          fname=os.path.join(eval_save_dir,
                             f'test_env{str(env_seed)}_{suffix}'),
      )
      game.on_execute()


if __name__ == "__main__":
  main()
