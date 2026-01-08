import os

from agent.mind.agent_new import get_agent, AgentSetting
from agent.gameenv_single_concur import GameEnv_Single_Concur, get_env
from agent.config import OvercookedExp1
from gym_cooking.utils.replay import Replay
from agent.il_agents.demonstrator_agent import get_priority_str, all_env_seeds
from agent.il_agents.iql.iql_agent import IQL_Agent, read_datasets

priority = [['David Soup'], ['Alice Soup']]
p = get_priority_str(priority)
p_llm = ''
model = 'gpt4o'
config = {
    'pref': p_llm,
    'operation': 'multiply',
    'il_model': 'none',
    'interpolation': False
}


def get_input_size():
  directory = f'demonstrations/{p}'
  env_seed = all_env_seeds[0]
  file_names = [f'{p}_demo_env{env_seed}_agent0.txt']
  file_names = [os.path.join(directory, f) for f in file_names]
  # print('Train files: ', file_names)
  _, input_size = read_datasets(
      fname_list=file_names,
      write=False,
      save_name=
      f'il_agents/iql/{p}/{str(num_demos)}demos/overcooked_ring_{str(num_demos)}demos.pkl'
  )
  return input_size


user_reward = False
eval_env_seeds = all_env_seeds[-3:]
suffixes = ['slow0']
for env_seed in eval_env_seeds:
  for suffix in suffixes:
    overcooked_env = get_env(OvercookedExp1, priority, seed=env_seed)
    agent_set = AgentSetting(mode='HierGuidedAgent',
                             hl_mode='',
                             ll_mode='',
                             prompt_style='',
                             speed=3,
                             pref=config['pref'],
                             operation=config['operation'],
                             interpolation=config['interpolation'],
                             user_reward=user_reward)
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
        eval_save_dir = f'il_agents/llm/{model}_results/hla/wo_{p}_{config["il_model"]}'
      else:
        eval_save_dir = f'il_agents/llm/{model}_results/hla/w_{p}_{config["il_model"]}'
    else:
      if p_llm == '':
        eval_save_dir = f'il_agents/llm/{model}_results/hla/wo_{p}_{config["operation"]}_{config["il_model"]}'
      else:
        eval_save_dir = f'il_agents/llm/{model}_results/hla/w_{p}_{config["operation"]}_{config["il_model"]}'
    os.makedirs(eval_save_dir, exist_ok=True)
    game = GameEnv_Single_Concur(
        env=overcooked_env,
        max_timesteps=10000,
        agent_type='ai',
        agent_model=agent_model,
        agent_fps=3,
        game_fps=5,
        p_str=p_llm,
        play=True,
        write=True,
        fname=os.path.join(eval_save_dir, f'test_env{str(env_seed)}_{suffix}'),
    )
    game.on_execute()
