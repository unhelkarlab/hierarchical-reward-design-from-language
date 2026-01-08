import os

from agent.mind.agent_new import QAgent_Overcooked
from agent.gameenv_single import GameEnv_Single, get_env
from agent.config import OvercookedExp1
from agent.il_agents.demonstrator_agent import all_env_seeds, get_priority_str

priority = [['Bob Soup'], ['Cathy Soup']]
p = get_priority_str(priority)
user_reward = False
user_reward_suffix = 'user' if user_reward else 'task'
os.makedirs(f'il_agents/llm/handcrafted_results/{p}/{user_reward_suffix}',
            exist_ok=True)
eval_env_seeds = all_env_seeds[-3:]
total_c_rewards = 0
for env_seed in eval_env_seeds:
  overcooked_env = get_env(OvercookedExp1, priority, seed=env_seed)
  q_agent = QAgent_Overcooked(user_reward)
  game = GameEnv_Single(env=overcooked_env,
                        max_timesteps=1000,
                        agent_type='ai',
                        agent_model=q_agent,
                        play=False)
  c_rewards = game.execute_agent(
      fps=3,
      sleep_time=0,
      fname=
      f'il_agents/llm/handcrafted_results/{p}/{user_reward_suffix}/test_env{env_seed}_0',
      write=False)
  total_c_rewards += c_rewards
print('Avg reward: ', total_c_rewards / len(eval_env_seeds))
