import random
import os

from agent.config import OvercookedExp1
from agent.gameenv_single import GameEnv_Single, get_env
from agent.mind.agent_new import SimHumanPref


def get_priority_str(priority):
  p = ''
  for inner_list_idx in range(len(priority)):
    inner_list = priority[inner_list_idx]
    inner_p = ''
    for order in inner_list:
      inner_p += order[0]
    if inner_list_idx < len(priority) - 1:
      p += (inner_p + '_')
    else:
      p += inner_p
  return p


random.seed(37)
all_env_seeds = random.sample(range(1, 1001), 50)
print('Overcooked env seeds: ', all_env_seeds)


def main():
  do_other = True  # Whether the agent should work on other soups outside the priority list
  agent_seed = 0
  priorities = [[["David Soup"], ["Alice Soup"]]]
  priority_str = get_priority_str(priorities[0])
  # print('priority str: ', priority_str)
  os.makedirs(f'demonstrations_new/{priority_str}', exist_ok=True)
  for env_seed in all_env_seeds[-3:]:
    priority = random.choice(priorities)
    overcooked_env = get_env(OvercookedExp1, priority=priority, seed=env_seed)
    model = SimHumanPref(seed=agent_seed,
                         do_other_orders=do_other,
                         priority=priority)

    game = GameEnv_Single(env=overcooked_env,
                          max_timesteps=1000,
                          agent_type='sim_h',
                          agent_seed=agent_seed,
                          agent_model=model,
                          play=False)
    os.makedirs(f'demonstrations_new/{priority_str}', exist_ok=True)
    game.execute_agent(
        fps=3,
        sleep_time=0,
        fname=
        f'demonstrations_new/{priority_str}/{priority_str}_demo_env{env_seed}_agent{agent_seed}.txt',
        write=False)


if __name__ == "__main__":
  main()
