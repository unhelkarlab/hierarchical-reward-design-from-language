import os
import time
import torch
import torch.nn.functional as F
from copy import deepcopy
from omegaconf import DictConfig, OmegaConf
import hydra
import heapq
import random

from agent.config import OvercookedExp1
from agent.il_agents.iql.iql_agent import read_datasets
from agent.il_agents.demonstrator_agent import get_priority_str, all_env_seeds
from agent.gameenv_single import GameEnv_Single
from agent.gameenv_single_concur import GameEnv_Single_Concur, get_env
from agent.mind.prompt_local import MOVE_TO_HT, ALL_MOVES
from agent.executor.high import HighTask
from agent.il_agents.agent_base import IL_Agent
from agent.il_agents.iql.iql_agent import IQL_Agent
from agent.executor.low import EnvState
from agent.mind.agent_new import GuidedAgent, AgentSetting
from gym_cooking.utils.replay import Replay

from aic_ml.baselines.IQLearn.agent.softq import SoftQ
from aic_ml.baselines.IQLearn.agent.softq_models import SimpleQNetwork

# num_non_chop_steps = 4


class Futures_Agent(IL_Agent):
  """
  An agent used for testing to see if the Q function learned by IQL actually
  reflects the reward in the real world.
  """

  def __init__(self, use_intent, init_recipe) -> None:
    super().__init__()
    # Previous intent index
    self.prev_intent_idx = 0
    # Index of the previous previous intent
    self.prev_prev_intent_idx = 0
    # The recipe we are following
    self.recipe = init_recipe
    # Index of the current step of the recipe
    self.cur_recipe_index = 0
    # Cumulative reward
    self.c_reward = 0
    # Reward
    self.reward = 0
    # Last state
    self.last_env_tensor = None
    # Whether enough of the future has been simulated
    self.done = False
    # Number of macro action taken before done
    self.num_steps = 0
    # Whether to use the previous intent to predict the next macro action
    self.use_intent = use_intent

  def reset(self, recipe, recipe_index):
    # Reset the agent to start a new sim.
    self.recipe = recipe
    self.cur_recipe_index = recipe_index
    self.c_reward = 0
    self.reward = 0
    self.done = False
    self._task = None
    self.num_steps = 0

  def load_model_direct(self, model):
    self.model = model
    self.model_type = 'iql'

  def compute_reward(self, new_env_tensor, alpha=0.1, gamma=0.99):
    """
    Compute the reward of taking the action at cur_recipe_idx at the current
    state.
    """
    # Get Q value.
    if self.use_intent:
      last_state = (torch.cat(
          (self.last_env_tensor, torch.tensor([self.prev_prev_intent_idx
                                               ])))).unsqueeze(0)
    else:
      last_state = self.last_env_tensor.unsqueeze(0)
    q = self.model.q_net(last_state)
    dist = F.softmax(q / torch.tensor(alpha), dim=1)
    q_val = dist[0][self.prev_intent_idx].item()

    # Get value of the next state
    if self.use_intent:
      cur_state = (torch.cat(
          (new_env_tensor, torch.tensor([self.prev_intent_idx])))).unsqueeze(0)
    else:
      cur_state = new_env_tensor.unsqueeze(0)
    q_next = self.model.q_net(cur_state)
    v = alpha * torch.logsumexp(q_next / alpha, dim=1,
                                keepdim=True)[0][0].item()

    self.reward = q_val - gamma * v
    if self.cur_recipe_index == len(self.recipe) - 1:
      self.c_reward += (gamma**self.num_steps) * q_val
    else:
      self.c_reward += (gamma**self.num_steps) * self.reward

  def step(self, env_state, env_tensor):
    while True:
      if self._task is None:
        self._task = deepcopy(
            MOVE_TO_HT[self.recipe[self.cur_recipe_index].replace(
                'Assemble', 'Prepare')])
        self.last_env_tensor = env_tensor

      state, move, _msg = self._task(env_state)
      if state == HighTask.Working:
        # prev_intent_index will be used to compute q vals
        self.prev_intent_idx = ALL_MOVES.index(
            self.recipe[self.cur_recipe_index].replace('Assemble', 'Prepare'))
        return move, None
      else:
        if self.cur_recipe_index < len(self.recipe) - 1:
          self.cur_recipe_index += 1
        else:
          self.cur_recipe_index = 0
          self.done = True
        self.compute_reward(env_tensor)
        self._task = None
        self.num_steps += 1
        self.prev_prev_intent_idx = self.prev_intent_idx
        if state == HighTask.Failed:
          return (0, 0), None


def run_one_sim(env_state, cur_macro_action, skill):
  '''
  Run one simulation (not currently used)
  '''
  env_state_copy = deepcopy(env_state)
  env_state_copy = env_state
  sim_agent = Futures_Agent()
  sim_agent.reset(skill, 0)
  game = GameEnv_Single(env=env_state_copy.env,
                        max_timesteps=1000,
                        agent_type='futures',
                        prev_macro_idx=cur_macro_action,
                        agent_model=sim_agent,
                        play=False)
  game.execute_agent(fps=3, sleep_time=0, fname='', write=False)
  return game.all_obs, game.all_hl_actions, game.all_next_obs


def compute_rewards(model,
                    state,
                    action,
                    next_state,
                    num_steps,
                    final,
                    alpha=0.1,
                    gamma=0.99):
  '''
  Compute rewards (not currently used)
  '''
  # Get Q value.
  q = model.q_net(state.unsqueeze(0))
  dist = F.softmax(q / torch.tensor(alpha), dim=1)
  q_val = dist[0][action].item()

  # Get value of the next state
  q_next = model.q_net(next_state.unsqueeze(0))
  v = alpha * torch.logsumexp(q_next / alpha, dim=1, keepdim=True)[0][0].item()

  if final:
    return (gamma**num_steps) * q_val
  else:
    return (gamma**num_steps) * (q_val - gamma * v)


class LLM_Futures_Agent(GuidedAgent):
  """
  """

  def __init__(self,
               setting: AgentSetting,
               replay: Replay,
               request_type='guided_hl',
               top_k=-1) -> None:
    super().__init__(setting, replay, request_type)
    # The type of prompts we are using
    self.hl_mode = self.setting.hl_mode
    self.set_skills_for_eval()
    # The recipe we are following
    self.recipe = self.composite_skills[0]
    # Index of the current step of the recipe
    self.cur_recipe_index = 0
    # Agent that simulates futures
    self.futures_agent = Futures_Agent(self.setting.prev_skill,
                                       self.composite_skills[0])
    # How many top composite skills to store as we evaluate them
    self.top_k = top_k
    # The best action sequences for each composite skill
    self.sit_pref_actions = {}
    # Whether we use q evaluation
    self.q_eval = self.setting.q_eval
    # How many top composite skills to use as LLM's input
    self.top_k_llm = self.setting.top_k_llm
    # Whether there is no high level action recommendation
    self.no_hl_rec = self.setting.no_hl_rec

  def set_skills_for_eval(self):
    if self.hl_mode == 'prompt+Qlearned-comp' or self.hl_mode == 'prompt':
      prep_soup_a = [[
          'Chop Lettuce', 'Chop Onion', 'Assemble Alice Ingredients',
          'Cook Alice Soup'
      ],
                     [
                         'Chop Onion', 'Chop Lettuce',
                         'Assemble Alice Ingredients', 'Cook Alice Soup'
                     ]]
      prep_soup_b = [[
          'Chop Tomato', 'Chop Lettuce', 'Assemble Bob Ingredients',
          'Cook Bob Soup'
      ],
                     [
                         'Chop Lettuce', 'Chop Tomato',
                         'Assemble Bob Ingredients', 'Cook Bob Soup'
                     ]]
      prep_soup_c = [[
          'Chop Tomato', 'Chop Onion', 'Assemble Cathy Ingredients',
          'Cook Cathy Soup'
      ],
                     [
                         'Chop Onion', 'Chop Tomato',
                         'Assemble Cathy Ingredients', 'Cook Cathy Soup'
                     ]]
      prep_soup_d = [[
          'Chop Tomato', 'Chop Lettuce', 'Chop Onion',
          'Assemble David Ingredients', 'Cook David Soup'
      ],
                     [
                         'Chop Tomato', 'Chop Onion', 'Chop Lettuce',
                         'Assemble David Ingredients', 'Cook David Soup'
                     ],
                     [
                         'Chop Onion', 'Chop Tomato', 'Chop Lettuce',
                         'Assemble David Ingredients', 'Cook David Soup'
                     ],
                     [
                         'Chop Onion', 'Chop Lettuce', 'Chop Tomato',
                         'Assemble David Ingredients', 'Cook David Soup'
                     ],
                     [
                         'Chop Lettuce', 'Chop Onion', 'Chop Tomato',
                         'Assemble David Ingredients', 'Cook David Soup'
                     ],
                     [
                         'Chop Lettuce', 'Chop Tomato', 'Chop Onion',
                         'Assemble David Ingredients', 'Cook David Soup'
                     ]]
      # manage_burned_soup = [['Putout', 'Drop']]
      get_soup_a = [['Plate Alice Soup', 'Serve Alice Soup']]
      get_soup_b = [['Plate Bob Soup', 'Serve Bob Soup']]
      get_soup_c = [['Plate Cathy Soup', 'Serve Cathy Soup']]
      get_soup_d = [['Plate David Soup', 'Serve David Soup']]
      self.composite_skills = [
          prep_soup_a, prep_soup_b, prep_soup_c, prep_soup_d, get_soup_a,
          get_soup_b, get_soup_c, get_soup_d
      ]
      self.skill_idx_to_name = {
          0: 'prepare and cook Alice soup',
          1: 'prepare and cook Bob soup',
          2: 'prepare and cook Cathy soup',
          3: 'prepare and cook David soup',
          4: 'plate and serve Alice soup',
          5: 'plate and serve Bob soup',
          6: 'plate and serve Cathy soup',
          7: 'plate and serve David soup'
      }
    elif self.hl_mode == 'prompt+Qlearned-skill':
      self.composite_skills = [[[move]] for move in ALL_MOVES]
      self.skill_idx_to_name = {
          i: inner_list[0][0]
          for i, inner_list in enumerate(self.composite_skills)
      }
    else:
      raise NotImplementedError('Prompt type not su')

  def load_model(self, model_path, cfg_path, input_size):
    """
    Load the IQL model. We will use its Q function directly here instead of
    using the policy derived from the Q function.
    """
    obs_dim = input_size
    action_dim = len(ALL_MOVES)
    discrete_obs = False
    q_net_base = SimpleQNetwork
    cfg = OmegaConf.load(cfg_path)
    self.model = SoftQ(cfg, obs_dim, action_dim, discrete_obs, q_net_base)
    self.model.load(model_path)
    self.model_type = 'iql'

    self.futures_agent.load_model_direct(self.model)

  def get_soup_vals(self, env_state: EnvState, action=None, num_to_sample=2):
    '''
    Get the values of each composite skill through simulations

    Args:
    - env_state: the state of the environment
    - action: if we would like to perform simulations preemptively, we can
              finish executing the agent's current action given by this
              parameter, and then start simulating each composite skill
    - num_two_sample: max number of high level action sequences to sample and
                      evaluate for each composite skill
    '''
    start_time = time.time()
    skill_vals = []
    skill_idx = []
    counter = 0
    # Iterate through each composite skill
    for skill_group in self.composite_skills:
      # print(f'==============={skill_idx_to_name[counter]}===============')
      best_val = float('-inf')
      best_idx = -1
      # As each composite skill can have more than one corresponding high level
      # action sequences, randomly sample a few high level action sequeces to
      # simulate
      num_to_sample = min(len(skill_group), num_to_sample)
      skill_indices = random.sample(range(len(skill_group)), num_to_sample)
      for idx in skill_indices:
        # Perform one simulation
        start_time_one_sim = time.time()
        env_state_copy = deepcopy(env_state)
        skill = skill_group[idx]
        if action is not None and action in ALL_MOVES:
          skill_copy = deepcopy(skill)
          skill_copy.insert(0, action)
        else:
          skill_copy = skill
        # print('Skill: ', skill)
        self.simulate_helper(env_state_copy.env, skill_copy, 0)
        # print('Val: ', self.futures_agent.c_reward
        # Update best high level action sequence info if needed
        if self.futures_agent.c_reward > best_val:
          best_val = self.futures_agent.c_reward
          best_idx = idx
        end_time_one_sim = time.time()
        # print('One sim time: ', end_time_one_sim - start_time_one_sim)

        # skill = skill_group[idx]
        # if action is not None and action in ALL_MOVES:
        #   skill_copy = deepcopy(skill)
        #   skill_copy.insert(0, action)
        # else:
        #   skill_copy = skill
        # all_obs, all_actions, all_next_obs = run_one_sim(
        #     env_state, self.prev_intent_idx, skill_copy)
        # c_rewards = 0
        # for i in range(len(all_obs)):
        #   final = (i == len(all_obs) - 1)
        #   c_rewards += compute_rewards(self.model, all_obs[i], all_actions[i],
        #                                all_next_obs[i], i, final)
        # if c_rewards > best_val:
        #   best_val = c_rewards
        #   best_idx = idx
      skill_vals.append(best_val)
      skill_idx.append(best_idx)
      counter += 1
    end_time = time.time()
    # print('Total time diff: ', end_time - start_time)
    return skill_vals, skill_idx

  def simulate_helper(self, environment, soup, action_start_idx):
    self.futures_agent.reset(soup, action_start_idx)
    game = GameEnv_Single(env=environment,
                          max_timesteps=1000,
                          agent_type='futures',
                          agent_model=self.futures_agent,
                          play=False)
    # t0 = time.time()
    game.execute_agent(fps=3, sleep_time=0, fname='', write=False)
    # t1 = time.time()
    # print('Inner one sim time: ', t1 - t0)

  def set_sit_pref(self, action=None):
    super().set_sit_pref()
    env_state = self._last_env
    if env_state is None or not self.q_eval:
      return

    skill_vals, skill_idx = self.get_soup_vals(env_state, action)
    print('skill vals: ', skill_vals)

    if self.top_k == -1:
      top_k = len(skill_vals)
    else:
      top_k = self.top_k
    # Get the soups with largest and second largest q vals as preferences
    top_indices = heapq.nlargest(top_k,
                                 range(len(skill_vals)),
                                 key=skill_vals.__getitem__)
    print('top indices: ', top_indices)

    pref = []
    pref_actions = {}
    for idx in top_indices:
      pref.append(self.skill_idx_to_name[idx])
      pref_actions[self.skill_idx_to_name[idx]] = self.composite_skills[idx][
          skill_idx[idx]]
    self.sit_pref = pref
    self.sit_pref_actions = pref_actions
    print('Sit pref: ', self.sit_pref)
    self.computing_sit_pref = False


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig):
  # Possible values for the parameters
  # Environment variations
  all_priorities = [[['Alice Soup']], [['Bob Soup'], ['Cathy Soup']],
                    [['David Soup'], ['Alice Soup']],
                    [['Alice Soup', 'Bob Soup', 'Cathy Soup', 'David Soup']]]
  all_other_rewards = [0, 5]

  # Agent variations
  all_hl_modes = [
      'prompt', 'prompt+Qtask', 'prompt+Qtaskuser', 'prompt+Qlearned-comp',
      'prompt+Qlearned-skill'
  ]
  all_ll_modes = ['']
  all_prompt_styles = ['lang', 'prog']
  all_num_demos = [1, 3, 10]
  all_text_desc = [True, False]

  # Agent hyperparameters
  all_filter_actions = [True, False]
  all_prev_skill = [True, False]
  all_q_evals = [True, False]
  all_top_k_llm = [2, 3, -1]
  all_no_hl_rec = [True, False]

  # ============================================================================

  # print(OmegaConf.to_yaml(cfg))
  # Legacy hyperparameters
  assert cfg.operation == 'cond'
  assert cfg.fast_il is False
  assert cfg.gen_mode == '5_unranked'
  assert cfg.interpolation is False
  operation = cfg.operation
  fast_il = cfg.fast_il
  gen_mode = cfg.gen_mode
  interpolation = cfg.interpolation

  # Common hyperparameters
  assert cfg.ll_mode == ""
  assert cfg.filter_actions is True
  assert cfg.prev_skill is True
  assert cfg.no_hl_rec is True
  ll_mode = cfg.ll_mode
  filter_actions = cfg.filter_actions
  prev_skill = cfg.prev_skill
  no_hl_rec = cfg.no_hl_rec

  # Create a dictionary of experiment specific parameter
  all_params = {}
  possible_exps = ['D_A', 'B_C', 'DABC']
  for exp in possible_exps:
    if exp in cfg:
      all_params.update(cfg[exp])
      print('params: ', cfg[exp])
  for name, params in all_params.items():
    print('Current model name: ', name)
    # Current parameters (Environment)
    priority = params['priority']
    p = get_priority_str(priority)
    other_reward = 0
    if params['eval_env_seed_start'] is None:
      eval_env_seeds = all_env_seeds[:params['eval_env_seed_end']]
    elif params['eval_env_seed_end'] is None:
      eval_env_seeds = all_env_seeds[params['eval_env_seed_start']:]
    else:
      eval_env_seeds = all_env_seeds[
          params['eval_env_seed_start']:params['eval_env_seed_end']]
    num_soup_orders = params['num_soup_orders']
    map_name = params['game_map']

    # Current parameters (Agent)
    hl_mode = params['hl_mode']
    prompt_style = params['prompt_style']
    num_demos = params['num_demos']
    text_desc = params['text_desc']
    q_eval = params['q_eval']
    top_k_llm = params['top_k_llm']
    model = params[
        'model']  # need to change in call.py to actually use a different model
    if hl_mode == 'prompt+Qtaskuser':
      user_reward = True
    else:
      user_reward = False

    # Create config dictionary
    config = {
        'pref': p,
        'operation': operation,
        'il_model': 'iql_' + str(num_demos),
        'fast_il': fast_il,
        'gen_mode': gen_mode,
        'interpolation': interpolation,
        'q_eval': q_eval,
        'top_k_llm': top_k_llm,
        'no_hl_rec': no_hl_rec,
        'use_intent': prev_skill,
        'skip': filter_actions,
        'num_suffix': 0,
        'hl_mode': hl_mode,
        'll_mode': ll_mode,
        'prompt_style': prompt_style,
        'text_desc': text_desc,
        'model': model,
        'user_reward': user_reward
    }
    # ============================================================================
    # Read a demonstration file to get the input size for neural networks
    directory = f'demonstrations_new/{p}'
    env_seed = 623
    file_names = [f'{p}_demo_env{env_seed}_agent0.txt']
    file_names = [os.path.join(directory, f) for f in file_names]
    expert_dataset, input_size = read_datasets(
        fname_list=file_names,
        concatenate=config['use_intent'],
        filter=config['skip'],
        write=False,
        save_name=
        f'il_agents/iql/{p}/{str(num_demos)}demos/overcooked_ring_{str(num_demos)}demos.pkl'
    )

    # Define some names for file saving
    intent_suffix = 'withprev' if config['use_intent'] else 'noprev'
    skip_suffix = 'skip' if config['skip'] else 'noskip'
    text_desc_note = 'w' if config['text_desc'] else 'wo'
    dir_suffix = '_fast' if config['fast_il'] else ''
    rec = 'noskillrec' if config['no_hl_rec'] else 'skillrec'
    suffixes = ['slow0']
    for env_seed in eval_env_seeds:
      for suffix in suffixes:
        # Init Overcooked Env
        OvercookedExp1.max_num_orders = num_soup_orders
        OvercookedExp1.game_map = map_name
        overcooked_env = get_env(OvercookedExp1,
                                 priority=priority,
                                 seed=env_seed)
        # Init Agent configurations
        agent_set = AgentSetting(mode='LLM_Futures_Agent',
                                 hl_mode=config['hl_mode'],
                                 ll_mode=config['ll_mode'],
                                 prompt_style=config['prompt_style'],
                                 top_k_llm=config['top_k_llm'],
                                 no_hl_rec=config['no_hl_rec'],
                                 text_desc=config['text_desc'],
                                 prev_skill=config['use_intent'],
                                 speed=3,
                                 pref=config['pref'],
                                 operation=config['operation'],
                                 fast_il=config['fast_il'],
                                 gen_mode=config['gen_mode'],
                                 interpolation=config['interpolation'],
                                 q_eval=config['q_eval'],
                                 user_reward=config['user_reward'])
        replay = Replay()  # not used
        # Init agent and load a IL model if needed
        llm_futures_agent = LLM_Futures_Agent(agent_set, replay)
        if config['hl_mode'] == 'prompt':
          llm_futures_agent.set_il_model('none', None)
        else:
          llm_futures_agent.load_model(
              f'il_agents/iql/{p}_{skip_suffix}_{intent_suffix}/{num_demos}demos/best_softq_{num_demos}demos',
              f'il_agents/iql/{p}_{skip_suffix}_{intent_suffix}/{num_demos}demos/config_utf.yaml',
              input_size=input_size)
        # Init save path
        eval_save_dir = f'il_agents/llm/{model}_results/{config["gen_mode"]}/{map_name}_{num_soup_orders}/{text_desc_note}_{p}_{config["operation"]}_iql{num_demos}_top{config["top_k_llm"]}_hl_{skip_suffix}_{intent_suffix}_{rec}_{config["hl_mode"]}_{config["prompt_style"]}{dir_suffix}{config["num_suffix"]}'
        os.makedirs(eval_save_dir, exist_ok=True)
        # Eval
        game = GameEnv_Single_Concur(
            env=overcooked_env,
            max_timesteps=10000,
            agent_type='ai',
            agent_model=llm_futures_agent,
            agent_fps=3,
            game_fps=5,
            play=True,
            write=True,
            fname=os.path.join(eval_save_dir,
                               f'test_env{str(env_seed)}_{suffix}'),
        )
        game.on_execute()


if __name__ == "__main__":
  main()
