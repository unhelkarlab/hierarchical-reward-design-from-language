import os
import torch
import torch.nn.functional as F
from copy import deepcopy
from omegaconf import OmegaConf

from agent.config import OvercookedExp1
from agent.il_agents.iql.iql_agent import read_datasets
from agent.il_agents.demonstrator_agent import get_priority_str
from agent.gameenv_single import GameEnv_Single, get_env
from agent.mind.prompt_local import MOVE_TO_HT, ALL_MOVES
from agent.executor.high import HighTask
from agent.il_agents.agent_base import IL_Agent

from aic_ml.baselines.IQLearn.agent.softq import SoftQ
from aic_ml.baselines.IQLearn.agent.softq_models import SimpleQNetwork

# Action sequences of completing each soup
soup_a = [
    'Chop Lettuce', 'Chop Onion', 'Prepare Alice Ingredients',
    'Cook Alice Soup', 'Plate Alice Soup', 'Serve Alice Soup'
]
soup_b = [
    'Chop Tomato', 'Chop Lettuce', 'Prepare Bob Ingredients', 'Cook Bob Soup',
    'Plate Bob Soup', 'Serve Bob Soup'
]
soup_c = [
    'Chop Tomato', 'Chop Onion', 'Prepare Cathy Ingredients', 'Cook Cathy Soup',
    'Plate Cathy Soup', 'Serve Cathy Soup'
]
soup_d = [
    'Chop Tomato', 'Chop Lettuce', 'Chop Onion', 'Prepare David Ingredients',
    'Cook David Soup', 'Plate David Soup', 'Serve David Soup'
]


class Test_Agent(IL_Agent):
  """
  An agent used for testing to see if the Q function learned by IQL actually
  reflects the reward in the real world.
  """

  def __init__(self) -> None:
    super().__init__()
    # Previous intent index
    self.prev_intent_idx = 0
    # The recipe we are following
    self.recipe = soup_c
    # Index of the current step of the recipe
    self.cur_recipe_index = 0
    # Cumulative reward
    self.c_reward = 0
    # Reward
    self.reward = 0
    # Last state
    self.last_env_tensor = None

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

  def load_model_direct(self, model):
    """
    We probably won't use this method.
    """
    self.model = model

  def compute_reward(self, new_env_tensor, alpha=0.1, gamma=0.99):
    """
    Compute the reward of taking the action at cur_recipe_idx at the current
    state.
    """
    # Get Q value.
    last_state = (torch.cat(
        (self.last_env_tensor, torch.tensor([self.prev_intent_idx
                                             ])))).unsqueeze(0)
    q = self.model.q_net(last_state)
    dist = F.softmax(q / torch.tensor(alpha), dim=1)
    q_val = dist[0][self.prev_intent_idx].item()

    # Get value of the next state
    cur_intent_idx = ALL_MOVES.index(self.recipe[self.cur_recipe_index])
    cur_state = (torch.cat(
        (new_env_tensor, torch.tensor([cur_intent_idx])))).unsqueeze(0)
    q_next = self.model.q_net(cur_state)
    v = alpha * torch.logsumexp(q_next / alpha, dim=1,
                                keepdim=True)[0][0].item()

    self.reward = q_val - gamma * v
    self.c_reward += self.reward
    print('c reward: ', self.c_reward)

  def step(self, env_state, env_tensor):
    while True:
      if self._task is None:
        self._task = deepcopy(MOVE_TO_HT[self.recipe[self.cur_recipe_index]])
        self.last_env_tensor = env_tensor

      state, move, msg = self._task(env_state)
      if state == HighTask.Working:
        self.prev_intent_idx = ALL_MOVES.index(
            self.recipe[self.cur_recipe_index])
        return move, None
      elif state == HighTask.Failed:
        print(f"Move Failed: {move}")
        if self.cur_recipe_index < len(self.recipe) - 1:
          self.cur_recipe_index += 1
        else:
          self.cur_recipe_index = 0
        # Only compute the reward if the move is successful (for now).
        self.compute_reward(env_tensor)
        self._task = None
        return (0, 0), None
      else:
        if self.cur_recipe_index < len(self.recipe) - 1:
          self.cur_recipe_index += 1
        else:
          self.cur_recipe_index = 0
        # Only compute the reward if the move is successful (for now).
        self.compute_reward(env_tensor)
        self._task = None


num_demos = 1
env_seed = 647
priority = [['David Soup'], ['Alice Soup']]
p = get_priority_str(priority)

directory = f'demonstrations/{p}'
file_names = [f'{p}_demo_env{env_seed}_agent0.txt']
file_names = [os.path.join(directory, f) for f in file_names]
expert_dataset, input_size = read_datasets(
    fname_list=file_names,
    write=False,
    save_name=
    f'il_agents/iql/{p}/{str(num_demos)}demos/overcooked_ring_{str(num_demos)}demos.pkl'
)

env_seed = 105
overcooked_env = get_env(OvercookedExp1, priority=priority, seed=env_seed)
test_agent = Test_Agent()
test_agent.load_model(
    f'il_agents/iql/{p}/{str(num_demos)}demos/best_softq_{str(num_demos)}demos',
    f'il_agents/iql/{p}/{str(num_demos)}demos/config_utf.yaml',
    input_size=input_size)
game = GameEnv_Single(env=overcooked_env,
                      max_timesteps=1000,
                      agent_type='iql',
                      agent_model=test_agent,
                      play=True)
game.execute_agent(fps=3, sleep_time=0.1, fname='', write=False)
