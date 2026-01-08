import torch
import time

from rw4t.rw4t_sim import AgentInfo
from rw4t.rw4t_game import RW4T_Game


class RW4T_Sim_Simple(RW4T_Game):
  '''
  An RW4T simulator for running experiments without any action delay.
  '''

  def __init__(
      self,
      env,
      agent_type: str,
      agent_model,
      reward_function=None,
      max_timesteps: float = float('inf'),
      game_fps: float = 2,
      play: bool = False,
  ):
    RW4T_Game.__init__(self, env, play=play)
    # Time steps
    self.max_timesteps = max_timesteps
    self.cur_timestep = 0
    # Game frame rate
    self.fps = game_fps
    # A wrapper around the agent's decision model
    self.agent_info = AgentInfo(agent_type, 0)
    self.agent_info.agent = agent_model
    # A (learned) reward function
    self.reward_function = reward_function
    # Whether we display the game
    self.play = play
    if self.play:
      self.on_init()
      self.on_render()

  def execute_agent(self, sleep_time=1e-10):
    '''
    Execute the agent until done.
    '''
    c_reward = 0
    done = False
    while not done and self.cur_timestep < self.max_timesteps:
      obs_tensor = torch.from_numpy(self.env.get_current_features()).float()
      action = self.agent_info.agent.step(self.env, obs_tensor)
      _obs, reward, done, _info = self.env.step(action,
                                                passed_time=1 / self.fps)
      if self.reward_function is None:
        c_reward += reward
      else:
        c_reward += self.reward_function(obs_tensor, action)
      if self.agent_info.type == 'futures' and self.agent_info.agent.done:
        break
      self.cur_timestep += 1
      if self.play:
        self.on_render()
      time.sleep(sleep_time)
    if self.play:
      self.on_cleanup()
    return c_reward
