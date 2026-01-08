import queue
import time
import threading
from copy import deepcopy
import torch

from rw4t_game import RW4T_Game
from rw4t_env import RW4TEnv
import utils


class AgentInfo():

  def __init__(self, type, idx) -> None:
    self.type = type
    self.idx = idx
    self.agent = None
    self.q = queue.Queue()


class RW4T_Sim(RW4T_Game):

  def __init__(
      self,
      env,
      agent_type: str,
      agent_model,
      agent_fps: float = 2,
      game_fps: float = 10,
      play: bool = False,
  ):
    RW4T_Game.__init__(self, env, play=play)
    # Game frame rate
    self.fps = game_fps
    # Agent info
    self.agent_fps = agent_fps
    self.agent_info = AgentInfo(agent_type, 0)
    self.agent_info.agent = agent_model
    # Whether the game is completed
    self._success = False
    # Queue for environment events
    self._q_env = queue.Queue()

  def _run_env(self):
    self.on_render()
    # Add environment info to the agent's queue
    self.agent_info.q.put_nowait(('Env', {
        "EnvState":
        self.env,
        "EnvTensor":
        torch.from_numpy(self.env.get_current_features()).float()
    }))

    seconds_per_step = 1 / self.fps
    action = None
    next_frame_time = time.perf_counter()
    while True:
      # Get environment events
      while not self._q_env.empty():
        event = self._q_env.get_nowait()
        event_type, args = event
        if event_type == 'Action' and args['agent'] == "ai":
          action = args['action']
        else:
          raise NotImplementedError

      # Set action and step in the environment
      action_processed = action if action is not None else utils.RW4T_LL_Actions.idle.value
      state, _, done, _ = self.env.step(
          action_processed,
          passed_time=seconds_per_step,
          hl_action=self.agent_info.agent.prev_intent_idx)
      # Update done info
      if done:
        self._success = True
      # If the action has taken a non noop action, put the environment info
      # into the agent's queue
      if self.env.current_action is None:
        self.agent_info.q.put(('Env', {
            "EnvState":
            self.env,
            "EnvTensor":
            torch.from_numpy(self.env.get_current_features()).float()
        }))
      # Reset action and render game
      action = None
      self.on_render()
      # Sleep to run the game at the desired frame rate
      next_frame_time += seconds_per_step
      sleep_time = next_frame_time - time.perf_counter()
      sleep_time = max(sleep_time, 0)
      time.sleep(sleep_time)
      if done:
        return

  def _run_ai(self):
    # Initialize variables
    time_per_step = 1 / self.agent_fps
    env = None
    state = None
    env_update = False
    chat = ''
    next_frame_time = time.perf_counter()
    while True:
      if self._success:
        break
      # Get environment update from queue
      event = self.agent_info.q.get()
      while True:
        event_type, args = event
        if event_type == 'Env':
          env = args['EnvState']
          state = args['EnvTensor']
          env_update = True
        elif event_type == 'Chat':
          chat = args['chat']
        if not self.agent_info.q.empty():
          event = self.agent_info.q.get()
        else:
          break

      # Execute action
      if env_update:
        move = self.agent_info.agent.step(env, state)
        self._q_env.put(('Action', {"agent": "ai", "action": move}))
        env_update = False

      # Sleep to run the game at the desired frame rate
      next_frame_time += time_per_step
      sleep_time = next_frame_time - time.perf_counter()
      sleep_time = max(sleep_time, 0)
      print('Reward: ', self.env.cumulative_reward)
      time.sleep(sleep_time)

  def on_execute(self):
    # Initialize pygame and start display
    self.on_init()
    # Start executing the environment in the background
    thread_env = threading.Thread(target=self._run_env, daemon=True)
    thread_env.start()
    # Start running the ai agent in the foreground
    self._run_ai()
    # Quit game when game is done
    self.on_cleanup()
    print('Reward: ', self.env.cumulative_reward)
    return self._success


# if __name__ == '__main__':
#   kit_pref = [(1, 0), (2, 4), (5, 2)]
#   danger_pref = [(4, 1), (1, 3), (2, 3)]
#   seeds = utils.rw4t_seeds[-6:]
#   map_size = 6
#   # seeds = list(range(71, 80))
#   for seed in seeds:
#     # print('seed: ', seed)
#     env = RW4TEnv(seed=seed, action_duration=0, write=False)
#     env.set_kit_pref(kit_pref)
#     env.set_danger_pref(danger_pref)
#     agent = IQL_Agent(hl=True, use_intent=False)
#     agent.load_model(
#         model_path=
#         f'il_agents/iql/{map_size}by{map_size}/10demos/best_softq_10demos',
#         cfg_path=f'il_agents/iql/{map_size}by{map_size}/10demos/config_utf.yaml',
#         input_size=38,
#         output_size=len(utils.RW4T_HL_Actions))
#     rw4t_sim = RW4T_Sim(env, agent_type='iql', agent_model=agent, play=True)
#     rw4t_sim.on_execute()
