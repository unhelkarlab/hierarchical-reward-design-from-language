import queue
import time
import threading
import torch
import os
from copy import deepcopy
from typing import List, Tuple
from gym_cooking.misc.game.game import Game
from gym_cooking.utils.replay import Replay
from gym_cooking.envs.overcooked_environment import OvercookedEnvironment
from agent.executor.low import EnvState
from agent.gameenv_single import Transition
from agent.il_agents.iql.iql_agent import IQL_Agent
from agent.mind.agent_new import SimHumanPref
from agent.mind.prompt_local import ALL_MOVES


class AgentInfo():

  def __init__(self, type, idx, fps=3) -> None:
    self.type = type
    self.idx = idx
    self.fps = fps

    self.q = queue.Queue()
    self.has_started = False
    self.agent = None
    self.time_last = time.time()


class GameSimAI(Game):
  """
  Game simulator with an AI agent.
  """

  def __init__(self,
               env: OvercookedEnvironment,
               priority: str,
               start_state: dict,
               agent_types: List[str],
               agent_models: List[object],
               sim_h_sets: List[dict],
               replay: Replay,
               save_path: str,
               agent_speed=3,
               play=False,
               need_log=True,
               is_practice=False,
               seed=0):
    super().__init__(env, play=play)
    self.num_agents = len(self.sim_agents)

    self.agent_types = agent_types
    self.agent_models = agent_models
    self.sim_h_sets = sim_h_sets
    self.replay = replay
    self.is_practice = is_practice

    self.human_idx = -1

    # # Game update frequency
    # self.fps = 10

    self.action_dict = {agent.name: None for agent in self.sim_agents}
    self._q_control = queue.Queue()
    self._q_env = queue.Queue()

    self.on_init()
    self.on_render()

    # Initialize agents
    self.agent_infos = []
    agent_idx = 0
    num_sim_hs = 0
    for agent_type in self.agent_types:
      print(self.sim_agents[agent_idx].name)
      agent_info = AgentInfo(agent_type, agent_idx, 3)
      if agent_type == 'ai' or agent_type == 'sim_h':
        info = self.env.get_ai_info()
        old_features = self.env.get_current_features()
        env = EnvState(world=info['world'],
                       agents=info['sim_agents'],
                       agent_idx=agent_idx,
                       order=info['order_scheduler'],
                       event_history=info['event_history'],
                       time=info['current_time'],
                       chg_grid=info['chg_grid'],
                       env=self.env)
        agent_info.q.put_nowait(('Env', {
            "EnvState":
            deepcopy(env),
            "EnvTensor":
            torch.from_numpy(old_features).float()
        }))
      if agent_type == 'ai':
        agent_info.agent = self.agent_models[agent_idx - num_sim_hs]
        agent_info.fps = agent_speed
      if agent_type == 'sim_h':
        agent_info.fps = agent_speed
        num_sim_hs += 1
        agent_info.agent = SimHumanPref(seed)
      if agent_type == 'h':
        self.human_idx = agent_idx
      self.agent_infos.append(agent_info)
      agent_idx += 1

    self.chats = {}
    self._success = False
    self._paused = 0

    self.need_log = need_log  # whether the current iteration needs to be logged
    self.fname = save_path

  def _run_env(self, elapsed_time: float):
    if self._success:
      return

    self.chats = {}
    elapsed_time = elapsed_time / 1000
    # seconds_per_step = 1 / self.fps
    old_state = self.env.get_current_state()
    old_features = self.env.get_current_features()
    while not self._q_env.empty():
      event = self._q_env.get_nowait()
      event_type, args = event
      if event_type == 'Action':
        self.action_dict[self.sim_agents[int(
            args['agent'])].name] = args['action']
      elif event_type == 'Pause':
        self._paused = True
      elif event_type == 'Continue':
        self._paused = False
      elif event_type == 'Chat':
        self.chats[args['agent_idx']] = args['chat']

    if not self._paused:
      ad = {
          k: v if v is not None else (0, 0)
          for k, v in self.action_dict.items()
      }
      # for agent_idx in range(self.num_agents):
      #   self.actions[agent_idx] = ad[self.sim_agents[agent_idx].name]
      state, reward, done, info = self.env.step(ad, passed_time=elapsed_time)
      features = self.env.get_current_features()
      if self.need_log:
        assert self.num_agents == 1
        move = list(ad.values())[0]
        macro_action = self.agent_infos[0].agent.get_cur_intent()
        if macro_action != 'None' and macro_action is not None:
          # print('macro: ', macro_action)
          macro_idx = ALL_MOVES.index(macro_action)
        else:
          # print('macro action is none')
          macro_idx = -1
        new_macro = True if (hasattr(self.agent_infos[0].agent, 'new_task')
                             and self.agent_infos[0].agent.new_task) else False
        transition = Transition(old_state, old_features, move, macro_action,
                                macro_idx, new_macro, reward, state, features,
                                done, info)
        self.write_transition(transition)
      self.on_render(paused=self._paused)
      if done:
        self._success = True
        self._q_control.put(('Quit', {}))
        return

      agent_idx = 0
      for agent_info in self.agent_infos:
        info = self.env.get_ai_info()
        state = self.env.get_current_state()
        old_features = self.env.get_current_features()
        env = EnvState(world=info['world'],
                       agents=info['sim_agents'],
                       agent_idx=agent_idx,
                       order=info['order_scheduler'],
                       event_history=info['event_history'],
                       time=info['current_time'],
                       chg_grid=info['chg_grid'],
                       env=self.env)
        if self.action_dict[self.sim_agents[agent_idx].name] is not None:
          agent_info.q.put_nowait(('Env', {
              "EnvState":
              deepcopy(env),
              "EnvTensor":
              torch.from_numpy(old_features).float()
          }))

        if hasattr(agent_info.agent, 'cumulative_reward'):
          agent_info.agent.cumulative_reward += self.reward
        agent_idx += 1
      self.action_dict = {agent.name: None for agent in self.sim_agents}

      if len(self.agent_infos) == 1:
        # If there is only one agent, the agent is following instructions, and
        # the agent does not have a current task, pause the simulation
        agent = self.agent_infos[0].agent
        if agent is not None and not isinstance(agent, IQL_Agent):
          if not agent.auto and agent._tasks == [] and agent._task is None:
            self._paused = True
    else:
      self.on_render(paused=self._paused)

  def _run_ai(self, agent_info: AgentInfo):
    time_per_step = 1 / agent_info.fps
    env = None
    state = None
    env_update = False
    chat = ''

    while True:
      if self._paused:
        continue
      event = agent_info.q.get()
      while True:
        event_type, args = event
        if event_type == 'Env':
          env = args['EnvState']
          state = args['EnvTensor']
          env_update = True
        elif event_type == 'Chat':
          chat = args['chat']
        elif event_type == "Action":
          human_act = True
        elif event_type == "Quit":
          return
        if not agent_info.q.empty():
          event = agent_info.q.get()
        else:
          break

      # if chat != '':
      #   agent_info.agent.high_level_infer(env, chat)
      #   chat = ''

      if env_update:
        move, chat_ret = agent_info.agent(env, state, chat)
        chat = ''
        self.sim_agents[agent_info.idx].set_cur_intent(
            agent_info.agent.get_cur_intent())
        self.sim_agents[agent_info.idx].set_intent_hist(
            agent_info.agent.get_intent_hist())

        # sleep
        sleep_time = max(time_per_step - (time.time() - agent_info.time_last),
                         0)
        time.sleep(sleep_time)
        agent_info.time_last = time.time()

        if chat_ret:
          self._q_env.put(('Chat', {
              'agent_idx': agent_info.idx,
              "chat": chat_ret
          }))

          for other_agent_info in self.agent_infos:
            if agent_info.idx != other_agent_info.idx:
              other_agent_info.q.put(('Chat', {
                  'agent_idx': agent_info.idx,
                  "chat": chat_ret
              }))
        self._q_env.put(('Action', {
            "agent": str(agent_info.idx),
            "action": move
        }))
        env_update = False

  def _run_sim_h(self, agent_info: AgentInfo):
    time_per_step = 1 / agent_info.fps
    env = None
    env_update = False
    chat_ret = None

    while True:
      if self._paused:
        continue

      chat = ''
      event = agent_info.q.get()
      while True:
        event_type, args = event
        if event_type == 'Env':
          env = args['EnvState']
          env_update = True
        elif event_type == 'Chat':
          if args['agent_idx'] == 0:
            # Only check for the AI's msg
            chat = args['chat']
        elif event_type == "Action":
          human_act = True
        elif event_type == "Quit":
          return
        if not agent_info.q.empty():
          event = agent_info.q.get()
        else:
          break

      if chat != '':
        chat_ret = 'message received'

      if env_update:
        move, _ = agent_info.agent.step(env)
        self.sim_agents[agent_info.idx].set_cur_intent(
            agent_info.agent.get_cur_intent())
        self.sim_agents[agent_info.idx].set_intent_hist(
            agent_info.agent.get_intent_hist())

        # sleep
        sleep_time = max(time_per_step - (time.time() - agent_info.time_last),
                         0)
        time.sleep(sleep_time)
        agent_info.time_last = time.time()

        if chat_ret:
          self._q_env.put(('Chat', {
              'agent_idx': agent_info.idx,
              "chat": chat_ret
          }))

          # for other_agent_info in self.agent_infos:
          #   if agent_info.idx != other_agent_info.idx:
          #     other_agent_info.q.put(('Chat', {
          #         'agent_idx': agent_info.idx,
          #         "chat": chat_ret
          #     }))
          chat_ret = None
        self._q_env.put(('Action', {
            "agent": str(agent_info.idx),
            "action": move
        }))
        env_update = False

  def on_execute(self, elapsed_time: float):
    self._run_env(elapsed_time)

    for agent_info in self.agent_infos:
      if agent_info.type == 'ai' and not agent_info.has_started:
        agent_info.has_started = True
        thread_ai = threading.Thread(target=self._run_ai,
                                     args=(agent_info, ),
                                     daemon=True)
        thread_ai.start()
      if agent_info.type == 'sim_h' and not agent_info.has_started:
        agent_info.has_started = True
        thread_human = threading.Thread(target=self._run_sim_h,
                                        args=(agent_info, ),
                                        daemon=True)
        thread_human.start()

    if self._success:
      for agent_info in self.agent_infos:
        if agent_info.type == 'ai':
          agent_info.agent.game_end()
    return self._success, self.chats

  def add_action(self, action: Tuple[int, int], agent_num: int):
    # TODO: Need modification
    self._q_env.put(('Action', {"agent": str(agent_num), "action": action}))

  def add_event(self, event: str):
    self._q_env.put((event, {}))

  def add_chat_to_queues(self, chat_in: str):
    # TODO: Need modification
    self._q_env.put(('ChatIn', {"chat": chat_in, "mode": "text"}))
    for other_agent_info in self.agent_infos:
      if other_agent_info.type != 'h':
        other_agent_info.q.put(('Chat', {'agent_idx': 1, "chat": chat_in}))

  def get_num_fires(self):
    return self.env.num_fires

  def get_num_bad_deliveries(self):
    return self.env.order_scheduler.num_bad_deliveries

  def get_score(self):
    return self.env.reward_sum

  def state_to_tensor(self, state):
    # Convert the state of a tensor
    state_map = torch.tensor(state['map'].tolist()).view(-1)
    state_orders = torch.tensor(state['current_orders'].tolist())
    state_holdings = torch.tensor(state['current_holdings']['agent-1'].tolist())
    state = torch.cat((state_map, state_orders, state_holdings))
    return state

  def write_transition(self, transition: Transition):
    header = ''
    if not os.path.isfile(self.fname):
      header += transition.get_header()
      header += '\n'

    data = str(transition)
    data += '\n'

    with open(self.fname, 'a+') as f:
      f.write(header)
      f.write(data)
      f.close()
