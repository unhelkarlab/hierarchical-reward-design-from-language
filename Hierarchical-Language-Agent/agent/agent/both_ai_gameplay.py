# modules for game
from gym_cooking.misc.game.game import Game
from gym_cooking.misc.game.utils import *
from gym_cooking.utils.gui import popup_text
from gym_cooking.utils.replay import Replay
from agent.executor.low import EnvState
from agent.mind.agent import get_agent, AgentSetting

# helpers
import pygame
import threading
import queue
import time

from copy import deepcopy as dcopy

# import vosk
# from vosk import Model, KaldiRecognizer
# import pyaudio
# import os

# def speak(text: str):
#     def _speak(text: str):
#         from win32com.client import Dispatch
#         speaker = Dispatch("SAPI.SpVoice")
#         speaker.Rate = 5
#         speaker.Speak(text)
#         pass
#
#     threading.Thread(target=_speak, args=(text,)).start()


class GamePlay(Game):

  def __init__(self,
               env,
               replay: Replay,
               agent_set_1: AgentSetting,
               agent_set_2: AgentSetting = None):
    Game.__init__(self, env, play=True)
    self.replay = replay
    self.agent_set_1 = agent_set_1
    if self.agent_set_2 is not None:
      self.agent_set_2 = agent_set_2

    # fps of human and ai
    self.fps = 10
    self.fps_ai = agent_set_1.speed

    self.idx_human = 1
    self.ai_1 = get_agent(self.agent_set_1, self.replay)
    self.ai_2 = get_agent(self.agent_set_2, self.replay)

    # concurrent control variables
    self._q_control = queue.Queue()  # receive
    self._q_env = queue.Queue()
    self._q_ai_1 = queue.Queue()
    self._q_ai_2 = queue.Queue()
    self._success = False

  def on_event(self, event):
    if event.type == pygame.QUIT:
      self._q_control.put(('Quit', {}))

    elif event.type == pygame.KEYDOWN:
      if event.key in KeyToTuple.keys():
        # Control
        action_dict = {agent.name: (0, 0) for agent in self.sim_agents}
        action = KeyToTuple[event.key]
        action_dict[self.current_agent.name] = action
        self._q_env.put(('Action', {"agent": "human", "action": action}))
        self._q_ai.put(('Action', {"agent": "human", "action": action}))

      if pygame.key.name(event.key) == "space":
        self._q_env.put(('Pause', {}))

        s = popup_text("Say to AI:")

        if s is not None:
          self._q_env.put(('ChatIn', {"chat": s, "mode": "text"}))
          self._q_ai.put(('Chat', dict(chat=s)))

        self._q_env.put(('Continue', {}))

  def _run_env(self):
    seconds_per_step = 1 / self.fps
    idx_human = self.idx_human
    paused = 0
    chat_in, chat_out = "", ""
    last_t = time.time()
    action_dict = {agent.name: None for agent in self.sim_agents}

    self.on_render(paused=paused)
    info = self.env.get_ai_info()
    e_1 = EnvState(world=info['world'],
                   agents=info['sim_agents'],
                   agent_idx=0,
                   order=info['order_scheduler'],
                   event_history=info['event_history'],
                   time=info['current_time'],
                   chg_grid=info['chg_grid'])
    self._q_ai_1.put_nowait(('Env', {"EnvState": e_1}))
    e_2 = EnvState(world=info['world'],
                   agents=info['sim_agents'],
                   agent_idx=1,
                   order=info['order_scheduler'],
                   event_history=info['event_history'],
                   time=info['current_time'],
                   chg_grid=info['chg_grid'])
    self._q_ai_2.put_nowait(('Env', {"EnvState": e_2}))

    while True:
      while not self._q_env.empty():
        event = self._q_env.get_nowait()
        event_type, args = event
        if event_type == 'Action':
          if args['agent'] == "human":
            action_dict[self.sim_agents[idx_human].name] = args['action']
          elif args['agent'] == 'ai_1':
            action_dict[self.sim_agents[0].name] = args['action']
          elif args['agent'] == 'ai_2':
            action_dict[self.sim_agents[1].name] = args['action']
        elif event_type == 'Pause':
          paused += 1
        elif event_type == 'Continue':
          paused -= 1
        elif event_type == 'ChatIn':
          chat_in = f"User Input: [{args['mode']}]\n\n" + \
              args['chat']
          chat_out = ""
        elif event_type == 'ChatOut':
          chat_out = "AI Output:\n\n" + args['chat']

      if not paused:
        ad = {k: v if v is not None else (0, 0) for k, v in action_dict.items()}
        self.replay.log('env.step', {
            'action_dict': ad,
            'passed_time': seconds_per_step
        })
        _, _, done, _ = self.env.step(ad, passed_time=seconds_per_step)
        if done:
          self._success = True
          self._q_control.put(('Quit', {}))
          return

        info = self.env.get_ai_info()
        e_1 = EnvState(world=info['world'],
                       agents=info['sim_agents'],
                       agent_idx=0,
                       order=info['order_scheduler'],
                       event_history=info['event_history'],
                       time=info['current_time'],
                       chg_grid=info['chg_grid'])
        e_2 = EnvState(world=info['world'],
                       agents=info['sim_agents'],
                       agent_idx=1,
                       order=info['order_scheduler'],
                       event_history=info['event_history'],
                       time=info['current_time'],
                       chg_grid=info['chg_grid'])
        if action_dict[self.sim_agents[0].name] is not None:
          self._q_ai_1.put(('Env', {"EnvState": dcopy(e_1)}))
        if action_dict[self.sim_agents[1].name] is not None:
          self._q_ai_2.put(('Env', {"EnvState": dcopy(e_2)}))
        action_dict = {agent.name: None for agent in self.sim_agents}

      sleep_time = max(seconds_per_step - (time.time() - last_t), 0)
      last_t = time.time()
      time.sleep(sleep_time)

      chat = chat_in + '\n\n' + chat_out

      if not paused:
        self.replay.log('on_render', {'paused': paused, 'chat': chat})
      self.on_render(paused=paused, chat=chat)

  def _run_ai_1(self):
    time_per_step = 1 / self.fps_ai
    time_last = time.time()
    human_act = True
    env = None
    env_update = False
    chat = ''
    while True:
      event = self._q_ai_1.get()
      while True:
        event_type, args = event
        if event_type == 'Env':
          env = args['EnvState']
          env_update = True
        elif event_type == 'Chat':
          chat = args['chat']
        elif event_type == "Action":
          human_act = True
        elif event_type == "Quit":
          return
        if not self._q_ai_1.empty():
          event = self._q_ai_1.get()
        else:
          break

      if chat != '':
        self.ai_1.high_level_infer(env, chat)
        chat = ''

      if env_update:
        move, chat_ret = self.ai_1(env)

        # sleep
        sleep_time = max(time_per_step - (time.time() - time_last), 0)
        time.sleep(sleep_time)
        time_last = time.time()

        if chat_ret:
          self._q_env.put(('ChatOut', {"chat": chat_ret}))
        self._q_env.put(('Action', {"agent": "ai_1", "action": move}))
        human_act = False
        env_update = False

  def _run_ai_2(self):
    time_per_step = 1 / self.fps_ai
    time_last = time.time()
    human_act = True
    env = None
    env_update = False
    chat = ''
    while True:
      event = self._q_ai_2.get()
      while True:
        event_type, args = event
        if event_type == 'Env':
          env = args['EnvState']
          env_update = True
        elif event_type == 'Chat':
          chat = args['chat']
        elif event_type == "Action":
          human_act = True
        elif event_type == "Quit":
          return
        if not self._q_ai_2.empty():
          event = self._q_ai_2.get()
        else:
          break

      if chat != '':
        self.ai_2.high_level_infer(env, chat)
        chat = ''

      if env_update:
        move, chat_ret = self.ai_2(env)

        # sleep
        sleep_time = max(time_per_step - (time.time() - time_last), 0)
        time.sleep(sleep_time)
        time_last = time.time()

        if chat_ret:
          self._q_env.put(('ChatOut', {"chat": chat_ret}))
        self._q_env.put(('Action', {"agent": "ai_2", "action": move}))
        human_act = False
        env_update = False

  # def _run_listen(self):
  #     ena_listen = False
  #     self._q_env.put(('Setting', {"ena_listen": ena_listen}))

  #     dir_path = os.path.dirname(os.path.realpath(__file__))
  #     model_path = dir_path + r"/vosk"
  #     model = Model(model_path)
  #     recognizer = KaldiRecognizer(model, 16000)

  #     mic = pyaudio.PyAudio()
  #     stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
  #     stream.start_stream()

  #     while True:
  #         while not self._q_listen.empty():
  #             event = self._q_listen.get_nowait()
  #             event_type, args = event
  #             if event_type == 'Quit':
  #                 return
  #             elif event_type == 'ListenSwitch':
  #                 ena_listen = not ena_listen
  #                 self._q_env.put(('Setting', {"ena_listen": ena_listen}))

  #         if not ena_listen:
  #             time.sleep(0.1)
  #             continue

  #         data = stream.read(10000)

  #         if recognizer.AcceptWaveform(data):
  #             text = recognizer.Result()
  #             s = text[14:-3]
  #             if s is not None and len(s) >= 4:
  #                 self._q_env.put(('ChatIn', {"chat": s, "mode": "speech"}))
  #                 self._q_ai.put(('Chat', dict(chat=s)))

  # def _run_human(self):
  #   while True:
  #     for event in pygame.event.get():
  #       self.on_event(event)
  #     if not self._q_control.empty():
  #       event, args = self._q_control.get_nowait()
  #       if event == 'Quit':
  #         self._q_ai.put(('Quit', {}))
  #         return

  def on_execute(self):
    if self.on_init() == False:
      exit()

    thread_env = threading.Thread(target=self._run_env, daemon=True)
    thread_ai_1 = threading.Thread(target=self._run_ai_1, daemon=True)
    thread_ai_2 = threading.Thread(target=self._run_ai_2, daemon=True)
    # thread_listen = threading.Thread(target=self._run_listen, daemon=True)
    thread_env.start()
    thread_ai_1.start()
    thread_ai_2.start()
    # thread_listen.start()

    # self._run_human()

    # clean up
    self.on_cleanup()

    # save history
    # if hasattr(self.ai, "_lock"):
    #   self.ai._lock.acquire()
    # if hasattr(self.ai, "_int_hist"):
    #   self.replay['int_hist'] = self.ai._int_hist
    # if hasattr(self.ai, "_llm_hist"):
    #   self.replay['llm_hist'] = self.ai._llm_hist
    # if hasattr(self.ai, "_mov_hist"):
    #   self.replay['mov_hist'] = self.ai._mov_hist
    # # log recipy infos
    # self.replay['order_result'] = dict(
    #     success=self.env.order_scheduler.successful_orders,
    #     fail=self.env.order_scheduler.failed_orders,
    #     reward=self.env.order_scheduler.reward)
    # if hasattr(self.ai, "_lock"):
    #   self.ai._lock.release()

    return self._success
