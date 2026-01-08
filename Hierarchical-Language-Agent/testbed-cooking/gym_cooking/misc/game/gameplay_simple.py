# modules for game
from gym_cooking.misc.game.game import Game
from gym_cooking.misc.game.utils import KeyToTuple2

# helpers
import pygame
import threading
import time
import queue


class GamePlaySimple(Game):

  def __init__(self, env, play=True):
    Game.__init__(self, env, play=play)
    # FPS of game
    self.fps = 10
    # Whether the game has ended
    self._success = False
    # Queue for application events (i.e. 'Quit')
    self._q_control = queue.Queue()
    # Queue for environment events (i.e. 'action')
    self._q_env = queue.Queue()

  def on_event(self, event):
    if event.type == pygame.QUIT:
      self._q_control.put(('Quit', {}))
    elif event.type == pygame.MOUSEBUTTONDOWN:
      # Toggle active state when clicking on the box
      if self.input_box_gui.collidepoint(event.pos):
        self.active = not self.active
      else:
        self.active = False
      self.color = self.color_active if self.active else self.color_inactive
    elif event.type == pygame.KEYDOWN:
      if event.key in KeyToTuple2.keys():
        # Agent movement
        action_dict = {agent.name: (0, 0) for agent in self.sim_agents}
        action = KeyToTuple2[event.key]
        action_dict[self.current_agent.name] = action
        self._q_env.put(('Action', {"agent": "1", "action": action}))
      else:
        # Typing in the option input box
        if self.active:
          if event.key == pygame.K_RETURN:
            self.text = ''  # Clear text after pressing Enter
          elif event.key == pygame.K_BACKSPACE:
            self.text = self.text[:-1]
          else:
            self.text += event.unicode

  def _run_env(self):
    seconds_per_step = 1 / self.fps
    last_t = time.time()
    action_dict = {agent.name: None for agent in self.sim_agents}
    paused = False
    self.on_render(paused=paused)

    while True:
      # Get the environment events since the last update
      while not self._q_env.empty():
        event = self._q_env.get_nowait()
        event_type, args = event
        if event_type == 'Action':
          if args['agent'] == "1":
            action_dict[self.sim_agents[0].name] = args['action']

      # Take a step in the environment
      if not paused:
        ad = {k: v if v is not None else (0, 0) for k, v in action_dict.items()}
        if ad[list(ad.keys())[0]] != (0, 0):
          try:
            option = int(self.text)
          except ValueError:
            option = self.env.all_moves_dict_with_wait['Wait']
          _, _, done, truncated, _ = self.env.step(ad, option)
          if done or truncated:
            self._success = True
            self._q_control.put(('Quit', {}))
            return
          action_dict = {agent.name: None for agent in self.sim_agents}

      # Put the thread to sleep so that we take inputs at the set frame rate
      sleep_time = max(seconds_per_step - (time.time() - last_t), 0)
      last_t = time.time()
      time.sleep(sleep_time)

      self.on_render(paused=paused, chat='')

  def _run_human(self):
    while True:
      for event in pygame.event.get():
        self.on_event(event)
      if not self._q_control.empty():
        event, args = self._q_control.get_nowait()
        if event == 'Quit':
          return

  def on_execute(self):
    # Initialize game screen
    if self.on_init() is False:
      exit()
    # Run the domain on a thread
    thread_env = threading.Thread(target=self._run_env, daemon=True)
    thread_env.start()
    # Take human inputs in the foreground
    self._run_human()
    # clean up
    self.on_cleanup()

    return self._success
