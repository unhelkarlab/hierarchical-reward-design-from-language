import queue
import pygame
import time
import threading

from rw4t_game import RW4T_Game
from rw4t_env import RW4TEnv
from map_config import pref_dicts
import utils


class RW4T_Manual(RW4T_Game):

  def __init__(self, env, low_level=False, play=True):
    RW4T_Game.__init__(self, env, play=play)
    # Game frame rate
    self.fps = 10
    # Whether the game is completed
    self._success = False
    # Queue for control events
    self._q_control = queue.Queue()
    # Queue for environment events
    self._q_env = queue.Queue()

    # Input box setup
    self.input_box = True
    self.color_active = pygame.Color('dodgerblue')
    self.color_inactive = self.GRAY
    self.color = self.color_inactive
    self.active = False
    self.text = ''

    # Whether we are working with low-level only
    self.low_level = low_level

    self.all_rewards = (0, 0, 0, 0, 0)

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
      if event.key in utils.Key_2_Action.keys():
        # If user clicks on one of the control keys
        action = utils.Key_2_Action[event.key]
        self._q_env.put(('Action', {"agent": "1", "action": action}))
      else:
        # If user clicks on any other key and the input box is active
        if self.active:
          if event.key == pygame.K_RETURN:
            self.text = ''  # Clear text after pressing Enter
          elif event.key == pygame.K_BACKSPACE:
            self.text = self.text[:-1]
          else:
            self.text += event.unicode

  def _run_env(self):
    self.on_render()

    seconds_per_step = 1 / self.fps
    action = None
    next_frame_time = time.perf_counter()
    while True:
      # Get environment events
      while not self._q_env.empty():
        event = self._q_env.get_nowait()
        event_type, args = event
        if event_type == 'Action' and args['agent'] == "1":
          action = args['action']
        else:
          raise NotImplementedError

      # Set action and step in the environment
      action = action if action is not None else utils.RW4T_LL_Actions.idle.value
      if not self.low_level:
        hl_action = int(self.text) if len(self.text) > 0 else -1
      else:
        hl_action = self.env.option
      if action != utils.RW4T_LL_Actions.idle.value:
        _, reward, done, truncated, _ = self.env.step(
            action, hl_action, passed_time=seconds_per_step)
        self.all_rewards = tuple([
            acc_sub_r + sub_r
            for acc_sub_r, sub_r in zip(self.all_rewards, reward)
        ])
        # print(self.all_rewards)
        # Update done info
        if not self.low_level and (done or truncated):
          self._success = True
          self._q_control.put(('Quit', {}))
          return

        if self.low_level and (done or truncated):
          self.env.reset()
          self.total_rewards = 0
      # Reset action and render game
      action = None
      self.on_render()
      # Sleep to run the game at the desired frame rate
      next_frame_time += seconds_per_step
      sleep_time = next_frame_time - time.perf_counter()
      sleep_time = max(sleep_time, 0)
      time.sleep(sleep_time)

  def _run_human(self):
    while True:
      # Get all pygame events
      for event in pygame.event.get():
        # Add event to control or environment queue
        self.on_event(event)
      # Check if game has ended, and return from this function if so
      if not self._q_control.empty():
        event, _args = self._q_control.get_nowait()
        if event == 'Quit':
          return

  def on_execute(self):
    # Initialize pygame and start display
    self.on_init()
    # Start listening to human inputs in the background
    thread_env = threading.Thread(target=self._run_human, daemon=True)
    thread_env.start()
    # Start executing the environment in the foreground
    self._run_env()
    # Quit game when game is done
    self.on_cleanup()
    return self._success


if __name__ == '__main__':
  low_level = False
  hl_pref_r = True
  pbrs_r = False
  for seed in utils.rw4t_seeds[:5]:
    pref_dict = pref_dicts['six_by_six_8_train_pref_dict']
    env = RW4TEnv(map_name='six_by_six_8_train_map',
                  pref_dict=pref_dict,
                  low_level=low_level,
                  hl_pref_r=hl_pref_r,
                  pbrs_r=pbrs_r,
                  seed=seed,
                  action_duration=0,
                  write=False,
                  fname=f'rw4t_demos_manual/manual_control_{seed}_test.txt')
    manual_control = RW4T_Manual(env, low_level=low_level)
    manual_control.on_execute()
