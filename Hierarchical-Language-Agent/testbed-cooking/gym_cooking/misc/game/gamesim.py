import queue
from web_experiment import socketio
from gym_cooking.misc.game.game import Game


class GameSim(Game):
  """
  Similar to GamePlay, but used as a simulator for the web game.
  """

  def __init__(self, env, play=False):
    super().__init__(env, play=play)

    self.fps = 20
    self._success = False
    self._paused = False

    self._q_control = queue.Queue()
    self._q_env = queue.Queue()

    self.on_init()
    self.on_render()

  def _run_env(self):
    if self._success:
      return

    action_dict = {agent.name: None for agent in self.sim_agents}
    seconds_per_step = 1 / self.fps
    chat = ''

    # self.on_render(paused=paused)
    while not self._q_env.empty():
      event = self._q_env.get_nowait()
      event_type, args = event
      if event_type == 'Action':
        if args['agent'] == "1":
          action_dict[self.sim_agents[0].name] = args['action']
        elif args['agent'] == "2":
          action_dict[self.sim_agents[1].name] = args['action']
      elif event_type == 'Pause':
        self._paused += 1
      elif event_type == 'Continue':
        self._paused -= 1
      elif event_type == 'ChatIn':
        chat = args['chat']

    if not self._paused:
      ad = {k: v if v is not None else (0, 0) for k, v in action_dict.items()}
      _, _, done, _ = self.env.step(ad, passed_time=seconds_per_step)
      if done:
        self._success = True
        self._q_control.put(('Quit', {}))
        return

      action_dict = {agent.name: None for agent in self.sim_agents}
    self.on_render(paused=self._paused, chat=chat)

  def on_execute(self):
    socketio.start_background_task(target=self._run_env)
    return self._success

  def add_action(self, action, agent_num):
    self._q_env.put(('Action', {"agent": str(agent_num), "action": action}))
