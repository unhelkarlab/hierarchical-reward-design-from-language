import time
import pygame
from copy import deepcopy

from gym_cooking.envs.overcooked_simple import OvercookedSimple, MapSetting
from gym_cooking.misc.game.game import Game
from agent.mind.prompt_local import MOVE_TO_HT
from agent.executor.high import HighTask
from agent.executor.low import EnvState


def test(render):
  # Initialize the environment
  map_set = MapSetting(**dict(level="new1", ))
  env = OvercookedSimple(map_set)
  env.reset()
  # Initialize the game
  # For some reason, I have to initialize game after calling env.reset()
  game = Game(env, play=True)

  num_episodes = 1
  for ep in range(num_episodes):
    # Reset the environment
    _obs, _info = env.reset()
    game.__init__(env, play=True)
    game.on_init()
    game.on_render()

    done = False
    truncated = False
    # Run one episode
    while not (done or truncated):
      # Let pygame handle window close events, etc.
      if render:
        for event in pygame.event.get():
          if event.type == pygame.QUIT:
            done = True
            truncated = True
            break
      # Get the option from user inputs
      option = int(input("Enter the next option: "))
      # Map the option number to option name
      ht = [
          name for name, val in env.all_moves_dict_with_wait.items()
          if val == option
      ][0]
      # Get the task that corresponds to the option name (can only be done once
      # if we would like to execute the option until it terminates)
      task = deepcopy(MOVE_TO_HT[ht])
      # Step the environment forward
      ll_done = False
      ll_truncated = False
      while not (ll_done or ll_truncated):
        # Get the state info to be used for the low-leel policy
        info = env.get_ai_info()
        env_state = EnvState(world=info['world'],
                             agents=info['sim_agents'],
                             agent_idx=0,
                             order=info['order_scheduler'],
                             event_history=info['event_history'],
                             time=info['current_time'],
                             chg_grid=info['chg_grid'],
                             env=env)
        # Get the action of the low-level policy
        state, move, _msg = task(env_state)
        # Step the environment forward if the option is valid and isn't done yet
        if state == HighTask.Working:
          action_dict = {}
          action_dict[env.sim_agents[0].name] = move
          action_dict = {
              k: v if v is not None else (0, 0)
              for k, v in action_dict.items()
          }
          _obs, reward, done, truncated, _info = env.step(action_dict, option)
          if render:
            # Render the updated state in the window
            game.on_render()
            # A small pause to see what's going on
            time.sleep(0.5)
          ll_done = False
          ll_truncated = truncated
        else:
          ll_done = True


if __name__ == '__main__':
  test(render=True)
