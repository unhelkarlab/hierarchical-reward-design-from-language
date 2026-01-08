from gym_cooking.misc.game.gameplay_simple import GamePlaySimple
from gym_cooking.envs.overcooked_simple import OvercookedSimple, MapSetting

import argparse


def parse_arguments():
  '''
  Parse command line arguments when calling this script.

  "map" is the only flag for this program and its value is defaulted to "ring"
  if the flag is not specified.
  '''
  parser = argparse.ArgumentParser("Overcooked argument parser")
  parser.add_argument("--map",
                      type=str,
                      choices=['ring', 'bottleneck', 'partition', 'quick'],
                      default='ring')

  return parser.parse_args()


MAP_SETTINGS = dict(
    ring=dict(level="new1", ),
    bottleneck=dict(level="new3", ),
    partition=dict(level="new2"),
    quick=dict(
        level="new5",
        max_num_orders=4,
    ),
)

if __name__ == '__main__':
  # Get the command line arguments
  arglist = parse_arguments()
  # Start the environment
  map_set = MapSetting(**MAP_SETTINGS[arglist.map])
  env = OvercookedSimple(map_set)
  env.reset()
  # Start the game
  game = GamePlaySimple(env)
  ok = game.on_execute()
  print(ok)
