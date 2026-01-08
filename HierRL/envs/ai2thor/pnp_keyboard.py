# Drive ThorPickPlaceEnv with global hotkeys while the Unity window is focused.
# WASD = strafe/move, Arrows = rotate/move, I/K = look, SPACE=Pickup, ENTER=Put, BACKSPACE=Drop, R=reset, Q/ESC=quit

import time
from queue import Queue, Empty

from pynput import keyboard

from HierRL.envs.ai2thor.pnp_env import ThorPickPlaceEnv
from HierRL.envs.ai2thor.pnp_training_utils import (PnP_HL_Actions,
                                                    PnP_LL_Actions)

# Map high-level commands (strings) to PnP_LL_Actions indices in your env
KEY2ACTION = {
    "MOVE_AHEAD": PnP_LL_Actions.index("MoveAhead"),
    # "MOVE_BACK": PnP_LL_Actions.index("MoveBack"),
    # "MOVE_LEFT": PnP_LL_Actions.index("MoveLeft"),
    # "MOVE_RIGHT": PnP_LL_Actions.index("MoveRight"),
    "ROTATE_LEFT": PnP_LL_Actions.index("RotateLeft"),
    "ROTATE_RIGHT": PnP_LL_Actions.index("RotateRight"),
    # "LOOK_UP": PnP_LL_Actions.index("LookUp"),
    # "LOOK_DOWN": PnP_LL_Actions.index("LookDown"),
    "PICKUP": PnP_LL_Actions.index("PickupNearestTarget"),
    "PUT": PnP_LL_Actions.index("PutHeldOnReceptacle"),
    # "DROP": PnP_LL_Actions.index("DropHeld"),
}

cmd_queue = Queue()
pressed = set()


def enqueue(cmd: str):
  try:
    cmd_queue.put_nowait(cmd)
  except Exception:
    pass


def on_press(key):
  # Debounce: fire once per physical press
  if key in pressed:
    return
  pressed.add(key)

  try:
    # Character keys
    if hasattr(key, "char") and key.char:
      c = key.char.lower()
      if c == "w":
        enqueue("MOVE_AHEAD")
      elif c == "s":
        enqueue("MOVE_BACK")
      elif c == "a":
        enqueue("MOVE_LEFT")
      elif c == "d":
        enqueue("MOVE_RIGHT")
      elif c == "i":
        enqueue("LOOK_UP")
      elif c == "k":
        enqueue("LOOK_DOWN")
      elif c == "r":
        enqueue("RESET")
      elif c == "q":
        enqueue("QUIT")
    else:
      # Special keys
      if key == keyboard.Key.up:
        enqueue("MOVE_AHEAD")
      elif key == keyboard.Key.down:
        enqueue("MOVE_BACK")
      elif key == keyboard.Key.left:
        enqueue("ROTATE_LEFT")
      elif key == keyboard.Key.right:
        enqueue("ROTATE_RIGHT")
      elif key == keyboard.Key.space:
        enqueue("PICKUP")
      elif key == keyboard.Key.enter:
        enqueue("PUT")
      elif key == keyboard.Key.backspace:
        enqueue("DROP")
      elif key == keyboard.Key.esc:
        enqueue("QUIT")
  except Exception:
    pass


def on_release(key):
  pressed.discard(key)


option_to_use = PnP_HL_Actions.drop_egg.value


def main():
  env = ThorPickPlaceEnv(low_level=False, option=option_to_use, hl_pref_r=True)
  # obs, info = env.reset()
  step_count = 0

  # Start global listener
  listener = keyboard.Listener(on_press=on_press, on_release=on_release)
  listener.start()

  print("Use WASD/Arrows, I/K, SPACE/ENTER/BACKSPACE, R, Q.")

  target_fps = 30.0
  target_dt = 1.0 / target_fps

  try:
    while True:
      t0 = time.time()

      # handle at most one command per tick
      try:
        cmd = cmd_queue.get_nowait()
      except Empty:
        cmd = None

      if cmd == "QUIT":
        break
      elif cmd == "RESET":
        obs, info = env.reset(options={'option': option_to_use})
        step_count = 0
      elif cmd:
        print(f"Action: {cmd}")
        # step the env for mapped actions
        a = KEY2ACTION.get(cmd)
        if a is not None:
          obs, reward, term, trunc, info = env.step(a, option_to_use)
          step_count += 1
          if term or trunc:
            print(f"Episode done: reward={reward}, term={term}, trunc={trunc}")
            obs, info = env.reset(options={'option': option_to_use})
            step_count = 0
      else:
        env.controller.step(action="Pass")

      # maintain ~30 FPS
      dt = time.time() - t0
      if dt < target_dt:
        time.sleep(target_dt - dt)
  finally:
    listener.stop()
    env.close()


if __name__ == "__main__":
  main()
