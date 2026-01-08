# ithor_global_keys.py
# Control AI2-THOR with global hotkeys while the Unity window is focused.
# WASD = strafe/move, Arrows = rotate/move, I/K = look, P = screenshot, Q / ESC = quit.

import time
import threading
from queue import Queue, Empty

from ai2thor.controller import Controller
from pynput import keyboard
from PIL import Image

# Map logical commands to THOR actions
ACTION = {
    "MOVE_AHEAD": "MoveAhead",
    "MOVE_BACK": "MoveBack",
    "MOVE_LEFT": "MoveLeft",  # strafe
    "MOVE_RIGHT": "MoveRight",  # strafe
    "ROTATE_LEFT": "RotateLeft",
    "ROTATE_RIGHT": "RotateRight",
    "LOOK_UP": "LookUp",
    "LOOK_DOWN": "LookDown",
}

# Create an infinite queue of commands
cmd_queue = Queue()
pressed = set()  # track held keys to avoid auto-repeat floods
shutdown = threading.Event()


def enqueue_once(cmd: str):
  """
  Put a command in the queue (non-blocking).
  """
  try:
    cmd_queue.put_nowait(cmd)
  except Exception:
    pass


def on_press(key):
  """
  Global key-down handler. Only enqueue on the *first* press (debounce).
  """
  # Ignore repeats while held
  if key in pressed:
    return
  pressed.add(key)

  try:
    # Character keys
    if hasattr(key, "char") and key.char:
      c = key.char.lower()
      if c == "w": enqueue_once("MOVE_AHEAD")
      elif c == "s": enqueue_once("MOVE_BACK")
      elif c == "a": enqueue_once("MOVE_LEFT")
      elif c == "d": enqueue_once("MOVE_RIGHT")
      elif c == "i": enqueue_once("LOOK_UP")
      elif c == "k": enqueue_once("LOOK_DOWN")
      elif c == "p": enqueue_once("SCREENSHOT")
      elif c == "q": enqueue_once("QUIT")
    else:
      # Special keys
      if key == keyboard.Key.up: enqueue_once("MOVE_AHEAD")
      elif key == keyboard.Key.down: enqueue_once("MOVE_BACK")
      elif key == keyboard.Key.left: enqueue_once("ROTATE_LEFT")
      elif key == keyboard.Key.right: enqueue_once("ROTATE_RIGHT")
      elif key == keyboard.Key.esc: enqueue_once("QUIT")
  except Exception:
    pass


def on_release(key):
  """
  Remove from held set so a new press will enqueue again.
  """
  pressed.discard(key)


def main():
  # Start Unity / AI2-THOR
  ctrl = Controller(
      scene="FloorPlan1",
      width=800,
      height=600,
      gridSize=0.25,
      rotateStepDegrees=90,
      renderDepthImage=False,
      renderInstanceSegmentation=False,
  )

  ev = ctrl.step(action="Pass")

  # Start global keyboard listener in a background thread
  listener = keyboard.Listener(on_press=on_press, on_release=on_release)
  listener.start()

  print("Global controls ready. Focus the Unity window and use keys:")
  print(
      "  Arrows rotate/move, WASD strafe+move, I/K look, P screenshot, Q/ESC quit."
  )

  target_fps = 30.0
  target_dt = 1.0 / target_fps
  try:
    while not shutdown.is_set():
      t0 = time.time()

      # Handle at most one queued command per tick (prevents bursty repeats)
      try:
        cmd = cmd_queue.get_nowait()
      except Empty:
        cmd = None

      if cmd == "QUIT":
        break
      elif cmd == "SCREENSHOT":
        Image.fromarray(ev.frame).save("ithor_frame.png")
        print('Saved "ithor_frame.png"')
      elif cmd:
        # Execute mapped action
        action = ACTION.get(cmd, None)
        if action:
          ev = ctrl.step(action=action)

      else:
        # Idle tick: keep Unity repainting so the window stays responsive
        ev = ctrl.step(action="Pass")

      # Simple HUD in stdout
      ok = ev.metadata.get("lastActionSuccess")
      err = ev.metadata.get("errorMessage", "")
      if err:
        print(f"success={ok} err={err}")

      # Rate control
      dt = time.time() - t0
      if dt < target_dt:
        time.sleep(target_dt - dt)

  finally:
    shutdown.set()
    listener.stop()
    ctrl.stop()


if __name__ == "__main__":
  main()
