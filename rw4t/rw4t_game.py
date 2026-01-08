import pygame
import imageio
import numpy as np
from pathlib import Path

import rw4t.utils as utils


class RW4T_Game:

  def __init__(self,
               env,
               play=False,
               record=False,
               record_path="recordings/gameplay.mp4"):
    pygame.init()

    self.record = record
    self.record_path = record_path
    self.frames = []

    # Load and scale images
    self.cell_size = 100
    self.obj_scale = self.cell_size * 2 / 3
    self.robot_imgs = {
        "default":
        pygame.image.load(f"{Path(__file__).parent}/icons/robot.png"),
        "pick_food":
        pygame.image.load(f"{Path(__file__).parent}/icons/robot-pick-food.png"),
        "pick_medkit":
        pygame.image.load(
            f"{Path(__file__).parent}/icons/robot-pick-medkit.png"),
        "with_food":
        pygame.image.load(f"{Path(__file__).parent}/icons/robot-with-food.png"),
        "with_medkit":
        pygame.image.load(
            f"{Path(__file__).parent}/icons/robot-with-medkit.png")
    }
    self.item_imgs = {
        "food":
        pygame.image.load(f"{Path(__file__).parent}/icons/food.png"),
        "medkit":
        pygame.image.load(f"{Path(__file__).parent}/icons/medical-kit.png")
    }
    for k in self.robot_imgs:
      self.robot_imgs[k] = pygame.transform.scale(
          self.robot_imgs[k], (self.cell_size, self.cell_size))
    for k in self.item_imgs:
      self.item_imgs[k] = pygame.transform.scale(
          self.item_imgs[k], (self.obj_scale, self.obj_scale))

    self.env = env
    self.play = play
    self._running = True

    self.map_size = self.env.map_size
    self.update_env_info()
    self.font = pygame.font.SysFont('Arial', 20)

    # input box inits
    self.input_box = False  # whether to display input box
    self.input_box_gui = pygame.Rect(self.map_size * self.cell_size + 20, 50,
                                     200, 50)
    self.WHITE = (255, 255, 255)
    self.BLACK = (0, 0, 0)
    self.GRAY = (200, 200, 200)
    self.ib_font = pygame.font.Font(None, 40)
    self.lbl_font = pygame.font.Font(None, 30)

    # Other UI inits
    self.extra_ui_height = 100
    self.domain_text_font = pygame.font.SysFont('Arial', 36)

    # Whether we would like to display the option
    self.low_level = False

  def on_init(self):
    # Initialize pygame
    pygame.init()
    width = self.map_size * self.cell_size + (250 if self.low_level
                                              or self.input_box else 0)
    height = self.map_size * self.cell_size + self.extra_ui_height
    if self.play:
      self.screen = pygame.display.set_mode((width, height))
      pygame.display.set_caption("RW4T")
    else:
      # Create a hidden surface
      self.screen = pygame.Surface((width, height))
    self._running = True

  def on_render(self):
    # Update environment info for rendering
    self.update_env_info()

    # Make background white
    if self.play:
      self.screen.fill((255, 255, 255))

    # Draw the grid
    for x in range(self.map_size):
      for y in range(self.map_size):
        rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                           self.cell_size, self.cell_size)
        if self.map[y, x] == utils.RW4T_State.obstacle.value:
          pygame.draw.rect(self.screen, (0, 0, 0), rect)
        elif (
            self.map[y, x] == utils.RW4T_State.circle.value
            and not (self.agent_pos == (x, y)
                     and self.agent_holding == utils.RW4T_State.circle.value)):
          rect = pygame.Rect(x * self.cell_size, (y + 1 / 3) * self.cell_size,
                             self.cell_size, self.cell_size)
          self.screen.blit(self.item_imgs["food"], rect)
        elif (
            self.map[y, x] == utils.RW4T_State.square.value
            and not (self.agent_pos == (x, y)
                     and self.agent_holding == utils.RW4T_State.square.value)):
          rect = pygame.Rect(x * self.cell_size, (y + 1 / 3) * self.cell_size,
                             self.cell_size, self.cell_size)
          self.screen.blit(self.item_imgs["medkit"], rect)
        elif self.map[y, x] == utils.RW4T_State.triangle.value:
          # cx, cy = (x * self.cell_size + self.cell_size // 2,
          #           y * self.cell_size + self.cell_size // 2)
          # points = [(cx, cy - self.cell_size // 2.5),
          #           (cx - self.cell_size // 2.5, cy + self.cell_size // 2.5),
          #           (cx + self.cell_size // 2.5, cy + self.cell_size // 2.5)]
          # pygame.draw.polygon(self.screen, (255, 0, 0), points)
          raise NotImplementedError
        elif self.map[y, x] == utils.RW4T_State.yellow_zone.value:
          pygame.draw.rect(self.screen, (255, 255, 0), rect)
        elif self.map[y, x] == utils.RW4T_State.orange_zone.value:
          pygame.draw.rect(self.screen, (255, 165, 0), rect)
        elif self.map[y, x] == utils.RW4T_State.red_zone.value:
          pygame.draw.rect(self.screen, (255, 0, 0), rect)
        elif self.map[y, x] == utils.RW4T_State.school.value:
          # text = self.font.render('S', True, (0, 0, 0))
          # text_rect = text.get_rect(
          #     center=(x * self.cell_size + (self.cell_size / 2),
          #             y * self.cell_size + (self.cell_size / 2)))
          # self.screen.blit(text, text_rect)
          pygame.draw.rect(self.screen, (255, 192, 203), rect)
        elif self.map[y, x] == utils.RW4T_State.hospital.value:
          text = self.font.render('H', True, (0, 0, 0))
          text_rect = text.get_rect(
              center=(x * self.cell_size + (self.cell_size / 2),
                      y * self.cell_size + (self.cell_size / 2)))
          self.screen.blit(text, text_rect)
        elif self.map[y, x] == utils.RW4T_State.park.value:
          text = self.font.render('P', True, (0, 0, 0))
          text_rect = text.get_rect(
              center=(x * self.cell_size + (self.cell_size / 2),
                      y * self.cell_size + (self.cell_size / 2)))
          self.screen.blit(text, text_rect)
        else:
          pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

    # Draw the agent as a blue triangle
    self._draw_agent()
    x, y = self.agent_pos

    # # Render the timer on the screen
    # timer_text = self.font.render(f'Time: {self.env.timer:.1f}s', True,
    #                               (0, 0, 0))
    # self.screen.blit(timer_text,
    #                  (10, 10))  # Position the timer at the top-left corner
    # # Render the score on the screen
    # score_text = self.font.render(f'Score: {self.env.c_task_reward}', True,
    #                               (0, 0, 0))
    # self.screen.blit(score_text,
    #                  ((self.env.map_size - 2.5) * self.cell_size,
    #                   10))  # Position the timer at the top-right corner

    # Render input box if needed
    if self.input_box:
      # Input box label
      lbl_surface = self.lbl_font.render('Option: ', True, self.BLACK)
      lbl_pos = (self.input_box_gui.x, self.input_box_gui.y - 30)
      self.screen.blit(lbl_surface, lbl_pos)
      # Input box itself
      txt_surface = self.ib_font.render(self.text, True, self.BLACK)
      self.screen.blit(txt_surface,
                       (self.input_box_gui.x + 5, self.input_box_gui.y + 10))
      pygame.draw.rect(self.screen, self.color, self.input_box_gui, 2)

    # Render the current selected option
    if self.low_level:
      display_text = f'{self.env.option} '
      if self.env.option in [
          item.value for item in self.env.rw4t_hl_actions_with_dummy
      ]:
        display_text += self.env.rw4t_hl_actions_with_dummy(
            self.env.option).name
      lbl_surface = self.lbl_font.render(display_text, True, self.BLACK)
      lbl_pos = (self.input_box_gui.x, self.input_box_gui.y + 70)
      self.screen.blit(lbl_surface, lbl_pos)

    # Draw "Rescue World" text centered below the grid
    domain_text = self.domain_text_font.render("Rescue World", True, (0, 0, 0))
    text_rect = domain_text.get_rect(
        center=(self.map_size * self.cell_size // 2,
                self.map_size * self.cell_size + self.extra_ui_height // 2))
    self.screen.blit(domain_text, text_rect)

    # Update display
    pygame.display.flip()

    # Capture frame if recording
    if self.record:
      frame = pygame.surfarray.array3d(self.screen)
      frame = np.transpose(frame, (1, 0, 2))  # (W, H, C) -> (H, W, C)
      self.frames.append(frame)

  def _draw_agent(self):
    x, y = self.agent_pos
    robot_img_name = 'default'
    if self.agent_holding == utils.RW4T_State.circle.value:
      robot_img_name = "pick_food"
    elif self.agent_holding == utils.RW4T_State.square.value:
      robot_img_name = "pick_medkit"
    robot_img = self.robot_imgs[robot_img_name]

    if robot_img_name == 'default':
      rect = pygame.Rect((x + 1 / 8) * self.cell_size, y * self.cell_size,
                         self.cell_size, self.cell_size)
    else:
      rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size,
                         self.cell_size)
    self.screen.blit(robot_img, rect)

  def on_cleanup(self):
    # Capture frame if recording
    if self.record and self.frames:
      Path(self.record_path).parent.mkdir(parents=True, exist_ok=True)
      imageio.mimsave(self.record_path, self.frames, fps=2)

    pygame.quit()

  def update_env_info(self):
    self.agent_pos = self.env.agent_pos
    self.agent_holding = self.env.agent_holding
    self.map = self.env.map
