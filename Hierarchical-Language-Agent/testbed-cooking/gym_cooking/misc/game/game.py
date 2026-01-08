import os
from pathlib import Path
import pygame.draw
import imageio

import gym_cooking
from gym_cooking.utils.core import *
from gym_cooking.misc.game.utils import *
import numpy as np

graphics_dir = 'misc/game/graphics'
_image_library = {}


def get_image(path):
  global _image_library
  image = _image_library.get(path)
  if image == None:
    canonicalized_path = os.path.join(
        Path(gym_cooking.__file__).absolute().parent,
        path.replace('/', os.sep).replace('\\', os.sep))
    image = pygame.image.load(canonicalized_path)
    _image_library[path] = image
  return image


class Game:

  def __init__(self,
               env,
               play=False,
               record=False,
               record_path="recordings/gameplay.mp4"):
    pygame.init()

    self.record = record
    self.record_path = record_path
    self.frames = []

    self._running = True
    self.env = env
    self.world = env.world
    self.sim_agents = env.sim_agents
    self.current_agent = self.sim_agents[0]
    self.order_scheduler = env.order_scheduler
    self.play = play

    # Visual parameters
    self.scale = 80  # num pixels per tile
    self.holding_scale = 0.5
    self.container_scale = 0.7
    self.width = self.scale * (self.world.width + 0)
    self.height = self.scale * (self.world.height + 2)
    self.world_size = (self.world.width, self.world.height)
    self.tile_size = (self.scale, self.scale)
    self.holding_size = tuple(
        (self.holding_scale * np.asarray(self.tile_size)).astype(int))
    self.container_size = tuple(
        (self.container_scale * np.asarray(self.tile_size)).astype(int))
    self.holding_container_size = tuple(
        (self.container_scale * np.asarray(self.holding_size)).astype(int))
    # self.font = pygame.font.SysFont('arialttf', 10)

    self.small_font = pygame.font.SysFont('Times', 20)
    self.font = pygame.font.SysFont('Times', 40)
    self.large_font = pygame.font.SysFont('Times', 100)

    self.__plot_elements = []

    # Input box to enter options
    self.input_box = False  # whether to display input box
    self.input_box_gui = pygame.Rect(self.width + 20, 50, 200, 50)
    self.WHITE = (255, 255, 255)
    self.BLACK = (0, 0, 0)
    self.GRAY = (200, 200, 200)
    self.ib_font = pygame.font.Font(None, 40)
    self.lbl_font = pygame.font.Font(None, 30)
    self.color_active = pygame.Color('dodgerblue')
    self.color_inactive = self.GRAY
    self.color = self.color_inactive
    self.active = False
    self.text = ''

    self.show_order = False
    self.show_domain_name = True
    self.domain_name_height = 100
    self.domain_text_font = pygame.font.SysFont('Arial', 36)

  def on_init(self):
    pygame.init()
    if self.play:
      width = self.width + (250 if self.input_box else 0)
      if self.show_domain_name:
        height = self.scale * self.world.height + self.domain_name_height
      elif not self.show_order:
        height = self.scale * self.world.height
      else:
        height = self.height
      self.screen = pygame.display.set_mode((width, height))
    else:
      # Create a hidden surface
      self.screen = pygame.Surface((self.width, self.height))
      # self.screen = None
    self._running = True

  def on_event(self, event):
    if event.type == pygame.QUIT:
      self._running = False

  def on_render(self, paused=False, chat='', replay=False):
    self.__plot_elements = []
    # try:
    #   self.screen.fill(Color.FLOOR)
    # except:
    #   print('yeehaw')
    #   return
    if self.play:
      self.screen.fill(Color.FLOOR)
    self.__plot_elements.append(('Fill', {'color': Color.FLOOR}))
    objs = {
        'normal': [],
        'chopping': [],
        'cook': [],
        'held': [],
    }

    # Draw gridsquares
    for o_list in self.world.objects.values():
      for o in o_list:
        if isinstance(o, GridSquare):
          # change grid square
          if not replay and o.name == self.env.chg_grid:
            o1 = Counter(o.location)
            self.draw_gridsquare(o1)
          else:
            self.draw_gridsquare(o)
        elif isinstance(self.world.get_gridsquare_at(o.location), Pot):
          objs['cook'].append(o)
        elif len(
            o.contents) == 1 and o.contents[0].full_name.startswith('Chopping'):
          objs['chopping'].append(o)
        elif o.is_held:
          objs['held'].append(o)
        else:
          objs['normal'].append(o)

    # Draw objects not held by agents
    for o in objs['normal']:
      self.draw_object(o)

    # Draw agents and their holdings
    for agent in self.sim_agents:
      self.draw_agent(agent)

    # Draw cooking objects
    for o in objs['cook']:
      if o.is_cooking():
        self.draw_cooking_object(o)
      else:
        assert o.is_cooked()
        self.draw_cooked_object_in_pot(o)

    # Draw chopping objects
    for o in objs['chopping']:
      self.draw_chopping_object(o)

    if self.show_order:
      # Draw current orders
      self.draw_current_orders()

      # Draw soup hint
      self.draw_soup_hint()

      # Draw current time
      self.draw_current_time()

    if self.show_domain_name:
      self.draw_domain_name()

    # print("[Plot]", self.__plot_elements, flush=True)

    if paused and self.play:
      self.draw_paused()
    if chat:
      self.draw_multiline(chat)

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

    if self.play:
      pygame.display.flip()
      pygame.display.update()

    if self.record:
      frame = pygame.surfarray.array3d(self.screen)
      frame = np.transpose(frame, (1, 0, 2))  # (W, H, C) -> (H, W, C)
      self.frames.append(frame)

  def draw_gridsquare(self, gs):
    sl = self.scaled_location(gs.location)
    fill = pygame.Rect(sl[0], sl[1], self.scale, self.scale)
    box = (sl[0], sl[1], self.scale, self.scale)

    if isinstance(gs, Counter):
      pygame.draw.rect(self.screen, Color.COUNTER, fill)
      self.__plot_elements.append(('Rect', {
          'color': Color.COUNTER,
          'box': box
      }))
      pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
      self.__plot_elements.append(('Rect', {
          'color': Color.COUNTER_BORDER,
          'box': box,
          'width': 1
      }))

    elif isinstance(gs, Delivery):
      pygame.draw.rect(self.screen, Color.DELIVERY, fill)
      self.__plot_elements.append(('Rect', {
          'color': Color.DELIVERY,
          'box': box
      }))
      self.draw('delivery', self.tile_size, sl)

    elif isinstance(gs, Bin) or isinstance(gs, Tile) or isinstance(
        gs, Cutboard) or isinstance(gs, Pot):
      pygame.draw.rect(self.screen, Color.COUNTER, fill)
      self.__plot_elements.append(('Rect', {
          'color': Color.COUNTER,
          'box': box
      }))
      pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
      self.__plot_elements.append(('Rect', {
          'color': Color.COUNTER_BORDER,
          'box': box,
          'width': 1
      }))
      self.draw(gs.name, self.tile_size, sl)

    return

  def draw(self, path, size, location):
    image_path = '{}/{}.png'.format(graphics_dir, path)
    image = pygame.transform.scale(get_image(image_path), size)
    self.screen.blit(image, location)
    self.__plot_elements.append(('Image', {
        'path': path,
        'size': size,
        'location': location
    }))

  def draw_bar(self, a, b, c, d, color):
    bar = pygame.Rect(a, b, c, d)
    pygame.draw.rect(self.screen, color, bar)
    self.__plot_elements.append(('Rect', {'color': color, 'box': (a, b, c, d)}))

  def put_text(self, font, text, color, loc):
    if font == self.small_font:
      size = 20
    elif font == self.font:
      size = 40
    elif font == self.large_font:
      size = 100
    else:
      raise NotImplementedError
    # width = int(font.size(text)[0]) + 1
    if self.play:
      t = font.render(text, True, color)
      self.screen.blit(t, loc)
    self.__plot_elements.append(('Text', {
        'text': text,
        'color': color,
        'location': loc,
        'size': size,
        'width': 100
    }))

  def draw_agent(self, agent):
    self.draw('agent-{}'.format(agent.color), self.tile_size,
              self.scaled_location(agent.location))
    self.draw_agent_object(agent.holding)

  def draw_agent_object(self, obj):
    # Holding shows up in bottom right corner.
    if obj is None:
      return
    if any([isinstance(c, Plate) for c in obj.contents]):
      self.draw('Plate', self.holding_size, self.holding_location(obj.location))
      if len(obj.contents) > 1:
        plate = obj.unmerge('Plate')
        self.draw(obj.full_name, self.holding_container_size,
                  self.holding_container_location(obj.location))
        obj.merge(plate)
    else:
      self.draw(obj.full_name, self.holding_size,
                self.holding_location(obj.location))

  def draw_object(self, obj):
    if obj is None:
      return
    if any([isinstance(c, Plate) for c in obj.contents]):
      self.draw('Plate', self.tile_size, self.scaled_location(obj.location))
      if len(obj.contents) > 1:
        plate = obj.unmerge('Plate')
        self.draw(obj.full_name, self.container_size,
                  self.container_location(obj.location))
        obj.merge(plate)
    else:
      self.draw(obj.full_name, self.tile_size,
                self.scaled_location(obj.location))

  def draw_cooking_object(self, obj):
    if obj is None:
      return
    self.draw_by_scale_loc(obj.full_name.replace('Cooking', 'Cooked'),
                           (obj.location[0], obj.location[1] - 0.05), 0.6)
    loc = self.scaled_location(obj.location)
    self.draw_bar(
        loc[0] + (0.6 - 0.4) * self.tile_size[0],
        loc[1] + (0.5 + 0.35) * self.tile_size[1], 0.7 * self.tile_size[0] *
        obj.contents[0].state._rest_steps / COOKING_TIME_SECONDS,
        0.1 * self.tile_size[1], (220, 70, 1))
    self.draw_by_scale_loc('pot_clippart',
                           (obj.location[0] - 0.45, obj.location[1] + 0.4), 0.4)

  def draw_cooked_object_in_pot(self, obj):
    if obj is None:
      return
    #  cooking -> cooked
    if 'Cooked' in obj.full_name:
      self.draw_by_scale_loc(obj.full_name,
                             (obj.location[0], obj.location[1] - 0.05), 0.6)
      # obj is cooked, show the bar before the cooked object is fired
      loc = self.scaled_location(obj.location)
      self.draw_bar(
          loc[0] + (0.6 - 0.4) * self.tile_size[0],
          loc[1] + (0.5 + 0.35) * self.tile_size[1], 0.7 * self.tile_size[0] *
          obj.contents[0].state._rest_steps / COOKED_BEFORE_FIRE_TIME_SECONDS,
          0.1 * self.tile_size[1], (220, 70, 1))
      self.draw_by_scale_loc('fire',
                             (obj.location[0] - 0.45, obj.location[1] + 0.4),
                             0.4)
    elif 'Fire' in obj.full_name:
      fire = obj.unmerge('Fire')
      self.draw_by_scale_loc(obj.full_name, (obj.location[0], obj.location[1]),
                             0.8)
      self.draw_by_scale_loc('full_fire', (obj.location[0], obj.location[1]),
                             0.8)
      ratio = fire.rest / FIRE_PUTOUT_TIME_SECONDS
      obj.merge(fire)
      loc = self.scaled_location(obj.location)
      self.draw_bar(loc[0] + (0.6 - 0.4) * self.tile_size[0],
                    loc[1] + (0.5 + 0.35) * self.tile_size[1],
                    0.7 * self.tile_size[0] * ratio, 0.1 * self.tile_size[1],
                    (220, 70, 1))
      # self.draw_by_scale_loc('fire', (obj.location[0] - 0.45, obj.location[1] + 0.4), 0.4)
    else:
      self.draw_by_scale_loc(obj.full_name,
                             (obj.location[0], obj.location[1] - 0.05), 0.6)

  def draw_chopping_object(self, obj):
    if obj is None:
      return
    self.draw(obj.full_name.replace('Chopping', 'Fresh'), self.container_size,
              self.container_location(obj.location))
    loc = self.scaled_location(obj.location)
    self.draw_bar(
        loc[0] + (0.5 - 0.4) * self.tile_size[0],
        loc[1] + (0.5 + 0.35) * self.tile_size[1], 0.7 * self.tile_size[0] *
        obj.contents[0].state._rest_steps / CHOPPING_NUM_STEPS,
        0.1 * self.tile_size[1], (220, 70, 1))

  def draw_by_scale_loc(self, path, loc, scale):
    loc = self.scaled_location(loc)
    loc = tuple((np.asarray(loc) + self.scale * (1 - scale) / 2).astype(int))
    size = tuple((scale * np.asarray(self.tile_size)).astype(int))
    self.draw(path, size, loc)

  def draw_current_orders(self):
    if self.world.arglist.user_recipy and self.order_scheduler is not None:
      for i, (order, restTime, timeLimit,
              bonus) in enumerate(self.order_scheduler.current_orders):
        self.draw_current_order(i, copy.deepcopy(order), restTime)

    # draw success and failed ones
    self.put_text(self.small_font, "Score", (40, 80, 180),
                  ((0 + 0.15) * self.tile_size[0],
                   (self.world.height + 1.6) * self.tile_size[1]))
    self.put_text(
        self.font,
        str(self.order_scheduler.reward + self.order_scheduler.p_reward),
        (40, 80, 180), ((0.5 + 0.3) * self.tile_size[0],
                        (self.world.height + 1.4) * self.tile_size[1]))

  def draw_current_order(self, idx, obj, t):
    # order
    obj_loc = (idx * 1.25, self.world.height)
    if any([isinstance(c, Plate) for c in obj.contents]):
      self.draw('Plate', self.tile_size, self.scaled_location(obj_loc))
      if len(obj.contents) > 1:
        plate = obj.unmerge('Plate')
        self.draw(obj.full_name, self.container_size,
                  self.container_location(obj_loc))
        obj.merge(plate)
    else:
      self.draw(obj.full_name, self.tile_size, self.scaled_location(obj_loc))
    colors = {
        0: (220, 70, 1),
        0.25: (249, 168, 37),
        0.5: (255, 201, 40),
        0.75: (124, 178, 66),
        1.0: (0, 138, 122),
    }
    w = t / MAX_ORDER_LENGTH_SECONDS
    color = (0, 0, 0)
    for l, r in zip([0., 0.25, 0.5, 0.75], [0.25, 0.5, 0.75, 1.0]):
      if l - 1e-3 <= w and w <= r + 1e-3:
        color = np.array(colors[l]) + (np.array(colors[r]) -
                                       np.array(colors[l])) * (w - l) / (r - l)
    self.draw_bar((idx * 1.25 + 0.1) * self.tile_size[0],
                  (self.world.height + 1) * self.tile_size[1],
                  int(self.tile_size[0] * 0.9 * t / MAX_ORDER_LENGTH_SECONDS),
                  self.tile_size[1] // 5, color)

    name = obj.full_name
    name = name.replace("CookedLettuce-CookedOnion-CookedTomato-Plate", "David")
    name = name.replace("CookedLettuce-CookedOnion-Plate", "Alice")
    # name = name.replace("CookedOnion-Plate", "Alice")
    name = name.replace("CookedLettuce-CookedTomato-Plate", "Bob")
    name = name.replace("CookedOnion-CookedTomato-Plate", "Cathy")
    pos = self.scaled_location(obj_loc)
    self.put_text(self.small_font, name, (40, 40, 100),
                  (pos[0] + self.tile_size[0] * 0.2, pos[1]))

  def draw_soup_hint(self):
    if self.world.arglist.user_recipy:
      return
    self.put_text(self.small_font, "Hint:", (0, 0, 0),
                  ((self.world.width - 3.3) * self.tile_size[0],
                   (self.world.height + 0.25) * self.tile_size[1]))

    self.put_text(self.small_font, "Alice", (40, 40, 100),
                  ((self.world.width - 2.6) * self.tile_size[0],
                   (self.world.height + 0.25) * self.tile_size[1]))
    self.draw("ChoppedLettuce-ChoppedOnion",
              (self.tile_size[0] / 2.5, self.tile_size[1] / 2.5),
              ((self.world.width - 2.0) * self.tile_size[0],
               (self.world.height + 0.2) * self.tile_size[1]))

    self.put_text(self.small_font, "Bob", (40, 40, 100),
                  ((self.world.width - 1.1) * self.tile_size[0],
                   (self.world.height + 0.25) * self.tile_size[1]))
    self.draw("ChoppedLettuce-ChoppedTomato",
              (self.tile_size[0] / 2.5, self.tile_size[1] / 2.5),
              ((self.world.width - 0.5) * self.tile_size[0],
               (self.world.height + 0.2) * self.tile_size[1]))

    self.put_text(self.small_font, "Cathy", (40, 40, 100),
                  ((self.world.width - 2.6) * self.tile_size[0],
                   (self.world.height + 0.75) * self.tile_size[1]))
    self.draw("ChoppedOnion-ChoppedTomato",
              (self.tile_size[0] / 2.5, self.tile_size[1] / 2.5),
              ((self.world.width - 2.0) * self.tile_size[0],
               (self.world.height + 0.7) * self.tile_size[1]))

    self.put_text(self.small_font, "David", (40, 40, 100),
                  ((self.world.width - 1.1) * self.tile_size[0],
                   (self.world.height + 0.75) * self.tile_size[1]))
    self.draw("ChoppedLettuce-ChoppedOnion-ChoppedTomato",
              (self.tile_size[0] / 2.5, self.tile_size[1] / 2.5),
              ((self.world.width - 0.5) * self.tile_size[0],
               (self.world.height + 0.7) * self.tile_size[1]))
    pass

  def draw_current_time(self):
    # time_render = f"Time: {self.env.current_time: .1f}/{60: .1f}"
    time_render = f"Time: {self.env.current_time: .1f}/{self.env.arglist.max_num_timesteps: .1f}"
    self.put_text(self.small_font, time_render, (220, 70, 1),
                  ((self.world.width - 2.0) * self.tile_size[0],
                   (self.world.height + 1.6) * self.tile_size[1]))

  def draw_paused(self):
    self.put_text(self.large_font, "PAUSED", (255, 0, 0),
                  (self.world.width * 0.2 * self.tile_size[0],
                   self.world.height // 2 * self.tile_size[1]))

  def draw_domain_name(self):
    domain_text = self.domain_text_font.render("Kitchen", True, (0, 0, 0))
    text_rect = domain_text.get_rect(center=(self.scale * self.world.width // 2,
                                             self.scale * self.world.height +
                                             self.domain_name_height // 2))
    self.screen.blit(domain_text, text_rect)

  def scaled_location(self, loc):
    """Return top-left corner of scaled location given coordinates loc, e.g. (3, 4)"""
    return tuple(self.scale * np.asarray(loc))

  def holding_location(self, loc):
    """Return top-left corner of location where agent holding will be drawn (bottom right corner) given coordinates loc, e.g. (3, 4)"""
    scaled_loc = self.scaled_location(loc)
    return tuple((np.asarray(scaled_loc) + self.scale *
                  (1 - self.holding_scale)).astype(int))

  def container_location(self, loc):
    """Return top-left corner of location where contained (i.e. plated) object will be drawn, given coordinates loc, e.g. (3, 4)"""
    scaled_loc = self.scaled_location(loc)
    return tuple((np.asarray(scaled_loc) + self.scale *
                  (1 - self.container_scale) / 2).astype(int))

  def holding_container_location(self, loc):
    """Return top-left corner of location where contained, held object will be drawn given coordinates loc, e.g. (3, 4)"""
    scaled_loc = self.scaled_location(loc)
    factor = (1 - self.holding_scale) + \
        (1 - self.container_scale) / 2 * self.holding_scale
    return tuple((np.asarray(scaled_loc) + self.scale * factor).astype(int))

  def on_cleanup(self):
    # pygame.display.quit()
    if self.record and self.frames:
      Path(self.record_path).parent.mkdir(parents=True, exist_ok=True)
      imageio.mimsave(self.record_path, self.frames, fps=2)

    pygame.quit()

  def get_visualization(self):
    return self.__plot_elements

  def draw_multiline(self, text: str):
    # pygame.draw.rect(self.screen, (255, 255, 255), (0, 0, 5000, 5000))
    font = pygame.font.SysFont("timesnewroman", 24, bold=True)  # 24
    pos = (10, 10)

    # 2D array where each row is a list of words.
    words = [words for words in text.splitlines()]
    space = font.size(' ')[0]  # The width of a space.
    # 500, self.screen.get_size()[1]
    max_width, max_height = self.screen.get_size()
    x, y = pos
    word_height = 1
    # color = (180, 0, 0)
    for line in words:
      # if "AI Output:" in line: color = (0, 0, 180)
      for word in line:
        word_surface = font.render(word, 0, (0, 0, 180))
        word_width, word_height = word_surface.get_size()
        if x + word_width >= max_width:
          x = pos[0]  # Reset the x.
          y += word_height  # Start on new row.
        self.screen.blit(word_surface, (x, y))
        x += word_width
      x = pos[0]  # Reset the x.
      y += word_height  # Start on new row.
