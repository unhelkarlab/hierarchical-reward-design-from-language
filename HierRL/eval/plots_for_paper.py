import matplotlib.pyplot as plt
import numpy as np


# Annotate bars with values
def annotate_bars(bars, ax):
  '''
  Take a set of bars from a bar plot and writes their numeric heights
  above each bar.
  '''
  for bar in bars:
    height = bar.get_height()
    ax.annotate(
        f'{height:.1f}',
        xy=(bar.get_x() + bar.get_width() / 2,
            height),  # (x, y) coordinate where the annotation is anchored
        xytext=(0, 3),  # offset in points from the (x, y) coordinate
        textcoords="offset points",
        ha='center',  # horizontal alignment
        va='bottom',  # vertical alignment
        fontsize=14,
        fontweight='bold')


# ── p-value brackets with real bracket ends ──────────────────────────────
def add_pval_bracket(i,
                     p_text,
                     x_vals,
                     y_means,
                     y_errs,
                     bar_w,
                     ax,
                     h_gap=0.3,
                     v_tick=0.12,
                     text_pad=0.15):
  """
  Draws a [ bracket ] between the Flat & Hier bars of group i and
  writes the p-value above it.

  Parameters
  ----------
  i        : metric index (0-3)
  p_text   : p-value string (e.g. '<0.001')
  x_vals   : np.arange vector of x positions
  y_means  : the 'data' list (bar heights)
  y_errs   : the 'stds' list (errorbar heights)
  bar_w    : bar width
  h_gap    : vertical gap between bar top and horizontal bar (axes units)
  v_tick   : length of the little vertical ticks   (axes units)
  text_pad : extra space above the bracket for the text  (axes units)
  """
  # x-positions of left / right bar centres
  xl = x_vals[i] - bar_w / 2
  xr = x_vals[i] + bar_w / 2

  # height of bar top + error bar
  yl = y_means[i][0] + y_errs[i][0]
  yr = y_means[i][1] + y_errs[i][1]
  y_horiz = max(yl, yr) + h_gap  # horizontal segment y

  # --- draw bracket: left tick, horiz, right tick ---------------------
  ax.plot([xl, xl], [y_horiz - v_tick, y_horiz], color='black', lw=1.5)
  ax.plot([xl, xr], [y_horiz, y_horiz], color='black', lw=1.5)
  ax.plot([xr, xr], [y_horiz - v_tick, y_horiz], color='black', lw=1.5)

  # --- add p-value text ----------------------------------------------
  ax.annotate(f'p {p_text}',
              xy=((xl + xr) / 2, y_horiz + text_pad),
              ha='center',
              va='bottom',
              fontsize=14,
              fontweight='bold')

  return y_horiz + text_pad  # tell caller how high we went


def draw_group_bracket(x_left,
                       x_right,
                       label,
                       ypos,
                       ax,
                       trans,
                       v_tick,
                       txt_offset=0.02):
  """
    Draws  ┌────┐ style bracket between x_left and x_right (axes coords)
    and puts *label* centered beneath.
    """
  # left vertical tick
  ax.plot([x_left, x_left], [ypos, ypos - v_tick],
          transform=trans,
          lw=1.5,
          color='black',
          clip_on=False)
  # horizontal bar
  ax.plot([x_left, x_right], [ypos - v_tick, ypos - v_tick],
          transform=trans,
          lw=1.5,
          color='black',
          clip_on=False)
  # right vertical tick
  ax.plot([x_right, x_right], [ypos, ypos - v_tick],
          transform=trans,
          lw=1.5,
          color='black',
          clip_on=False)
  # text label
  ax.text((x_left + x_right) / 2,
          ypos - v_tick - txt_offset,
          label,
          transform=trans,
          ha='center',
          va='top',
          fontsize=16,
          fontweight='bold')


def bar_plot(x_labels,
             data,
             methods,
             colors=None,
             stds=None,
             p_values=None,
             x_group=None,
             bar_width=0.3,
             y_ticks=None,
             y_label='Score',
             y_anno=0,
             y_line=-0.10,
             figsize=(8, 4),
             tight_layout=True,
             layout_padding=[0, 0, 1, 1],
             legend_loc='lower right',
             legend_fontsize=14,
             title=None):

  if colors is None:
    # Default colors for the bars
    colors = ['salmon', 'skyblue']

  # Plot settings
  x = np.arange(len(x_labels))

  fig, ax = plt.subplots(figsize=figsize)

  err_style = dict(
      ecolor='dimgrey',
      elinewidth=1.5,
      capsize=5,
      capthick=1.5,
      #  zorder=0,
  )

  bar_containers = []
  for i, method in enumerate(methods):
    shift = (i - (len(methods) - 1) / 2) * bar_width
    if stds is not None:
      bar_container = ax.bar(x + shift, [d[i] for d in data],
                             yerr=[s[i] for s in stds],
                             width=bar_width,
                             label=method,
                             color=colors[i],
                             edgecolor='black',
                             linewidth=2,
                             error_kw=err_style)
    else:
      bar_container = ax.bar(x + shift, [d[i] for d in data],
                             width=bar_width,
                             label=method,
                             color=colors[i],
                             edgecolor='black',
                             linewidth=2)
    bar_containers.append(bar_container)

  for bar_container in bar_containers:
    annotate_bars(bar_container, ax)

  if p_values is not None:
    # draw every bracket & remember the tallest one
    y_top = 0
    for idx, p in enumerate(p_values):
      y_top = max(y_top, add_pval_bracket(idx, p, x, data, stds, bar_width, ax))
  else:
    y_top = max(bar.get_height() for bar_container in bar_containers
                for bar in bar_container)

  # finally enlarge the y-axis so every bracket/text fits comfortably
  current_bottom, _ = ax.get_ylim()
  ax.set_ylim(current_bottom, y_top + y_anno)

  # Axes and labels
  ax.set_ylabel(y_label, fontweight='bold', fontsize=16)
  ax.set_xticks(x)
  ax.set_xticklabels(x_labels, rotation=0, ha='center', fontsize=16)
  if y_ticks is not None:
    ax.set_yticks(y_ticks)
  ax.tick_params(axis='y', labelsize=16)
  ax.legend(loc=legend_loc, prop={'weight': 'bold', 'size': legend_fontsize})

  # Grid and Styling
  ax.grid(axis='y', linestyle='--', alpha=0.7, linewidth=1.5)
  ax.tick_params(axis='both', width=3, length=6)

  for spine in ax.spines.values():
    spine.set_linewidth(3)
  for label in ax.get_xticklabels():
    label.set_fontweight('bold')
  for label in ax.get_yticklabels():
    label.set_fontweight('bold')

  if x_group is not None:
    # ── grouped-label brackets under the x-tick labels ────────────────────────
    trans = ax.get_xaxis_transform()  # positions in axes coords
    v_tick = 0.015  # length of vertical ticks

    # coordinates for bracket endpoints (all in data units → convert to axes coords)
    # Convert the data x to axes coordinates: (x_data) / (plot width in data)
    # but easier: use the same data coordinates as x because transform='data' property;
    # However we used axis transform. For convenience use those data positions, convert to axis transform.

    # Group 1: "Persistence", "Safety", "Overall"
    x_left_group1 = x[x_group[0]['idx'][0]] - bar_width + x_group[0]['pos'][0]
    x_right_group1 = x[x_group[0]['idx']
                       [-1]] + bar_width + x_group[0]['pos'][-1]
    draw_group_bracket(x_left_group1, x_right_group1, x_group[0]['name'],
                       y_line, ax, trans, v_tick)

    # Group 2: "Chopping"
    x_left_group2 = x[x_group[1]['idx'][0]] - bar_width + x_group[1]['pos'][0]
    x_right_group2 = x[x_group[1]['idx']
                       [-1]] + bar_width + x_group[1]['pos'][-1]
    draw_group_bracket(x_left_group2, x_right_group2, x_group[1]['name'],
                       y_line, ax, trans, v_tick)

  if tight_layout:
    plt.tight_layout(rect=layout_padding)

  if title:
    ax.set_title(title, fontsize=18, fontweight='bold')

  plt.show()


def draw_user_eval():
  # Data from the table
  x_labels = ['Persistence', 'Safety', 'Overall', 'Chopping']
  data = [
      [2.42, 4.76],  # Persistence
      [4.67, 4.62],  # Safety
      [3.46, 4.64],  # Overall
      [1.70, 4.47]  # Chopping
  ]
  methods = ['Flat', 'Hier']
  stds = [
      [1.07, 0.54],  # Persistence
      [0.55, 0.62],  # Safety
      [0.57, 0.57],  # Overall
      [0.88, 0.93]  # Chopping
  ]
  p_values = ['< 0.001', '= 0.86', '< 0.001', '< 0.001']

  bar_plot(x_labels=x_labels,
           data=data,
           methods=methods,
           stds=stds,
           p_values=p_values,
           x_group=[{
               'name': 'Rescue World',
               'idx': [0, 2],
               'pos': [-0.15, 0]
           }, {
               'name': 'Kitchen',
               'idx': [3, 3],
               'pos': [-0.075, 0.075]
           }],
           y_ticks=[1, 2, 3, 4, 5],
           y_anno=0.75)


def draw_user_eval_perfect():
  # Data from the table
  x_labels = ['Persistence', 'Safety', 'Overall', 'Chopping']
  data = [
      [12.50, 76.92],  # Persistence
      [50.00, 61.54],  # Safety
      [12.50, 53.85],  # Overall
      [10.00, 71.43]  # Chopping
  ]
  methods = ['Flat', 'Hier']
  title = 'Proportion of rewards that received a perfect \nalignment score from all participants'

  bar_plot(x_labels=x_labels,
           data=data,
           methods=methods,
           x_group=[{
               'name': 'Rescue World',
               'idx': [0, 2],
               'pos': [-0.15, 0]
           }, {
               'name': 'Kitchen',
               'idx': [3, 3],
               'pos': [-0.075, 0.075]
           }],
           y_label='Percentage (%)',
           y_anno=9,
           y_line=-0.12,
           tight_layout=True,
           layout_padding=[0, 0, 1, 0.9],
           title=title)


def draw_code_gen_error():
  # Data from the table
  x_labels = ['Rescue', 'iTHOR', 'Kitchen']
  data = [
      [29.2, 12.5],
      [58.3, 9.1],
      [29.2, 16.7],
  ]
  title = 'Syntax Error (%)'

  bar_plot(x_labels=x_labels,
           data=data,
           methods=['Flat', 'Hier'],
           bar_width=0.25,
           y_label='Error (%)',
           y_anno=9,
           y_ticks=[0, 20, 40, 60, 80],
           figsize=(6, 2.7),
           tight_layout=True,
           legend_loc='upper right')


def draw_task_feasibility():
  # Data from the table
  x_labels = ['All', 'Syn', 'All', 'Syn']
  data = [[33.33, 54.17], [47.06, 61.90], [41.67, 58.33], [58.83, 70.00]]
  mthods = ['Flat', 'Hier']
  title = 'Task Feasibility (%)'

  bar_plot(x_labels=x_labels,
           data=data,
           methods=mthods,
           x_group=[{
               'name': 'Rescue World',
               'idx': [0, 1],
               'pos': [-0.075, 0.075]
           }, {
               'name': 'Kitchen',
               'idx': [2, 3],
               'pos': [-0.075, 0.075]
           }],
           bar_width=0.3,
           y_label='Feasibility (%)',
           y_anno=8,
           y_line=-0.13,
           figsize=(8, 4),
           tight_layout=True,
           layout_padding=[0, 0, 1, 0.95],
           title=title)


def draw_task_feasibility_new():
  # Data from the table
  x_labels = ['Rescue', 'iTHOR', 'Kitchen']
  data = [[47.1, 61.9], [30.0, 72.7], [58.8, 70.0]]
  title = 'Task Completion (%)'

  bar_plot(x_labels=x_labels,
           data=data,
           methods=['Flat', 'Hier'],
           bar_width=0.25,
           y_label='Completion (%)',
           y_anno=11,
           y_ticks=[0, 20, 40, 60, 80],
           figsize=(6, 2.7),
           tight_layout=True)


def draw_numerical_eval():
  # Data from the table
  x_labels = ['Persistence', 'Safety', 'Overall', 'Chopping']
  data = [[20.00, 12.50, 76.92], [0.00, 62.50, 76.92], [0.00, 12.50, 69.23],
          [0.00, 10.00, 92.86]]
  methods = ['Task', 'Flat', 'Hier']
  title = 'Numerical Evaluation'

  bar_plot(x_labels=x_labels,
           data=data,
           methods=methods,
           colors=['lightgray', 'salmon', 'skyblue'],
           x_group=[{
               'name': 'Rescue World',
               'idx': [0, 2],
               'pos': [-0.15, 0]
           }, {
               'name': 'Kitchen',
               'idx': [3, 3],
               'pos': [-0.075, 0.075]
           }],
           bar_width=0.25,
           y_label='Percentage (%)',
           y_anno=12,
           y_line=-0.12,
           tight_layout=True,
           layout_padding=[0, 0, 1, 0.9],
           legend_loc='upper left',
           legend_fontsize=12,
           title=title)


# draw_user_eval()
# draw_user_eval_perfect()
draw_code_gen_error()
# draw_task_feasibility()
draw_task_feasibility_new()
# draw_numerical_eval()
