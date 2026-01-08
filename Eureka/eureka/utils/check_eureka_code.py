import ast
import astor


def extract_eureka_code(code_string):
  lines = code_string.split("\n")
  in_function = False
  func_indent = None
  func_lines = []

  for line in lines:
    stripped = line.lstrip()
    indent = len(line) - len(stripped)

    if not in_function:
      if stripped.startswith("def "):
        in_function = True
        func_indent = indent
        func_lines.append(stripped)
    else:
      # End of function if we hit a line with less indentation
      if stripped and indent <= func_indent:
        break
      func_lines.append(line)

  return "\n".join(func_lines)


def check_for_statefulness(code: str):
  """
  Checks if the code defines function attributes or uses global variables.
  Returns a dict indicating what was detected.
  """
  tree = ast.parse(code)
  result = {
      "uses_function_attributes": False,
      "uses_global_variables": False,
      "function_names": [],
      "details": []
  }

  class StatefulnessDetector(ast.NodeVisitor):

    def __init__(self):
      self.current_function = None

    def visit_FunctionDef(self, node):
      self.current_function = node.name
      result["function_names"].append(node.name)
      self.generic_visit(node)

    def visit_Attribute(self, node):
      # Detects things like reward.prev_state or function_name.attr
      if isinstance(node.value, ast.Name):
        if node.value.id in result["function_names"]:
          result["uses_function_attributes"] = True
          result["details"].append(
              f"Function attribute used: {astor.to_source(node).strip()}")
      self.generic_visit(node)

    def visit_Global(self, node):
      result["uses_global_variables"] = True
      for name in node.names:
        result["details"].append(f"Global variable declared: {name}")
      self.generic_visit(node)

    def visit_Name(self, node):
      # Catch any names used that look like external vars
      self.generic_visit(node)

  detector = StatefulnessDetector()
  detector.visit(tree)
  return result


if __name__ == "__main__":
  code_1 = """
x = 5

def foo():
    print("Hello")
    y = 10
    return y

z = 20
"""
  code_2 = """
def foo():
    print("Hello")
    y = 10
    return y

z = 20
"""
  code_3 = """
x = 5

def foo():
    print("Hello")
    y = 10
    return y
"""
  code_4 = """
def foo():
    print("Hello")
    y = 10
    return y
"""
  #   print(extract_eureka_code(code_4))

  func_1 = '''def get_user_pref_reward(state: Dict, action: int) -> Tuple[float, Dict[str, float]]:
    """
    state: the current state of the environment.
    action: the (low-level) action that the agent is about to perform in the current state.

    The user preference focuses on two aspects:
    1. Encourages picking up an object type that's the same as the last delivered object type if possible.
    2. Penalizes the agent for going through danger zones while delivering.
    """
    global counter
    current_position = state['pos']
    holding = state['holding']
    map_state = state['map']

    # Danger zone states
    danger_zones = {
        rw4t_utils.RW4T_State.orange_zone.value,
        rw4t_utils.RW4T_State.red_zone.value
    }
    
    # Delivered object type (tracking agent's last delivery)
    if not hasattr(get_user_pref_reward, "last_delivered"):
        get_user_pref_reward.last_delivered = None
    
    reward = 0.0
    reward_components = {}
    
    # Component 1: Encourage picking up the same type of object last delivered
    if action == rw4t_utils.RW4T_LL_Actions.pick.value and holding == rw4t_utils.Holding_Obj.empty.value:
        x, y = current_position
        current_cell = map_state[y, x]
        
        if current_cell in [rw4t_utils.RW4T_State.circle.value, rw4t_utils.RW4T_State.square.value]:
            if get_user_pref_reward.last_delivered is not None:
                if current_cell == get_user_pref_reward.last_delivered:
                    reward += 5.0
                    reward_components['same_type_pickup'] = 5.0
                else:
                    reward += 1.0
                    reward_components['diff_type_pickup'] = 1.0
            else:
                reward_components['no_prior_delivery'] = 0.0

    # Component 2: Penalize being in danger zones during delivery
    if holding != rw4t_utils.Holding_Obj.empty.value:
        x, y = current_position
        if map_state[y, x] in danger_zones:
            reward -= 10.0
            reward_components['danger_zone_penalty'] = -10.0

    if action == rw4t_utils.RW4T_LL_Actions.drop.value:
        get_user_pref_reward.last_delivered = holding

    return reward, reward_components
'''
  print(check_for_statefulness(func_1))
