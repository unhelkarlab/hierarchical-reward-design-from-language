from agent.executor.low import EnvState


class IL_Agent:

  def __init__(self) -> None:
    self.cur_intent = None
    self._task = None
    self.model = None

  def load_model(self, model_path):
    raise NotImplementedError

  def step(self, env: EnvState):
    raise NotImplementedError

  def get_cur_intent(self):
    return self.cur_intent
