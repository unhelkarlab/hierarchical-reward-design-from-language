import os
from dataclasses import dataclass

from gym_cooking.utils.replay import Replay

from agent.mind.agent_new import AgentSetting
from agent.il_agents.iql.iql_agent import IQL_Agent
from agent.il_agents.llm.llm_futures_agent import LLM_Futures_Agent


@dataclass
class Agent_Config:
  # Agent type should be in the format of llm-type_il-type
  # llm-type can be hier, reg, or future
  # il-type only supports iql
  agent_type: str
  speed: float = 3

  # LLM agents
  auto: bool = True
  pref: str = ''
  human_instr: str = ''
  q_val: bool = False
  interpolation: bool = False
  fast: bool = False

  # IL agents
  num_demos: int = 3
  input_size: int = 0


def get_ai_agent(agent_config: Agent_Config, replay, p_str):
  agent_type = agent_config.agent_type
  assert '_' in agent_type and len(agent_type.split('_')) == 2
  llm_type = agent_type.split('_')[0]
  il_type = agent_type.split('_')[1]
  assert llm_type != 'none' or il_type != 'none'

  # Load llm agent
  llm_agent = None
  if llm_type != 'none':
    if llm_type == 'future':
      agent_set = AgentSetting(mode='LLM_Futures_Agent',
                               hl_mode='prompt',
                               ll_mode='',
                               prompt_style='lang',
                               top_k_llm=-1,
                               no_hl_rec=True,
                               text_desc=False,
                               prev_skill=True,
                               speed=3,
                               pref=p_str,
                               operation='cond',
                               fast_il=False,
                               gen_mode='5_unranked',
                               interpolation=False,
                               q_eval=False,
                               user_reward=False)
      replay = Replay()  # not used
      # Init agent and load a IL model if needed
      llm_agent = LLM_Futures_Agent(agent_set, replay)
    else:
      raise NotImplementedError

  # Load imitation learning agent
  if il_type != 'none':
    raise NotImplementedError
  else:
    if llm_agent is not None:
      llm_agent.set_il_model('none', None)

  return llm_agent
