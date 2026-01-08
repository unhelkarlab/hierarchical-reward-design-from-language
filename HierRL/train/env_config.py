from pathlib import Path

PNP_HL_ENV_PARAMS_TASK_PREF = {
    'algo': 'dqn',
    'hl_pref': 'task',  # all / high / task
    'hl_pref_r': True,
    'll_model_name': 'll_model_wo_llpref',
    'scene': 'FloorPlan20',
}

PNP_HL_ENV_PARAMS_HIGH_PREF = {
    'algo': 'dqn',
    'hl_pref': 'high',  # all / high / task
    'hl_pref_r': True,
    'll_model_name': 'll_model_w_llpref',
    'scene': 'FloorPlan20',
}

PNP_HL_ENV_PARAMS_FLATSA_PREF = {
    'algo': 'VariableStepDQN',
    'hl_pref': 'high',  # all / high / task
    'hl_pref_r': False,
    'll_model_name': 'll_model_wflat_llpref',
    'scene': 'FloorPlan20',
}

PNP_HL_ENV_PARAMS_ALL_PREF = {
    'algo': 'dqn',
    'hl_pref': 'all',  # all / high / task
    'hl_pref_r': True,
    'll_model_name': 'll_model_w_llpref',
    'scene': 'FloorPlan20',
}

PNP_LL_ENV_PARAMS_TASK_PREF = {
    'algo': 'ppo',
    'll_pref': False,
    'one_network': False,
    'option_to_use': 0,
    'scene': 'FloorPlan20'
}

PNP_LL_ENV_PARAMS_LOW_PREF = {
    'algo': 'ppo',
    'll_pref': True,
    'one_network': False,
    'option_to_use': 1,
    'scene': 'FloorPlan20',
}

PNP_LL_ENV_PARAMS_FLATSA_PREF = {
    'algo': 'ppo',
    'll_pref': True,
    'hl_pref_r': False,
    'one_network': False,
    'option_to_use': 0,
    'scene': 'FloorPlan20'
}

RW4T_HL_ENV_PARAMS_TASK_PREF = {
    'algo': 'dqn',
    'hl_pref': 'task',  # all / high / task
    'hl_pref_r': True,
    'pbrs_r': False,
    'll_model_name': 'll_model_wo_llpref',
    'map_num': 8,
    'convenience_features': False
}

RW4T_HL_ENV_PARAMS_HIGH_PREF = {
    'algo': 'dqn',
    'hl_pref': 'high',  # all / high / task
    'hl_pref_r': True,
    'pbrs_r': False,
    'll_model_name': 'll_model_w_llpref',
    'map_num': 8,
    'convenience_features': False
}

RW4T_HL_ENV_PARAMS_FLATSA_PREF_EUREKA = {
    'algo': 'VariableStepDQN',
    'hl_pref': 'high',  # all / high / task
    'hl_pref_r': True,
    'pbrs_r': False,
    'll_model_name': 'll_model_w_llpref',
    'map_num': 8,
    'convenience_features': False
}

RW4T_HL_ENV_PARAMS_FLATSA_PREF_NONEUREKA = {
    'algo': 'VariableStepDQN',
    'hl_pref': 'high',  # all / high / task
    'hl_pref_r': False,
    'pbrs_r': False,
    'll_model_name': 'll_model_wflat_llpref',
    'map_num': 8,
    'convenience_features': False
}

RW4T_HL_ENV_PARAMS_ALL_PREF = {
    'algo': 'dqn',
    'hl_pref': 'all',  # all / high / task
    'hl_pref_r': True,
    'pbrs_r': False,
    'll_model_name': 'll_model_w_llpref',
    'map_num': 8,
    'convenience_features': False
}

RW4T_LL_ENV_PARAMS_TASK_PREF = {
    'algo': 'ppo',
    'll_pref': False,
    'one_network': True,
    'option_to_use': None,
    'map_num': 8,
    'convenience_features': False
}

RW4T_LL_ENV_PARAMS_LOW_PREF = {
    'algo': 'ppo',
    'll_pref': True,
    'one_network': True,
    'option_to_use': None,
    'map_num': 8,
    'convenience_features': False
}

RW4T_LL_ENV_PARAMS_FLATSA_PREF_EUREKA = {
    'algo': 'ppo',
    'll_pref': True,
    'one_network': True,
    'option_to_use': None,
    'map_num': 8,
    'convenience_features': False
}

RW4T_LL_ENV_PARAMS_FLATSA_PREF_NONEUREKA = {
    'algo': 'ppo',
    'll_pref': True,
    'hl_pref_r': False,
    'one_network': True,
    'option_to_use': None,
    'map_num': 8,
    'convenience_features': False
}

OC_HL_ENV_PARAMS_TASK_PREF = {
    'algo': 'dqn',
    'hl_pref': False,
    'eval_with_hl_pref': False,
    'hl_pref_r': False,
    'pbrs_r': False,
    'ez': True,
    'salad': True,
    'serve': False,
    'convenience_features': False
}

OC_HL_ENV_PARAMS_HIGH_PREF = {
    'algo': 'dqn',
    'hl_pref': True,
    'eval_with_hl_pref': True,
    'hl_pref_r': True,
    'pbrs_r': False,
    'ez': True,
    'salad': True,
    'serve': False,
    'convenience_features': False
}

OC_HL_ENV_PARAMS_FLATSA_PREF_EUREKA = {
    'algo': 'dqn',
    'hl_pref': True,
    'eval_with_hl_pref': True,
    'hl_pref_r': True,
    'pbrs_r': False,
    'ez': True,
    'salad': True,
    'serve': False,
    'convenience_features': False
}

OC_HL_ENV_PARAMS_FLATSA_PREF_NONEUREKA = {
    'algo': 'dqn',
    'hl_pref': True,
    'eval_with_hl_pref': True,
    'hl_pref_r': False,
    'pbrs_r': False,
    'ez': True,
    'salad': True,
    'serve': False,
    'convenience_features': False
}
