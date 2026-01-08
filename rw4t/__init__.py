from gymnasium.envs.registration import register

register(id='rw4t-v0',
         entry_point='rw4t_env:RW4TEnv',
         kwargs={'map_name': 'six_by_six_6_train_map'})
