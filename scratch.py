import numpy as np
import os
# Create a 1D array of strings

# data_dir = "/home/jupyter-msiper/bootstrapping-pcgrl/goal_maps/expert_zelda"
# write_dir = "/home/jupyter-msiper/bootstrapping-pcgrl/goal_maps/zelda_expert"


# for idx, file in enumerate(os.listdir(data_dir)):
#     print(f"{file}")
#     lvl_data = None
#     with open(f"{data_dir}/{file}", "r") as f:
#         lvl_data = f.read()

#     lvl_arr = np.array([c for c in lvl_data])
#     lvl_arr = lvl_arr.reshape(7, 11)

#     with open("{write_dir}/{idx}.txt".format(write_dir=write_dir, idx=str(idx)), "w") as f:
#         for row in lvl_arr:
#             f.write(''.join(row)+"\n")


import gym
import numpy as np

import collections
import pickle

# import d4rl


datasets = []

__data = None
# with open('/home/jupyter-msiper/bootstrapping-pcgrl/halfcheetah-expert-v2.pkl', 'rb') as f:
#     __data = pickle.load(f)
    
    
"""Data Schema (list of 1000 dictionaries long; inside each dictionary element are the keys 'observations', 'actions', 'next_obsevations', 'terminals', 'rewards' each 1000-element long lists

[
    {
        "observations": [
        
        ],
        "actions": [
        
        ]
    },
]
"""

# print(__data) 
# import pdb; pdb.set_trace()


# dataset = __data

# N = dataset['rewards'].shape[0]
# data_ = collections.defaultdict(list)

# use_timeouts = False
# if 'timeouts' in dataset:
#     use_timeouts = True

# episode_step = 0
# paths = []
# for i in range(N):
#     done_bool = bool(dataset['terminals'][i])
#     if use_timeouts:
#         final_timestep = dataset['timeouts'][i]
#     else:
#         final_timestep = (episode_step == 1000-1)
#     for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
#         data_[k].append(dataset[k][i])
#     if done_bool or final_timestep:
#         episode_step = 0
#         episode_data = {}
#         for k in data_:
#             episode_data[k] = np.array(data_[k])
#         paths.append(episode_data)
#         data_ = collections.defaultdict(list)
#     episode_step += 1

# returns = np.array([np.sum(p['rewards']) for p in paths])
# num_samples = np.sum([p['rewards'].shape[0] for p in paths])
# print(f'Number of samples collected: {num_samples}')
# print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

# with open(f'test.pkl', 'wb') as f:
#     pickle.dump(paths, f)

# import pdb; pdb.set_trace()
# from the_dt_impl_plan import to_2d_array_level, find_closest_goal_map, generate_start_goal_map_pairs, int_arr_from_str_arr
# import numpy as np
# import os
# from gym_pcgrl.envs.pcgrl_env import PcgrlEnv
# from gym_pcgrl.envs.probs.zelda_prob import ZeldaProblem
# from gym_pcgrl.envs.helper import (
#     get_tile_locations,
#     calc_num_regions,
#     calc_certain_tile,
#     run_dikjstra,
#     get_string_map,
# )
# ZELDA_PROBLEM_OBJ = ZeldaProblem()

# PROJECT_ROOT = os.getenv("PROJECT_ROOT")


# path_to_goal_maps_dir=f"{PROJECT_ROOT}/goal_maps/expert_zelda_lg"
# all_goal_maps = closest_map = curr_goal_map = [int_arr_from_str_arr(
#         to_2d_array_level(f"{path_to_goal_maps_dir}/{file}")
#     ) for file in os.listdir(path_to_goal_maps_dir) if ".txt" in file]

# for goal_map in all_goal_maps:
#     new_map_stats = ZELDA_PROBLEM_OBJ.get_stats(get_string_map(np.array(goal_map).reshape(7,11), ZELDA_PROBLEM_OBJ.get_tile_types()))
#     print(f"{ZELDA_PROBLEM_OBJ.get_episode_over(new_map_stats, None)}")




__data = None
with open('/home/jupyter-msiper/bootstrapping-pcgrl/raw_dataset_dt_lg_10K_episodes.pkl', 'rb') as f:
    __data = pickle.load(f)
    
import pdb; pdb.set_trace()