""" 
This is the plan we will use to implement a DT
on PoD trajectories.

1) Ensure that the PoD transitions are generated correctly
2) Make sure that the rtg / input to the DT is correct (is the statement correctly calculated, and the reward is normalized correctly).
3) Training correctly
4) Implement inference code for the DT according to the cs25 tutorial
5) The DT can generate solvable levels at inference
"""

import os
import math
import copy
from collections import OrderedDict
import random
from timeit import default_timer as timer
from datetime import timedelta

import numpy as np
import pandas as pd
from gym import error
import json
import tqdm

from utils import make_vec_envs as mkvenv, make_env as mkenv
from gym_pcgrl.envs.reps.narrow_rep import NarrowRepresentation
from gym_pcgrl.envs.probs.zelda_prob import ZeldaProblem


from gym_pcgrl.wrappers import CroppedImagePCGRLWrapper
from gym_pcgrl.envs.helper import (
    get_tile_locations,
    calc_num_regions,
    calc_certain_tile,
    run_dikjstra,
    get_string_map,
)
import constants
import utils


ZELDA_PROBLEM_OBJ = ZeldaProblem()
NUM_EPISODES = 5000
PROCESS_STATE_AS_ONEHOT = True
LEGACY_STABLE_BASELINES_EXPERT_DATASET = True
PROJECT_ROOT = os.getenv("PROJECT_ROOT")

from gym_pcgrl.envs.pcgrl_env import PcgrlEnv



if not PROJECT_ROOT:
    raise RuntimeError("The env var `PROJECT_ROOT` is not set.")
    
    



def gen_random_map(random, width, height, prob):
    map = random.choice(
        list(prob.keys()), size=(height, width), p=list(prob.values())
    ).astype(np.uint8)
    return map


def compute_hamm_dist(random_map, goal):
    hamming_distance = 0.0
    for i in range(len(random_map)):
        for j in range(len(random_map[0])):
            if random_map[i][j] != goal[i][j]:
                hamming_distance += 1
    return float(hamming_distance / (len(random_map) * len(random_map[0])))

def int_arr_from_str_arr(map):
    int_map = []
    for row_idx in range(len(map)):
        new_row = []
        for col_idx in range(len(map[0])):
            new_row.append(constants.INT_MAP_ZELDA[map[row_idx][col_idx]])
        int_map.append(new_row)
    return int_map

# def to_2d_array_level(filename):
#     level = []

#     with open(filename, "r") as f:
#         rows = f.readlines()
#         for row in rows:
#             new_row = []
#             for char in row:
#                 if char != "\n":
#                     new_row.append(constants.TILES_MAP_ZELDA[char])
#             level.append(new_row)

#     return level

def to_2d_array_level(filename):
    level = []

    with open(filename, "r") as f:
        rows = f.readlines()
        for row in rows:
            new_row = []
            for char in row:
                if char != "\n":
                    new_row.append(constants.TILES_MAP_ZELDA[char])
            level.append(new_row)

    return np.array(level).reshape(7, 11)


# def find_closest_goal_map(random_map, goal_set_idxs):
#     smallest_hamming_dist = math.inf
#     filepath = constants.DOMAIN_VARS_ZELDA["goal_maps_filepath"]
#     closest_map = curr_goal_map = int_arr_from_str_arr(
#         to_2d_array_level(filepath.format(goal_set_idxs[0]))
#     )

#     for curr_idx in goal_set_idxs:
#         temp_hamm_distance = compute_hamm_dist(random_map, curr_goal_map)
#         if temp_hamm_distance < smallest_hamming_dist:
#             closest_map = curr_goal_map
#             smallest_hamming_dist = temp_hamm_distance

#         curr_goal_map = int_arr_from_str_arr(
#             to_2d_array_level(filepath.format(curr_idx))
#         )
#     return closest_map


def find_closest_goal_map(random_map, goal_maps):
    smallest_hamming_dist = math.inf
    # filepath = constants.DOMAIN_VARS_ZELDA["goal_maps_filepath"]
    closest_map = None

    for curr_idx, curr_goal_map in enumerate(goal_maps):
        temp_hamm_distance = compute_hamm_dist(random_map, curr_goal_map)
        if temp_hamm_distance < smallest_hamming_dist:
            closest_map = curr_goal_map
            smallest_hamming_dist = temp_hamm_distance

        # curr_goal_map = int_arr_from_str_arr(
        #     to_2d_array_level(filepath.format(curr_idx))
        # )
    return closest_map


def transform_state(obs, x, y, obs_size):
    map = obs
    size = obs_size
    pad = obs_size // 2
    padded = np.pad(map, pad, constant_values=1)
    cropped = padded[y: y + size, x: x + size]
    
    return cropped


def transform_action(target_action, act_dim):
    oh = [0]*act_dim
    oh[target_action] = 1
    return np.array(oh)


# def generate_start_goal_map_pairs(num_episodes):
#     goal_maps_set = [i for i in range(0, len(os.listdir(constants.ZELDA_GOAL_MAPS_ROOT)))]
#     random.shuffle(goal_maps_set)
#     goal_set_idxs = goal_maps_set
#     rng, _ = utils.np_random(None)

#     start_maps = [
#         gen_random_map(
#             rng,
#             constants.DOMAIN_VARS_ZELDA["env_x"],
#             constants.DOMAIN_VARS_ZELDA["env_y"],
#             constants.DOMAIN_VARS_ZELDA["action_pronbabilities_map"],
#         )
#         for _ in range(num_episodes)
#     ]
#     goal_maps = [
#         np.array(find_closest_goal_map(start_map, goal_set_idxs)) for start_map in start_maps
#     ]
#     return zip(start_maps, goal_maps)


def generate_start_goal_map_pairs(num_episodes, path_to_goal_maps_dir):
    goal_maps_set = [i for i in range(0, len(os.listdir(constants.ZELDA_GOAL_MAPS_ROOT)))]
    random.shuffle(goal_maps_set)
    goal_set_idxs = goal_maps_set
    rng, _ = utils.np_random(None)

    start_maps = [
        gen_random_map(
            rng,
            constants.DOMAIN_VARS_ZELDA["env_x"],
            constants.DOMAIN_VARS_ZELDA["env_y"],
            constants.DOMAIN_VARS_ZELDA["action_pronbabilities_map"],
        )
        for _ in range(num_episodes)
    ]
    
    all_goal_maps = closest_map = curr_goal_map = [int_arr_from_str_arr(
        to_2d_array_level(f"{path_to_goal_maps_dir}/{file}")
    ) for file in os.listdir(path_to_goal_maps_dir) if ".txt" in file]
    goal_maps = [
        np.array(find_closest_goal_map(start_map, all_goal_maps)) for start_map in start_maps
    ]
    return zip(start_maps, goal_maps)


def one_hot_encode(arr):
    """Converts a 2D numpy array of integers to one-hot encoding."""

    # Find the maximum value in the array
    max_value = 7 #np.max(arr)

    # Create a one-hot encoded array filled with zeros
    # import pdb; pdb.set_trace()
    one_hot = np.zeros((arr.shape[0], arr.shape[0], max_value + 1))

    # Set the appropriate indices to 1
    for i in range(arr.shape[0]):
        for j in range(arr.shape[0]):
            one_hot[i, j, arr[i, j]] = 1

    return one_hot


def generate_episode(ep_len):
    kwargs = {
        'render_rank': 0,
        'render': False,
        'cropped_size': 22,
        'resume': False
    }
    pcgrl_env = PcgrlEnv(prob="zelda", rep="narrow")
    num_episodes = 1
    start_goal_map_pairs_tuples = generate_start_goal_map_pairs(num_episodes)
    obs_size = 22
    traj_len = 77
    action_dim = (8,)
    obs_dim = (obs_size, obs_size, 8)
    reward_approximator_trajectory = []
    pcgrl_env.reset()
    for epsiode_num, (start_map, goal_map) in enumerate(start_goal_map_pairs_tuples):
        pcgrl_env._rep._map = start_map
        steps = 0
        next_state = None
        rewards_for_episode = []
        current_state_ohs = []
        target_action_ohs = []
        next_state_ohs = []
        doness = []
        # while not np.array_equal(goal_map, next_state):
        for _ in range(ep_len):
            # get the next x,y position
            x, y, = pcgrl_env._rep._x, pcgrl_env._rep._y

            # import pdb; pdb.set_trace()
            # get the current map state from env._prob._map
            current_state = pcgrl_env._rep._map
            current_state_oh = transform_state(current_state, x, y, obs_size)
            current_state_ohs.append(current_state_oh)
            # get the target action from the goal map
            target_action = goal_map[y, x]
            target_action_oh = transform_action(target_action, action_dim[0])
            target_action_ohs.append(target_action_oh)

            # apply the action on the map
            obs, reward, dones, info = pcgrl_env.step(target_action + 1)
            # print(f"reward: {reward}, dones: {dones}, info: {info}")
            rewards_for_episode.append(float(reward))
            doness.append(int(dones))

            # get the reward and the next state
            next_state = pcgrl_env._rep._map
            next_state_oh = transform_state(next_state, x, y, obs_size)
            next_state_ohs.append(next_state_oh)


        # Here calculate the returns to go
        cumulative_reward = sum(rewards_for_episode)
        running_cum_reward = cumulative_reward
        rtgs = []
        for r in rewards_for_episode:
            rtgs.append(running_cum_reward)
            running_cum_reward -= r
            
        # print(f"rtgs: {rtgs}")
        current_state_ohs = [one_hot_encode(np.array(c)) for c in current_state_ohs]
        next_state_ohs = [one_hot_encode(np.array(n)) for n in next_state_ohs]
        
        return np.array(current_state_ohs), np.array(target_action_ohs), np.array(rewards_for_episode), np.array(next_state_ohs), np.array(rtgs), np.array(doness) 
    
    
def generate_episode_hf(ep_len=100):
    kwargs = {
        'render_rank': 0,
        'render': False,
        'cropped_size': 22,
        'resume': False
    }
    pcgrl_env = PcgrlEnv(prob="zelda", rep="narrow")
    num_episodes = 1
    start_goal_map_pairs_tuples = generate_start_goal_map_pairs(num_episodes)
    obs_size = 22
    traj_len = 77
    action_dim = (8,)
    obs_dim = (obs_size, obs_size, 8)
    reward_approximator_trajectory = []
    pcgrl_env.reset()
    for epsiode_num, (start_map, goal_map) in enumerate(start_goal_map_pairs_tuples):
        pcgrl_env._rep._map = start_map
        steps = 0
        next_state = None
        rewards_for_episode = []
        current_state_ohs = []
        target_action_ohs = []
        next_state_ohs = []
        doness = []
        # while not np.array_equal(goal_map, next_state):
        for _ in range(ep_len):
            # get the next x,y position
            x, y, = pcgrl_env._rep._x, pcgrl_env._rep._y

            # import pdb; pdb.set_trace()
            # get the current map state from env._prob._map
            current_state = pcgrl_env._rep._map
            current_state_oh = transform_state(current_state, x, y, obs_size)
            
            current_state_ohs.append((current_state_oh/8.0).reshape(-1))
            # get the target action from the goal map
            target_action = goal_map[y, x]
            target_action_oh = transform_action(target_action, action_dim[0])
            # import pdb; pdb.set_trace()
            target_action_ohs.append(target_action_oh/8.0)
            

            # apply the action on the map
            obs, reward, dones, info = pcgrl_env.step(target_action + 1)
            # print(f"reward: {reward}, dones: {dones}, info: {info}")
            rewards_for_episode.append(float(reward))
            doness.append(dones)

            # get the reward and the next state
            next_state = pcgrl_env._rep._map
            next_state_oh = transform_state(next_state, x, y, obs_size)
            next_state_ohs.append(next_state_oh)


        # Here calculate the returns to go
        cumulative_reward = sum(rewards_for_episode)
        running_cum_reward = cumulative_reward
        rtgs = []
        for r in rewards_for_episode:
            rtgs.append(running_cum_reward)
            running_cum_reward -= r
            
        # print(f"rtgs: {rtgs}")
        current_state_ohs = [np.array(c) for c in current_state_ohs]
        # next_state_ohs = [np.array(n) for n in next_state_ohs]
        
        # return {"observaions": np.array(current_state_ohs), "actions": np.array(target_action_ohs), "rewards": np.array(rewards_for_episode), "dones": np.array(doness)}
        return {"observations": current_state_ohs, "actions": np.array(target_action_ohs), "rewards": np.array(rewards_for_episode), "dones": np.array(doness)}
    
    
def generate_multiple_episode_hf(ep_len=100, num_episodes=10, path_to_goal_maps_dir=f"{PROJECT_ROOT}/goal_maps/expert_zelda_lg"):
    kwargs = {
        'render_rank': 0,
        'render': False,
        'cropped_size': 22,
        'resume': False
    }
    pcgrl_env = PcgrlEnv(prob="zelda", rep="narrow")
    # num_episodes = 1
    start_goal_map_pairs_tuples = generate_start_goal_map_pairs(num_episodes, path_to_goal_maps_dir)
    obs_size = 22
    traj_len = 77
    action_dim = (8,)
    obs_dim = (obs_size, obs_size, 8)
    reward_approximator_trajectory = []
    pcgrl_env.reset()
    all_rewards_for_episode = []
    all_current_state_ohs = []
    all_target_action_ohs = []
    all_next_state_ohs = []
    all_doness = []
    for epsiode_num, (start_map, goal_map) in enumerate(start_goal_map_pairs_tuples):
        pcgrl_env._rep._map = start_map
        steps = 0
        next_state = None
        rewards_for_episode = []
        current_state_ohs = []
        target_action_ohs = []
        next_state_ohs = []
        doness = []
        solved = False
        total_reward = 0.0
        while not solved:
        # for _ in range(ep_len):
            # get the next x,y position
            x, y, = pcgrl_env._rep._x, pcgrl_env._rep._y

            # import pdb; pdb.set_trace()
            # get the current map state from env._prob._map
            current_state = pcgrl_env._rep._map
            current_state_oh = transform_state(current_state, x, y, obs_size)
            
            current_state_ohs.append((current_state_oh/8.0).reshape(-1))
            # get the target action from the goal map
            target_action = goal_map[y, x]
            target_action_oh = transform_action(target_action, action_dim[0])
            # import pdb; pdb.set_trace()
            target_action_ohs.append(target_action_oh/8.0)
            

            # apply the action on the map
            obs, reward, dones, info = pcgrl_env.step(target_action + 1)
            # print(f"reward: {reward}, dones: {dones}, info: {info}")
            rewards_for_episode.append(float(reward))
            print(f"reward: {reward}")

            # get the reward and the next state
            next_state = pcgrl_env._rep._map
            next_state_oh = transform_state(next_state, x, y, obs_size)
            next_state_ohs.append(next_state_oh)
            if np.array_equal(goal_map, next_state) or dones is True:
                solved = True
                doness.append(solved)
            else:
                doness.append(dones)

        # Here calculate the returns to go
        cumulative_reward = sum(rewards_for_episode)
        running_cum_reward = cumulative_reward
        rtgs = []
        for r in rewards_for_episode:
            rtgs.append(running_cum_reward)
            running_cum_reward -= r
            
        # print(f"rtgs: {rtgs}")
        current_state_ohs = [np.array(c) for c in current_state_ohs]
        # next_state_ohs = [np.array(n) for n in next_state_ohs]
        
        all_rewards_for_episode.extend(rewards_for_episode)
        all_current_state_ohs.extend(current_state_ohs)
        all_target_action_ohs.extend(target_action_ohs)
        # all_next_state_ohs.append()
        all_doness.extend(doness)
        
        
        pcgrl_env.reset()
        # return {"observaions": np.array(current_state_ohs), "actions": np.array(target_action_ohs), "rewards": np.array(rewards_for_episode), "dones": np.array(doness)}
    return {"observations": all_current_state_ohs, "actions": np.array(all_target_action_ohs), "rewards": np.array(all_rewards_for_episode), "dones": np.array(all_doness)}
    
    
# output = generate_episode_hf(100)


# def generate_dataset_for_dt_wrapper():
    


def generate_dataset_for_dt(num_episodes=10, path_to_goal_maps_dir=f"{PROJECT_ROOT}/goal_maps/expert_zelda_lg"):
    kwargs = {
        'render_rank': 0,
        'render': False,
        'cropped_size': 22,
        'resume': False
    }
    pcgrl_env = PcgrlEnv(prob="zelda", rep="narrow")
    # num_episodes = 1
    start_goal_map_pairs_tuples = generate_start_goal_map_pairs(num_episodes, path_to_goal_maps_dir)
    # print(len(start_goal_map_pairs_tuples))
    obs_size = 22
    traj_len = 77
    action_dim = (8,)
    obs_dim = (obs_size, obs_size, 8)
    reward_approximator_trajectory = []
    pcgrl_env.reset()
    all_rewards_for_episode = []
    all_current_state_ohs = []
    all_target_action_ohs = []
    all_next_state_ohs = []
    all_doness = []
    paths = []
    for epsiode_num, (start_map, goal_map) in enumerate(start_goal_map_pairs_tuples):
        pcgrl_env._rep._map = start_map
        steps = 0
        next_state = None
        rewards_for_episode = []
        current_state_ohs = []
        target_action_ohs = []
        next_state_ohs = []
        doness = []
        solved = False
        total_reward = 0.0
        while not solved:
        # for _ in range(ep_len):
            # get the next x,y position
            x, y, = pcgrl_env._rep._x, pcgrl_env._rep._y

            # import pdb; pdb.set_trace()
            # get the current map state from env._prob._map
            current_state = pcgrl_env._rep._map
            current_state_oh = transform_state(current_state, x, y, obs_size)
            
            current_state_ohs.append((current_state_oh/8.0).reshape(-1))
            # get the target action from the goal map
            target_action = goal_map[y, x]
            target_action_oh = transform_action(target_action, action_dim[0])
            # import pdb; pdb.set_trace()
            target_action_ohs.append(target_action_oh/8.0)
            
            # apply the action on the map
            obs, reward, dones, info = pcgrl_env.step(target_action + 1)
            # print(f"reward: {reward}, dones: {dones}, info: {info}")
            rewards_for_episode.append(float(reward))
            # print(f"reward: {reward}")

            # get the reward and the next state
            next_state = pcgrl_env._rep._map
            next_state_oh = transform_state(next_state, x, y, obs_size)
            next_state_ohs.append(next_state_oh)
            if np.array_equal(goal_map, next_state) or dones is True:
                solved = True
                doness.append(solved)
                paths.append({"observations": [np.array(c) for c in current_state_ohs], "actions": np.array(target_action_ohs), "rewards": np.array(rewards_for_episode), "dones": np.array(doness)})
                rewards_for_episode = []
                current_state_ohs = []
                target_action_ohs = []
                next_state_ohs = []
                doness = []
            else:
                doness.append(dones)
        
        pcgrl_env.reset()

    return paths


def generate_dataset_for_dt_newest(num_episodes=10, path_to_goal_maps_dir=f"{PROJECT_ROOT}/goal_maps/expert_zelda_lg"):
    kwargs = {
        'render_rank': 0,
        'render': False,
        'cropped_size': 22,
        'resume': False
    }
    pcgrl_env = PcgrlEnv(prob="zelda", rep="narrow")
    # num_episodes = 1
    # start_goal_map_pairs_tuples = generate_start_goal_map_pairs(num_episodes, path_to_goal_maps_dir)
    # print(len(start_goal_map_pairs_tuples))
    all_goal_maps = [int_arr_from_str_arr(
        to_2d_array_level(f"{path_to_goal_maps_dir}/{file}")
    ) for file in os.listdir(path_to_goal_maps_dir) if ".txt" in file]
    obs_size = 22
    traj_len = 77
    action_dim = (8,)
    obs_dim = (obs_size, obs_size, 8)
    # reward_approximator_trajectory = []
    # pcgrl_env.reset()
    # all_rewards_for_episode = []
    # all_current_state_ohs = []
    # all_target_action_ohs = []
    # all_next_state_ohs = []
    # all_doness = []
    paths = []
    for i in tqdm.tqdm(range(num_episodes)):
        start_map = pcgrl_env.reset()['map']
        goal_map = np.array(find_closest_goal_map(start_map, all_goal_maps)).reshape(7, 11)
        steps = 0
        next_state = None
        all_rewards_for_episode = []
        all_current_state_ohs = []
        all_target_action_ohs = []
        all_next_state_ohs = []
        all_doness = []
        rewards_for_episode = []
        current_state_ohs = []
        target_action_ohs = []
        next_state_ohs = []
        doness = []
        solved = False
        total_reward = 0.0
        while steps < 1000:
        # for _ in range(ep_len):
            # get the next x,y position
            x, y, = pcgrl_env._rep._x, pcgrl_env._rep._y

            # import pdb; pdb.set_trace()
            # get the current map state from env._prob._map
            current_state = pcgrl_env._rep._map
            current_state_oh = transform_state(current_state, x, y, obs_size)
            
            current_state_ohs.append((current_state_oh/8.0).reshape(-1))
            # get the target action from the goal map
            target_action = goal_map[y, x]
            target_action_oh = transform_action(target_action, action_dim[0])
            # import pdb; pdb.set_trace()
            target_action_ohs.append(target_action_oh/8.0)
            
            # apply the action on the map
            obs, reward, dones, info = pcgrl_env.step(target_action + 1)
            # print(f"reward: {reward}, dones: {dones}, info: {info}")
            total_reward += float(reward)
            # rewards_for_episode.append(float(reward))
            rewards_for_episode.append(total_reward)
            # print(f"reward: {reward}")

            # get the reward and the next state
            next_state = pcgrl_env._rep._map
            next_state_oh = transform_state(next_state, x, y, obs_size)
            next_state_ohs.append(next_state_oh)
            if info['solved']:
                solved = True
                doness.append(solved)
                solved = False
            elif dones is True:
                doness.append(dones)
                dones = False
            else:
                doness.append(dones)
                
                
                
            if solved or dones:
                all_rewards_for_episode.extend(rewards_for_episode)
                all_current_state_ohs.extend(current_state_ohs)
                all_target_action_ohs.extend(target_action_ohs)
                all_next_state_ohs.extend(next_state_ohs)
                all_doness.extend(doness)
                
                
                rewards_for_episode = []
                current_state_ohs = []
                target_action_ohs = []
                next_state_ohs = []
                doness = []
                total_reward = 0.0
                
                start_map = pcgrl_env.reset()['map']
                goal_map =  np.array(find_closest_goal_map(start_map, all_goal_maps)).reshape(7, 11)
                
            steps += 1   
        
        paths.append({"observations": all_current_state_ohs, "actions": np.array(all_target_action_ohs), "rewards": np.array(all_rewards_for_episode), "dones": np.array(all_doness)})


    return paths


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



def generate_dataset_for_dt_newest_normal_distribution(num_episodes=10, path_to_goal_maps_dir=f"{PROJECT_ROOT}/goal_maps/expert_zelda_lg"):
    kwargs = {
        'render_rank': 0,
        'render': False,
        'cropped_size': 22,
        'resume': False
    }
    pcgrl_env = PcgrlEnv(prob="zelda", rep="narrow")
    # num_episodes = 1
    # start_goal_map_pairs_tuples = generate_start_goal_map_pairs(num_episodes, path_to_goal_maps_dir)
    # print(len(start_goal_map_pairs_tuples))
    all_goal_maps = [int_arr_from_str_arr(
        to_2d_array_level(f"{path_to_goal_maps_dir}/{file}")
    ) for file in os.listdir(path_to_goal_maps_dir) if ".txt" in file]
    obs_size = 22
    traj_len = 77
    action_dim = (8,)
    obs_dim = (obs_size, obs_size, 8)

    paths = []
    for i in tqdm.tqdm(range(num_episodes)):
        start_map = pcgrl_env.reset()['map']
        goal_map = np.array(find_closest_goal_map(start_map, all_goal_maps)).reshape(7, 11)
        steps = 0
        next_state = None
        all_rewards_for_episode = []
        all_current_state_ohs = []
        all_target_action_ohs = []
        all_next_state_ohs = []
        all_doness = []
        rewards_for_episode = []
        current_state_ohs = []
        target_action_ohs = []
        next_state_ohs = []
        doness = []
        solved = False
        total_reward = 0.0
        while steps < 1000:
        # for _ in range(ep_len):
            # get the next x,y position
            x, y, = pcgrl_env._rep._x, pcgrl_env._rep._y

            # import pdb; pdb.set_trace()
            # get the current map state from env._prob._map
            current_state = pcgrl_env._rep._map
            current_state_oh = transform_state(current_state, x, y, obs_size)
            
            current_state_ohs.append((current_state_oh/8.0).reshape(-1))
            # get the target action from the goal map
            target_action = goal_map[y, x]
            target_action_oh = transform_action(target_action, action_dim[0])
            # import pdb; pdb.set_trace()
            target_action_ohs.append(target_action_oh/8.0)
            
            # apply the action on the map
            obs, reward, dones, info = pcgrl_env.step(target_action + 1)
            # print(f"reward: {reward}, dones: {dones}, info: {info}")
            total_reward += float(reward)
            # rewards_for_episode.append(float(reward))
            rewards_for_episode.append(total_reward)
            # print(f"reward: {reward}")

            # get the reward and the next state
            next_state = pcgrl_env._rep._map
            next_state_oh = transform_state(next_state, x, y, obs_size)
            next_state_ohs.append(next_state_oh)
            if info['solved']:
                solved = True
                doness.append(solved)
                solved = False
            elif dones is True:
                doness.append(dones)
                dones = False
            else:
                doness.append(dones)
                
                
                
            if solved or dones:
                all_rewards_for_episode.extend(rewards_for_episode)
                all_current_state_ohs.extend(current_state_ohs)
                all_target_action_ohs.extend(target_action_ohs)
                all_next_state_ohs.extend(next_state_ohs)
                all_doness.extend(doness)
                
                
                rewards_for_episode = []
                current_state_ohs = []
                target_action_ohs = []
                next_state_ohs = []
                doness = []
                total_reward = 0.0
                
                start_map = pcgrl_env.reset()['map']
                goal_map =  np.array(find_closest_goal_map(start_map, all_goal_maps)).reshape(7, 11)
                
            steps += 1   
        
        paths.append({"observations": all_current_state_ohs, "actions": np.array(all_target_action_ohs), "rewards": np.array(all_rewards_for_episode), "dones": np.array(all_doness)})


    return paths



import pickle
paths = generate_dataset_for_dt_newest(num_episodes=1000)
with open('/home/jupyter-msiper/bootstrapping-pcgrl/raw_dataset_dt_lg_10K_episodes.pkl', 'wb') as f:
    pickle.dump(paths, f)


# import pdb; pdb.set_trace()

