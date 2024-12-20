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


def find_closest_goal_map(random_map, goal_set_idxs):
    smallest_hamming_dist = math.inf
    filepath = constants.DOMAIN_VARS_ZELDA["goal_maps_filepath"]
    closest_map = curr_goal_map = int_arr_from_str_arr(
        to_2d_array_level(filepath.format(goal_set_idxs[0]))
    )

    for curr_idx in goal_set_idxs:
        temp_hamm_distance = compute_hamm_dist(random_map, curr_goal_map)
        if temp_hamm_distance < smallest_hamming_dist:
            closest_map = curr_goal_map
            smallest_hamming_dist = temp_hamm_distance

        curr_goal_map = int_arr_from_str_arr(
            to_2d_array_level(filepath.format(curr_idx))
        )
    return closest_map


def gen_pod_transitions(random_map, goal_map, traj_len, obs_size, controllable=False, randomize_sequence=True):
    string_map_for_map_stats = str_arr_from_int_arr(goal_map)

    # Targets
    if controllable:
        new_map_stats_dict = ZELDA_PROBLEM_OBJ.get_stats(string_map_for_map_stats)
        num_enemies = new_map_stats_dict["enemies"]
        nearest_enemy = new_map_stats_dict["nearest-enemy"]
        path_length = new_map_stats_dict["path-length"]
        conditional_diffs = []      
    
    xys = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10)]
    
    if randomize_sequence:
        random.shuffle(xys)

    steps = 0
    obs = copy.copy(random_map)
    cumulative_reward = 0.0
    returns = []
    states = []
    actions = []
    dones = []
    next_states = []
    is_start = True
    rewards = []
    is_starts = []
    while steps < traj_len:
        steps += 1
        y, x = xys.pop()
        action = goal_map[y][x]
        prev_obs = copy.copy(obs)
        row = obs[y]
        row[x] = action 
        obs[y] = row
        current_reward = ZELDA_PROBLEM_OBJ.get_reward(ZELDA_PROBLEM_OBJ.get_stats(str_arr_from_int_arr(transform(obs, x, y, obs_size))), ZELDA_PROBLEM_OBJ.get_stats(str_arr_from_int_arr(transform(prev_obs, x, y, obs_size))))
        cumulative_reward += current_reward
        is_done = ZELDA_PROBLEM_OBJ.get_episode_over(ZELDA_PROBLEM_OBJ.get_stats(str_arr_from_int_arr(transform(obs, x, y, obs_size))), ZELDA_PROBLEM_OBJ.get_stats(str_arr_from_int_arr(transform(prev_obs, x, y, obs_size))))
        dones.append(is_done)
        rewards.append(current_reward)
        if is_start:
            is_starts.append(is_start)
            is_start = False
        else:
            is_starts.append(is_start)
        
        states.append(transform(prev_obs, x, y, obs_size))
        next_states.append(transform(obs, x, y, obs_size))
        actions.append(action)


        if controllable:
            string_map_for_map_stats = str_arr_from_int_arr(obs)
            new_map_stats_dict = ZELDA_PROBLEM_OBJ.get_stats(string_map_for_map_stats)
            enemies_diff = num_enemies - new_map_stats_dict["enemies"]
            if enemies_diff > 0:
                enemies_diff = 3
            elif enemies_diff < 0:
                enemies_diff = 1
            else:
                enemies_diff = 2
    
            nearest_enemies_diff = nearest_enemy - new_map_stats_dict["nearest-enemy"]
            if nearest_enemies_diff > 0:
                nearest_enemies_diff = 3
            elif nearest_enemies_diff < 0:
                nearest_enemies_diff = 1
            else:
                nearest_enemies_diff = 2
    
            path_diff = path_length - new_map_stats_dict["path-length"]
            if path_diff > 0:
                path_diff = 3
            elif path_diff < 0:
                path_diff = 1
            else:
                path_diff = 2
    
            conditional_diffs.append(
                (enemies_diff, nearest_enemies_diff, path_diff)
            )     

    returns = [cumulative_reward/float(steps) for _ in range(len(states))]
     
    if controllable:
        return next_states, states, actions, returns, conditional_diffs, dones, is_starts, rewards
    else:
        return next_states, states, actions, returns, dones, is_starts, rewards


def transform(obs, x, y, obs_size):
    map = obs
    size = obs_size
    pad = obs_size // 2
    padded = np.pad(map, pad, constant_values=1)
    cropped = padded[y: y + size, x: x + size]
    
    return cropped
    

def to_2d_array_level(file_name):
    level = []

    with open(file_name, "r") as f:
        rows = f.readlines()
        for row in rows:
            new_row = []
            for char in row:
                if char != "\n":
                    new_row.append(constants.TILES_MAP_ZELDA[char])
            level.append(new_row)

    return level
    
    
def int_arr_from_str_arr(map):
    int_map = []
    for row_idx in range(len(map)):
        new_row = []
        for col_idx in range(len(map[0])):
            new_row.append(constants.INT_MAP_ZELDA[map[row_idx][col_idx]])
        int_map.append(new_row)
    return int_map


def str_arr_from_int_arr(map):
    translation_map = {v: k for k, v in constants.INT_MAP_ZELDA.items()}
    str_map = []
    for row_idx in range(len(map)):
        new_row = []
        for col_idx in range(len(map[0])):
            new_row.append(translation_map[map[row_idx][col_idx]])
        str_map.append(new_row)

    return str_map


def convert_state_to_int(state, obs_dim=(21,21, 8)):
    new_state = []
    new_state_oh = []
    for row in range(len(state)):
        new_row = []
        for col in range(len(state[0])):
            cell_oh = [0]*8
            cell_oh[int(state[row][col])]=1
            # new_row.append(cell_oh)
            new_state_oh.append(cell_oh)
            
            # new_state.append(constants.DOMAIN_VARS_ZELDA["int_map"][state[row][col]])
    return np.array(new_state_oh).reshape(obs_dim)


def int_from_oh(state, obs_dim=(21, 21, 8)):
    int_map = []
    for row in range(len(state)):
        new_row = []
        for col in range(len(state[0])):
            cell = np.argmax(state[row][col])
            new_row.append(cell)

        int_map.append(new_row)
    return np.array(int_map).reshape(obs_dim[0], obs_dim[1])


def gen_trajectories(id):
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
        for _ in range(NUM_EPISODES)
    ]
    goal_maps = [
        find_closest_goal_map(start_map, goal_set_idxs) for start_map in start_maps
    ]
    
    
    obs_size = 22
    traj_len = 77
    action_dim = (8,)
    obs_dim = (obs_size, obs_size, 8)
    reward_approximator_trajectory = []
    
    for epsiode_num, (start_map, goal_map) in enumerate(zip(start_maps, goal_maps)):
        next_states, states, actions, returns, dones, is_starts, rewards = gen_pod_transitions(start_map, goal_map, traj_len, obs_size)
        for next_state, state, action, ret, done, is_start, reward in zip(next_states, states, actions, returns, dones, is_starts, rewards):
            reward_approximator_trajectory.append({
                "next_state": next_state,
                "state": state,
                "action": action,
                "return": ret,
                "done": done,
                "is_start": is_start,
                "rewards": reward
            })

        print(f"generated {epsiode_num+1} episodes")
    
    tuple_trajectories = []
    for json_dict in reward_approximator_trajectory:
        ret = json_dict["return"]
        done = json_dict["done"]
        if PROCESS_STATE_AS_ONEHOT:
            action_oh = [0]*8
            action_oh[json_dict["action"]] = 1
            tuple_trajectories.append((convert_state_to_int(json_dict["next_state"]), convert_state_to_int(json_dict["state"]), np.array(action_oh), np.array([ret]), np.array([done]), np.array([json_dict["is_start"]]), np.array([json_dict["rewards"]])))
        else:
            tuple_trajectories.append((json_dict["next_state"], json_dict["state"], json_dict["action"], np.array([ret]), np.array([done]), np.array([json_dict["is_start"]]), np.array([json_dict["rewards"]])))
    
    random.shuffle(tuple_trajectories)

    if PROCESS_STATE_AS_ONEHOT:
        expert_observations = np.empty((len(tuple_trajectories),) + obs_dim)
        expert_next_observations = np.empty((len(tuple_trajectories),) + obs_dim)
        expert_actions = np.empty((len(tuple_trajectories),) + action_dim)
    else:
        expert_observations = np.empty((len(tuple_trajectories),) + (obs_size,obs_size,))
        expert_next_observations = np.empty((len(tuple_trajectories),) + (obs_size,obs_size,))
        expert_actions = np.empty((len(tuple_trajectories),) + (1,))    

    expert_returns = np.empty((len(tuple_trajectories),) + (1,))
    expert_dones = np.empty((len(tuple_trajectories),) + (1,))
    expert_is_starts = np.empty((len(tuple_trajectories),) + (1,))
    expert_rewards = np.empty((len(tuple_trajectories),) + (1,))

    reward_approximator_trajectory.append({
        "next_state": next_state,
        "state": state,
        "action": action,
        "rewards": ret,
        "done": done,
        "is_start": is_start,
        "rewards": rewards
    })

    for i, (next_obs, obs, act, returns, done, is_start, reward) in enumerate(tuple_trajectories):
        expert_next_observations[i] = next_obs
        expert_observations[i] = obs
        expert_actions[i] = act
        expert_rewards[i] = reward
        expert_dones[i] = done
        expert_is_starts[i] = is_start
        expert_rewards[i] = reward

    numpy_archive_filename = f"{PROJECT_ROOT}/data/zelda/trajectories/lg_expert_goalset_traj_{id}.npz"
    if LEGACY_STABLE_BASELINES_EXPERT_DATASET:
        np.savez_compressed(
            numpy_archive_filename,
            actions=expert_actions,
            episode_returns=expert_returns,
            rewards=expert_rewards,
            obs=expert_observations,
            episode_starts=expert_is_starts
        )
    else:
        np.savez_compressed(
            numpy_archive_filename,
            expert_next_observations=expert_next_observations,
            expert_actions=expert_actions,
            expert_observations=expert_observations,
            expert_rewards=expert_rewards,
            expert_dones=expert_dones
        )

    print(f"Saved file as {numpy_archive_filename}")
    
# NOTE: to run this --> PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python gen_expert_traj.py

for id in range(1,3):
    gen_trajectories(id)
    
        