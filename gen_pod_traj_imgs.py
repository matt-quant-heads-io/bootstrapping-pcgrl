import os
import math
import copy
from collections import OrderedDict, defaultdict
import random
from timeit import default_timer as timer
from datetime import timedelta
import argparse
import datetime
import shutil

import numpy as np
import pandas as pd
from gym import error
import json


from gym_pcgrl.envs.reps.narrow_rep import NarrowRepresentation
from gym_pcgrl.envs.probs.zelda_prob import ZeldaProblem
from gym_pcgrl.envs.probs.loderunner_prob import LRProblem
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


DOMAIN_TO_PROB_OBJ_MAP = {
    "loderunner": LRProblem,
    "zelda": ZeldaProblem
}
DOMAIN_TO_INT_TILE_MAP = {
    "zelda": {
        "empty": 0,
        "solid": 1,
        "player": 2,
        "key": 3,
        "door": 4,
        "bat": 5,
        "scorpion": 6,
        "spider": 7,
    },
    "loderunner": {
        "empty": 0,
        "brick": 1, 
        "ladder": 2, 
        "rope": 3, 
        "solid": 4, 
        "gold": 5, 
        "enemy": 6, 
        "player": 7
    }
}
DOMAIN_TO_CHAR_TO_STR_TILE_MAP = {
    "zelda":{
        "g": "door",
        "+": "key",
        "A": "player",
        "1": "bat",
        "2": "spider",
        "3": "scorpion",
        "w": "solid",
        ".": "empty",
    },
    "loderunner": {
        ".":"empty",
        "b":"brick", 
        "#":"ladder", 
        "-":"rope", 
        "B":"solid", 
        "G":"gold", 
        "E":"enemy", 
        "M":"player"
    }
}
os.system("source ./set_project_root.sh")
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


def find_closest_goal_map(random_map, goal_set_idxs, goal_maps_filepath, char_to_string_tiles_map, str_to_int_tiles_map):
    smallest_hamming_dist = math.inf
    filepath = goal_maps_filepath
    goal_maps = [f for f in os.listdir(goal_maps_filepath) if '.txt' in f]
    closest_map = curr_goal_map = int_arr_from_str_arr(
        to_2d_array_level(f"{goal_maps_filepath}/{goal_maps[0]}", char_to_string_tiles_map), str_to_int_tiles_map
    )

    for next_curr_goal_map_fp in goal_maps:
        next_curr_goal_map = int_arr_from_str_arr(
            to_2d_array_level(f"{goal_maps_filepath}/{next_curr_goal_map_fp}", char_to_string_tiles_map), str_to_int_tiles_map
        )
        temp_hamm_distance = compute_hamm_dist(random_map, next_curr_goal_map)
        if temp_hamm_distance < smallest_hamming_dist:
            closest_map = next_curr_goal_map
            smallest_hamming_dist = temp_hamm_distance

    return closest_map


def gen_pod_transitions(random_map, goal_map, traj_len, obs_size, controllable=False, randomize_sequence=True, prob_obj=None, str_to_int_map=None, xys=None):
    string_map_for_map_stats = str_arr_from_int_arr(goal_map, str_to_int_map)

    # Targets
    if controllable:
        new_map_stats_dict = prob_obj.get_stats(string_map_for_map_stats)
        num_enemies = new_map_stats_dict["enemies"]
        nearest_enemy = new_map_stats_dict["nearest-enemy"]
        path_length = new_map_stats_dict["path-length"]
        conditional_diffs = []
            
    # import pdb; pdb.set_trace()
    # xys = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10)]
    
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
        y, x = xys[steps]
        steps += 1
        
        action = goal_map[y][x]
        prev_obs = copy.copy(obs)
        row = obs[y]
        row[x] = action 
        obs[y] = row
        # print(f"stats1:{}")
        # print(f"stats2: {}")
        current_reward = prob_obj.get_reward(prob_obj.get_stats(str_arr_from_int_arr(transform(obs, x, y, obs_size),str_to_int_map), ), prob_obj.get_stats(str_arr_from_int_arr(transform(prev_obs, x, y, obs_size),str_to_int_map)))
        cumulative_reward += current_reward
        is_done = prob_obj.get_episode_over(prob_obj.get_stats(str_arr_from_int_arr(transform(obs, x, y, obs_size),str_to_int_map)), prob_obj.get_stats(str_arr_from_int_arr(transform(prev_obs, x, y, obs_size),str_to_int_map)))
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
            new_map_stats_dict = prob_obj.get_stats(string_map_for_map_stats)
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
    size = obs_size[0]
    pad = obs_size[0] // 2
    padded = np.pad(map, pad, constant_values=1)
    cropped = padded[y: y + size, x: x + size]
    
    return cropped
    

def to_2d_array_level(file_name, tiles_map):
    level = []

    with open(file_name, "r") as f:
        rows = f.readlines()
        for row in rows:
            new_row = []
            for char in row:
                if char != "\n":
                    new_row.append(tiles_map[char])
            level.append(new_row)

    return level
    
    
def int_arr_from_str_arr(map, int_arr_from_str_arr):
    int_map = []
    for row_idx in range(len(map)):
        new_row = []
        for col_idx in range(len(map[0])):
            new_row.append(int_arr_from_str_arr[map[row_idx][col_idx]])
        int_map.append(new_row)
    return int_map


def str_arr_from_int_arr(map, str_to_int_map):
    translation_map = {v: k for k, v in str_to_int_map.items()}
    str_map = []
    for row_idx in range(len(map)):
        new_row = []
        for col_idx in range(len(map[0])):
            new_row.append(translation_map[map[row_idx][col_idx]])
        str_map.append(new_row)

    return str_map


def convert_state_to_onehot(state, obs_dim, act_dim):
    new_state = []
    new_state_oh = []
    for row in range(len(state)):
        new_row = []
        for col in range(len(state[0])):
            cell_oh = [0]*act_dim
            cell_oh[int(state[row][col])]=1
            # new_row.append(cell_oh)
            print(cell_oh)
            new_state_oh.append(cell_oh)
            
            # new_state.append(constants.DOMAIN_VARS_ZELDA["int_map"][state[row][col]])
            
    # import pdb; pdb.set_trace()
    return np.array(new_state_oh).reshape((obs_dim[0], obs_dim[1], act_dim))


def int_from_oh(state, obs_dim):
    int_map = []
    for row in range(len(state)):
        new_row = []
        for col in range(len(state[0])):
            cell = np.argmax(state[row][col])
            new_row.append(cell)

        int_map.append(new_row)
    return np.array(int_map).reshape(obs_dim[0], obs_dim[1])


def convert_int_map_to_str_map(state, int_to_str_tiles_map):
    str_map = []
    for row_i, row in enumerate(state):
        for col_i, col in enumerate(row):
            str_map.append(int_to_str_tiles_map[state[row_i][col_i]])
    return np.array(str_map).reshape(state.shape)


def gen_trajectories(domain, num_episodes, mode_of_output, action_dim, obs_dim, traj_len):
    goal_maps_root_dir = f"{PROJECT_ROOT}/goal_maps/{domain}"
    goal_maps_set = [i for i in range(0, len(sorted(os.listdir(goal_maps_root_dir))[:50]))]
    print(f"goal_maps_set: {goal_maps_set}")
    img_dir = "/home/jupyter-msiper/bootstrapping-pcgrl/data/loderunner/images"
    shutil.rmtree(img_dir)
    os.mkdir(img_dir)
    with open(f"{img_dir}/goal_maps_used.txt", "w") as f:
        for goal_map_idx in goal_maps_set:
            f.write(str(goal_map_idx))
            f.write("\n")
    prob_obj = DOMAIN_TO_PROB_OBJ_MAP[domain]()
    random.shuffle(goal_maps_set)
    goal_set_idxs = goal_maps_set
    rng, _ = utils.np_random(None)
    char_to_string_tiles_map = DOMAIN_TO_CHAR_TO_STR_TILE_MAP[domain]
    str_to_int_tiles_map = DOMAIN_TO_INT_TILE_MAP[domain]
    int_to_str_tiles_map = {v:k for k,v in str_to_int_tiles_map.items()}
    start_maps = [
        gen_random_map(
            rng,
            prob_obj._width,
            prob_obj._height,
            {str_to_int_tiles_map[s]: p for s,p in prob_obj._prob.items()},
        )
        for _ in range(num_episodes)
    ]
    goal_maps = [
        find_closest_goal_map(start_map, goal_set_idxs, goal_maps_root_dir, char_to_string_tiles_map, str_to_int_tiles_map) for start_map in start_maps
    ]
    xys = []
    for row_i, row in enumerate(start_maps[0]):
        for col_i, col in enumerate(row):
            xys.append((row_i, col_i))
    
    action_dim = (action_dim,)
    obs_dim = (obs_dim, obs_dim, action_dim[0])
    reward_approximator_trajectory = []
    freq_action_dict = defaultdict(int)
    labels_dict_for_csv = {"episode_id": [], "image_path": [], "action": [], "reward": []}
    for k,v in char_to_string_tiles_map.items():
        freq_action_dict[v] = 0
        
    
    for epsiode_num, (start_map, goal_map) in enumerate(zip(start_maps, goal_maps)):
        next_states, states, actions, returns, dones, is_starts, rewards = gen_pod_transitions(start_map, goal_map, traj_len, obs_dim, prob_obj=prob_obj, str_to_int_map=str_to_int_tiles_map, xys=xys)
        
        # convert the state to jpg image
        for next_state, state, action, _return, done, is_start, reward in zip(next_states, states, actions, returns, dones, is_starts, rewards):
            str_map = convert_int_map_to_str_map(state, int_to_str_tiles_map)
            img = prob_obj.render(str_map)
            # import pdb; pdb.set_trace()
            # increment the freq_action_dict with the action
            action_str = int_to_str_tiles_map[action]
            freq_action_dict[action_str] += 1
            
            # get the image filename (the action as string + <action_as_int>.jpg)
            img_filename = f"{action_str}{freq_action_dict[action_str]}.jpg"
            # Add the image filename and the action_as_int to labels_dict_for_csv
            labels_dict_for_csv["episode_id"].append(epsiode_num+1)
            labels_dict_for_csv["image_path"].append(img_filename)
            labels_dict_for_csv["action"].append(action)
            labels_dict_for_csv["reward"].append(reward)

            
            # save the image in the img_dir
            img = img.convert('RGB')
            img.save(f"{img_dir}/{img_filename}")
        
    df = pd.DataFrame(labels_dict_for_csv)
    df.to_csv(f"{img_dir}/labels.csv", index=False)

    
# PROJECT_ROOT=/home/jupyter-msiper/bootstrapping-pcgrl python gen_pod_traj_imgs.py --domain loderunner --num_episodes 10 --mode_of_output image --action_dim 8 --obs_dim 64 --traj_len 5
def get_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-d", "--domain", help="domain with which to run PoD", type=str, default="loderunner", choices=["lego", "loderunner", "zelda"])
    argparser.add_argument("-n", "--num_episodes", help="number of episodes to run PoD", type=int, default="loderunner")
    argparser.add_argument("-m", "--mode_of_output", help="Output mode of the PoD algorithm", type=str, default="image", choices=["onehot", "integer", "image", "string"])
    argparser.add_argument("-a", "--action_dim", help="Action dimension", type=int)
    argparser.add_argument("-o", "--obs_dim", help="Observation dimension", type=int)
    argparser.add_argument("-t", "--traj_len", help="Trajectory length", type=int)
    
    return argparser.parse_args()
    

import base64
from io import BytesIO
from PIL import Image

def pil_to_base64(pil_img):
    """Convert a PIL Image to base64 string."""
    img_buffer = BytesIO()
    pil_img.save(img_buffer, format="JPEG")  # Adjust format as needed
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode('utf-8')
    return base64_str



def gen_pod_trajectories(args):
    gen_trajectories(args.domain, args.num_episodes, args.mode_of_output, args.action_dim, args.obs_dim, args.traj_len)


def main():
    args = get_args()
    gen_pod_trajectories(args)
    
    
# NOTE: to run this --> PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python gen_expert_traj.py

main()
    
        