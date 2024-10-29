import numpy as np
import constants
from tqdm import tqdm, trange
import os
import pandas as pd

def hamm_dist(lvl1, lvl2):
    # print(f"lvl1: {lvl1}")
    # print(f"lvl2: {lvl2}")
    value = np.abs(lvl1.flatten() - lvl2.flatten())
    value[value > 0] = 1
    return value.sum()

def l1_dist(im1, im2):
    return abs(im1-im2).mean()

def direction_diversity(lvls, goals, distfn=hamm_dist):
    dists = []
    for lvl in lvls:
        min_dist = -1
        for goal in goals:
            new_dist = distfn(lvl, goal)
            if min_dist < 0 or new_dist < min_dist:
                min_dist = new_dist
        dists.append(min_dist)
    return np.array(dists)

def get_subset_diversity(lvls, goal, cuttoff, distfn=hamm_dist):
    dists = direction_diversity(lvls, goal, distfn)
    return lvls[dists >= cuttoff]

def greedy_set_diversity(lvls, cuttoff, distfn=hamm_dist):
    indeces = set()
    for i,lvl in enumerate(tqdm(lvls, leave=False)):
        if i in indeces:
            continue
        values = direction_diversity(lvls, [lvl], distfn)
        temp = np.where(values < cuttoff)[0]
        if len(temp) > 1:
            temp = temp[temp > i]
            indeces.update(temp)
    return np.delete(lvls, list(indeces), axis=0)

def ogreedy_set_diversity(lvls, cuttoff, distfn=hamm_dist):
    extra_info = []
    repeat = np.zeros(len(lvls))
    for i,lvl in enumerate(tqdm(lvls, leave=False, desc="Sorting By Repeatition")):
        values = direction_diversity(lvls, [lvl], distfn)
        values[i] = 10000
        repeat[i] = (values < cuttoff).sum()
        
        temp_info = {}
        temp_info['cuttoff'] = (values < cuttoff).sum()
        temp_info['worse'] = values.min()
        temp_info['identical'] = (values == 0).sum()
        temp_info['worse_identical'] = (values == temp_info['worse']).sum()
        extra_info.append(temp_info)
    return greedy_set_diversity(lvls[repeat.argsort()], cuttoff, distfn), extra_info



# Reads in .txt playable map and converts it to int[][]
def read_in_charmap_as_int_level(file_name):
    level = []

    with open(file_name, "r") as f:
        rows = f.readlines()
        for row in rows:
            new_row = []
            for char in row:
                if char != "\n":
                    new_row.append(constants.CHAR_INT_MAP[char])
            level.append(new_row)

    # Remove the border
    truncated_level = level[1 : len(level) - 1]
    new_level = []
    for row in truncated_level:
        new_row = row[1 : len(row) - 1]
        new_level.append(new_row)
    return np.array(new_level)


# def read_in_chartestmap_as_int_level(file_name):
#     level = []

#     with open(file_name, "r") as f:
#         level = f.readlines()[0]

#     new_level = []
#     for l in level:
#         new_level.append(constants.CHAR_INT_MAP[l])
#     # print(f"new_level: {new_level}")

#     new_level = np.array(new_level)
#     return new_level.reshape((7,11))

import json

CHAR_INT_MAP = {
    "g": 4,
    "+": 3,
    "A": 2,
    "1": 5,
    "2": 7,
    "3": 6,
    "w": 1,
    ".": 0
}

def read_in_chartestmap_as_int_level(file_name):
    levels = []
    solved_maps = None
    with open(file_name + "/inference_results.json", "r") as f:
        solved_maps = json.loads(f.read())['solved_maps']

    new_level = []
    for solved_map in solved_maps:
        new_level = []
        for l in solved_map:
            new_level.append(CHAR_INT_MAP[l])
        new_level = np.array(new_level)
        new_level = new_level.reshape((7,11))
        levels.append(new_level)

    return np.array(levels)


def load_goal_maps(path):
    int_maps = []
    for goal_file in os.listdir(path):
        if goal_file.endswith('.txt'):
            # print(f"found GOAL map at{f'{path}/{goal_file}'}")
            int_maps.append(read_in_charmap_as_int_level('{path}/{goal_file}'))

    # print(f"GOAL MAPS: {int_maps}")
    return np.array(int_maps)


# def load_test_maps(path):
#     int_maps = []
#     for goal_subdir in os.listdir(path):
#         for goal_dir in os.listdir(f"{path}/{goal_subdir}"):
#             for goal_file in os.listdir(f"{path}/{goal_subdir}/{goal_dir}"):
#                 if goal_file.endswith('.txt'):
#                     int_maps.append(read_in_chartestmap_as_int_level(f'{path}/{goal_subdir}/{goal_dir}/{goal_file}'))

#     return np.array(int_maps)


obs_size = 21

# TEST_MAPS_DIR = f'/home/jupyter-msiper/controllable_pod/generated_maps_char_maps'
# GOAL_MAPS_DIR = '/home/jupyter-msiper/controllable_pod/playable_maps'


def calculate_results(test_map_dir):
    test_lvls = read_in_chartestmap_as_int_level(test_map_dir)
    # goal_lvls = load_goal_maps(goal_map_dir) 
    inference_results_json = None
    with open(test_map_dir + "/inference_results.json", "r") as f:
        inference_results_json = json.loads(f.read())
    
    width = 11
    height = 7
    num_boostrap_epochs = inference_results_json['num_boostrap_epochs'][0]
    num_ppo_timesteps = inference_results_json["num_ppo_timesteps"][0]
    train_process = inference_results_json["train_process"][0]
    bootstrap_total_time = inference_results_json["bootstrap_total_time"][0]
    total_train_time = inference_results_json["total_train_time"][0]
    ppo_total_time = inference_results_json["ppo_total_time"][0]
    num_boostrap_episodes = inference_results_json["num_boostrap_episodes"][0]

    results_dict = {
        "num_boostrap_epochs": [
            
        ],
        "num_ppo_timesteps": [
            
        ],
        "train_process": [
            
        ],
        "bootstrap_total_time": [
            
        ],
        "total_train_time": [
            
        ],
        "ppo_total_time": [
            
        ],
        "num_boostrap_episodes": [
            
        ],
        "obs": [
            
        ],
        "threshold": [
            
        ],
        "min num diff tiles": [
            
        ],
        "Diversity (wrt internal)": [
            
        ]
    }
    thresholds = [0.05]
    for threshold in thresholds:
        cuttoff = threshold * width * height  
        unique_ogreedy_lvls, _ = ogreedy_set_diversity(test_lvls, cuttoff)

        results_dict["obs"].append(obs_size)
        results_dict["threshold"].append(threshold)
        results_dict["min num diff tiles"].append(int(cuttoff))
        results_dict["Diversity (wrt internal)"].append(len(unique_ogreedy_lvls) / float(len(test_lvls)))
        results_dict["num_boostrap_epochs"].append(num_boostrap_epochs)
        results_dict["num_ppo_timesteps"].append(num_ppo_timesteps)
        results_dict["train_process"].append(train_process)
        results_dict["bootstrap_total_time"].append(bootstrap_total_time)
        results_dict["total_train_time"].append(total_train_time)
        results_dict["ppo_total_time"].append(ppo_total_time)
        results_dict["num_boostrap_episodes"].append(num_boostrap_episodes)


    import pdb; pdb.set_trace()
    df = pd.DataFrame(results_dict)
    df.to_csv(test_map_dir + "/final_results.json", index=False)
    
    print("Finished generating results!")


# Cutoff 10%
# Diversity with respect to Goal then Internal: 122 / 309

# Cutoff 10%
# Diversity with respect to Goal then Internal: 122 / 309



"""
Next steps:
===========
- pos (1), negative (-1), neutral (0) [1-hot encoded] for conditional inputs, every stps
    calc params needed, network learns which trait to increse/decrease
    - compute this each step during trajectory generation

    - can play with the destructor since now we have these local updates 
        different sequences of updates can produce the same output traits 
        so how can we destroy in a smart way?

    - counting network stays the same

    - 3D minecraft

- Binary?
- Start overleaf

"""