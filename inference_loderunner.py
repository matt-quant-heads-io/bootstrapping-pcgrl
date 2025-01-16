"""
Run a trained agent and get generated maps
"""
import json
import model

from utils import make_env, make_vec_envs
import os
import argparse
import torch as th
import pathlib
from sl_model import CNN
from gen_pod_traj import transform

from gym_pcgrl.envs.pcgrl_env import PcgrlEnv
import numpy as np



PROJECT_ROOT = os.getenv("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise RuntimeError("The env var `PROJECT_ROOT` is not set.")

DOMAIN_TO_INT_TILE_MAP = {
        "empty": 0,
        "brick": 1, 
        "ladder": 2, 
        "rope": 3, 
        "solid": 4, 
        "gold": 5, 
        "enemy": 6, 
        "player": 7
}
DOMAIN_TO_CHAR_TO_STR_TILE_MAP =  {
        ".":"empty",
        "b":"brick", 
        "#":"ladder", 
        "-":"rope", 
        "B":"solid", 
        "G":"gold", 
        "E":"enemy", 
        "M":"player"
}

# from stable_baselines import PPO2
from gym import envs
import gym_pcgrl

[env.id for env in envs.registry.all() if "gym_pcgrl" in env.entry_point]
print([env.id for env in envs.registry.all() if "gym_pcgrl" in env.entry_point])

REV_TILES_MAP = {
    "door": "g",
    "key": "+",
    "player": "A",
    "bat": "1",
    "spider": "2",
    "scorpion": "3",
    "solid": "w",
    "empty": ".",
}

def str_arr_from_int_arr(map, str_to_int_map):
    translation_map = {v: k for k, v in str_to_int_map.items()}
    str_map = []
    for row_idx in range(len(map)):
        new_row = []
        for col_idx in range(len(map[0])):
            new_row.append(translation_map[map[row_idx][col_idx]])
        str_map.append(new_row)

    return str_map


def str_map_to_char_map(str_map):
    level_str = ""
    for row in str_map:
        for col in row:
            level_str += REV_TILES_MAP[col]

    return level_str


def convert_int_map_to_str_map(state, int_to_str_tiles_map):
    str_map = []
    for row_i, row in enumerate(state):
        for col_i, col in enumerate(row):
            str_map.append(int_to_str_tiles_map[state[row_i][col_i]])
    return np.array(str_map).reshape(state.shape)

def convert_str_map_to_int_map(state, str_to_int_tiles_map):
    int_map = []
    for row_i, row in enumerate(state):
        for col_i, col in enumerate(row):
            int_map.append(str_to_int_tiles_map[col])

    import pdb; pdb.set_trace()
    return np.array(int_map).reshape(*state.shape)


def convert_onehot_map_to_int_map(state):
    int_map = []
    for row_i, row in enumerate(state):
        for col_i, col in enumerate(row):
            idx = np.where(col == 1)
            int_map.append(idx)
    return np.array(int_map).reshape(*state.shape[:2])


def infer(game, representation, model_path, **kwargs):
    """
     - max_trials: The number of trials per evaluation.
     - infer_kwargs: Args to pass to the environment.
    """
    env_name = '{}-{}-v0'.format(game, representation)
    if game == "binary":
        model.FullyConvPolicy = model.FullyConvPolicyBigMap
        kwargs['cropped_size'] = 28
    elif game == "zelda":
        model.FullyConvPolicy = model.FullyConvPolicyBigMap
        kwargs['cropped_size'] = 22
    elif game == "sokoban":
        model.FullyConvPolicy = model.FullyConvPolicySmallMap
        kwargs['cropped_size'] = 10
    elif game == "loderunner":
        kwargs['cropped_size'] = 64

    kwargs['render'] = True

    env = make_vec_envs(env_name, representation, None, 1, **kwargs)
    # import pdb; pdb.set_trace()
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    
    if game != "loderunner":
        agent = PPO2.load(model_path)
    else:
        model = CNN(device)
        agent = model.load_state_dict(th.load(model_path, weights_only=True))

    env = PcgrlEnv(prob="loderunner", rep="narrow")

    data = np.load("/home/jupyter-msiper/bootstrapping-pcgrl/data/loderunner/pod_trajs_2025-01-11 19:56:45.897043.npz")
    start_state = data["obs"].astype(np.float32)[0]
    int_state = convert_onehot_map_to_int_map(start_state)
    int_to_str_map = {v:k for k,v in DOMAIN_TO_INT_TILE_MAP.items()}
    str_map = convert_int_map_to_str_map(int_state, int_to_str_map)
    
    # print(f"obs: {obs.shape}")
    obs = env.reset()['map']
    env._rep._map = int_state[42:, 32:]
    # str_arr_from_int_arr()
    # import pdb; pdb.set_trace()
    dones = False
    solved = 0
    unsolved = 0
    rewards = []
    results = {"solved_maps": [], 'num_boostrap_episodes': [], 'train_process': [], 'num_ppo_timesteps': [], 'num_boostrap_epochs': [], 'bootstrap_total_time': [], 'ppo_total_time': [], 'total_train_time': []}
    for i in range(kwargs.get('trials', 1)):
        while not dones:
            if game != "loderunner":
                action, _ = agent.predict(obs)
            else:
                
                
                # _obs = convert_str_map_to_int_map(env._rep._map, DOMAIN_TO_INT_TILE_MAP)
                # import pdb; pdb.set_trace()
                _obs = transform(env._rep._map, env._rep._x, env._rep._y, (64, 64, 8))
                
                
                output = model(th.nn.functional.one_hot(th.from_numpy(_obs).long(), num_classes=8).float().view(-1, 64, 64, 8))
                action = th.argmax(output[0]).item()
                # print(f"output: {output}")
            obs, _, dones, info = env.step(action+1)
            if kwargs.get('verbose', False):
                pass
            # print(f"info: {info}")
            if dones:
                if info["solved"]:
                    import pdb; pdb.set_trace()
                    solved += 1
                    solved_pct = round(float(solved) / (float(solved)+float(unsolved)),2)*100
                    print("Solve %: {solved_pct} Solved {solved}, Unsolved: {unsolved}".format(solved_pct=solved_pct, solved=solved, unsolved=unsolved))
                    reward = info["reward"]
                    rewards.append(reward)
                    
                    print("Mean reward: {reward}".format(reward=sum(rewards)/len(rewards)))
                    char_map = str_map_to_char_map(info["final_map"])
                    results["solved_maps"].append(char_map)
                    results["num_boostrap_episodes"].append(0)
                    results["num_ppo_timesteps"].append(0)
                    results["train_process"].append(0)
                    results["num_boostrap_epochs"].append(0)
                    results["bootstrap_total_time"].append(0)
                    results["ppo_total_time"].append(0)
                    results["total_train_time"].append(0)
                else:
                    unsolved += 1
                    solved_pct = round(float(solved) / (float(solved)+float(unsolved)),2)*100
                    print("Solve %: {solved_pct} Solved {solved}, Unsolved: {unsolved}".format(solved_pct=solved_pct, solved=solved, unsolved=unsolved))
                break
        dones = False
        obs = env.reset()['map']
        os.system("chmod 777 {experiment_path}/inference_results.json".format(experiment_path=kwargs['experiment_path']))
        with open(kwargs['experiment_path']+"/inference_results.json", "w") as f:
            f.write(json.dumps(results))
        # time.sleep(0.2)


def parse_args():
    parser = argparse.ArgumentParser(
        prog='Inference script',
        description='This is the inference script for evaluating agent performance'
    )
    parser.add_argument('--game', '-g', choices=['zelda', 'loderunner'], default="zelda") 
    parser.add_argument('--representation', '-r', default='narrow')
    parser.add_argument('--results_path', default="ppo_100M_steps")
    parser.add_argument('--model_path', required=True, type=str)
    parser.add_argument('--experiment', default="1", type=str)
    parser.add_argument('--chg_pct', default=1.0, type=float)
    parser.add_argument('--trials', default=500, type=int)
    parser.add_argument('--verbose', default=True, type=bool)

    return parser.parse_args()


################################## MAIN ########################################
if __name__ == '__main__':
    args = parse_args()
    game = args.game 
    representation = args.representation
    model_path = args.model_path
    
    experiment_path = PROJECT_ROOT + "/experiments/" + args.game + "/" + args.experiment + "/inference_results"
    experiment_filepath = pathlib.Path(experiment_path)
    if not experiment_filepath.exists():
        os.makedirs(str(experiment_filepath))

    
    kwargs = {
        'change_percentage': args.chg_pct,
        'trials': args.trials,
        'verbose': args.verbose,
        'experiment_path': experiment_path
    }
    
    
    infer(game, representation, model_path, **kwargs)