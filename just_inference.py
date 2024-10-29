"""
Run a trained agent and get generated maps
"""
import json
import model
from stable_baselines import PPO2

import time
from utils import make_vec_envs
import os
import uuid

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


def str_map_to_char_map(str_map):
    level_str = ""
    for row in str_map:
        for col in row:
            level_str += REV_TILES_MAP[col]

    return level_str


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
        kwargs['cropped_size'] = 21
    elif game == "sokoban":
        model.FullyConvPolicy = model.FullyConvPolicySmallMap
        kwargs['cropped_size'] = 10
    kwargs['render'] = False

    agent = PPO2.load(model_path)
    env = make_vec_envs(env_name, representation, None, 1, **kwargs)
    obs = env.reset()
    obs = env.reset()
    dones = False
    solved = 0
    unsolved = 0
    results = {"solved_maps": [], 'num_boostrap_episodes': [], 'train_process': [], 'num_ppo_timesteps': [], 'num_boostrap_epochs': [], 'bootstrap_total_time': [], 'ppo_total_time': [], 'total_train_time': []}
    for i in range(kwargs.get('trials', 1)):
        while not dones:
            action, _ = agent.predict(obs)
            # print("action: {action}".format(action=action))
            obs, _, dones, info = env.step(action)
            # obs, _, dones, info = env.step(action)
            if kwargs.get('verbose', False):
                # print(info[0])
                pass
            if dones:
                if info[0]["solved"]:
                    solved += 1
                    solved_pct = round(float(solved) / (float(solved)+float(unsolved)),2)*100
                    print("Solve %: {solved_pct} Solved {solved}, Unsolved: {unsolved}".format(solved_pct=solved_pct, solved=solved, unsolved=unsolved))
                    char_map = str_map_to_char_map(info[0]["final_map"])
                    results["solved_maps"].append(char_map)
                    results["num_boostrap_episodes"].append(0)
                    results["num_ppo_timesteps"].append(0)
                    results["train_process"].append(0)
                    results["num_boostrap_epochs"].append(0)
                    results["bootstrap_total_time"].append(0)
                    results["ppo_total_time"].append(0)
                    results["total_train_time"].append(0)
                    open("/home/jupyter-msiper/bootstrapping-pcgrl/goal_maps/expert_zelda/" + str(uuid.uuid1()) +".txt", "w").write(char_map)
                else:
                    unsolved += 1
                    solved_pct = round(float(solved) / (float(solved)+float(unsolved)),2)*100
                    print("Solve %: {solved_pct} Solved {solved}, Unsolved: {unsolved}".format(solved_pct=solved_pct, solved=solved, unsolved=unsolved))

                # print("info {info}".format(info=info))
                break
        dones = False
        obs = env.reset()
        # os.system("chmod 777 {experiment_path}/inference_results.json".format(experiment_path=kwargs['experiment_path']))
        # with open(kwargs['experiment_path']+"/inference_results.json", "w") as f:
        #     f.write(json.dumps(results))
        # time.sleep(0.2)

################################## MAIN ########################################
game = 'zelda'
representation = 'narrow'
model_path = 'models/{}/{}/model_1.pkl'.format(game, representation)
experiment_path = "/home/jupyter-msiper/bootstrapping-pcgrl/data/zelda/models/ppo/50_5000_100"
experiment_path = "/home/jupyter-msiper/bootstrapping-pcgrl/data/zelda/models/bootstrap_then_ppo/50_5000_100"
model_path = "{experiment_path}/model.zip".format(experiment_path=experiment_path) #"/home/jupyter-msiper/gym-pcgrl/runs/zelda_narrow_1_1_log/best_model2.pkl"#"/home/jupyter-msiper/gym-pcgrl/runs/zelda_narrow_1_1_log/best_model.pkl" #"/home/jupyter-msiper/gym-pcgrl/ppo2_zelda_wide_10_epochs.zip"

# model_path = "/home/jupyter-msiper/bootstrapping-pcgrl/models/zelda/narrow/model_1.pkl"
model_path = "/home/jupyter-msiper/bootstrapping-pcgrl/data/zelda/models/ppo/5000_1000000_100/best_model.pkl"
model_path = "/home/jupyter-msiper/bootstrapping-pcgrl/data/zelda/models/bootstrap_then_ppo/10000_1000000_100/best_model.pkl"
# model_path = "/home/jupyter-msiper/bootstrapping-pcgrl/data/zelda/models/bootstrap_then_ppo/5000_615000_100/best_model.pkl"
kwargs = {
    'change_percentage': 0.4,
    'trials': 100,
    'verbose': True,
    'experiment_path': experiment_path
}

if __name__ == '__main__':
    infer(game, representation, model_path, **kwargs)
