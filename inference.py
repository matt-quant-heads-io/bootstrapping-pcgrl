"""
Run a trained agent and get generated maps
"""
import json
import model
from stable_baselines import PPO2

import time
from utils import make_vec_envs
import os


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
            # obs, _, dones, info = env.step(action+1)
            obs, _, dones, info = env.step(action)
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
                    results["num_boostrap_episodes"].append(kwargs['num_boostrap_episodes'])
                    results["num_ppo_timesteps"].append(kwargs['num_ppo_timesteps'])
                    results["train_process"].append(kwargs['train_process'])
                    results["num_boostrap_epochs"].append(kwargs['num_boostrap_epochs'])
                    results["bootstrap_total_time"].append(kwargs['bootstrap_total_time'])
                    results["ppo_total_time"].append(kwargs['ppo_total_time'])
                    results["total_train_time"].append(kwargs['total_train_time'])
                else:
                    unsolved += 1
                    solved_pct = round(float(solved) / (float(solved)+float(unsolved)),2)*100
                    print("Solve %: {solved_pct} Solved {solved}, Unsolved: {unsolved}".format(solved_pct=solved_pct, solved=solved, unsolved=unsolved))

                # print("info {info}".format(info=info))
                break
        dones = False
        obs = env.reset()
        os.system("chmod 777 {experiment_path}/inference_results.json".format(experiment_path=kwargs['experiment_path']))
        with open(kwargs['experiment_path']+"/inference_results.json", "w") as f:
            f.write(json.dumps(results))
        # time.sleep(0.2)

################################## MAIN ########################################
# game = 'zelda'
# representation = 'narrow'
# model_path = 'models/{}/{}/model_1.pkl'.format(game, representation)
# model_path = "/home/jupyter-msiper/gym-pcgrl/runs/zelda_narrow_1_1_log/best_model2.pkl"#"/home/jupyter-msiper/gym-pcgrl/runs/zelda_narrow_1_1_log/best_model.pkl" #"/home/jupyter-msiper/gym-pcgrl/ppo2_zelda_wide_10_epochs.zip"
# kwargs = {
#     'change_percentage': 1,
#     'trials': 1000,
#     'verbose': True
# }

# if __name__ == '__main__':
#     infer(game, representation, model_path, **kwargs)
