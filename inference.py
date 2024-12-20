"""
Run a trained agent and get generated maps
"""
import json
import model

from utils import make_vec_envs
import os
import argparse

PROJECT_ROOT = os.getenv("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise RuntimeError("The env var `PROJECT_ROOT` is not set.")

from stable_baselines import PPO2


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
        kwargs['cropped_size'] = 22
    elif game == "sokoban":
        model.FullyConvPolicy = model.FullyConvPolicySmallMap
        kwargs['cropped_size'] = 10
    kwargs['render'] = False

    env = make_vec_envs(env_name, representation, None, 1, **kwargs)
    agent = PPO2.load(model_path)
    obs = env.reset()
    obs = env.reset()
    dones = False
    solved = 0
    unsolved = 0
    rewards = []
    results = {"solved_maps": [], 'num_boostrap_episodes': [], 'train_process': [], 'num_ppo_timesteps': [], 'num_boostrap_epochs': [], 'bootstrap_total_time': [], 'ppo_total_time': [], 'total_train_time': []}
    for i in range(kwargs.get('trials', 1)):
        while not dones:
            action, _ = agent.predict(obs)
            obs, _, dones, info = env.step(action)
            if kwargs.get('verbose', False):
                # print(info[0])
                pass
            if dones:
                if info[0]["solved"]:
                    solved += 1
                    solved_pct = round(float(solved) / (float(solved)+float(unsolved)),2)*100
                    print("Solve %: {solved_pct} Solved {solved}, Unsolved: {unsolved}".format(solved_pct=solved_pct, solved=solved, unsolved=unsolved))
                    reward = info[0]["reward"]
                    rewards.append(reward)
                    
                    print("Mean reward: {reward}".format(reward=sum(rewards)/len(rewards)))
                    char_map = str_map_to_char_map(info[0]["final_map"])
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
        obs = env.reset()
        os.system("chmod 777 {experiment_path}/generated_maps/inference_results.json".format(experiment_path=kwargs['experiment_path']))
        with open(kwargs['experiment_path']+"/generated_maps/inference_results.json", "w") as f:
            f.write(json.dumps(results))
        # time.sleep(0.2)


def parse_args():
    parser = argparse.ArgumentParser(
        prog='Inference script',
        description='This is the inference script for evaluating agent performance'
    )
    parser.add_argument('--game', '-g', choices=['zelda', 'loderunner'], default="zelda") 
    parser.add_argument('--representation', 'r', default='narrow')
    parser.add_argument('--results_path', default="1")
    parser.add_argument('--model_path', required=True, type=str)
    parser.add_argument('--chg_pct', default=0.4, type=float)
    parser.add_argument('--trials', default=500, type=int)
    parser.add_argument('--verbose', default=True, type=bool)

    return parser.parse_args()


################################## MAIN ########################################
if __name__ == '__main__':
    args = parse_args()
    game = args.game 
    representation = args.representation
    model_path = args.model_path
    
    results_path = PROJECT_ROOT + "/data/" + args.game + "/experiments/" + args.results_path
    kwargs = {
        'change_percentage': args.chg_pct,
        'trials': args.trials,
        'verbose': args.verbose,
        'experiment_path': results_path
    }
    
    
    infer(game, representation, model_path, **kwargs)