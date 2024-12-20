import os
import argparse
import datetime
import pathlib

import gym
import tensorflow as tf

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines.gail.dataset.dataset import ExpertDataset
from stable_baselines.results_plotter import load_results, ts2xy

import matplotlib 
# matplotlib.use('TkAgg') 

from pathlib import Path
from utils import load_model
from model import FullyConvPolicyBigMap, FullyConvPolicySmallMap, CustomPolicyBigMap, CustomPolicySmallMap

# from utils import get_exp_name, max_exp_idx, load_model, make_vec_envs
from stable_baselines import PPO2
from stable_baselines.results_plotter import load_results, ts2xy

import tensorflow as tf
import numpy as np
import os

import numpy as np
from utils import make_vec_envs as mkvenv, make_env as mkenv

from model import FullyConvPolicyBigMap, FullyConvPolicySmallMap, CustomPolicyBigMap, CustomPolicySmallMap

np.seterr(all='raise') 


PROJECT_ROOT = os.getenv("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise RuntimeError("The env var `PROJECT_ROOT` is not set.")
    

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward

    # Print stats every 1000 calls
    if (n_steps + 1) % 10 == 0:
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 100:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, we save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(os.path.join(log_dir, 'best_model.pkl'))
            else:
                print("Saving latest model")
                _locals['self'].save(os.path.join(log_dir, 'latest_model.pkl'))
        else:
            print('{} monitor entries'.format(len(x)))
            pass
    n_steps += 1
    # Returning False will stop training early
    return True


def main(game, representation, experiment, steps, n_cpu, render, logging, experiment_path, tb_log_dir, **kwargs):
    resume = kwargs.get('resume', False)
    if game == "binary":
        kwargs['cropped_size'] = 28
    elif game == "zelda":
        kwargs['cropped_size'] = 22
    elif game == "sokoban":
        kwargs['cropped_size'] = 10

    global log_dir

    log_dir = kwargs["logdir"]
    if resume:
        model = load_model(experiment_path)
        
    kwargs = {
        **kwargs,
        'render_rank': 0,
        'render': render,
    }
    env = mkvenv(game+"-" + representation +"-v0", "narrow", experiment_path, 1, **kwargs)
    if not resume or model is None:
        model = PPO2(CustomPolicyBigMap, env, verbose=1, tensorboard_log=experiment_path+"/")
    else:
        model.set_env(env)
        
    if not logging:
        model.learn(total_timesteps=int(steps), tb_log_name=tb_log_dir)
    else:
        model.learn(total_timesteps=int(steps), tb_log_name=tb_log_dir, callback=callback)


def parse_args():
    parser = argparse.ArgumentParser(
        prog='Training script',
        description='This is the training script for a non-bootstrapped agent'
    )
    parser.add_argument('--game', '-g', choices=['zelda', 'loderunner'], default="zelda") 
    parser.add_argument('--representation', '-r', default='narrow')
    parser.add_argument('--experiment', default="1")
    parser.add_argument('--n_steps', default=0, type=int)
    parser.add_argument('--steps', default=1000000, type=int)
    parser.add_argument('--render', default=False, type=bool)
    parser.add_argument('--logging', default=True, type=bool)
    parser.add_argument('--n_cpu', default=1, type=int)
    parser.add_argument('--tb_log_dir', default="ppo_1000_steps", type=str)
    parser.add_argument('--resume', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    game = args.game
    representation = args.representation
    experiment = args.experiment
    n_steps = args.n_steps
    steps = args.steps
    render = args.render
    logging = args.logging
    n_cpu = args.n_cpu
    kwargs = {
        'resume': args.resume
    }
    best_mean_reward = -np.inf
    experiment_path = PROJECT_ROOT + "/data/" + args.game + "/experiments/" + args.experiment
    experiment_filepath = pathlib.Path(experiment_path)
    if not experiment_filepath.exists():
        os.makedirs(str(experiment_filepath))

    kwargs["logdir"] = experiment_path
    log_dir = experiment_path
    main(game, representation, experiment, steps, n_cpu, render, logging, experiment_path, args.tb_log_dir, **kwargs)