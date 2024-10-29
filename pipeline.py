import os
import argparse
import datetime

import gym
import tensorflow as tf

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines.gail.dataset.dataset import ExpertDataset

import numpy as np
from utils import make_vec_envs as mkvenv, make_env as mkenv

from model import FullyConvPolicyBigMap, FullyConvPolicySmallMap, CustomPolicyBigMap, CustomPolicySmallMap
from inference import infer
from compute_results import calculate_results

from train import callback

np.seterr(all='raise') 


PROJECT_ROOT = os.getenv("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise RuntimeError("The env var `PROJECT_ROOT` is not set.")


def load_trajectories(path_to_trajectories):
    data1 = np.load("/home/jupyter-msiper/bootstrapping-pcgrl/data/zelda/trajectories/lg_expert_goalset_traj_1.npz")
    data2 = np.load("/home/jupyter-msiper/bootstrapping-pcgrl/data/zelda/trajectories/lg_expert_goalset_traj_2.npz")
    expert_actions = np.concatenate([data1["actions"].astype(np.float64), data2["actions"].astype(np.float64)])
    acts = np.array([[np.argmax(a)+1] for a in expert_actions])
    expert_rewards = np.concatenate([data1["rewards"].astype(np.float64), data2["rewards"].astype(np.float64)])
    expert_obs = np.concatenate([data1["obs"].astype(np.float64), data2["obs"].astype(np.float64)])
    expert_starts = np.concatenate([data1["episode_starts"].astype(np.float64), data2["episode_starts"].astype(np.float64)])

    return {
        "actions": acts,
        "episode_returns": expert_rewards,
        "rewards": expert_rewards,
        "obs": expert_obs,
        "episode_starts": expert_starts,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        prog='Experiment Pipeline',
        description='This program constructs the experiment pipeline to be executed bases on the input arguments'
    )
    parser.add_argument('--train_process', choices=['ppo', 'bootstrap_then_ppo']) 
    parser.add_argument('--goals_for_traj_gen', choices=['standard', 'expert'], default='standard')
    parser.add_argument('--num_bootstrap_epochs', type=int)
    parser.add_argument('--num_ppo_timesteps', type=int)
    parser.add_argument('--num_boostrap_episodes', type=int)
    parser.add_argument('--domain', choices=['zelda', 'loderunner'], default='zelda')
    parser.add_argument('--num_boostrap_epochs', default=100, type=int)
    parser.add_argument('--resume', action='store_true')

    return parser.parse_args()

"""
EXPERIMENT 1: 1 MILLION TOTAL STEPS

Bootstrap:
==========
RDMAV_FORK_SAFE=1 python pipeline.py --resume --train_process bootstrap_then_ppo --goals_for_traj_gen standard  --num_bootstrap_epochs 100 --num_ppo_timesteps 615000 --num_boostrap_episodes 5000 --domain zelda

args.num_boostrap_episodes == 5000, n_epochs == 100, num_ppo_timesteps == 615000 --> 1M total steps

PPO:
==========
num_ppo_timesteps == 1000000 --> 1000000 total steps

RDMAV_FORK_SAFE=1 python pipeline.py --train_process ppo --goals_for_traj_gen standard  --num_bootstrap_epochs 100 --num_ppo_timesteps 1000000 --num_boostrap_episodes 5000 --domain zelda 


"""



def construct_experiment(args):
    kwargs = {
        'resume': False,
        "cropped_size": 21
    }
    experiment_path = "data/{domain}/models/{train_process}/{num_boostrap_episodes}_{num_ppo_timesteps}_{num_boostrap_epochs}".format(domain=args.domain, train_process=args.train_process, num_boostrap_episodes=str(args.num_boostrap_episodes), num_ppo_timesteps=str(args.num_ppo_timesteps), num_boostrap_epochs=str(args.num_boostrap_epochs))
    
    
    if args.train_process == "ppo":
        model_save_path = "{experiment_path}/model.zip".format(experiment_path=experiment_path)
    else:
        model_save_path = "{experiment_path}/best_model.pkl".format(experiment_path=experiment_path)

    tb_loag_path = experiment_path
    os.makedirs(experiment_path, exist_ok=True)
    env = mkenv("zelda-narrow-v0", "narrow", experiment_path, 1, **kwargs)
    if args.resume:
        # model = PPO2.load(model_save_path)
        model = PPO2.load("/home/jupyter-msiper/bootstrapping-pcgrl/data/zelda/models/bootstrap_then_ppo/10000_1000000_100/best_model.pkl") 
    else:
        model = PPO2(CustomPolicyBigMap, env, verbose=2, tensorboard_log=experiment_path, full_tensorboard_log=True)
        model.set_env(env)
    
    bootstrap_total_time = None
    if args.train_process == "bootstrap_then_ppo": #and not args.resume:
        traj_path = "{PROJECT_ROOT}/data/{domain}/trajectories/expert_goalset_traj.npz".format(PROJECT_ROOT=PROJECT_ROOT, domain=args.domain)
        # os.makedirs(traj_path, exist_ok=True)
        trajectories = load_trajectories(traj_path)
        dataset = ExpertDataset(traj_data=trajectories, traj_limitation=args.num_boostrap_episodes, batch_size=64,train_fraction=.99, verbose=2)   
        bootstrap_start_time = datetime.datetime.now()
        print("Calling pretrain!")
        model.pretrain(dataset, n_epochs=args.num_boostrap_epochs, save_path=model_save_path)  
        bootstrap_end_time = datetime.datetime.now()
        bootstrap_total_time = (bootstrap_end_time - bootstrap_start_time).total_seconds()
        model.save(model_save_path)


    # ppo_start_time = datetime.datetime.now()
    # model.learn(total_timesteps=args.num_ppo_timesteps, tb_log_name="ppo_log", callback=callback)
    # ppo_end_time = datetime.datetime.now()
    # ppo_total_time = (ppo_end_time - ppo_start_time).total_seconds()
    
    # model.save(model_save_path)
    # inference_save_path = "{tb_loag_path}/{num_boostrap_episodes}_{num_ppo_timesteps}_{num_boostrap_epochs}.json".format(tb_loag_path=tb_loag_path, num_boostrap_episodes=str(args.num_boostrap_episodes), num_ppo_timesteps=str(args.num_ppo_timesteps), num_boostrap_epochs=str(args.num_boostrap_epochs))

    # kwargs = {
    #     'change_percentage': 1,
    #     'trials': 1000,
    #     'verbose': True,
    #     'train_process': args.train_process,
    #     'num_boostrap_episodes': args.num_boostrap_episodes if args.num_boostrap_episodes else -1,
    #     'num_ppo_timesteps': args.num_ppo_timesteps,
    #     'inference_save_path': "{PROJECT_ROOT}/{inference_save_path}".format(PROJECT_ROOT=PROJECT_ROOT, inference_save_path=inference_save_path),
    #     'num_boostrap_epochs': args.num_boostrap_epochs,
    #     'experiment_path': experiment_path,
    #     'bootstrap_total_time': bootstrap_total_time if bootstrap_total_time else 0,
    #     'ppo_total_time': ppo_total_time,
    #     'total_train_time': bootstrap_total_time + ppo_total_time  if bootstrap_total_time else ppo_total_time 
    
    # }
    # # infer(args.domain, "narrow", "{experiment_path}/best_model.pkl".format(experiment_path=experiment_path), **kwargs)
    # infer(args.domain, "narrow", model_save_path, **kwargs)
    # calculate_results(experiment_path)


def main():
    args = parse_args()
    construct_experiment(args)


main()


