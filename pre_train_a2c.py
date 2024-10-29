import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines.gail.dataset.dataset import ExpertDataset

import numpy as np
from utils import make_vec_envs as mkenv


from model import FullyConvPolicyBigMap, FullyConvPolicySmallMap, CustomPolicyBigMap, CustomPolicySmallMap

import numpy as np

np.seterr(all='raise') 

def sample_my_expert_transitions():
    data = np.load("/home/jupyter-msiper/bootstrapping_rl/lg_expert_traj_2.npz")
    expert_actions = data["actions"].astype(np.float64)
    acts = np.array([[np.argmax(a)] for a in expert_actions])
    expert_returns = data["episode_returns"].astype(np.float64)
    expert_rewards = data["rewards"].astype(np.float64)
    expert_obs = data["obs"].astype(np.float64)
    expert_starts = data["episode_starts"].astype(np.float64)

    return {
        "actions": acts,
        "episode_returns": expert_rewards,
        "rewards": expert_rewards,
        "obs": expert_obs,
        "episode_starts": expert_starts,
    }



# sample_my_expert_transitions()
# multiprocess environment
# env = make_vec_env('CartPole-v1', n_envs=4)
game = 'binary'
representation = 'narrow'
experiment = None
steps = 1e8
render = False
logging = True
n_cpu = 50
kwargs = {
    'resume': False,
    "cropped_size": 21
}

# NOTE: to run this --> RDMAV_FORK_SAFE=1 python pre_train_a2c.py

env = mkenv("zelda-narrow-v0", "narrow", "/home/jupyter-msiper/gym-pcgrl", 1, **kwargs)

data = sample_my_expert_transitions()
dataset = ExpertDataset(traj_data=data, traj_limitation=5000, batch_size=64,train_fraction=.95, verbose=2) #ExpertDataset(expert_path="/home/jupyter-msiper/bootstrapping_rl/lg_expert_traj_2.npz")


model = PPO2(CustomPolicyBigMap, env, verbose=2, tensorboard_log="/home/jupyter-msiper/gym-pcgrl", full_tensorboard_log=True)
model.set_env(env)
model.pretrain(dataset, n_epochs=100)

# NOTE: Check the HPC-related email
# NOTE: a) Controlability (aesthetics), b) speed increases (training), c) higher quality, d) diversity (?)
# NOTE: Use load running instead of Zelda!!
# model.learn(total_timesteps=100000)
# model.learn(total_timesteps=1000)
# model.save("ppo2_zelda_narrow_100_epochs")

# model.pretrain(dataset, n_epochs=20)

# del model # remove to demonstrate saving and loading

# model = PPO2.load("ppo2_zelda_wide_100K_epochs", env=env)

# Enjoy trained agent
solved = 0
failed = 0
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if dones:   
        solved = info[0]['solved']
        if solved:
            solved += 1
            print("{solved}".format(solved=solved))
            print("{failed}".format(failed=failed))
        else:
            failed += 1 
    # env.render()