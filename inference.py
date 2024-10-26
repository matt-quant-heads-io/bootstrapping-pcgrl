"""
Run a trained agent and get generated maps
"""
import model
from stable_baselines import PPO2

import time
from utils import make_vec_envs

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
                    # input('')
                else:
                    unsolved += 1
                    solved_pct = round(float(solved) / (float(solved)+float(unsolved)),2)*100
                    print("Solve %: {solved_pct} Solved {solved}, Unsolved: {unsolved}".format(solved_pct=solved_pct, solved=solved, unsolved=unsolved))

                # print("info {info}".format(info=info))
                break
        dones = False
        obs = env.reset()
        # time.sleep(0.2)

################################## MAIN ########################################
game = 'zelda'
representation = 'narrow'
model_path = 'models/{}/{}/model_1.pkl'.format(game, representation)
model_path = "/home/jupyter-msiper/gym-pcgrl/runs/zelda_narrow_1_1_log/best_model2.pkl"#"/home/jupyter-msiper/gym-pcgrl/runs/zelda_narrow_1_1_log/best_model.pkl" #"/home/jupyter-msiper/gym-pcgrl/ppo2_zelda_wide_10_epochs.zip"
kwargs = {
    'change_percentage': 1,
    'trials': 1000,
    'verbose': True
}

if __name__ == '__main__':
    infer(game, representation, model_path, **kwargs)
