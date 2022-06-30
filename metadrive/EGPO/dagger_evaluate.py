from __future__ import print_function
import time
from egpo_utils.dagger.exp_saver import Experiment

from egpo_utils.common import get_expert_action
from egpo_utils.expert_guided_env import ExpertGuidedEnv
import torch
from egpo_utils.dagger.utils import store_data, read_data, train_model, evaluation
import numpy as np
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from egpo_utils.dagger.model import Model
import os
from egpo_utils.common import evaluation_config
import os.path as osp
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import random
# require loguru imageio easydict tensorboardX pyyaml pytorch==1.5.0 stable_baselines3, cudatoolkit==9.2

# hyperpara
evaluation_episode_num = 50
device = "cuda"

# test env config
eval_config = evaluation_config["env_config"]
eval_config["horizon"] = 1000
force_seed = 510

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def evaluate(env, model, evaluation_episode_num=30, device="cuda"):
    with torch.no_grad():
        print("... evaluation")
        episode_reward = 0
        episode_cost = 0
        success_num = 0
        episode_num = 0
        velocity = []
        episode_overtake = []
        reward = 0
        state = env.reset(force_seed=force_seed)
        set_seed(episode_num)
        while episode_num < evaluation_episode_num:
            prediction = model(torch.tensor(state).to(device).float())
            next_state, r, done, info = env.step(prediction.detach().cpu().numpy().flatten())
            state = next_state
            episode_reward += r
            reward += r
            episode_cost += info["native_cost"]
            velocity.append(info["velocity"])
            if done:
                episode_overtake.append(info["overtake_vehicle_num"])
                episode_num += 1
                if info["arrive_dest"]:
                    success_num += 1
                env.reset(force_seed=force_seed)
                set_seed(episode_num)
                reward = 0

        res = dict(
            mean_episode_reward=episode_reward / episode_num,
            mean_episode_cost=episode_cost / episode_num,
            mean_success_rate=success_num / episode_num,
            mean_velocity=np.mean(velocity),
            mean_episode_overtake_num=np.mean(episode_overtake)
        )
        return res

if __name__ == "__main__":

    eval_env = ExpertGuidedEnv(eval_config)

    obs_shape = eval_env.observation_space.shape[0]
    action_shape = eval_env.action_space.shape[0]
    # print(obs_shape)
    # print(action_shape)

    # agent
    agent = Model(obs_shape, action_shape, (256, 256)).to(device).float()
    agent.load('/home/jvy5516/project/metadrive/EGPO/dagger_models/model_5.pth')
    eval_res = evaluate(eval_env, agent, evaluation_episode_num=evaluation_episode_num)
    print(eval_res)
