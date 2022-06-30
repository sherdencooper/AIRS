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
import pandas as pd
import argparse
# require loguru imageio easydict tensorboardX pyyaml pytorch==1.5.0 stable_baselines3, cudatoolkit==9.2

# hyperpara
device = "cuda"

# test env config
eval_config = evaluation_config["env_config"]
eval_config["horizon"] = 1000
force_seed = 510

eval_env = ExpertGuidedEnv(eval_config)

obs_shape = eval_env.observation_space.shape[0]
action_shape = eval_env.action_space.shape[0]
# print(obs_shape)
# print(action_shape)
# agent
agent = Model(obs_shape, action_shape, (256, 256)).to(device).float()
agent.load('/home/jvy5516/project/metadrive/EGPO/dagger_models/model_5.pth')
num = 50
path = './observation/reward.txt'
df = pd.read_table(path, sep=';', header=None)
reward_origin = df.values.squeeze()


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def read_loc(method):
    summary = 'summary_explain/' + method + '.txt'
    summary_pd = pd.read_table(summary, sep=';', header=None)
    return summary_pd[0].values

def process(loc,num,reward_origin,thres=200):
    out = replay(num,loc)
    dif = np.abs(out-reward_origin[0:num])
    sum = 0
    num = 0
    for i in range(dif.shape[0]):
        if 0<=dif[i]<thres:
            sum+=dif[i]
            num+=1
    return sum/num

def replay(num,loc):
    eval_env = ExpertGuidedEnv(eval_config)
    with torch.no_grad():
        print("... replay")
        reward = 0
        timestep = 0
        replay_reward = []
        episode_num = 0
        state = eval_env.reset(force_seed=force_seed)
        set_seed(episode_num)
        while episode_num < num:
            prediction = agent(torch.tensor(state).to(device).float())
            if loc[episode_num]<=timestep<=loc[episode_num]+50:
                prediction[0] = np.random.uniform(low=-0.05,high=0.05)
                prediction[1] = np.random.uniform(low=0.5,high=1.5)
            next_state, r, done, info = eval_env.step(prediction.detach().cpu().numpy().flatten())
            state = next_state
            reward += r
            timestep += 1
            if done:
                episode_num += 1
                eval_env.reset(force_seed=force_seed)
                set_seed(episode_num)
                replay_reward.append(reward)
                reward = 0
        eval_env.close()
        return np.array(replay_reward)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fidelity test.')

    parser.add_argument('--method', default='concept', type=str,
                        help='method for choosing the critical steps')
    parser.add_argument('--num', default=50,type=int,
                        help='number of fidelity test episodes')
    args = parser.parse_args()

    if args.method == 'random':
        loc = np.random.randint(322,622,size=[num])
    else:
        loc = read_loc(args.method)
    set_seed(10)

    diff = process(loc,args.num,reward_origin)
    print("The diff of %s is %s" % (args.method,diff))
