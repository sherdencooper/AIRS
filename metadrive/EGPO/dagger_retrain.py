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
from dagger_evaluate import evaluate
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import random
import pandas as pd
# require loguru imageio easydict tensorboardX pyyaml pytorch==1.5.0 stable_baselines3, cudatoolkit==9.2

# hyperpara
learning_rate = 5e-6
batch_size = 64
beta_i = 0.3  # expert mix ratio
T = 20000  # batch
evaluation_episode_num = 50
num_epoch = 1  # sgd epoch on data set
train_loss_threshold = 0.5
device = "cuda"
force_seed = 510
# training env_config
training_config = dict(
    vehicle_config=dict(
        use_saver=False,
        free_level=100),
    safe_rl_env=True,
    horizon=1000,
)

# test env config
eval_config = evaluation_config["env_config"]
eval_config["horizon"] = 1000

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def read_loc(method):
    summary = 'summary_explain/' + method + '.txt'
    summary_pd = pd.read_table(summary, sep=';', header=None)
    return summary_pd[0].values

def make_env(env_cls, config, seed=0):
    def _init():
        env = env_cls(config)
        return env

    return _init


expert_weights = osp.join(osp.dirname(__file__), "egpo_utils/expert.npz")

if __name__ == "__main__":
    if not os.path.exists("dagger_models"):
        os.mkdir("dagger_models")
    tm = time.localtime(time.time())
    tm_stamp = "%s-%s-%s-%s-%s-%s" % (tm.tm_year, tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec)
    log_dir = os.path.join(
        "dagger_lr_{}_bs_{}_sgd_iter_{}_dagger_batch_size_{}".format(learning_rate, batch_size, num_epoch, T), tm_stamp)
    training_env = SubprocVecEnv([make_env(ExpertGuidedEnv, config=eval_config)])  # seperate with eval env

    eval_env = ExpertGuidedEnv(eval_config)

    obs_shape = eval_env.observation_space.shape[0]
    action_shape = eval_env.action_space.shape[0]

    # agent
    agent = Model(obs_shape, action_shape, (256, 256)).to(device).float()
    model_number = 6
    agent.load('/home/jvy5516/project/metadrive/EGPO/dagger_models/model_5.pth')
    old_model_number = 5
    # dagger buffer
    samples = {
        "state": [],
        "next_state": [],
        "reward": [],
        "action": [],
        "terminal": [],
    }
    loc = read_loc('concept')
    steps = 0    
    curr_beta = beta_i ** model_number
    if model_number != old_model_number:
        old_model_number = model_number
    episode_reward = 0
    success_num = 0
    episode_cost = 0
    done_num = 0
    done = False
    state = training_env.reset()[0]
    sample_start = time.time()
    episode_step = 0
    set_seed(done_num)
    model_number += 1
    while True:
        # preprocess image and find prediction ->  policy(state)
        prediction = agent(torch.tensor(state).to(device).float())
        expert_a = get_expert_action(training_env)[0]
        pi = curr_beta * expert_a + (1 - curr_beta) * prediction.detach().cpu().numpy().flatten()
        next_state, r, done, info = training_env.step([pi])
        episode_step += 1
        next_state = next_state[0]
        r = r[0]
        done = done[0]
        info = info[0]
        episode_reward += r
        episode_cost += info["native_cost"]
        if episode_step>=loc[done_num]:     #only train over these critical timesteps
            sample = 2
        else:
            sample = 1
        for num in range(sample):
            samples["state"].append(state)
            samples["action"].append(np.array(expert_a))
            samples["next_state"].append(next_state)
            samples["reward"].append(r)
            samples["terminal"].append(done)
        state = next_state
        steps += 1
        # train after T steps
        if steps > T and done:
            if info["arrive_dest"]:
                success_num += 1
            done_num += 1

            train_start = time.time()
            store_data(samples, "./data")
            X_train, y_train = read_data("./data", "data_dagger.pkl.gzip")
            loss, last_epoch_loss, epoch_num = train_model(agent, X_train, y_train,
                                                           "dagger_models/model_{}.pth".format(model_number + 1),
                                                           num_epochs=num_epoch * (model_number + 1),
                                                           batch_size=batch_size,
                                                           learning_rate=learning_rate,
                                                           early_terminate_loss_threshold=train_loss_threshold,
                                                           device=device)
            eval_res = evaluate(eval_env, agent, evaluation_episode_num=evaluation_episode_num)
            print(eval_res)
            model_number += 1
            break
        if done:
            if info["arrive_dest"]:
                success_num += 1
            done_num += 1
            training_env.reset()
            set_seed(done_num)
            episode_step = 0
    training_env.close()
    eval_env.close()
