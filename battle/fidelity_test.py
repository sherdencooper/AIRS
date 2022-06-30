import torch
import gym
import logging
import sys
import asciichartpy
import argparse
import _env.cyberbattle_env as cyberbattle_env
from agents.baseline.agent_wrapper import Verbosity
import agents.baseline.agent_dql as dqla
import agents.baseline.agent_wrapper as w
import agents.baseline.plotting as p
import agents.baseline.learner as learner
import numpy as np
import pandas as pd
import glob
import random
from __init__ import *
import sys
sys.argv = ['']
parser = argparse.ArgumentParser(description='Run simulation with DQL baseline agent.')

parser.add_argument('--training_episode_count', default=50, type=int,
                    help='number of training epochs')

parser.add_argument('--eval_episode_count', default=10, type=int,
                    help='number of evaluation epochs')

parser.add_argument('--iteration_count', default=69, type=int,
                    help='number of simulation iterations for each epoch')

parser.add_argument('--reward_goal', default=2700, type=int,
                    help='minimum target rewards to reach for the attacker to reach its goal')

parser.add_argument('--ownership_goal', default=1.0, type=float,
                    help='percentage of network nodes to own for the attacker to reach its goal')

parser.add_argument('--rewardplot_with', default=80, type=int,
                    help='width of the reward plot (values are averaged across iterations to fit in the desired width)')

parser.add_argument('--chain_size', default=20, type=int,
                    help='size of the chain of the CyberBattleChain sample environment')

parser.add_argument('--random_agent', dest='run_random_agent', action='store_false', help='run the random agent as a baseline for comparison')
parser.add_argument('--no-random_agent', dest='run_random_agent', action='store_true', help='do not run the random agent as a baseline for comparison')
parser.set_defaults(run_random_agent=False)

args = parser.parse_args()

logging.basicConfig(stream=sys.stdout, level=logging.ERROR, format="%(levelname)s: %(message)s")

print(f"torch cuda available={torch.cuda.is_available()}")

cyberbattlechain = gym.make('CyberBattleChain-v0',
                            size=args.chain_size,
                            attacker_goal=cyberbattle_env.AttackerGoal(
                                own_atleast_percent=args.ownership_goal,
                                reward=args.reward_goal))

ep = w.EnvironmentBounds.of_identifiers(
    maximum_total_credentials=22,
    maximum_node_count=22,
    identifiers=cyberbattlechain.identifiers
)

def process(loc,num,reward_origin,thres=100):
    out = replay(num,loc)
    dif = np.abs(out-reward_origin[0:num])
    sum = 0
    num = 0
    for i in range(dif.shape[0]):
        if 0<=dif[i]<thres:
            sum+=dif[i]
            num+=1
    return sum/num

def read_loc(method):
    summary = 'summary_explain/' + method + '.txt'
    summary_pd = pd.read_table(summary, sep=';', header=None)
    return summary_pd[0].values

def set_seed(seed):
    cyberbattlechain.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def replay(num,loc):
    replay_reward = []
    for i in range(num):
        set_seed(i)
        reward = learner.epsilon_greedy_search(
        cyberbattle_gym_env=cyberbattlechain,
        environment_properties=ep,
        learner=dqla.DeepQLearnerPolicy(
            ep=ep,
            gamma=0.015,
            replay_memory_size=10000,
            target_update=10,
            batch_size=512,
            learning_rate=0.0),  # torch default is 1e-2
        # episode_count=args.training_episode_count,
        episode_count=1,
        iteration_count=args.iteration_count,
        epsilon=0.1,
        render=False,
        # epsilon_multdecay=0.75,  # 0.999,
        epsilon_exponential_decay=5000,  # 10000
        epsilon_minimum=0.1,
        verbosity=Verbosity.Quiet,
        title="DQL",
        seed = i,
        write_log=False,
        fidelity_test=True,
        replay_index=loc[i]
        )
        replay_reward.append(reward)
    return np.array(replay_reward)


num = 100
path = './observation/reward.txt'
df = pd.read_table(path, sep=';', header=None)
reward_origin = df.values.squeeze()


con_loc = read_loc('concept')
att_loc = read_loc('attention')
the_loc = read_loc('theta')
sal_loc = read_loc('saliency')
smo_loc = read_loc('smooth')
int_loc = read_loc('integrated')

set_seed(10)
ran_loc = np.random.randint(36,48,size=[num])

con_diff = process(con_loc,num,reward_origin)
att_diff = process(att_loc,num,reward_origin)
the_diff = process(the_loc,num,reward_origin)
sal_diff = process(sal_loc,num,reward_origin)
smo_diff = process(smo_loc,num,reward_origin)
int_diff = process(int_loc,num,reward_origin)
ran_diff = process(ran_loc,num,reward_origin)


print(con_diff,att_diff,the_diff,sal_diff,smo_diff,int_diff,ran_diff)