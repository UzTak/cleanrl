# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

# import gymnasium as gym
import gym
import gym_examples
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

import sys
# sys.path.append('/home/cyrus/aa203_cleanrl')
# root_dir = '/home/cyrus/aa203_cleanrl'
sys.path.append('C:/Users/yujit/github/cleanrl')
root_dir = 'C:/Users/yujit/github/cleanrl'

from plot_misc import plot_sol_qw, plot_input
import matplotlib.pyplot as plt

from ppo_continuous_action import * 
from ppo_eval import *


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    
    # your pretrained model path 
    run_name = "RPO_Detumble3DEnv-v0"
    model_path = root_dir + f"/runs/gym_examples/{run_name}/ppo_continuous_action.cleanrl_model"

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    agent = Agent(envs).to(device)
    
    print(f"loading a trained model from {model_path}...")

    episodic_returns, all_obs, all_actions = evaluate_dummy(
        model_path,
        make_env,
        args.env_id,
        eval_episodes=10,
        run_name=f"{run_name}-eval",
        Model=Agent,
        device=device,
        gamma=args.gamma,
    )
    
    print(episodic_returns)

    J_targ = np.array([[562.07457,   0.     ,   0.0],
              [  0.     , 562.07465,   0.     ],
              [  0.0,   0.     , 192.29662]])
    for i in range(1):#range(len(all_obs)):
        plot_sol_qw(all_obs[i][:,0:7].T, range(len(all_obs[i][:,0:7])), J_targ)

        # plot_input(all_actions[i].T,range(len(all_actions[i])))
    plt.show()
    print("done")
