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
sys.path.append('C:/Users/yujit/github/cleanrl')
root_dir = 'C:/Users/yujit/github/cleanrl'

from cleanrl.ppo_continuous_action import * 
from cleanrl_utils.evals.ppo_eval import *
from plot_misc import * 


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    
    # your pretrained model path 
    run_name = "RPO_Detumble2DEnv-v0__ppo_continuous_action__1__1717398428"
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

    eval_episodes = 5
    episodic_returns, obs_history, actions, tf, utot = evaluate_dummy(
        model_path,
        make_env,
        args.env_id,
        eval_episodes=eval_episodes,
        run_name=f"{run_name}-eval",
        Model=Agent,
        device=device,
        gamma=args.gamma,
        use_NN=False,
        seed_vec=np.arange(0, eval_episodes)
    )
    
    print(f"tf: {tf}")
    print(f"utot: {np.round(utot,2)}")
    
    fig = plt.figure(figsize=(10, 6))
    color_list = get_color_list('viridis', eval_episodes)
    
    tf_max = int(np.max(tf)) 
    
    # moment of inertia (RSO, body frame)
    J = np.array([[562.07457,   0.     ,   0.0],
              [  0.     , 562.07465,   0.     ],
              [  0.0,   0.     , 192.29662]])
    
    for i in range(eval_episodes):
        # fig = plot_sol_qw2(fig, obs_history[i, :tf_max+2, :7].T, actions[i, :tf_max+1], range(np.shape(obs_history[i, :tf_max+2])[0]), None, c=color_list[i])
        tf_i = int(tf[i])
        qw = obs_history[i, :tf_i, :7].T
        a  = actions[i, :tf_i-1]
        t  = range(np.shape(obs_history[i, :tf_i])[0])
        fig = plot_sol_qw(fig, qw, a, t, J, c=color_list[i])
    
    # print(episodic_returns)
    print("done")
    plt.tight_layout()
    plt.show()
