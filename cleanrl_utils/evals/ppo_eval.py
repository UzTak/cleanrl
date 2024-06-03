from typing import Callable

# import gymnasium as gym
import gym 
import torch
import numpy as np 


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
    gamma: float = 0.99,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, capture_video, run_name, gamma)])
    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        obs, _ = envs.reset()
        r = 0 
        while True:
            actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
            next_obs, reward, terminated, _, infos = envs.step(actions.cpu().numpy())
            # if "final_info" in infos:
            #     for info in infos["final_info"]:
            #         if "episode" not in info:
            #             continue
            #         print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
            #         episodic_returns += [info["episode"]["r"]]
            r += reward.item() 
            obs = next_obs
            if terminated:
                break
        episodic_returns.append(r)

    return episodic_returns



def evaluate_dummy(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
    gamma: float = 0.99,
):
    
    """
    Return: 
       episodic_returns: (eval_episodes, )
       obs_history: (eval_episodes, 20, 10)
    """
    
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, capture_video, run_name, gamma)])
    
    episodic_returns = []
    
    obs_history = np.zeros((eval_episodes, envs.envs[0].env.num_burn_max, 10))   
    
    for i in range(eval_episodes):
        
        obs, _ = envs.reset()
        obs_history[i, 0] = obs
        
        r = 0 
        j = 0    # index 
        for j in range(envs.envs[0].env.num_burn_max):
            actions, = torch.tensor([[1.0]])
            next_obs, reward, terminated, _, infos = envs.step(actions.cpu().numpy())
            # if "final_info" in infos:
            #     for info in infos["final_info"]:
            #         if "episode" not in info:
            #             continue
            #         print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
            #         episodic_returns += [info["episode"]["r"]]
            r += reward.item() 
            obs = next_obs
            obs_history[i, j] = obs
            if terminated:
                break
            
            # j += 1
        
        episodic_returns.append(r)

    return episodic_returns, obs_history



if __name__ == "__main__":
    from huggingface_hub import hf_hub_download

    from cleanrl.ppo_continuous_action import Agent, make_env

    model_path = hf_hub_download(
        repo_id="sdpkjc/Hopper-v4-ppo_continuous_action-seed1", filename="ppo_continuous_action.cleanrl_model"
    )
    evaluate(
        model_path,
        make_env,
        "Hopper-v4",
        eval_episodes=10,
        run_name=f"eval",
        Model=Agent,
        device="cpu",
        capture_video=False,
    )
