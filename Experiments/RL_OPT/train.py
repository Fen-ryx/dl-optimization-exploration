import torch
import warnings
import numpy as np
import scipy.linalg
import gymnasium as gym

from utils import (
    rollout,
    calculate_gaes,
    compute_rewards,
    create_Qmat_cvec
)
from ppo import PPOTrainer
from actor import ActorNetwork
from quadopt_env import QuadraticEnv, DIM

torch.manual_seed(42)
warnings.filterwarnings("ignore")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_fn(
    env,
    model,
    n_episodes,
    print_freq
):
    episodic_rewards = []
    
    for episode_num in range(n_episodes):
        train_data, reward = rollout(model, env)
        episodic_rewards.append(reward)
        
        permute_idxs = np.random.permutation(len(train_data[0]))
        obs = torch.tensor(train_data[0][permute_idxs], dtype=torch.float32, device=DEVICE)
        acts = torch.tensor(train_data[1][permute_idxs], dtype=torch.float32, device=DEVICE)
        gaes = torch.tensor(train_data[3][permute_idxs], dtype=torch.float32, device=DEVICE)
        act_log_probs = torch.tensor(train_data[4][permute_idxs], dtype=torch.float32, device=DEVICE)
        
        returns = compute_rewards(train_data[2])[permute_idxs]
        returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
        
        ppo.train_policy(obs, acts, gaes, act_log_probs)
        if ((episode_num+1) % print_freq == 0):
            print('Episode {} | Avg Reward {:.1f}'.format(
                episode_num + 1, torch.mean(torch.tensor(episodic_rewards[-print_freq:])).item()))
            print(f"Final Episode Reward: {episodic_rewards[-1]}")
            print(f"Final Optimum: {env._agent_location}")


if __name__ == "__main__":
    info = create_Qmat_cvec()
    env = QuadraticEnv(Q_mat=info["Q"], c_mat=info["c"])
    model = ActorNetwork(obs_state_space=DIM+1, action_state_space=DIM, hidden_dim=64).to(DEVICE)
    train_data, reward = rollout(model, env)
    print("Enviroment, Model created. Rollout Executed.")
    
    n_episodes, print_freq = 90, 10
    ppo = PPOTrainer(
        actor=model,
        policy_lr=1e-3,
        clip_factor=0.2,
        policy_kldiv_bound=0.01,
        max_policy_train_steps=50
    )
    print("Proximal Policy Optimization Object created.")
    
    train_fn(env, model, n_episodes, print_freq)
    solution = scipy.linalg.solve(info["Q"], -info["c"], assume_a="pos")
    print(solution)