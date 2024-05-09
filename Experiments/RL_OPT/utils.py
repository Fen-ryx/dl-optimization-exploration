# I define rewards as (old_obj_value - new_obj_value) and values as (-current_obj_value)
import sys
import torch
import numpy as np
import scipy.stats
import scipy.linalg

from quadopt_env import DIM
from torch.distributions.multivariate_normal import MultivariateNormal

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_Qmat_cvec():
    eig_vecs = torch.tensor(
        scipy.stats.ortho_group.rvs(dim=DIM), dtype=torch.float
    )
    eig_vals = torch.rand(DIM) * 29 + 1
    
    Q = eig_vecs @ torch.diag(eig_vals) @ eig_vecs.T
    c = torch.normal(0, 1 / np.sqrt(DIM), size=(DIM,))
    
    optimal_x = torch.tensor(scipy.linalg.solve(Q.numpy(), c.numpy(), assume_a="pos"))
    optimal_val = 0.5 * optimal_x.T @ Q @ optimal_x + c.T @ optimal_x
    
    return {
        "Q": Q,
        "c": c,
        "optimal_x": optimal_x,
        "optimal_val": optimal_val
    }

def compute_rewards(rewards, gamma=1.):
    new_rewards = [float(rewards[-1])]
    for i in reversed(range(len(rewards)-1)):
        new_rewards.append(float(rewards[i]) + gamma * new_rewards[-1])
    return np.array(new_rewards[::-1])

def calculate_gaes(rewards, values, gamma=1., decay=0.97):
    next_values = np.concatenate([values[1:], [0]])
    deltas = [rew + gamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]

    gaes = [deltas[-1]]
    for i in reversed(range(len(deltas)-1)):
        gaes.append(deltas[i] + decay * gamma * gaes[-1])

    return np.array(gaes[::-1])

def create_batched_covariance_matrix(variance):
    matrices = []
    for v in variance:
        matrices.append(torch.diag(v))
    return torch.stack(matrices, dim=0)

def rollout(model, env, max_steps=100, eps=1e-6):
    obs = env.reset()
    ep_reward = 0.
    train_data = [[], [], [], [], []] # obs, actions, rewards, values, action_log_probs
    
    for _ in range(max_steps):
        obs_vector = torch.cat(
            (torch.tensor([obs['objective']], device=DEVICE), obs['current_value']),
            dim=0).clamp(-100., 100.)
        
        mean, variance = model.forward(torch.tensor(obs_vector.view(size=(1, -1)), device=DEVICE))
        mean, variance = mean.squeeze(), variance.squeeze()
        gaussian_sampler = MultivariateNormal(
            loc=mean, 
            covariance_matrix=torch.diag(variance) + eps * torch.eye(
                mean.size()[0], 
                dtype=torch.float32, 
                device=DEVICE
            )
        )
        
        action = gaussian_sampler.sample().clamp(-10., 10.)
        action_log_prob = gaussian_sampler.log_prob(action).item()

        value = -obs["objective"].clamp(-100, 100).item()
        new_obs, reward, terminated = env.step(action)
        
        for i, item in enumerate((obs_vector.cpu(), action.cpu(), reward.cpu(), value, action_log_prob)):
            train_data[i].append(item)
        
        obs = new_obs
        ep_reward += reward
        if terminated:
            break
    
    train_data = [np.asarray(d) for d in train_data]
    train_data[3] = calculate_gaes(train_data[2], train_data[3])
    return train_data, ep_reward  


if __name__ == "__main__":
    variance = torch.randn((100, 3))
    output = create_batched_covariance_matrix(variance)
    import ipdb; ipdb.set_trace()