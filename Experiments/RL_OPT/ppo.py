import torch

from torch import optim
from quadopt_env import DIM
from utils import create_batched_covariance_matrix
from torch.distributions.multivariate_normal import MultivariateNormal

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PPOTrainer():
    def __init__(self,
                 actor,
                 policy_lr,
                 clip_factor,
                 policy_kldiv_bound,
                 max_policy_train_steps,
                 ):
        self.ac = actor
        self.lr = policy_lr
        self.clip = clip_factor
        self.kldiv_bound = policy_kldiv_bound
        self.max_policy_train_steps = max_policy_train_steps
        
        self.optimizer = optim.Adam(self.ac.parameters(), lr=self.lr)
    
    def train_policy(self, obs, acts, gaes, old_log_probs, eps=1e-6):
        for _ in range(self.max_policy_train_steps):
            self.optimizer.zero_grad()
            
            mean, variance = self.ac.forward(obs)
            mean, variance = mean.squeeze(), variance.squeeze()
            variance_matrix = create_batched_covariance_matrix(variance)
            
            gaussian_sampler = MultivariateNormal(
                loc=mean, 
                covariance_matrix=variance_matrix + eps * torch.eye(
                    mean.size()[1], 
                    dtype=torch.float32, 
                    device=DEVICE
                )
            )
            
            new_log_probs = gaussian_sampler.log_prob(acts)
            
            full_ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = full_ratio.clamp(1 - self.clip, 1 + self.clip)
            
            policy_loss = - torch.min(full_ratio * gaes, clipped_ratio * gaes).mean()
            policy_loss.backward()
            self.optimizer.step()
            
            kl_div = (old_log_probs - new_log_probs).mean()
            if (kl_div >= self.kldiv_bound):
                break