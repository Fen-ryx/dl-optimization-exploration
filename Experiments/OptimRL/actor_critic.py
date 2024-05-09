import torch
from torch import nn


class PolicyValueNetwork(nn.Module):
    def __init__(self, input_dim, param_space=1, batch_size=1, hidden_dim=64):
        super().__init__()
        
        self.shared_layers = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        )
        self.policy_layers = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=param_space * batch_size)
        )
        self.value_layers = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=1)
        )
    
    def value(self, x):
        return self.value_layers(self.shared_layers(x))
    
    def policy(self, x):
        return self.policy_layers(self.shared_layers(x))
    
    def forward(self, x):
        latent = self.shared_layers(x)
        policy, value = self.policy_layers(latent), self.value_layers(latent)
        return policy, value


if __name__ == "__main__":
    network = PolicyValueNetwork(input_dim=20)
    policy, value = network.forward(torch.randn(size=(20,)))
    print(policy, value)