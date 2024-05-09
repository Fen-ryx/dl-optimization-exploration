import torch
import torch.nn as nn


class ActorNetwork(nn.Module):
    def __init__(self, obs_state_space, action_state_space, hidden_dim):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.obs_state_space = obs_state_space
        self.action_state_space = action_state_space
        
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=obs_state_space, out_features=self.hidden_dim),
            nn.ReLU()
        )
        
        self.dim_increase_layers = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=2*self.hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=2*self.hidden_dim, out_features=2*self.action_state_space),
        )
        
        self.relu = nn.ReLU()

    def forward(self, obs):
        out = self.dim_increase_layers(self.linear_layers(obs))
        mean, variance = out[:, :self.action_state_space], self.relu(out[:, :self.action_state_space])
        return mean, variance


if __name__ == "__main__":
    actor = ActorNetwork(obs_state_space=4, action_state_space=3, hidden_dim=64)
    obs = torch.rand(size=(1, 4))
    mean, variance = actor.forward(obs)
    import ipdb; ipdb.set_trace()
    print(mean.size(), variance.size())