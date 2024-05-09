import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

# Define the neural network architecture
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # return F.log_softmax(x, dim=1)
        return F.log_softmax(x, dim=0)

# PPO Agent
class PPOAgent:
    def __init__(self, input_size, hidden_size, output_size, lr, clip_param, value_coeff, entropy_coeff):
        self.policy_network = PolicyNetwork(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.clip_param = clip_param
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff

    def select_action(self, state):
        state = torch.FloatTensor(state)
        probs = self.policy_network(state)
        action = torch.multinomial(probs.exp(), 1)
        return action.item(), probs[:, action].item()

    def update_policy(self, states, actions, rewards, old_probs, advantages):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        old_probs = torch.FloatTensor(old_probs)
        advantages = torch.FloatTensor(advantages)

        # Calculate advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(5):  # Number of optimization epochs
            # Calculate new probabilities
            new_probs = self.policy_network(states).gather(1, actions.unsqueeze(1))

            # Clipped surrogate objective
            ratio = new_probs.exp() / old_probs.exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Total loss
            loss = policy_loss

            # Update policy network
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# Function to compute advantages
def compute_advantages(rewards, gamma=0.99, tau=0.95):
    advantages = np.zeros_like(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] - rewards[t - 1] if t > 0 else rewards[t]  # Assume rewards[t - 1] is the baseline
        gae = delta + gamma * tau * gae
        advantages[t] = gae
    return advantages

# Main training loop
def train():
    # Hyperparameters
    input_size = 28 * 28
    hidden_size = 128
    output_size = 10
    lr = 0.001
    clip_param = 0.2
    num_epochs = 10
    batch_size = 64
    value_coeff = 0.5
    entropy_coeff = 0.01

    # Initialize environment and agent
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)
    agent = PPOAgent(input_size, hidden_size, output_size, lr, clip_param, value_coeff, entropy_coeff)

    # Training loop
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Flatten images
            images = images.view(images.size(0), -1)

            # Select action and get log probability
            actions, old_probs = [], []
            for image in images:
                action, prob = agent.select_action(image)
                actions.append(action)
                old_probs.append(prob)

            # Perform action and get rewards
            rewards = [1 if pred == label else -1 for pred, label in zip(actions, labels)]

            # Update policy
            advantages = compute_advantages(rewards)
            agent.update_policy(images, actions, rewards, old_probs, advantages)

            # Print training information
            # if i % 100 == 0:
            #     print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')

if __name__ == "__main__":
    train()