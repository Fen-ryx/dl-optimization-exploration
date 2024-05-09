import torch

from actor_critic import PolicyValueNetwork
from biquad_env import BiQuadratic_Environment
from torch.distributions.multivariate_normal import MultivariateNormal


MAX_POLICY_OUTPUT = 1.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def computeObjChange(history, new_reward):
    obj_change = torch.sum(history[:, 0]) + new_reward
    return obj_change

def computeNewHistory(history, new_reward, grad):
    obj_change = torch.sum(history[:, 0]) + new_reward
    history = torch.cat((torch.tensor([obj_change, grad]).view(-1, 2), history[1:]), dim=0)
    return history

def sampleTrajectory(env, model, loc, val, grad, var, hist_len=10):
    '''
    A function that generates a trajectory for a particular objective function.
    Inputs: 
        env: The environment
        policy_model: The actor network, a simple feedforward NN
        hist_len: The length of the history which will be input to policy_model
    Outputs:
        rewards: A list of the rewards obtained in each timestep
        metrics: A list of lists that contains the location in the state space, as well as the objective value at each timestep
    '''
    done, i = False, 0
    
    history = torch.zeros((hist_len, 2)) # Input to the NN. Row i: [objective at (current - i - 1)th timestep - objective at current timestep, gradient at (current-i-1)th timestep]
    metrics = [[loc.item(), val.item()]] # Storing locations, objective values and action for logging purposes
    total_history = torch.tensor([0, grad]).view(-1, 2)
    value_history = torch.cat((torch.tensor([0, grad]).view(-1, 2), history), dim=0)
    rewards, actions, values, action_log_probs = [], [], [], []
    
    while (not done):
        i += 1
        logits, value = model.forward(history.view(-1).to(DEVICE)).cpu()
        
        action_distribution = MultivariateNormal(loc=logits, covariance_matrix=var*torch.eye(n=logits.size()[0]))
        action = action_distribution.sample()
        action_log_prob = action_distribution.log_prob(action).item()
        ###########################
        # print(f"Action: {action}")#
        ###########################
        
        loc, val, grad, reward, done = env.step(action)
        
        values.append(value)
        rewards.append(reward)
        actions.append(action)
        action_log_probs.append(action_log_prob)
        metrics.append([loc.item(), val.item()])
        
        obj_change = computeObjChange(total_history, reward)
        history = computeNewHistory(history, reward, grad)
        total_history = torch.cat((torch.tensor([obj_change, grad]).view(-1, 2), total_history), dim=0)
        value_history = torch.cat((torch.tensor([obj_change, grad]).view(-1, 2), value_history), dim=0)
        if (i % hist_len == 0):
            yield torch.tensor(values), torch.tensor(rewards), torch.tensor(actions), torch.tensor(action_log_probs), total_history[1:], value_history[1:], metrics, done
    
    return None
    # return torch.tensor(values), torch.tensor(rewards), torch.tensor(actions), total_history[1:], metrics

def computeGAEs(rewards, values, gamma=1., decay=0.97):
    values_next = torch.cat((values[1:], torch.tensor([values[0]])))
    deltas = torch.tensor([(reward + gamma * val_next - val) for reward, val_next, val in zip(rewards, values_next, values)])
    
    gaes = [deltas[-1]]
    for i in reversed(range(len(deltas) - 1)):
        gaes.append(deltas[i] + gamma * decay * gaes[-1])
    gaes = torch.tensor(gaes)
    return gaes.flip(0)

def discountRewards(rewards, gamma=1.):
    rewards_new = [rewards[-1]]
    for i in reversed(range(len(rewards) - 1)):
        rewards_new.append(rewards[i] + gamma * rewards_new[-1])
    rewards_new = torch.tensor(rewards_new)
    return rewards_new.flip(0)


if __name__ == "__main__":
    env, model, hist_len = BiQuadratic_Environment(max_steps=20), PolicyValueNetwork(input_dim=20), 10
    values, rewards, actions, total_history, metrics = sampleTrajectory(env, torch.tensor(0), torch.tensor(0), torch.tensor(0), model)
    import ipdb; ipdb.set_trace()
    gaes = computeGAEs(rewards, values)
    discounted_rewards = discountRewards(rewards)