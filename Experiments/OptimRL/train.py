import torch

from argparse import ArgumentParser
from actor_critic import PolicyValueNetwork
from biquad_env import BiQuadratic_Environment
from utils import (
    sampleTrajectory,
    computeGAEs,
    computeNewHistory,
    discountRewards,
    MAX_POLICY_OUTPUT
)
from torch.distributions.multivariate_normal import MultivariateNormal


torch.manual_seed(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PPOTrainer():
    def __init__(
        self,
        env,
        var,
        model,
        hist_len=10,
        batch_size=16,
        value_lr=5e-4,
        policy_lr=5e-4,
        num_train_fns=20,
        ppo_clip_val=0.2,
        num_test_fns=10,
        target_kl_div=0.01,
        value_train_iters=10,
        max_policy_train_iters=20
    ):
        self.env = env
        self.var = var
        self.hist_len = hist_len
        self.batch_size = batch_size
        self.model = model.to(DEVICE)
        self.ppo_clip_val = ppo_clip_val
        self.num_test_fns = num_test_fns
        self.target_kl_div = target_kl_div
        self.num_train_fns = num_train_fns
        self.value_train_iters = value_train_iters
        self.max_policy_train_iters = max_policy_train_iters
        
        policy_params = list(self.model.shared_layers.parameters()) + \
            list(self.model.policy_layers.parameters())
        self.policy_optim = torch.optim.Adam(params=policy_params, lr=policy_lr)
        
        value_params = list(self.model.shared_layers.parameters()) + \
            list(self.model.policy_layers.parameters())
        self.value_optim = torch.optim.Adam(params=value_params, lr=value_lr)
    
    def metaTrainer(self):
        for _ in range(self.num_train_fns):
            done = False
            loc, val, grad = self.env.reset()
            print("Environment reset. Training on new instance begins...")
            
            sampler = sampleTrajectory(self.env, self.model, loc, val, grad, self.var, self.hist_len)
            print("Sampler created to generate data.")
            
            while (not done):
                values, rewards, actions, action_log_probs, history, value_hist, metrics, done = next(sampler)
                
                gaes = computeGAEs(rewards, values)
                returns = discountRewards(rewards)

                self.trainStepPolicy(history, actions, action_log_probs, gaes)
                self.trainStepValue(value_hist, returns)
                import ipdb; ipdb.set_trace()
        
    def metaTester(self):
        print("Commencing Testing...")
        for _ in range(50000000):
            pass
        
        for _ in range(self.num_test_fns):
            i, done = 0, False
            loc, val, grad = self.env.reset()
            print("Environment reset. Testing on new instance...")
            
            metrics = [[loc.item(), val.item()]]
            history = torch.cat((torch.tensor([0., grad]).view(-1, 2), torch.zeros(size=(self.hist_len-1, 2))))
            
            while (not done):
                i += 1
                logits = self.model.policy(history.view(-1).to(DEVICE)).cpu()
                
                action_distribution = MultivariateNormal(loc=logits, covariance_matrix=self.var*torch.eye(n=logits.size()[0]))
                action = action_distribution.sample()
                loc, val, grad, reward, done = env.step(action)
                metrics.append([loc.item(), val.item()])
                
                history = computeNewHistory(history, reward, grad)
                done = (i == 40) or (torch.abs(loc - env.b / env.a) <= 1e-6)
            
            if (torch.abs(loc - env.b / env.a) <= 1e-6):
                print("Optimum Found!")
                print(f"Actual Solution: {(env.b / env.a).item()}")
                print(f"Model Solution: {loc.item()}")
    
    def trainStepPolicy(self, history, actions, old_action_log_probs, gaes):
        for _ in range(self.max_policy_train_iters):
            self.policy_optim.zero_grad()
            
            new_logits = self.model.policy(history[:10].view(-1).to(DEVICE))
            new_logits = new_logits.clamp(-MAX_POLICY_OUTPUT, MAX_POLICY_OUTPUT).cpu()
            new_logits = MultivariateNormal(loc=new_logits, covariance_matrix=self.var*torch.eye(new_logits.size()[0]))
            
            # new_action_log_probs = new_logits.log_prob(torch.tensor([actions[-1]]))
            new_action_log_probs = []
            for i in range(self.hist_len):
                new_action_log_probs.append(new_logits.log_prob(torch.tensor([actions[-self.hist_len+i]])))
            new_action_log_probs = torch.stack(new_action_log_probs)
            
            policy_ratio = torch.exp(new_action_log_probs - old_action_log_probs[-self.hist_len:])
            clipped_ratio = policy_ratio.clamp(1-self.ppo_clip_val, 1+self.ppo_clip_val)
            
            full_loss = policy_ratio * gaes[-self.hist_len:]
            clipped_loss = clipped_ratio * gaes[-self.hist_len:]
            policy_loss = -torch.min(full_loss, clipped_loss).mean()
            
            #####################################
            print(f"Policy Loss: {policy_loss}")#
            #####################################
            
            policy_loss.backward()
            self.policy_optim.step()
            
            kl_div_estimate = (old_action_log_probs[-self.hist_len:] - new_action_log_probs).mean()
            if (kl_div_estimate >= self.target_kl_div):
                print("KL-Divergence Exceeded, breaking...")
                break
    
    def trainStepValue(self, history, returns):
        for _ in range(self.value_train_iters):
            self.value_optim.zero_grad()
            
            values = []
            for i in range(len(history) - 10):
                value = self.model.value(history[i:i+10].view(-1).to(DEVICE)).cpu()
                values.append(value)
            
            values = torch.stack(values)
            value_loss = ((values - returns) ** 2).mean()
            
            value_loss.backward()
            self.value_optim.step()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--variance", type=float, default=0.1)
    parser.add_argument("--num-train-fns", type=int, default=20)
    parser.add_argument("--max-env-steps", type=int, default=40)
    parser.add_argument("--history-length", type=int, default=10)
    parser.add_argument("--param-space-dim", type=int, default=1)
    args = parser.parse_args()
    
    env = BiQuadratic_Environment(
        max_steps=args.max_env_steps,
        param_space=args.param_space_dim
    )
    print("Environment Instantiated")
    
    model = PolicyValueNetwork(
        input_dim=2*args.history_length,
        param_space=args.param_space_dim,
        batch_size=args.batch_size,
    )
    print("Model Instantiated")
    
    ppotrainer = PPOTrainer(
        env,
        args.variance,
        model,
        hist_len=args.history_length,
        num_train_fns=args.num_train_fns
    )
    print("Trainer Object Instantiated.\nTraining begins...")
    ppotrainer.metaTrainer()
    
    print("Training Completed.\nTesting begins...")
    ppotrainer.metaTester()