import torch

class BiQuadratic_Environment():
    def __init__(self, max_steps, param_space=1, tol=1e-6):
        self.tolerance = tol
        self.max_steps = max_steps
        self.param_space = param_space
    
    def reset(self):
        self.num_steps = 0
        self.agent = torch.rand((self.param_space,), requires_grad=True)
        self.a, self.b = torch.rand((self.param_space,)), torch.rand((1,))
        
        self.objective_value, gradient = self.computeObjectiveAndGradient()
        return self.agent, self.objective_value, gradient
    
    def step(self, update):
        self.num_steps += 1
        self.agent.grad.zero_()
        done = (self.num_steps >= self.max_steps) or (torch.abs(self.agent - self.b / self.a) <= self.tolerance)
        
        with torch.no_grad():
            self.agent -= update
        objective, gradient = self.computeObjectiveAndGradient()
        reward, self.objective_value = self.objective_value - objective, objective
        
        return self.agent, self.objective_value, gradient, reward, done
    
    def computeObjectiveAndGradient(self):
        objective = torch.square(self.a * self.agent - self.b)
        objective.backward()
        return objective, self.agent.grad


if __name__ == "__main__":
    env = BiQuadratic_Environment(max_steps=40)
    loc, val, grad = env.reset()
    print(loc, val, grad)