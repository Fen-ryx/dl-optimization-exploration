import torch
import numpy as np
import gymnasium as gym

from gym import spaces

DIM = 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class QuadraticEnv(gym.Env):
    def __init__(self, Q_mat, c_mat):
        self.Q = Q_mat.to(device=DEVICE)
        self.c = c_mat.to(device=DEVICE)
        
        self.threshold = 0.1
        self.render_mode = None
        self._old_objective = None
        self.num_steps_below_threshold = 0
        self.max_num_steps_below_threshold = 3
        
        self.action_space = spaces.Box(low=-10., high=10., shape=(DIM+1,), dtype=float)
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=-100., high=100., shape=(DIM+1,), dtype=float)
            }
        )
    
    def _get_obs(self):
        return {
            "objective": 0.5 * self._agent_location.T @ self.Q @ self._agent_location + self.c.T @ self._agent_location,
            "current_value": self._agent_location
        }
    
    def _termination(self, obj):
        if (abs(self._old_objective - obj) <= self.threshold):
            self.num_steps_below_threshold += 1
        else:
            self.num_steps_below_threshold = 0
        
        self._old_objective = obj
        if (self.num_steps_below_threshold == self.max_num_steps_below_threshold):
            return True
        else:
            return False
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        mean = np.zeros(DIM)
        cov_mat = np.eye(N=DIM)
        self._agent_location = torch.tensor(
            self.np_random.multivariate_normal(mean=mean, cov=cov_mat), 
            dtype=torch.float32,
            device=DEVICE
        )
        
        obs = self._get_obs()
        self._old_objective = obs["objective"]
        return obs
    
    def step(self, action):
        self._agent_location = self._agent_location + action
        obs = self._get_obs()
        reward = self._old_objective - obs["objective"]
        terminated = self._termination(obs["objective"])
        return obs, reward, terminated


if __name__ == "__main__":
    from utils import create_Qmat_cvec
    info = create_Qmat_cvec()
    
    env = QuadraticEnv(info["Q"], info["c"])
    obs = env.reset()
    print(obs)
    import ipdb; ipdb.set_trace()