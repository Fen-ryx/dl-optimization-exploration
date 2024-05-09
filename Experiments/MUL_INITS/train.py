# Standard imports and configurations
import torch
import torch.nn.functional as F
from torch.optim import Adam
from argparse import ArgumentParser
parser = ArgumentParser()

# Imports for logging
import wandb

from model import SRCNN, init_weights # init_weights allows models to be initialized with different weights
from dataloader import vsr_dataloader # Standard dataloader for my VSR tasks
from torchmetrics.image import PeakSignalNoiseRatio # For computation of validation metrics
psnr_metric = PeakSignalNoiseRatio()

# Seeds and important initializations
torch.manual_seed(42)
c_dev = torch.device('cpu')
g_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class Multiple_Initializations():
    def __init__(self, instances, train_dataloader, val_dataloader, num_epochs, num_elim_steps, reduce_factor):
        self.model = None
        self.optimizer = None
        self.instances = instances
        self.val_dataloader = val_dataloader
        self.train_dataloader = train_dataloader
        
        self.num_epochs = num_epochs
        self.reduce_factor = reduce_factor
        self.num_elim_steps = num_elim_steps
    
    # Function that implements a standard training loop
    def train_steps(self, mode):
        if (mode == 'elim'):
            steps = self.num_elim_steps
        else:
            steps = len(self.train_dataloader)
        
        for _ in range(steps):
            batch = next(iter(self.train_dataloader))
            x, y = batch['image'].to(g_dev), batch['target'].to(g_dev)
            output = self.model(x)
            loss = F.mse_loss(output, y)
            loss.backward()
            self.optimizer.zero_grad()

    # Function that implements a standard validation loop
    def validate_steps(self):
        loss = torch.Tensor(0).to(g_dev)
        for _ in range(len(self.val_dataloader)):
            batch = next(iter(self.val_dataloader))
            x, y = batch['image'].to(g_dev), batch['target'].to(g_dev)
            output = self.model(x)
            loss += F.mse_loss(output, y)
        loss.to(c_dev)
        return loss

    # The `train` function: This function trains every model for a set few number of steps given by `num_elim_steps`. Once every model has been trained so, we can then eliminate inferior initializations of this model by simply comparing the loss values. The decrease in the number of models is geometric in nature, so we move into the regular training paradigm very quickly.
    def train(self):
        num_steps = 0 # In this formulation, num_steps helps keep track of how many epochs one has trained
        
        while (num_steps // len(self.train_dataloader) < self.num_epochs):
            
            print(f"Number of models is {len(self.instances)}")
            for i, instance in enumerate(self.instances):
                self.model, _, self.optimizer = instance
                self.model.to(g_dev)
                # self.train_steps(mode='elim' if len(instances) > 1 else 'std') # Implementing elements of a standard training loop
                # train_steps() function
                if (len(self.instances) > 1):
                    steps = self.num_elim_steps
                else:
                    steps = len(self.train_dataloader)
                
                for _ in range(steps):
                    self.optimizer.zero_grad()
                    batch = next(iter(self.train_dataloader))
                    x, y = batch['image'].to(g_dev), batch['target'].to(g_dev)
                    output = self.model(x)
                    loss = F.mse_loss(output, y)
                    loss.backward()
                    self.optimizer.step()
                # train_steps() function
                
                print(f"Training loop for model {i+1} completed")
                # loss = self.validate_steps() # Implementing elements of a standard validation loop
                # validate_steps() function
                loss = 0.
                self.model.eval()
                with torch.no_grad():
                    for k in range(len(self.val_dataloader)):
                        batch = next(iter(self.val_dataloader))
                        x, y = batch['image'].to(g_dev), batch['target'].to(g_dev)
                        output = self.model(x)
                        loss += F.mse_loss(output, y).item()
                # validate_steps() function
                
                print(f"Validation loop for model {i+1} completed")
                self.model.to(c_dev) # Remove model from GPU
                self.instances[i] = [self.model, loss / (k + 1), self.optimizer] # Update instance parameters
                # torch.cuda.empty_cache()
            
            print(f"Since {i} is {len(self.instances)}, this means all models have been trained for {self.num_elim_steps} steps")
            self.instances.sort(key = lambda x: x[1])
            if (len(self.instances) > 1):
                num_steps += self.num_elim_steps
                size = (int)(len(self.instances) * self.reduce_factor)
                self.instances = self.instances[:size]
            else:
                num_steps += len(self.train_dataloader)
            
            print(f"New number of models after elimination is {len(self.instances)}")


if __name__ == "__main__":
    # User inputs
    parser.add_argument("--run-name", default="debug", type=str)
    parser.add_argument("--num-models", default=10, type=int)
    parser.add_argument("--num-epochs", default=10, type=int)
    parser.add_argument("--reduce-factor", default=0.1, type=float)
    parser.add_argument("--num-elim-steps", default=10, type=int)
    args = parser.parse_args()
    
    
    # wandb.init(
    #     project='Optim_Alg',
    #     name=args.run_name
    # )
    
    # Initializing all models
    instances = list()
    for i in range(args.num_models):
        model = SRCNN()
        model.apply(init_weights)
        optimizer = Adam(model.parameters(), lr=1e-3)
        instances.append([model, 0, optimizer])
    print(f"{len(instances)} models initialized for training")
    
    # Initializing dataloaders
    val_dataloader = vsr_dataloader(mode='val')
    test_dataloader = vsr_dataloader(mode='test')
    train_dataloader = vsr_dataloader(mode='train')
    print("Training dataloader created")
    
    print("Initializing Multiple Initalizations Object")
    multiple_inits = Multiple_Initializations(
        instances=instances, 
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader, 
        num_epochs=args.num_epochs, 
        num_elim_steps=args.num_elim_steps, 
        reduce_factor=args.reduce_factor
        )
    print("Calling train function")
    multiple_inits.train()