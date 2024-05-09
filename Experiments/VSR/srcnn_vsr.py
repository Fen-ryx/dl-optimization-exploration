from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import wandb
from argparse import ArgumentParser
parser = ArgumentParser()

import lightning.pytorch as pl
from pytorch_lightning.loggers import WandbLogger

import torch
from torch import optim, nn
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import random_split, DataLoader
from dataloader import vsr_dataloader

class SRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=2, padding_mode='replicate'),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=2, padding_mode='replicate'),
            nn.ReLU()
        )
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, padding=2, padding_mode='replicate')
    
    def forward(self, x):
        return self.conv3(self.conv2(self.conv1(x)))

class LightningModule(pl.LightningModule):
    def __init__(self, model, optim_choice):
        super().__init__()
        self.model = model
        self.optimizer = optim_choice
        self.psnr_metric = PeakSignalNoiseRatio()
        self.save_hyperparameters(ignore=['model'])
    
    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['target']
        output = self.model(x)
        loss = F.mse_loss(output, y)
        if batch_idx == 0:
            x_log = [wandb.Image(x[i].clone().detach().view(-1, 480, 320, 3).cpu().numpy()) for i in range(4)]
            y_log = [wandb.Image(y[i].clone().detach().view(-1, 480, 320, 3).cpu().numpy()) for i in range(4)]
            op_log = [wandb.Image(output[i].clone().detach().view(-1, 480, 320, 3).cpu().numpy()) for i in range(4)]
            self.logger.log_table(
                key='train',
                columns=['inp_img', 'pred_img', 'target_img'],
                data=[[x_log[i], op_log[i], y_log[i]] for i in range(4)]
            )
        
        self.log('train_loss', loss)
        self.log('train_psnr', self.psnr_metric(output, y))
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['target']
        output = self.model(x)
        loss = F.mse_loss(output, y)
        if batch_idx == 0:
            x_log = [wandb.Image(x[i].clone().detach().view(-1, 480, 320, 3).cpu().numpy()) for i in range(4)]
            y_log = [wandb.Image(y[i].clone().detach().view(-1, 480, 320, 3).cpu().numpy()) for i in range(4)]
            op_log = [wandb.Image(output[i].clone().detach().view(-1, 480, 320, 3).cpu().numpy()) for i in range(4)]
            self.logger.log_table(
                key='train',
                columns=['inp_img', 'pred_img', 'target_img'],
                data=[[x_log[i], op_log[i], y_log[i]] for i in range(4)]
            )
        self.log('val_loss', loss)
        self.log('val_psnr', self.psnr_metric(output, y))
    
    def test_step(self, batch, batch_idx):
        x, y = batch['image'], batch['target']
        output = self.model(x)
        if batch_idx == 0 or batch_idx == 1:
            x_log = [wandb.Image(x[i].clone().detach().view(-1, 480, 320, 3).cpu().numpy()) for i in range(4)]
            y_log = [wandb.Image(y[i].clone().detach().view(-1, 480, 320, 3).cpu().numpy()) for i in range(4)]
            op_log = [wandb.Image(output[i].clone().detach().view(-1, 480, 320, 3).cpu().numpy()) for i in range(4)]
            self.logger.log_table(
                key='train',
                columns=['inp_img', 'pred_img', 'target_img'],
                data=[[x_log[i], op_log[i], y_log[i]] for i in range(4)]
            )
        self.log('test_psnr', self.psnr_metric(output, y))
    
    def configure_optimizers(self):
        if self.optimizer == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=1e-3)
        elif self.optimizer == 'SGD':
            optimizer = optim.SGD(self.parameters(), lr=1e-3)
        elif self.optimizer == 'AdaGrad':
            optimizer = optim.Adagrad(self.parameters(), lr=1e-3)
        elif self.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(self.parameters(), lr=1e-3)
        elif self.optimizer == 'AdamW':
            optimizer = optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.01)
        elif self.optimizer == 'SGD-Momentum':
            optimizer = optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)
        else:
            optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    pl.seed_everything(42)
    torch.set_float32_matmul_precision('high')
    parser.add_argument("--opt", default="Adam", type=str)
    parser.add_argument("--run-name", default="debug", type=str)
    args = parser.parse_args()
    
    model = LightningModule(SRCNN(), args.opt)
    
    train_dl = vsr_dataloader('train')
    val_dl, test_dl = vsr_dataloader('val'), vsr_dataloader('test')
    early_stop_callback = pl.callbacks.EarlyStopping('val_loss', patience=3)
    
    wandb.init(
        project='Optim_Alg',
        name=args.run_name
        )
    wandb_logger = WandbLogger()
    
    trainer = pl.Trainer(
        callbacks=[early_stop_callback],
        logger=wandb_logger,
        max_epochs=100,
        log_every_n_steps=2
        )
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    trainer.test(model=model, dataloaders=test_dl)
    wandb.finish()