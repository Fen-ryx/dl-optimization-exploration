import wandb
from argparse import ArgumentParser
parser = ArgumentParser()

import lightning.pytorch as pl
from pytorch_lightning.loggers import WandbLogger

import torch
from torch import optim, nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, ToPILImage
from torch.utils.data import random_split, DataLoader


class FeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
            )
    
    def forward(self, x):
        return self.model(x)

class LightningModule(pl.LightningModule):
    def __init__(self, model, optim_choice):
        super().__init__()
        self.model = model
        self.optimizer = optim_choice
        self.acc_metric = Accuracy(task='multiclass', num_classes=10)
        self.save_hyperparameters(ignore=['model'])
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        if (batch_idx % 1000) == 0:
            img = [wandb.Image(x[i].cpu().numpy()) for i in range(4)]
            labels = [y[i].item() for i in range(4)]
        x = x.view(x.size(0), -1)
        target = F.one_hot(y, num_classes=10).float()
        
        output = self.model(x)
        loss = nn.CrossEntropyLoss()(output, target)
        if (batch_idx % 1000) == 0:
            pred = [torch.argmax(output[i]).item() for i in range(4)]
            self.logger.log_table(key='train', columns=['image', 'label', 'pred'], data=[[img[i], labels[i], pred[i]] for i in range(4)])
        self.log('train_loss', loss, prog_bar=True, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        if (batch_idx % 100) == 0:
            img = [wandb.Image(x[i].cpu().numpy()) for i in range(4)]
            labels = [y[i].item() for i in range(4)]
        x = x.view(x.size(0), -1)
        target = F.one_hot(y, num_classes=10).float()
        
        output = self.model(x)
        loss = nn.CrossEntropyLoss()(output, target)
        if (batch_idx % 100) == 0:
            pred = [torch.argmax(output[i]).item() for i in range(4)]
            self.logger.log_table(key='val', columns=['image', 'label', 'pred'], data=[[img[i], labels[i], pred[i]] for i in range(4)])
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_acc', self.acc_metric(torch.argmax(output, dim=1), y))
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        if (batch_idx % 100) == 0:
            img = [wandb.Image(x[i].cpu().numpy()) for i in range(4)]
            labels = [y[i].item() for i in range(4)]
        x = x.view(x.size(0), -1)
        target = F.one_hot(y, num_classes=10).float()
        
        output = self.model(x)
        if (batch_idx % 100) == 0:
            pred = [torch.argmax(output[i]).item() for i in range(4)]
            self.logger.log_table(key='test', columns=['image', 'label', 'pred'], data=[[img[i], labels[i], pred[i]] for i in range(4)])
        self.log('test_acc', self.acc_metric(torch.argmax(output, dim=1), y))
    
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
            optimizer = optim.AdamW(self.parameters(), lr=1e-3)
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
    
    model = LightningModule(FeedForwardNet(), args.opt)
    
    trainset = MNIST(root='./data/train', train=True, download=True, transform=ToTensor())
    valset, testset = random_split(MNIST(root='./data/val', train=False, download=True, transform=ToTensor()), [0.5, 0.5], torch.Generator().manual_seed(42))
    
    train_dl = DataLoader(trainset, batch_size=32, num_workers=8)
    val_dl, test_dl = DataLoader(valset, batch_size=32, num_workers=8), DataLoader(testset, batch_size=32, num_workers=8)
    
    early_stop_callback = pl.callbacks.EarlyStopping('val_loss', patience=3)
    
    wandb.init(
        project='Optim_Alg',
        name=args.run_name
        )
    wandb_logger = WandbLogger()
    
    trainer = pl.Trainer(
        callbacks=[early_stop_callback],
        logger=wandb_logger,
        max_epochs=15
        )
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    trainer.test(model=model, dataloaders=test_dl)
    wandb.finish()