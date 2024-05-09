import os
import torch
from PIL import Image

from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader, random_split
torch.manual_seed(42)

PATH = r'C:\Users\Gaurav\OptimAlg\Experiments\VSR\BSDS300-images'


class VSR_DataLoader(Dataset):
    def __init__(self, mode):
        super().__init__()
        
        self.sub_mode = 'test' if mode == 'val' else mode
        self.directory = os.path.join(PATH, 'images', self.sub_mode)
        with open(os.path.join(PATH, f'iids_{self.sub_mode}.txt')) as f:
            self.data = f.read().split('\n')[:-1]
        
        if mode == 'val':
            self.data, _ = random_split(dataset=self.data, lengths=[0.5, 0.5], generator=torch.Generator().manual_seed(42))
        elif mode == 'test':
            _, self.data = random_split(dataset=self.data, lengths=[0.5, 0.5], generator=torch.Generator().manual_seed(42))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_uid = self.data[index]
        target_image = Image.open(os.path.join(self.directory, f"{img_uid}.jpg")).resize(size=(480, 320))
        
        image = target_image.resize(size=(64, 96))
        input_image = image.resize(size=(480, 320), resample=Image.BICUBIC)
        
        x, y = ToTensor()(input_image) / 255.0, ToTensor()(target_image) / 255.0
        return {
            "image": x,
            "target": y
        }

def vsr_dataloader(mode, batch_size=16, drop_last=True):
    dataloader = DataLoader(
        VSR_DataLoader(mode=mode),
        batch_size=batch_size,
        shuffle=True if mode=='train' else False,
        num_workers=8,
        persistent_workers=True,
        drop_last=drop_last
    )
    return dataloader