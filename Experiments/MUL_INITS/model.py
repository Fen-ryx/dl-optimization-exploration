import torch
import torch.nn as nn

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


def init_weights(module):
    if isinstance(module, nn.Conv2d):
        torch.nn.init.normal_(module.weight)
        module.bias.data.fill_(0.01)


if __name__ == "__main__":
    model1, model2 = SRCNN(), SRCNN()
    
    model1.apply(init_weights)
    model2.apply(init_weights)
    model1.cuda()
    model2.cuda()
    
    test = torch.randn(size=(3, 320, 480)).cuda()
    o1, o2 = model1(test), model2(test)
    print(o1 == o2)