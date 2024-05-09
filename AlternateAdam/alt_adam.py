import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

torch.manual_seed(42)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=True, transform=transform),
    batch_size=64
)

model = SimpleNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print('Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()
            ))
            with open(r'C:\Users\Gaurav\OptimAlg\AlternateAdam\train_logs_og.txt', 'a') as f:
                f.write(str(loss.item())+"\n")
    
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            val_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    val_accuracy = 100. * correct / len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        val_loss, correct, len(val_loader.dataset), val_accuracy
    ))
    with open(r'C:\Users\Gaurav\OptimAlg\AlternateAdam\val_loss_logs_og.txt', 'a') as f:
        f.write(str(val_loss)+"\n")
    with open(r'C:\Users\Gaurav\OptimAlg\AlternateAdam\val_acc_logs_og.txt', 'a') as f:
        f.write(str(val_accuracy)+"\n")

torch.save(model.state_dict(), 'mnist_model.pth')