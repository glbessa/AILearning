import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights

class CNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential([
            self._cnn_block(in_channels),
            self._cnn_block(in_channels*2),
            self._cnn_block(in_channels*4),
            self._cnn_block(in_channels*8),
            nn.AvgPool2d(2),
            nn.Linear(128),
            nn.ReLU()
        ])

    def forward(self, x):
        return self.net(x)

    def _cnn_block(self, in_channels):
        return nn.Sequential([
            nn.Conv2d(in_channels, in_channels*2, 3, 1),
            nn.BatchNorm2d(in_channels*2),
            nn.ReLU()
        ])


class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNN(28)

    def forward(self, x):
        pass

class MNISTRecognition(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        return None

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 10
    lr = 1e-3
    image_size = 28
    batch_size = 12

    trans = transforms.Compose([

    ])

    dataset = datasets.MNIST(root='dataset', train=True, transform=trans, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SiameseNetwork()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):
        for batch_idx, x, y in enumerate(loader):
            pred = model(x)
            loss = pred - y

            # Zerando gradientes
            model.zero_grad()
            # Retropropagando a perda
            loss.backward()
            # Atualizando os gradientes
            optimizer.step()

            # Imprimindo resultados
            if batch_idx % 100 == 0:
                with torch.no_grad():
                    pred = model(x)

