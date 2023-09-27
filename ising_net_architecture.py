import torch
import torch.nn as nn
import torch.nn.functional as F

# simple convolutional neural net used for analyzing the Ising dataset
class SimpleCNN(nn.Module):
    def __init__(self, img_dim=60, n_categories=200, n_hidden=128, n_kernels=16):
        super(SimpleCNN, self).__init__()
        self.pool = nn.AvgPool2d(2, 2)
        self.img_dim = img_dim
        self.n_hidden = n_hidden
        self.n_kernels = n_kernels
        
        self.conv1 = nn.Conv2d(1, self.n_kernels, kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.n_kernels, 2*self.n_kernels, kernel_size=2, stride=1, padding=1)
        self.fc1 = nn.Linear(2*self.n_kernels * (self.img_dim//4) * (self.img_dim//4), self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, n_categories - 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 2*self.n_kernels * (self.img_dim//4) * (self.img_dim//4))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
