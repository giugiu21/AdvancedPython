import torch
import torch.nn as nn
import torch.nn.functional as F

#Defining the MLP model class
class MLP(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.flatten = nn.Flatten()

        self.layer1 = nn.Linear(28 * 28, 128)
        self.layer2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.flatten(x)      #input size: (batch, 1, 28, 28) flattens to (batch, 784)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.output(x)
        return x

#Defining the CNN model class
class CNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        self.layer1 = nn.Linear(32 * 7 * 7, 128)
        self.output = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # input size: (batch, 1, 28, 28) after 1st convolution output size: (batch, 16, 28, 28) after 1st pooling output size: (batch, 16, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))   # input size: (batch, 16, 14, 14) after 2nd convolution output: (batch, 32, 7, 7) after 2nd pooling output size: (batch, 32, 7, 7)

        x = torch.flatten(x, start_dim=1)      # input size: (batch, 32, 7, 7) after flattening layer output size: (batch, 1568)

        x = F.relu(self.layer1(x))
        x = self.output(x)
        return x