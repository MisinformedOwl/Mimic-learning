import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms import transforms
from torch import optim, nn
import numpy as np
import graphviz
from torchviz import make_dot as viz


class CNNModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(1,3, kernel_size=3)
        self.layer2 = nn.MaxPool2d(2,2)
        self.layer3 = nn.Conv2d(3, 3, kernel_size=3)
        self.layer4 = nn.MaxPool2d(2,2)
        self.layer5 = nn.Conv2d(3, 1, kernel_size=3)
        
        self.flatten = nn.Flatten()
        
        self.lin1 = nn.Linear(15376, 256)
        self.lin2 = nn.Linear(256, 128)
        self.lin3 = nn.Linear(128, 64)
        self.lin4 = nn.Linear(64, 32)
        self.lin5 = nn.Linear(32, 16)
        self.lin6 = nn.Linear(16, 8)
        self.lin7 = nn.Linear(8, 2)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        x = F.relu(self.layer5(x))
        
        x = self.flatten(x)
        
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = F.relu(self.lin4(x))
        x = F.relu(self.lin5(x))
        x = F.relu(self.lin6(x))

    def displayCNN(self, x):
        viz(x, params=dict(list(self.named_parameters()))).render("cnn_viz", format="png")
