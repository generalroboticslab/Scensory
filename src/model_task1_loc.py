import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import h5py
import numpy as np
import os
import torch.nn.functional as F
import random

class SensorBranch(nn.Module):
    def __init__(self):
        super(SensorBranch, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=9, out_channels=16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.fc = nn.Linear(32 * 7, 32)

    def forward(self, x):
        # (batch, 10, 30)
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))  # (batch, 16, 30)
        x = self.pool1(x)                    # (batch, 16, 15)
        x = F.relu(self.bn2(self.conv2(x)))  # (batch, 32, 15)
        x = self.pool2(x)                    # (batch, 32, 7)
        
        x = x.reshape(x.size(0), -1)         # (batch, 32*7)
        x = self.fc(x)                       # (batch, 32)
        return x

class MultiBranchFusion(nn.Module):
    def __init__(self, num_branches=6, hidden_dim=32, num_classes=4):
        super(MultiBranchFusion, self).__init__()
        self.branches = nn.ModuleList([SensorBranch() for _ in range(num_branches)])
        
        # num_branches * hidden_dim
        self.fc_merge = nn.Linear(num_branches * hidden_dim, 64)
        self.fc_out   = nn.Linear(64, num_classes)

    def forward(self, x):
        embeddings = []
        for i, branch in enumerate(self.branches):
            # take each sensor array (batch, 30, 10)
            sensor_i = x[:, :, i, :]
            emb_i = branch(sensor_i)
            embeddings.append(emb_i)
        
        # (batch, 6*32)
        fused = torch.cat(embeddings, dim=1)
        
        fused = F.relu(self.fc_merge(fused))  # (batch, 64)
        logits = self.fc_out(fused)          # (batch, 4)
        return logits
