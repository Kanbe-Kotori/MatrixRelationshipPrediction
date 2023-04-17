import torch
import torch.nn as nn


class Model5150(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=5, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(5),
            nn.ReLU(),
            nn.Conv1d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5150, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(128, 100),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.dense(x)
        return self.output(x)


class ModelHist(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.dense = nn.Sequential(
            nn.Linear(64*4*4, 64),
            nn.ReLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.dense(x)
        return self.output(x)


class ImbalancedLoss(nn.Module):
    def __init__(self, alpha):
        super(ImbalancedLoss, self).__init__()
        self.alpha = alpha

    def forward(self, inputs, targets):
        pos_weight = self.alpha * targets + (1 - targets)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = criterion(inputs, targets)
        return loss
