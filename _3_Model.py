import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(1),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(0.3)
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5150, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.output = nn.Sequential(
            nn.Linear(128, 100),
            # nn.Sigmoid()
        )

    def drop(self, x):
        x1, x2 = x[:, :100], x[:, 100:]
        x2 = self.dropout(x2)
        return torch.cat((x1, x2), 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.drop(x)
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
