import torch
import torch.nn as nn


class Residual(nn.Module):
    def __init__(self):
        super(Residual, self).__init__()
        self.attn = nn.MultiheadAttention(100, 5, 0.1)
        self.norm = nn.LayerNorm(100)
        self.mlp = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 100),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        attn, _ = self.attn(x, x, x)
        norm1 = self.norm(x + attn)
        mlp = self.mlp(norm1)
        return x + mlp


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=7, out_channels=7, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(7),
            nn.ReLU(),
            nn.Conv1d(in_channels=7, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.residual = Residual()
        self.output = nn.Sequential(
            # nn.Linear(100, 100),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        # x = self.drop(x)
        for i in range(5):
            x = self.residual(x)
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
