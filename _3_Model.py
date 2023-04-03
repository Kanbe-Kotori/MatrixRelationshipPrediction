import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(1),
            nn.ReLU()
        )
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5050, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.output = nn.Sequential(
            nn.Linear(128, 100),
            # nn.Sigmoid()
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
