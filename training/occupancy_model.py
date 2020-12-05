import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy
from PIL import Image
import io


class ResNetBlock(nn.Module):
    def __init__(self):
        super(ResNetBlock, self).__init__()
        self.fc1 = nn.Conv2d(256, 256, kernel_size=1)
        self.fc2 = nn.Conv2d(256, 256, kernel_size=1)

    def forward(self, x):
        x = self.fc1(F.relu(x))
        x = self.fc2(F.relu(x))


class PointNetEncoder(nn.Module):
    def __init__(self):
        super(PointNetEncoder, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 128)

    def forward(self, x):
        x = x.squeeze()
        n, c, k = x.size()
        x = x.permute(0, 2, 1)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        n, k, c = x.size()
        x = x.permute(0, 2, 1)

        pooled = F.max_pool1d(x, k).expand(x.size())
        x = torch.cat([x, pooled], dim=1)

        x = x.permute(0, 2, 1)

        x = F.relu(x)

        x = self.fc3(x)

        n, k, c = x.size()

        x = x.permute(0, 2, 1)

        pooled = F.max_pool1d(x, k)
        pooled = pooled.expand(x.size())

        x = torch.cat([x, pooled], dim=1)

        x = x.permute(0, 2, 1)

        x = F.relu(x)
        x = self.fc4(x)

        n, k, c = x.size()

        x = x.permute(0, 2, 1)

        x = F.max_pool1d(x, k)

        x = x.squeeze()

        mean = self.mean_fc(x)
        stddev = self.logstddev_fc(x)

        return mean, stddev


class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.fc1 = nn.Conv2d(256, 256, kernel_size=1)
        self.fc2 = nn.Conv2d(256, 256, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(256, affine=False, track_running_stats=True)
        self.bn2 = nn.BatchNorm2d(256, affine=False, track_running_stats=True)
        self.gammaLayer1 = nn.Conv1d(128, 256, kernel_size=1)
        self.gammaLayer2 = nn.Conv1d(128, 256, kernel_size=1)
        self.betaLayer1 = nn.Conv1d(128, 256, kernel_size=1)
        self.betaLayer2 = nn.Conv1d(128, 256, kernel_size=1)

    def forward(self, y):
        x = y['ex']
        n, c, k, d = x.size()

        encoding = y['enc']
        gamma = self.gammaLayer1(encoding)

        # Need to stack the beta and gamma
        # so that we multiply all the points for one mesh
        # by the same value
        gamma = torch.stack([gamma for _ in range(k)], dim=2)

        beta = self.betaLayer1(encoding)
        beta = torch.stack([beta for _ in range(k)], dim=2)

        # First apply Conditional Batch Normalization
        out = gamma * self.bn1(x) + beta
        # Then ReLU activation function
        out = F.relu(out)
        # fully connected layer
        out = self.fc1(out)
        # Second CBN layer
        gamma = self.gammaLayer2(encoding)
        gamma = torch.stack([gamma for _ in range(k)], dim=2)

        beta = self.betaLayer2(encoding)
        beta = torch.stack([beta for _ in range(k)], dim=2)

        out = gamma * self.bn2(out) + beta
        # RELU activation
        out = F.relu(out)
        # 2nd fully connected
        out = self.fc2(out)
        # Add to the input of the ResNet Block
        out = x + out

        return {'ex': out, 'enc': encoding}

class OccupancyModel(nn.Module):
    def __init__(self, encoder):
        super(OccupancyModel, self).__init__()
        self.blocks = self.makeBlocks()
        self.encoderModel = encoder
        self.gammaLayer = nn.Conv1d(128, 256, kernel_size=1)
        self.betaLayer = nn.Conv1d(128, 256, kernel_size=1)
        self.cbn = nn.BatchNorm2d(256, affine=False, track_running_stats=True)
        self.fc1 = nn.Conv2d(3, 256, kernel_size=1)
        self.fc2 = nn.Conv2d(256, 1, kernel_size=1)

    def makeBlocks(self):
        blocks = []
        for _ in range(5):
            blocks.append(Block())
        return nn.Sequential(*blocks)

    def sampleFromZDist(self, z):
        mean, logstddev = z
        std = logstddev.mul(0.5).exp_()
        eps = torch.randn_like(logstddev, requires_grad=True)
        return eps.mul(std).add_(mean)

    def forward(self, x, z_eval=None):
        if self.training:
            z_dist = self.encoderModel(x)
            z = self.sampleFromZDist(z_dist)
            z = z.unsqueeze(-1)
        else:
            z = z_eval
            z_dist = (0, 1)
        x = self.fc1(x)
        # 5 pre-activation ResNet-blocks
        x = self.blocks({'enc': z, 'ex': x})
        x = x['ex']
        n, c, k, d = x.size()

        # CBN
        gamma = self.gammaLayer(z)

        gamma = torch.stack([gamma for _ in range(k)], dim=2)

        beta = self.betaLayer(z)
        beta = torch.stack([beta for _ in range(k)], dim=2)

        x = gamma.mul(self.cbn(x)).add_(beta)
        x = F.relu(x)
        x = self.fc2(x)
        # x = x.view(-1,1)
        # x = torch.sigmoid(x)
        return x, z_dist
