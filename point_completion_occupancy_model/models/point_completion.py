import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetBlock(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        size_h = min(size_in, size_out)

        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.fc_2 = nn.Linear(size_in, size_out, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        dx = self.fc_0(self.relu(x))
        dx = self.fc_1(self.relu(dx))

        x_skip = self.fc_2(x)

        return x_skip + dx


class PointNetEncoder(nn.Module):
    def __init__(self):
        super(PointNetEncoder, self).__init__()
        self.fc1 = nn.Linear(3, 512)
        self.resnet_1 = ResnetBlock(512, 256)
        self.resnet_2 = ResnetBlock(512, 256)
        self.resnet_3 = ResnetBlock(512, 256)
        self.resnet_4 = ResnetBlock(512, 256)
        self.fc_final = nn.Linear(256, 256)

    def forward(self, x):
        x = x.squeeze()
        # print(x.shape)
        x = x.permute(0, 2, 1)
        # print(x.shape)

        x = F.relu(self.fc1(x))

        x = self.resnet_1(x)

        n, k, c = x.size()
        x = x.permute(0, 2, 1)

        pooled = F.max_pool1d(x, k).expand(x.size())
        x = torch.cat([x, pooled], dim=1)

        x = x.permute(0, 2, 1)

        x = F.relu(x)

        x = self.resnet_2(x)

        n, k, c = x.size()

        x = x.permute(0, 2, 1)

        pooled = F.max_pool1d(x, k).expand(x.size())
        x = torch.cat([x, pooled], dim=1)

        x = x.permute(0, 2, 1)

        x = F.relu(x)
        x = self.resnet_3(x)

        n, k, c = x.size()

        x = x.permute(0, 2, 1)

        pooled = F.max_pool1d(x, k).expand(x.size())
        x = torch.cat([x, pooled], dim=1)

        x = x.permute(0, 2, 1)

        x = self.resnet_4(x)
        n, k, c = x.size()
        x = x.permute(0, 2, 1)

        # print(f"x.shape is {x.shape}")
        # print(f"k is {k}")
        x = F.max_pool1d(x, k)

        x = x.squeeze()
        # print(f"x.shape is {x.shape}")
        pts = self.fc_final(x)

        return pts


class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.fc1 = nn.Conv2d(256, 256, kernel_size=1)
        self.fc2 = nn.Conv2d(256, 256, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(256, affine=False, track_running_stats=True)
        self.bn2 = nn.BatchNorm2d(256, affine=False, track_running_stats=True)
        self.gammaLayer1 = nn.Conv1d(256, 256, kernel_size=1)
        self.gammaLayer2 = nn.Conv1d(256, 256, kernel_size=1)
        self.betaLayer1 = nn.Conv1d(256, 256, kernel_size=1)
        self.betaLayer2 = nn.Conv1d(256, 256, kernel_size=1)

    def forward(self, y):
        x = y['ex']
        n, c, k, d = x.size()
        # n, c, k = x.size()

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
    def __init__(self):
        super(OccupancyModel, self).__init__()
        self.blocks = self.makeBlocks()
        self.encoderModel = PointNetEncoder()
        # self.fc_enc = nn.Linear(512, 256)  # there's a fc layer in the pointnetencoder so don't know if we need this.
        self.gammaLayer = nn.Conv1d(256, 256, kernel_size=1)
        self.betaLayer = nn.Conv1d(256, 256, kernel_size=1)
        self.cbn = nn.BatchNorm2d(256, affine=False, track_running_stats=True)
        self.fc1 = nn.Conv2d(3, 256, kernel_size=1)
        self.fc2 = nn.Conv2d(256, 1, kernel_size=1)

    def makeBlocks(self):
        blocks = []
        for _ in range(5):
            blocks.append(Block())
        return nn.Sequential(*blocks)

    def forward(self, x, pointcloud):
        n, c, k, d = x.size()
        # print(f"x.shape  at beginning of forward is {x.shape}")
        print(f"pointcloud.shape  at beginning of forward is {pointcloud.shape}")
        pt_cloud = self.encoderModel(pointcloud)
        # pts = self.fc_enc(x)
        # print("View effect is:")
        # print(pt_cloud.shape)
        pt_cloud = pt_cloud.view(-1, 256, 1)  # Add's another dimension? dunno why
        # print(pt_cloud.shape)
        x = self.fc1(x)
        # 5 pre-activation ResNet-blocks
        x = self.blocks({'enc': pt_cloud, 'ex': x})
        x = x['ex']
        # CBN
        gamma = self.gammaLayer(pt_cloud)
        gamma = torch.stack([gamma for _ in range(k)], dim=2)
        beta = self.betaLayer(pt_cloud)
        beta = torch.stack([beta for _ in range(k)], dim=2)
        x = gamma.mul(self.cbn(x)).add_(beta)
        x = F.relu(x)
        x = self.fc2(x)
        x = x.view(-1, 1)
        x = torch.sigmoid(x)
        # print(f"x.shape at end of forward is {x.shape}")
        return x
