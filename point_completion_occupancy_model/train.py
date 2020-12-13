import os

import torch
import torch.nn as nn
import torch.optim as optim

from utils.data_loader import generate_data_loader
from utils.constants import K, BATCH_SIZE, POINTCLOUD_N
from models.point_completion import OccupancyModel

SHAPENET_DIR = "/scratch/rag551/occupancy_networks/ShapeNet"
SHAPENET_CLASS = '04401088'  # electronics
SHAPENET_CLASS_DIR = os.path.join(SHAPENET_DIR, SHAPENET_CLASS)

MODEL_FILENAME = 'point_completion_model.pth'


def train(epoch, model, train_loader, optimizer):
    # decoderLoss = nn.BCEWithLogitsLoss(reduction='sum')  # Dunno if i need this loss or what
    modelCriterion = nn.BCELoss()
    model.train()

    for batch_idx, data in enumerate(train_loader):
        pts, occupancies, pointcloud = data
        # print(f"pts.shape is {pts.shape}")
        # print(f"occupancies.shape is {occupancies.shape}")
        # print(f"pointcloud.shape is {pointcloud.shape}")
        # Each batch size contains batch_size sets of "K" points
        # pointcloud = pointcloud.view(BATCH_SIZE*POINTCLOUD_N, 3, 1).cuda()
        # pts = pts.view(BATCH_SIZE * K, 3, 1).cuda()
        # occupancies = occupancies.view(BATCH_SIZE * K, 1).cuda()
        # not sure what i should be doing here
        pointcloud = pointcloud.view(-1, POINTCLOUD_N, 3, 1).permute(0, 2, 1, 3).cuda()
        # print(pointcloud.shape)
        pts = pts.view(-1, K, 3, 1).permute(0, 2, 1, 3).cuda()
        occupancies = occupancies.view(BATCH_SIZE*K, 1).cuda()

        # print(f"pts.shape is {pts.shape}")
        # print(f"occupancies.shape is {occupancies.shape}")
        optimizer.zero_grad()

        pred = model(pts, pointcloud)
        # print(f"pred.shape before permute {pred.shape}")
        pred = pred.squeeze(-1)
        # print(f"pred.shape after permute {pred.shape}")

        loss = modelCriterion(pred, occupancies)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader), 100. * batch_idx / len(train_loader),
                loss.item()))
        if batch_idx % 100 == 0:
            print(f"Saving to {MODEL_FILENAME}")
            torch.save(model.state_dict(), MODEL_FILENAME)


if __name__ == "__main__":
    shapenet_class_dir = os.path.join(SHAPENET_DIR, SHAPENET_CLASS)

    # catalogue all of the directories with the chosen category
    print(f"loading train.lst for dir {shapenet_class_dir}")
    train_loader = generate_data_loader(shapenet_class_dir, 'train.lst')

    model = OccupancyModel()
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1):
        train(epoch, model, train_loader, optimizer)
