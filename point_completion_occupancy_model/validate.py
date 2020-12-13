import torch
import torch.nn as nn

from utils.data_loader import DataSetClass, load_list_dirs, generate_data_loader
from utils.constants import K, BATCH_SIZE, DEVICE, POINTCLOUD_N
from models.point_completion import OccupancyModel
from train import SHAPENET_CLASS_DIR, MODEL_FILENAME


def validation(model, val_loader):
    model.eval()
    modelCriterion = nn.BCELoss()

    validation_loss = 0
    correct = 0

    for batch_idx, data in enumerate(val_loader):
        pts, occupancies, pointcloud = data

        pointcloud = pointcloud.view(-1, POINTCLOUD_N, 3, 1).permute(0, 2, 1, 3).cuda()
        pts = pts.view(-1, K, 3, 1).permute(0, 2, 1, 3).cuda()
        occupancies = occupancies.view(BATCH_SIZE*K, 1).cuda()

        pred = model(pts, pointcloud)

        pred = pred.squeeze(-1)
        pred = torch.sigmoid(pred)

        loss = modelCriterion(pred, occupancies)
        validation_loss += loss.item()

        threshold = 0.6
        roundedOut = [1 if out > threshold else 0 for out in pred.view(-1)]
        roundedOut = torch.tensor(roundedOut).cuda()
        correctNow = roundedOut.eq(occupancies.view(-1)).sum()
        correct += correctNow

        validation_loss /= len(val_loader.dataset)

        print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            validation_loss, correctNow, pts.size()[0] * K, 100. * correctNow / (pts.size()[0] * K)))


if __name__ == "__main__":
    # catalogue all of the directories with the chosen category
    print(f"loading val.lst for dir {SHAPENET_CLASS_DIR}")
    val_loader = generate_data_loader(SHAPENET_CLASS_DIR, 'val.lst')

    model = OccupancyModel()
    model.cuda()
    model.load_state_dict(torch.load(MODEL_FILENAME, map_location=DEVICE))

    validation(model, val_loader)
