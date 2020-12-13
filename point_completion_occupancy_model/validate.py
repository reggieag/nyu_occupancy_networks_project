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
        pts, occupancies, pointcloud, full_pointcloud = data

        pointcloud = pointcloud.view(-1, POINTCLOUD_N, 3, 1).permute(0, 2, 1, 3).cuda()
        pts = pts.view(-1, K, 3, 1).permute(0, 2, 1, 3).cuda()
        occupancies = occupancies.view(BATCH_SIZE*K).cuda()

        pred = model(pts, pointcloud)

        pred = pred.squeeze(-1)
        pred = torch.sigmoid(pred)
        # print(pred.shape)
        # print(occupancies.shape)
        # print(pred)
        # print(occupancies)
        loss = modelCriterion(pred, occupancies)
        # print(loss)
        validation_loss += loss.item()

        threshold = 0.6
        # print(pred.view(-1))
        roundedOut = [1 if out > threshold else 0 for out in pred.view(-1)]
        roundedOut = torch.tensor(roundedOut).cuda()
        correctNow = roundedOut.eq(occupancies.view(-1)).sum()

        occupied_correct = roundedOut.eq(1).logical_and(occupancies.view(-1).eq(1)).sum()
        predicted_occupied = roundedOut.eq(1).sum()
        actual_occupied = occupancies.view(-1).eq(1).sum()

        correct += correctNow

        validation_loss /= len(val_loader.dataset)

        if actual_occupied > 0:
            # print(pts)
            # print(pts.shape)
            # print(occupancies)
            # print(pred)
            print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                validation_loss, correctNow, pts.size()[0] * K, 100. * correctNow / (pts.size()[0] * K)))

            print(f'Occupied correct: {occupied_correct}, actual occupied: {actual_occupied}, predicted_occupied: {predicted_occupied}')


if __name__ == "__main__":
    # catalogue all of the directories with the chosen category
    print(f"loading val.lst for dir {SHAPENET_CLASS_DIR}")
    val_loader = generate_data_loader(SHAPENET_CLASS_DIR, 'val.lst', batch_size=BATCH_SIZE)

    model = OccupancyModel()
    model.cuda()
    model.load_state_dict(torch.load(MODEL_FILENAME, map_location=DEVICE))

    validation(model, val_loader)
