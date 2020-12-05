import torch
import io

from utils.data_loader import DataSetClass
from utils.constants import K, BATCH_SIZE, DEVICE

topdir = "/home/andrea/Documents/GradSchool/OccupancyNetworks/occupancy_networks"
couchesDirectory = f"{topdir}/data/ShapeNet/04256520"


if __name__ == '__main__()':

    # catalogue all of the directories with the chosen category
    trainingDirs = []
    with io.open(f"{couchesDirectory}/train.lst") as trainlist:
        for traindir in trainlist.readlines():
            trainingDirs.append(f"{couchesDirectory}/{traindir.strip()}")
    dataSets = []
    for tdir in trainingDirs:
        dataSets.append(DataSetClass(tdir))
    data = torch.utils.data.ConcatDataset(dataSets)
    train_loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    # Get the validation data
    valDirs = []
    with io.open(f"{couchesDirectory}/val.lst") as vallist:
        for valdir in vallist.readlines():
            valDirs.append(f"{couchesDirectory}/{valdir.strip()}")
    dataSets = []
    for vdir in valDirs:
        dataSets.append(DataSetClass(vdir))
    val_data = torch.utils.data.ConcatDataset(dataSets)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)