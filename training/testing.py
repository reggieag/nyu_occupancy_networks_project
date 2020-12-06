import torch
import torch.nn as nn
import os

from utils.data_loader import DataSetClass, load_list_dirs, generate_data_loader
from utils.constants import K, BATCH_SIZE, DEVICE

train_loader = generate_data_loader('/scratch/rag551/occupancy_networks/ShapeNet/04256520', 'train.lst')

for batch_idx, data in enumerate(train_loader):
    (pts, occupancies) = data
    break
    print(pts)
    print(occupancies)
