import torch
import os

from utils.data_loader import DataSetClass
from utils.constants import K, BATCH_SIZE, DEVICE

SHAPENET_DIR = "/scratch/rag551/occupancy_networks/ShapeNet"

# SHAPNET_CLASSES = [
#     '02958343'
# ]

SHAPENET_CLASS = '02958343'


def load_list_dirs(dir, list_file):
    with open(os.path.join(dir, list_file)) as train_list:
        return [train_dir for train_dir in train_list.readlines()]


def generate_data_loader(dir, list_file):
    dirs = load_list_dirs(dir, list_file)
    datasets = [DataSetClass(dir) for dir in dirs]
    data = torch.utils.data.ConcatDataset(datasets)
    return torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

if __name__ == '__main__()':
    shapenet_class_dir = os.path.join(SHAPENET_DIR, SHAPENET_CLASS)

    # catalogue all of the directories with the chosen category
    train_loader = generate_data_loader(shapenet_class_dir, 'train.lst')

    # Get the validation data
    validation_loader = generate_data_loader(shapenet_class_dir, 'val.lst')
