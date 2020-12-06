from utils.constants import K, BATCH_SIZE
import torch
import numpy
import os


# One DataSetClass per subdirectory in a category, will return "K" point samples and a single image randomly
# drawn from the 23 available
class DataSetClass(torch.utils.data.Dataset):
    def __init__(self, d):
        self.dir = d
        with numpy.load(f"{d}/points.npz") as data:
            self.pts = torch.tensor(data["points"], dtype=torch.float)
            self.occupancies = torch.tensor(numpy.unpackbits(data["occupancies"])[:self.pts.size()[0]],
                                            dtype=torch.float)
        self.K = K
        self.length = int(self.occupancies.size()[0] / self.K)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.pts[idx * self.K:(idx * self.K + self.K)], self.occupancies[idx * self.K:(idx * self.K + self.K)]


def load_list_dirs(top_dir, list_file):
    with open(os.path.join(top_dir, list_file)) as train_list:
        return [os.path.join(top_dir, train_dir) for train_dir in train_list.readlines()]


def generate_data_loader(top_dir, list_file):
    image_dirs = load_list_dirs(top_dir, list_file)
    datasets = [DataSetClass(image_dir.rstrip()) for image_dir in image_dirs]
    data = torch.utils.data.ConcatDataset(datasets)
    return torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
