import os

import numpy
import torch

from .constants import K, BATCH_SIZE, POINTCLOUD_N

# One DataSetClass per subdirectory in a category, will return "K" point samples and a pointcloud with POINTCLOUD_N points
class DataSetClass(torch.utils.data.Dataset):
    def __init__(self, d):
        self.dir = d
        with numpy.load(f"{d}/points.npz") as data:
            self.pts = torch.tensor(data["points"], dtype=torch.float)
            self.occupancies = torch.tensor(numpy.unpackbits(data["occupancies"])[:self.pts.size()[0]],
                                            dtype=torch.float)

        with numpy.load(f"{d}/pointcloud.npz") as ptcloud_data:
            print('loading point cloud')
            self.point_cloud = torch.tensor(ptcloud_data['points'], dtype=torch.float)
            print(self.point_cloud)
            print(self.point_cloud.shape)
        self.K = K
        self.length = int(self.occupancies.size()[0] / self.K)
        self.point_cloud = None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # print(f"getting dir {self.dir}")
        # pick POINTCLOUD_N random points for the pointcloud
        point_cloud_indicies = numpy.random.randint(self.point_cloud.shape[0], size=POINTCLOUD_N)
        point_cloud_sample = self.point_cloud[point_cloud_indicies, :]

        # return pts, occupancies, and a pointcloud
        return self.pts[idx * self.K:(idx * self.K + self.K)], self.occupancies[idx * self.K:(idx * self.K + self.K)], point_cloud_sample, self.point_cloud


def load_list_dirs(top_dir, list_file):
    with open(os.path.join(top_dir, list_file)) as train_list:
        return [os.path.join(top_dir, train_dir) for train_dir in train_list.readlines()]


def generate_data_loader(top_dir, list_file, batch_size=BATCH_SIZE):
    image_dirs = load_list_dirs(top_dir, list_file)
    datasets = [DataSetClass(image_dir.rstrip()) for image_dir in image_dirs]
    data = torch.utils.data.ConcatDataset(datasets)
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
