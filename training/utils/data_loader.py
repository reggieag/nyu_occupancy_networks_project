from utils.constants import K
import torch
import numpy


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