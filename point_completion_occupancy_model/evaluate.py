from functools import partial
from random import randint

import torch
import numpy

from models.point_completion import OccupancyModel
from train import MODEL_FILENAME, SHAPENET_CLASS_DIR
from utils.constants import DEVICE
from utils.data_loader import generate_data_loader


def generate_grid(ncuts, xl, xr, yl, yr, zl, zr):
    # The unit cube centered at 0
    # Subdivided into a grid of 32^3 "voxels"
    x = numpy.linspace(xl, xr, ncuts)
    y = numpy.linspace(yl, yr, ncuts)
    z = numpy.linspace(zl, zr, ncuts)
    xg, yg, zg = numpy.meshgrid(x, y, z)
    x = torch.tensor(xg)
    y = torch.tensor(yg)
    z = torch.tensor(zg)
    # Convert to a grid of 3 dimensional coordinate
    tgrid = torch.stack([x, y, z], dim=3).permute(1, 0, 2, 3)
    # A cube is made up the 8 vertices
    # Convert to a list where every 8 coords denote a cube
    gridpts = torch.zeros(8*(ncuts-1)*(ncuts-1)*(ncuts-1), 3)

    '''
    Vertex Order for marching cubes is 
    (0,0,0):(1,0,0):(1,1,0):(0,1,0):(0,0,1):(1,0,1):(1,1,1):(0,1,1)
    '''
    gpt = 0
    for i in range(ncuts-1):
        for j in range(ncuts-1):
            for k in range(ncuts-1):
                gridpts[gpt] = tgrid[i][j][k]
                gridpts[gpt+1] = tgrid[i+1][j][k]
                gridpts[gpt+2] = tgrid[i+1][j+1][k]
                gridpts[gpt+3] = tgrid[i][j+1][k]
                gridpts[gpt+4] = tgrid[i][j][k+1]
                gridpts[gpt+5] = tgrid[i+1][j][k+1]
                gridpts[gpt+6] = tgrid[i+1][j+1][k+1]
                gridpts[gpt+7] = tgrid[i][j+1][k+1]
                gpt = gpt + 8

    return gridpts


def generate_adaptive_grid(ncuts, xl, xh, yl, yh, zl, zh, limit, mesh_funct, onCuda):
    if not limit:
        return None
    g = generate_grid(ncuts, xl, xh, yl, yh, zl, zh)
    ag = g.view(-1, 3)

    final_grid = []
    # divide list of coordinates into cubes
    for i in range(0, int(ag.size()[0]), 8):
        # marking one active if occupancies differ on the vertices
        active = False
        for k in range(0, 8):
            coord = ag[i + k]
            if onCuda:
                coord = coord.cuda()
            active ^= mesh_funct(coord)
        if (active):
            # near left coordinate is v0
            nl = ag[i]
            # top right coordinate is v6
            tr = ag[i + 6]

            # subdivide this cube into 8 subvoxels
            g = generate_adaptive_grid(3, nl[0], tr[0], nl[1], tr[1], nl[2], tr[2], limit - 1, mesh_funct, onCuda)
            # replace this grid where the cube was
            if g is not None:
                final_grid.append(g)
            # or keep this cube
            else:
                final_grid.append(ag[i: i + 8])
        else:
            final_grid.append(ag[i:i + 8])
    for i in range(len(final_grid)):
        final_grid[i] = final_grid[i].view(-1, 3)
    final_grid = torch.cat(final_grid)

    return final_grid


def over_model_threshold(model, point_cloud, pt):
    x = model(pt.view(1, 3, 1), point_cloud)
    return (x > 0.2).item()


if __name__ == "__main__":
    model = OccupancyModel()
    model.load_state_dict(torch.load(MODEL_FILENAME, map_location=DEVICE))
    model.cuda()
    model.eval()

    test_loader = generate_data_loader(SHAPENET_CLASS_DIR, 'test.lst')

    for batch_idx, data in enumerate(test_loader):
        # print(f"evaluating {data.dir}")

        pts, occupancies, pointcloud = data

        f = partial(over_model_threshold, model, pointcloud)

        g = generate_adaptive_grid(32, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, 3, f, True)
        numpy.savetxt('electronic_ag_32_3.txt', g.detach().numpy())
        numpy.savetxt('incomplete_point_cloud.txt', pointcloud.detach().numpy())

        break
