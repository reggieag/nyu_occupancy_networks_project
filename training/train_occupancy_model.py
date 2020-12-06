import torch
import torch.nn as nn
import os

from utils.data_loader import DataSetClass, load_list_dirs, generate_data_loader
from utils.constants import K, BATCH_SIZE, DEVICE

SHAPENET_DIR = "/scratch/rag551/occupancy_networks/ShapeNet"

# SHAPNET_CLASSES = [
#     '02958343'
# ]

SHAPENET_CLASS = '04256520'  # couches


def train(epoch, model, trainloader, optimizer):
    # decoderLoss = nn.BCEWithLogitsLoss(reduction='sum')
    #
    # model.train()
    for batch_idx, data in enumerate(train_loader):
        (pts, occupancies) = data
        # Each batch size contains batch_size sets of "K" points
        pts = pts.view(-1, K, 3, 1).permute(0, 2, 1, 3).cuda()
        print(pts)
        occupancies = occupancies.view(-1, K, 1).cuda()
        print(occupancies)
        # optimizer.zero_grad()
        #
        # pred, z_dist = model(pts)  # a probability for each point, and the dist parameters of latent distribution
        # pred = pred.permute(0, 2, 1, 3).squeeze(-1)
        # # targetNormal = torch.stack((torch.zeros_like(z_dist[0]),torch.ones_like(z_dist[1])))
        # # encloss = encoderLoss(torch.stack(z_dist),targetNormal)
        # mu, log_var = z_dist
        # encloss = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
        # encloss = torch.sum(encloss).mul_(-0.5)
        # # encloss = -0.5*torch.sum(1+z_dist[1] + z_dist[0].pow(2) - z_dist[1].exp())
        # # print(encloss)
        # decloss = decoderLoss(pred, occupancies)
        #
        # loss = (encloss + decloss / K) / batch_size
        # loss.backward()
        # optimizer.step()
        # if batch_idx % 10 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader), 100. * batch_idx / len(train_loader),
        #         loss.item()))
        #     # print(f"Reconstruction Loss {decloss/(K*batch_size)}")
        # if batch_idx % 100 == 0:
        #     print("Saving to model3.pth")
        #     torch.save(model.state_dict(), "unconditional_model3.pth")

if __name__ == '__main__()':
    shapenet_class_dir = os.path.join(SHAPENET_DIR, SHAPENET_CLASS)

    # catalogue all of the directories with the chosen category
    print(f"loading train.lst for dir {shapenet_class_dir}")
    train_loader = generate_data_loader(shapenet_class_dir, 'train.lst')

    # Get the validation data
    print(f"loading val.lst for dir {shapenet_class_dir}")
    validation_loader = generate_data_loader(shapenet_class_dir, 'val.lst')

    train(1, 'test', train_loader, )
