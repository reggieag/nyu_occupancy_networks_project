#!/usr/bin/env python
# coding: utf-8

# In[4]:


import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18 as _resnet18
from PIL import Image
import io
import numpy
import random

imageFiles = ["000.jpg","001.jpg", "002.jpg","003.jpg", "004.jpg", "005.jpg", "006.jpg", "007.jpg", "008.jpg",
             "009.jpg", "010.jpg", "011.jpg", "012.jpg", "013.jpg", "014.jpg", "015.jpg", "016.jpg", "017.jpg",
             "018.jpg", "019.jpg", "020.jpg", "023.jpg"]

import pdb

class Block(nn.Module):
    def __init__(self):
        super(Block,self).__init__()
        self.fc1 = nn.Conv1d(256,256,kernel_size=1)
        self.fc2 = nn.Conv1d(256,256,kernel_size=1)
        self.bn1 = nn.BatchNorm1d(256, affine=False, track_running_stats=True)
        self.bn2 = nn.BatchNorm1d(256, affine=False, track_running_stats=True)
        self.gammaLayer1 = nn.Conv1d(256,256,kernel_size=1)
        self.gammaLayer2 = nn.Conv1d(256,256,kernel_size=1)
        self.betaLayer1 = nn.Conv1d(256,256,kernel_size=1)
        self.betaLayer2 = nn.Conv1d(256,256,kernel_size=1)

    def forward(self,y):
        x = y['ex']
        encoding = y['enc']
        gamma = self.gammaLayer1(encoding)
        beta = self.betaLayer1(encoding)
        #First apply Conditional Batch Normalization
        out = gamma*self.bn1(x) + beta
        #Then ReLU activation function
        out = F.relu(out)
        #fully connected layer
        out = self.fc1(out)
        #Second CBN layer
        gamma = self.gammaLayer2(encoding)
        beta = self.betaLayer2(encoding)
        out = gamma*self.bn2(out) + beta
        #RELU activation
        out = F.relu(out)
        #2nd fully connected
        out = self.fc2(out)
        #Add to the input of the ResNet Block
        out = x + out

        return {'ex':out, 'enc':encoding}

class OccupancyModel(nn.Module):
    def __init__(self):
        super(OccupancyModel,self).__init__()
        self.blocks = self.makeBlocks()
        self.encoderModel = _resnet18(pretrained=True)
        self.fc_enc = nn.Linear(1000, 256)
        self.gammaLayer = nn.Conv1d(256,256,kernel_size=1)
        self.betaLayer = nn.Conv1d(256,256,kernel_size=1)
        self.cbn = nn.BatchNorm1d(256, affine=False, track_running_stats=True)
        self.fc1 = nn.Conv1d(3,256,kernel_size=1)
        self.fc2 = nn.Conv1d(256,1,kernel_size=1)

    def makeBlocks(self):
        blocks = []
        for _ in range(5):
            blocks.append(Block())
        return nn.Sequential(*blocks)


    def forward(self,x,img):
        img = self.encoderModel(img)
        img = self.fc_enc(img)
        img = img.view(-1,256,1)
        x = self.fc1(x)
        #5 pre-activation ResNet-blocks
        x = self.blocks({'enc':img , 'ex':x })
        x = x['ex']
        #CBN
        gamma = self.gammaLayer(img)
        beta = self.betaLayer(img)
        x = gamma*self.cbn(x) + beta
        x = F.relu(x)
        x = self.fc2(x)
        x = x.view(-1,1)
        x = torch.sigmoid(x)
        return x


# In[12]:


def generateGrid():
    #The unit cube centered at 0
    #Subdivided into a grid of 32^3 "voxels"
    ncuts = 32
    x = numpy.linspace(-0.5,0.5,ncuts)
    y = numpy.linspace(-0.5,0.5,ncuts)
    z = numpy.linspace(-0.5,0.5,ncuts)
    xg,yg,zg = numpy.meshgrid(x,y,z)
    x = torch.tensor(xg)
    y = torch.tensor(yg)
    z = torch.tensor(zg)
    #Convert to a grid of 3 dimensional coordinate
    tgrid = torch.stack([x,y,z], dim=3).permute(1,0,2,3)
    #A cube is made up the 8 vertices
    #Convert to a list where every 8 coords denote a cube
    gridpts = torch.zeros(8*(ncuts*ncuts*ncuts),3)

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

#One DataSetClass per subdirectory in a category,
#will return grid samples and a single image randomly
#drawn from the 23 available
class MeshTestClass(torch.utils.data.Dataset):
    def __init__(self, d):
        self.dir = d
    def __len__(self):
        return 1

    def __getitem__(self,idx):
        #pick an image randomly to be used an observation
        imageFile = imageFiles[random.randint(0, len(imageFiles)-1)]
        with Image.open(f"{self.dir}/img_choy2016/{imageFile}") as image:
                image = numpy.array(image)
                image = torch.tensor(image,dtype=torch.float)
                #if the image is grey scale, stack 3 to conform dimensions
                if len(image.size()) < 3:
                    image = torch.stack([image, image, image])
                else:
                    image = image.permute(2,0,1)
        return image, generateGrid()


def generate_occ_vals(data, model):
    img, pts = data
    pts = pts.view(-1,3,1)
    occs = model(pts,img)
    torch.save(occs,'predictions.pt' )
    torch.save(pts, 'coords.pt')

is_cuda = (torch.cuda.is_available())
device = torch.device("cuda" if is_cuda else "cpu")


# Get the test Dataset
topdir = "/home/andrea/Documents/GradSchool/OccupancyNetworks/occupancy_networks"
benchesDirectory=f"{topdir}/data/ShapeNet/02828884"
testDirs = []
with io.open(f"{benchesDirectory}/test.lst") as testlist:
    for testdir in testlist.readlines():
        testDirs.append(f"{benchesDirectory}/{testdir.strip()}")

#dataSets = []
#for testDir in testDirs:
#    dataSets.append(DataSetClass(testDir))
#testData= torch.utils.data.ConcatDataset(dataSets)

#A single random mesh
testData = MeshTestClass(testDirs[random.randint(0,len(testDirs)-1)])

testLoader = torch.utils.data.DataLoader(testData, batch_size=1, shuffle=True)

# load the Model
model = OccupancyModel()
modelpath= "/home/andrea/Documents/GradSchool/OccupancyNetworks/nyu_occupancy_networks_project/training/modelcopy.pth"
model.load_state_dict(torch.load(modelpath))
#model.cuda()
model.eval()



for it, data in enumerate(testLoader):
    mesh = generate_occ_vals(data, model)

p = torch.load('predictions.pt')
numpy.savetxt('predictions.txt', p.detach().numpy())

c = torch.load('coords.pt')
c = c.view(-1,3)
numpy.savetxt('coords.txt', c.detach().numpy())
