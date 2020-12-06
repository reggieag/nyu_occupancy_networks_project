# Occupancy Networks
Authors: Andrea Malleo, Reggie Gomez

## Environment Setup
Make sure anaconda is installed.

```bash
conda env create -f requirements.yaml
conda activate occupancy_networks
```

## Dataset
We use renderings and voxelizations from [Choy2016](http://3d-r2n2.stanford.edu/).

```bash
bash scripts/download_choy.sh 
```

## Prince Notes

Request a gpu node
```bash
srun --gres=gpu:1 --pty /bin/bash
```