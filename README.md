# Occupancy Networks
Authors: Andrea Malleo, Reggie Gomez

# Results

## Single Image Mesh Reconstruction of Benches
![benchResult1](https://github.com/reggieag/nyu_occupancy_networks_project/blob/master/report/benchImages/pervertexbench.gif)
![benchResult2](https://github.com/reggieag/nyu_occupancy_networks_project/blob/master/report/benchImages/wirebench.gif)

## Interpolation in Latent Variable Space of Couches

![couchInterp1](https://github.com/reggieag/nyu_occupancy_networks_project/blob/master/report/latentInterpGifs/latentInterp_front_threshold3.gif)
![couchInterp2](https://github.com/reggieag/nyu_occupancy_networks_project/blob/master/report/latentInterpGifs/latentInterp_side_threshold3.gif)

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

## Prince Notes

Request a gpu node
```bash
srun --gres=gpu:1 --time=02:00:00 --mem=8GB --pty /bin/bash
```
