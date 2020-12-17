# Occupancy Networks
Authors: Andrea Malleo, Reggie Gomez

# Results

## Single Image Mesh Reconstruction of Benches
<p float="left">
<img src="/report/benchImages/pervertexbench.gif" width="350" height="350"/> 
<img src="/report/benchImages/wirebench.gif" width="350" height="350"/>
</p>

## Point Completion of Cell Phone
<p float="left">
<img src="/report/phoneImages/phone_pervertex.gif" width="350" height="350"/>
<img src="/report/phoneImages/phone_wireframe.gif" width="350" height="350"/>
</p>


## Interpolation in Latent Variable Space of Couches
<p float="left">
<img src="/report/latentInterpGifs/latentInterp_front_threshold3.gif" width="350" height="350"/> 
<img src="/report/latentInterpGifs/latentInterp_side_threshold3.gif" width="350" height="350"/>
</p>

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
srun --gres=gpu:1 --time=02:00:00 --mem=8GB --pty /bin/bash
```
