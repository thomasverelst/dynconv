# DynConv - Human pose estimation

This repository contains the implementation of DynConv ( [https://arxiv.org/abs/1912.03203](https://arxiv.org/abs/1912.03203) ) on human pose estimation.
The code is based on the code of Fast Pose Estimation ( [https://github.com/ilovepose/fast-human-pose-estimation.pytorch](https://github.com/ilovepose/fast-human-pose-estimation.pytorch) ). Note that not all features of the original repository are still working (e.g. Tensorboard logging, evaluation with flip augmentation).

Just looking for the CUDA code? https://github.com/thomasverelst/dynconv/blob/master/pose/lib/dynconv/cuda.py

## Installation
### Requirements
The main requirements of this work are:
> python 3.6  
> PyTorch 1.2  
> CuPy

These can be easily installed by creating a Conda environment. Otherwise look up instructions on how to install CuPy.

     conda create -n dynconv_pose python=3.6 cupy=7 -c conda-forge -c pytorch
    
Install Pytorch:

    pip install torch==1.2.0+cu92 torchvision==0.4.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
     
The other requirements (matplotlib, scipy...) can be installed using pip:

    pip install -r requirements.txt

### Dataset
MPII dataset can be downloaded on [http://human-pose.mpi-inf.mpg.de/](http://human-pose.mpi-inf.mpg.de/). 
unzip images and annotations to a folder, using the following structure:

    root
        images/
	        xxxxxxxxx.jpg
	        ...
        mpii_human_pose_v1_u12_1.mat

## Notes
Only tested on a single GPU! If your machine has multiple, add `CUDA_VISIBLE_DEVICES=0` in front of the Python command.

## Testing a pretrained model (accuracy)
Test a 4-stack model with computational budget 12.5%:

    python tools/test.py --cfg experiments/4stack/s0125.yaml DATASET.ROOT /path/to/mpii/root
This should give as output

> | Arch | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Mean |Mean@0.1 |   
> | hourglass_mn | 95.873 | 94.243 | 87.046 | 80.452 | 86.792 | 81.262 | 76.665 | 86.680 | 34.374 |   
> \# PARAMS: 6.885433 M   
> \# FLOPS (MACs): 1.706458359561866 G on 2958 images   
> \# FLOPS (MACs) used percentage per layer: 11, 6, 0, 58, 0, 75, 77, 72, 64, 63, 80, 97, 62, 37, 26, 1, 21, 25, 33, 42, 66, 31, 61, 46, 87, 97, 42, 31, 22, 1, 14, 18, 27, 31, 32, 27, 59, 45, 95, 50, 34, 31, 21, 0, 4, 10, 28, 15, 29, 26, 51, 47, 58, 44, 20, 21, 13,    
> \# conditional FLOPS (MACs) used: 0.9767140558539553G out of 6.157430784G                 (0.15862363542793423 %)

where the Mean number is the PCKh@0.5 (86.68 here), PARAMS the number of parameters in the model, FLOPS is the number of GMacs per image, conditional FLOPS the number of conditional FLOPS executed over the total number of available FLOPS.


## Testing a pretrained model (speed)

Speedtest mode disables unneccessary operations and adds a warmup-phase, especially needed because of changing tensor sizes of DynConv, which can be difficult for Pytorch's CUDA benchmark mode (which selects the optimal algorithm for CUDA backend operations). Speedtest a 4-stack model with computational budget 12.5%:

    python -O tools/speedtest.py --cfg experiments/4stack/s0125.yaml DATASET.ROOT /path/to/mpii/root

This should give as output (here on a NVIDIA GTX 1050 Ti):

> \>> WARMUP  
>\>> updating DynConv Gumbel temperature at epoch 100 to 1.0 (Gumbel noise: False)  
>\# PARAMS  6.885433 M  
>\# FLOPS:  1.7312132685217392 G on 1472 images (batch_count= 1472 )  
>\>> SPEEDTEST  
>ELAPSED TIME: 18.192158330231905s, SAMPLES PER SECOND: 81.68354590068353 ON 1486 SAMPLES
giving 82 samples per second.

## Testing a pretrained model (visualizing ponder cost)

Show the ponder cost maps of a sparse model:

    python tools/test.py --cfg experiments/4stack/s0125.yaml DEBUG.PONDER True DATASET.ROOT /path/to/mpii/root

which generates the computation heatmaps of Figure 9 in the paper.

## Testing other models
See the experiments folder for other modles
    
    b_x10.yaml: large baseline (x1.0 base features)
    b_x050.yaml: small baseline (x0.50 base features)
    s0125.yaml: dynconv with 0.125 target sparsity
    s025.yaml: dynconv with 0.25 target sparsity
    s050.yaml: dynconv with 0.50 target sparsity

## Training a model
Create a new experiment (e.g. s0125_2.yaml) and run:

     python tools/train.py --cfg experiments/4stack/s0125_2.yaml DEBUG.PONDER True DATASET.ROOT /path/to/mpii/root

