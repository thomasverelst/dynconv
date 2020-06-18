# DynConv

This repository contains the implementation of DynConv ( [https://arxiv.org/abs/1912.03203](https://arxiv.org/abs/1912.03203) ) on classification. For simplicity, the code on classification does not include CUDA optimization for faster inference. To get that part, check the code on human pose estimation ( [https://github.com/thomasverelst/dynconv/tree/master/pose](https://github.com/thomasverelst/dynconv/tree/master/pose) ). This version just masks out spatial positions (e.g. similar to other works like Spatially Adaptive Computation Time).



## Installation
### Requirements
The main requirements of this work are:
> Python 3.6  
> PyTorch 1.2  

These can be easily installed by creating a Conda environment. Otherwise look up instructions on how to install CuPy.

    conda create -n dynconv python=3.6 pytorch=1.2 torchvision cupy tensorboardX=1.6 -c conda-forge -c pytorch
The other requirements (matplotlib, tqdm, numpy...) can be installed using pip:

    pip install -r requirements.txt

### Trained models
Our trained models can be found here: [Microsoft OneDrive link](https://1drv.ms/u/s!ApImBF1PK3gnjoocGrSm908HR9-xuw?e=BWkRZF)

Unzip and place them into the root of the classification folder, so the folder structure looks as follows:
> ./main_cifar.py  
> ./main_imagenet.py  
> ./exp/cifar/resnet32/...  
> ./exp/cifar/resnet26/...  
> ./exp/imagenet/...  
> ...  


### Evaluate a trained sparse DynConv network

    python main_cifar.py --model resnet32 -r exp/cifar/resnet32/sparse07/checkpoint_best.pth --budget 0.7 -e

Note that the `-e` flag sets evaluation mode. Note that the budget gives the amount of desired conditional computations (between 0 and 1). Setting it to -1 results in a non-adaptive model (standard ResNet baseline, no DynConv).

This should output:
>\* Epoch 347 - Prec@1 93.580
>\* FLOPS (multiply-accumulates, MACs) per image:  55.428526 MMac

or test a more sparse model:

    python main_cifar.py --model resnet32 -r exp/cifar/resnet32/sparse03/checkpoint_best.pth --budget 0.3 -e

use the `--plot_ponder` flag to visualize the ponder cost maps (computation heatmaps)

    python main_cifar.py --model resnet32 -r exp/cifar/resnet32/sparse07/checkpoint_best.pth --budget 0.7 --plot_ponder -e


### Evaluate a pretrained baseline

    python main_cifar.py --model resnet32 -r exp/cifar/resnet32/base/checkpoint_best.pth -e

smaller models are Resnet 26, 20, 14 and 8, e.g.
    
    python main_cifar.py --model resnet20 -r exp/cifar/resnet20/base/checkpoint_best.pth -e

### Train a sparse network:

    python main_cifar.py --model resnet32 --save_dir exp/your_new_run -r exp/your_new_run --budget 0.5

### Train a baseline network:

    python main_cifar.py --model resnet32 --save_dir exp/your_newer_run -r exp/your_newer_run --budget -1



## ResNet-101 on ImageNet-1K


### Evaluate a pretrained sparse DynConv network

    python main_imagenet.py --batchsize 64 -r exp/imagenet/resnet101/sparse03/checkpoint_best.pth --budget 0.3 -e
    
should result in 

>\* Epoch 97 - Prec@1 75.710
>\* FLOPS (multiply-accumulates, MACs) per image:  2997.180928 MMac

use the `--plot_ponder` flag to visualize the ponder cost maps (computation heatmaps):
    python main_imagenet.py --batchsize 64 -r exp/imagenet/resnet101/sparse03/checkpoint_best.pth --budget 0.3 -e --plot_ponder


Likewise:

    python main_imagenet.py --batchsize 64 -r exp/imagenet/resnet101/sparse05/checkpoint_best.pth --budget 0.5 -e    

    python main_imagenet.py --batchsize 64 -r exp/imagenet/resnet101/sparse07/checkpoint_best.pth --budget 0.7 -e

    python main_imagenet.py --batchsize 64 -r exp/imagenet/resnet101/sparse08/checkpoint_best.pth --budget 0.8 -e



### Evaluate a pretrained baseline

    python main_imagenet.py --batchsize 64 -r exp/imagenet/resnet101/base/checkpoint_best.pth -e
