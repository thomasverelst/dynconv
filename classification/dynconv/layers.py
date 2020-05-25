import torch
import torch.nn as nn
import torch.nn.functional as F

# these wrappers register the FLOPS of each layer, which
# will be used in the sparsity criterion to restrict the 
# amount of executed conditoinal operations. 
# In the pose repository, these wrappers are also used to
# efficiently execute sparse layers.


## CONVOLUTIONS

def conv1x1(conv_module, x, mask, fast=False):
    w = conv_module.weight.data
    mask.flops_per_position += w.shape[0]*w.shape[1]
    conv_module.__mask__ = mask
    return conv_module(x)

def conv3x3_dw(conv_module, x, mask_dilate, mask, fast=False):
    w = conv_module.weight.data
    mask.flops_per_position += w.shape[0]*w.shape[1]*w.shape[2]*w.shape[3]
    conv_module.__mask__ = mask
    return conv_module(x)

def conv3x3(conv_module, x, mask_dilate, mask, fast=False):
    w = conv_module.weight.data
    mask.flops_per_position += w.shape[0]*w.shape[1]*w.shape[2]*w.shape[3]
    conv_module.__mask__ = mask
    return conv_module(x)


## BATCHNORM and RELU
def bn_relu(bn_module, relu_module, x, mask, fast=False):
    bn_module.__mask__ = mask
    if relu_module is not None:
        relu_module.__mask__ = mask

    x = bn_module(x)
    x = relu_module(x) if relu_module is not None else x
    return x

