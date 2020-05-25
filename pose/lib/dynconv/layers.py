import torch
import torch.nn as nn
import torch.nn.functional as F
import dynconv.cuda


## CONVOLUTIONS

def conv1x1(conv_module, x, mask, fast=False):
    # Conv1x1 gets slow when batch size is large, using linear per ppixel is equivalent 
    w = conv_module.weight.data
    mask.flops_per_position += w.shape[0]*w.shape[1]
    conv_module.__mask__ = mask
    if not fast:
        return conv_module(x)
    else:
        assert x.shape[2] == 1 and x.shape[3] == 1, x.shape
        assert not conv_module.training
        w = conv_module.weight.data
        bias = conv_module.bias.data if conv_module.bias else None
        assert w.shape[2] == 1, 'expected kernel size 1'
        assert w.shape[3] == 1, 'expected kernel size 1'
        
        z = x.squeeze(3).squeeze(2)
        out = F.linear(z, w.squeeze(3).squeeze(2), bias).unsqueeze(2).unsqueeze(3)

        # manually register FLOPS
        if hasattr(conv_module, '__flops_handle__'):
            conv_module.__flops_function__(conv_module, x, out)
        return out

def conv3x3_dw(conv_module, x, mask_dilate, mask, fast=False):
    w = conv_module.weight.data
    mask.flops_per_position += w.shape[0]*w.shape[1]*w.shape[2]*w.shape[3]
    conv_module.__mask__ = mask

    if not fast:
        return conv_module(x)
    else:
        assert mask is not None
        assert x.shape[2] == 1 and x.shape[3] == 1, x.shape
        assert not conv_module.training
        assert conv_module.bias is None
        assert w.shape[1] == 1, 'expected depthwise convolution'
        assert w.shape[2] == 3, 'expected kernel size 3'
        assert w.shape[3] == 3, 'expected kernel size 3'

        out = dynconv.cuda.masked_conv_dw(
            x, w, mask.active_positions_list, mask_dilate.active_positions_list_inverted, mask.size(), 
            conv_module.stride, padding=conv_module.padding, dilation=conv_module.dilation
        )

        # manually register FLOPS
        if hasattr(conv_module, '__flops_handle__'):
            conv_module.__flops_function__(conv_module, x, out)
        return out

def conv3x3(conv_module, x, mask_dilate, mask, fast=False):
    w = conv_module.weight.data
    mask.flops_per_position += w.shape[0]*w.shape[1]*w.shape[2]*w.shape[3]
    conv_module.__mask__ = mask
    if not fast:
        return conv_module(x)
    else:
        raise NotImplementedError('Fast 3x3 conv not implemented')




## BATCHNORM and RELU

def bn_relu(bn_module, relu_module, x, mask, fast=False):
    bn_module.__mask__ = mask
    if relu_module is not None:
        relu_module.__mask__ = mask
    if not fast:
        x = bn_module(x)
        x = relu_module(x) if relu_module is not None else x
        return x
    else:
        assert x.shape[2] == 1 and x.shape[3] == 1
        assert not bn_module.training
        assert relu_module is None or not relu_module.training
        relu = isinstance(relu_module, nn.ReLU)
        relu6 = isinstance(relu_module, nn.ReLU6)
        
        out = dynconv.cuda.batchnorm(x, bn_module.weight.data, bn_module.bias.data, bn_module.running_mean, bn_module.running_var, 
            relu=relu, relu6=relu6, hswish=False)
        
        # manually register FLOPS
        if hasattr(bn_module, '__flops_handle__'):
            bn_module.__flops_function__(bn_module, x, out)
        if relu_module is not None and hasattr(relu_module, '__flops_handle__'):
            relu_module.__flops_function__(relu_module, x, out)
        return out

