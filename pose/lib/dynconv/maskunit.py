import torch
import torch.nn as nn
import torch.nn.functional as F


class Mask():
    '''
    Class that holds the mask properties

    hard: the hard/binary mask (1 or 0), 4-dim tensor
    soft (optional): the float mask, same shape as hard
    active_positions: the amount of positions where hard == 1
    total_positions: the total amount of positions 
                        (typically batch_size * output_width * output_height)
    active_positions_list: bookkeeping (see function below)
    active_positions_list_inverted: bookkeeping (see function below)
    '''
    def __init__(self, hard, soft=None):
        assert hard.dim() == 4
        assert hard.shape[1] == 1
        assert soft is None or soft.shape == hard.shape

        self.hard = hard
        self.active_positions = torch.sum(hard) # this must be kept backpropagatable!
        self.total_positions = hard.numel()
        self.soft = soft
        
        self.active_positions_list = None
        self.active_positions_list_inverted = None

        self.flops_per_position = 0
    
    def size(self):
        return self.hard.shape

    def __repr__(self):
        return f'Mask with {self.active_positions}/{self.total_positions} positions, and {self.flops_per_position} accumulated FLOPS per position'

class MaskUnit(nn.Module):
    ''' 
    Generates the mask and applies the gumbel softmax trick 
    '''

    def __init__(self, channels, stride=1, dilate_stride=1):
        super(MaskUnit, self).__init__()
        # self.maskconv = Squeeze(channels=channels, stride=stride)
        self.maskconv = nn.Conv2d(channels, 1, stride=stride, kernel_size=1, padding=0)
        self.gumbel = Gumbel()
        self.expandmask = ExpandMask(stride=dilate_stride)

    def forward(self, x, meta):
        soft = self.maskconv(x)
        hard = self.gumbel(soft, meta['gumbel_temp'], meta['gumbel_noise'])
        mask = Mask(hard, soft)

        hard_dilate = self.expandmask(mask.hard)
        mask_dilate = Mask(hard_dilate)
        
        mask.active_positions_list = make_active_positions_list(mask.hard)
        mask_dilate.active_positions_list = make_active_positions_list(mask_dilate.hard)
        mask_dilate.active_positions_list_inverted = make_active_positions_list_inverted(mask_dilate.hard, mask_dilate.active_positions_list)
        
        m = {'std': mask, 'dilate': mask_dilate}
        if meta['save_masks']:
            meta['masks'].append(m)
        return m, meta

@torch.jit.script
def make_active_positions_list(mask_bool):
    ''' 
    returns the positions to be executed,
    determined by mask_bool, as a vector,
    where each element is the linearized/flattened position
    in the mask_bool tensor.
    The returned active_positions_list is needed for efficient
    gather/scatter/dw-conv in CUDA.
    '''
    return torch.nonzero(mask_bool.flatten()).squeeze(1).int()

@torch.jit.script
def make_active_positions_list_inverted(mask_bool, active_positions_list):
    '''
    returns a tensor of the same shape of mask_bool
    where every element/position gives the flattened/linearized position of the corresponding
    value in the given active_positions_list. If the position was not in
    active_positions_list, a random number is inserted.
    Needed as lookup table in the efficient depthwise convolution/
    '''
    active_positions_list_inverted = torch.empty(mask_bool.shape, dtype=torch.int, device='cuda')
    active_positions_list_inverted = active_positions_list_inverted.masked_scatter_(mask_bool, torch.arange(len(active_positions_list), device='cuda', dtype=torch.int) )
    return active_positions_list_inverted

## Gumbel

class Gumbel(nn.Module):
    ''' 
    Returns differentiable discrete outputs. Applies a Gumbel-Softmax trick on every element of x. 
    '''
    def __init__(self, eps=1e-8):
        super(Gumbel, self).__init__()
        self.eps = eps

    def forward(self, x, gumbel_temp=1.0, gumbel_noise=True):
        if not self.training:  # no Gumbel noise during inference
            return (x >= 0).float()

        if gumbel_noise:
            eps = self.eps
            U1, U2 = torch.rand_like(x), torch.rand_like(x)
            g1, g2 = -torch.log(-torch.log(U1 + eps)+eps), - \
                torch.log(-torch.log(U2 + eps)+eps)
            x = x + g1 - g2

        soft = torch.sigmoid(x / gumbel_temp)
        hard = ((soft >= 0.5).float() - soft).detach() + soft

        assert not torch.any(torch.isnan(hard))
        return hard

class Squeeze(nn.Module):
    """ 
    Squeeze module to predict masks 
    """

    def __init__(self, channels, stride=1):
        super(Squeeze, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, 1, bias=True)
        self.conv = nn.Conv2d(channels, 1, stride=stride,
                              kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, 1, 1, 1)
        z = self.conv(x)
        return z + y.expand_as(z)

class ExpandMask(nn.Module):
    '''
    dilates a given boolean tensor
    if stride > 1,
        the output width/height is multiplied by stride
    '''
    def __init__(self, stride, padding=1): 
        super(ExpandMask, self).__init__()
        self.stride=stride
        self.padding = padding
        if self.stride > 1:
            self.pad_kernel = torch.zeros( (1,1,self.stride, self.stride), device='cuda')
            self.pad_kernel[0,0,0,0] = 1
        self.dilate_kernel = torch.ones((1,1,1+2*self.padding,1+2*self.padding), device='cuda')

    def forward(self, x):
        assert x.shape[1] == 1
        x = x.float()
        if self.stride > 1:
            x = F.conv_transpose2d(x, self.pad_kernel, stride=self.stride, groups=x.size(1))
        x = F.conv2d(x, self.dilate_kernel, padding=self.padding, stride=1)
        return x > 0.5
