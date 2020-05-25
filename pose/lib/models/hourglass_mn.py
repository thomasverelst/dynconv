'''
Hourglass network inserted in the pre-activated Resnet 
Use lr=0.01 for current version
(c) YANG, Wei 

modified by Thomas Verelst
ESAT-PSI, KU LEUVEN, 2020
'''

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import dynconv



BN_MOMENTUM = 0.1

class InvertedResidual(nn.Module):
    expansion = 2
    def __init__(self, cfg, inp, oup, stride=1, expand_ratio=6, sparse=False):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        self.expand_ratio = expand_ratio
        self.sparse = sparse
        print(f'Inverted Residual - sparse: {sparse}: inp {inp}, hidden_dim {hidden_dim}, ' + 
              f'oup {oup}, stride {stride}, expand_ratio {expand_ratio}')
        
        layers = []
        if expand_ratio != 1:
            layers.extend([
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim, momentum=BN_MOMENTUM),
                nn.ReLU6(inplace=True)
            ])
        layers.extend([
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim, momentum=BN_MOMENTUM),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup, momentum=BN_MOMENTUM),
        ])
        self.conv = nn.Sequential(*layers)
        
        if sparse:
            assert self.identity
            assert expand_ratio != 1
            self.masker = dynconv.MaskUnit(inp, stride=stride, dilate_stride=stride)
        else:
            self.masker = None

    def forward(self, v):
        x, meta = v
        if not self.sparse:
            out = self.conv(x)
            if self.identity:
                out += x
            return out, meta
        else:    
            assert self.identity and self.expand_ratio != 1
            m, meta = self.masker(x, meta)
            mask, mask_dilate = m['std'], m['dilate']

            fast_inference = not self.training

            out = x.clone() # clone should not be needed, but otherwise seems to be bugged
            if fast_inference:
                x = dynconv.gather(x, mask_dilate)
            
            x = dynconv.conv1x1(   self.conv[0],               x, mask_dilate,       fast=fast_inference)
            x = dynconv.bn_relu(   self.conv[1], self.conv[2], x, mask_dilate,       fast=fast_inference)
            x = dynconv.conv3x3_dw(self.conv[3],               x, mask_dilate, mask, fast=fast_inference)
            x = dynconv.bn_relu(   self.conv[4], self.conv[5], x, mask,              fast=fast_inference)
            x = dynconv.conv1x1(   self.conv[6],               x, mask,              fast=fast_inference)
            x = dynconv.bn_relu(   self.conv[7], None,         x, mask,              fast=fast_inference)
            
            if fast_inference:
                out = dynconv.scatter(x, out, mask, sum_out=True)
            else:
                out = out + dynconv.apply_mask(x, mask)
            return out, meta




class Hourglass(nn.Module):
    def __init__(self, cfg, block, num_blocks, planes, depth, sparse):
        super(Hourglass, self).__init__()
        self.cfg = cfg
        self.sparse = sparse
        self.depth = depth
        self.block = block
        self.upsample = nn.Upsample(scale_factor=2)
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for _ in range(0, num_blocks):
            layers.append(block(self.cfg, planes*block.expansion, planes*2, sparse=self.sparse))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for _ in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, v):
        x, masks = v
        up1, masks = self.hg[n-1][0]( (x, masks))
        low1 = F.max_pool2d(x, 2, stride=2)
        low1, masks = self.hg[n-1][1]( (low1, masks) )

        if n > 1:
            low2, masks = self._hour_glass_forward(n-1, (low1,masks))
        else:
            low2, masks = self.hg[n-1][3]( (low1, masks))
        low3, masks = self.hg[n-1][2]( (low2, masks))
        up2 = self.upsample(low3)
        out = up1 + up2
        return out, masks

    def forward(self, v):
        return self._hour_glass_forward(self.depth, v)


class HourglassNet(nn.Module):
    '''Hourglass model from Newell et al ECCV 2016'''
    def __init__(self, block, cfg, **kwargs):
        super(HourglassNet, self).__init__()
        # Parameters: num_feats=256, num_stacks=8, num_blocks=1, num_classes=16
        extra = cfg.MODEL.EXTRA
        num_feats = extra.NUM_FEATURES
        num_stacks = extra.NUM_STACKS
        num_blocks = extra.NUM_BLOCKS
        num_classes = cfg.MODEL.NUM_JOINTS
        self.sparse = cfg.DYNCONV.ENABLED
        self.cfg = cfg

        self.inplanes = int(num_feats/4) 
        self.num_feats = int(num_feats/2)  
        self.num_stacks = num_stacks  
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes, momentum=BN_MOMENTUM) 
        self.relu = nn.ReLU(inplace=True)
        # Parameters: planes=64, blocks=1, stride=1, output channels = 128
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # Build hourglass modules
        ch = self.num_feats*block.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(num_stacks):
            # Parameters: num_blocks=4, self.num_feats=128
            hg.append(Hourglass(self.cfg, block, num_blocks, self.num_feats, 4, sparse=self.sparse))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score.append(nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))
            if i < num_stacks-1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1, bias=True))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_) 
        self.score_ = nn.ModuleList(score_)

    def _make_residual(self, block, planes, num_blocks, stride=1):
        layers = []
        layers.append(block(self.cfg, self.inplanes, planes*2, stride=stride, sparse=(self.sparse and stride==1 and self.inplanes == planes*2)))
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.cfg, self.inplanes, planes*2, sparse=self.sparse))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(
                conv,
                bn,
                self.relu,
            )

    def forward(self, x, meta):
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x) 

        x,meta = self.layer1((x,meta))  
        x = self.maxpool(x)
        x,meta = self.layer2((x,meta))  
        x,meta = self.layer3((x,meta))  

        for i in range(self.num_stacks):
            y, meta = self.hg[i]( (x,meta) )
            y, meta = self.res[i]( (y,meta) )
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)
            if i < self.num_stacks-1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_
        return out, meta


def get_pose_net(cfg, is_train, **kwargs):
    model = HourglassNet(InvertedResidual, cfg, **kwargs)
    return model
