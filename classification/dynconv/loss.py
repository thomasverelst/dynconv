import utils.logger as logger
import math
import torch
import torch.nn as nn


class SparsityCriterion(nn.Module):
    ''' 
    Defines the sparsity loss, consisting of two parts:
    - network loss: MSE between computational budget used for whole network and target 
    - block loss: sparsity (percentage of used FLOPS between 0 and 1) in a block must lie between upper and lower bound. 
    This loss is annealed.
    '''

    def __init__(self, sparsity_target, num_epochs):
        super(SparsityCriterion, self).__init__()
        self.sparsity_target = sparsity_target
        self.num_epochs = num_epochs

    def forward(self, meta):

        p = meta['epoch'] / (0.33*self.num_epochs)
        progress = math.cos(min(max(p, 0), 1) * (math.pi / 2))**2
        upper_bound = (1 - progress*(1-self.sparsity_target))
        lower_bound = progress*self.sparsity_target

        dicts = [meta[k] for k in ['masks', 'masks_dilate', 'flops_per_position','flops_per_position_dilate']]
        for i, (mask, mask_dilate, flops, flops_dilate) in enumerate(zip(*dicts)):
            if i == 0:
                loss_block = torch.tensor(.0).to(device=mask.device)
                cost, total = torch.tensor(.0).to(device=mask.device), torch.tensor(.0).to(device=mask.device)
            batchsize = mask.shape[0]
            mask_active_positions = mask.view(batchsize, -1).sum(1)
            mask_dilate_active_positions = mask_dilate.view(batchsize, -1).sum(1)

            c = torch.sum(mask_active_positions * flops) + \
                torch.sum(mask_dilate_active_positions * flops_dilate)
            t = torch.sum(mask[0].numel() * flops) + \
                torch.sum(mask_dilate[0].numel() * flops_dilate)

            layer_perc = c / t
            logger.add('layer_perc_'+str(i), layer_perc.item())
            assert layer_perc >= 0 and layer_perc <= 1, layer_perc
            loss_block += max(0, layer_perc - upper_bound)**2  # upper bound
            loss_block += max(0, lower_bound - layer_perc)**2  # lower bound

            cost += c
            total += t

        perc = cost/total
        assert perc >= 0 and perc <= 1, perc
        loss_block /= len(meta['masks'])
        loss_network = (perc - self.sparsity_target)**2

        logger.add('upper_bound', upper_bound)
        logger.add('lower_bound', lower_bound)
        logger.add('cost_perc', perc.item())
        logger.add('loss_sp_block', loss_block.item())
        logger.add('loss_sp_network', loss_network.item())
        return loss_network + loss_block