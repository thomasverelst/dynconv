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

        loss_block = torch.tensor(.0).to(device=meta['device'])
        cost, total = torch.tensor(.0).to(device=meta['device']), torch.tensor(.0).to(device=meta['device'])

        for i, mask in enumerate(meta['masks']):
            m_dil = mask['dilate']
            m = mask['std']

            c = m_dil.active_positions * m_dil.flops_per_position + \
                m.active_positions * m.flops_per_position
            t = m_dil.total_positions * m_dil.flops_per_position + \
                m.total_positions * m.flops_per_position

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