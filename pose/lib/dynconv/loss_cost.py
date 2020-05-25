# import utils.logger as logger
import math
import torch
import torch.nn as nn


class SparsityCriterion(nn.Module):
    ''' 
    Defines the sparsity loss, consisting of two parts:
    - network loss: MSE between computational budget used for whole network and target 
    - block loss: sparsity (percentage of used FLOPS between 0 and 1) in a block must 
            lie between upper and lower bound. This loss is annealed.
    '''

    def __init__(self, sparsity_target, num_epochs, weight, anneal_speed=0.33):
        super(SparsityCriterion, self).__init__()
        self.sparsity_target = sparsity_target
        self.num_epochs = num_epochs
        self.weight = weight

        # epoch where annealing of upper and lower bound is finished
        self.anneal_finish_epoch = anneal_speed*self.num_epochs


    def forward(self, meta):

        p = meta['epoch'] / self.anneal_finish_epoch
        progress = math.cos(min(max(p, 0), 1) * (math.pi / 2))**2
        upper_bound = (1 - progress*(1-self.sparsity_target))
        lower_bound = progress*self.sparsity_target

        loss_block = torch.tensor(.0).cuda()
        cost, total = torch.tensor(.0).cuda(), torch.tensor(.0).cuda()

        for i, mask in enumerate(meta['masks']):
            m_dil = mask['dilate']
            m = mask['std']

            c = m_dil.active_positions * m_dil.flops_per_position + \
                m.active_positions * m.flops_per_position
            t = m_dil.total_positions * m_dil.flops_per_position + \
                m.total_positions * m.flops_per_position


            layer_perc = c / t
            assert layer_perc >= 0 and layer_perc <= 1, layer_perc
            loss_block += max(0, layer_perc - upper_bound)**2  # upper bound on FLOPS in block
            loss_block += max(0, lower_bound - layer_perc)**2  # lower bound on FLOPS in block

            cost += c
            total += t

        perc = cost/total
        assert perc >= 0 and perc <= 1, perc
        loss_block /= len(meta['masks'])
        loss_network = (perc - self.sparsity_target)**2 # mse of FLOPS over wole network

        sparsity_meta = {}
        sparsity_meta['upper_bound'] = upper_bound
        sparsity_meta['lower_bound'] = lower_bound
        sparsity_meta['cost_perc'] = perc.item()
        sparsity_meta['loss_sp_block'] = loss_block.item()
        sparsity_meta['loss_sp_network'] = loss_network.item()
        meta['sparsity_meta'] = sparsity_meta

        loss_sparsity =  loss_network + loss_block
        return self.weight*loss_sparsity, meta