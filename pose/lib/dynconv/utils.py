import torch
import torch.nn.functional as F

def apply_mask(x, mask):
    mask_hard = mask.hard
    assert mask_hard.shape[0] == x.shape[0]
    assert mask_hard.shape[2:4] == x.shape[2:4], (mask_hard.shape, x.shape)
    return mask_hard.float().expand_as(x) * x

def ponder_cost_map(masks):
    """ takes in the mask list and returns a 2D image of ponder cost """
    if masks is None or len(masks) == 0:
        return None
    assert isinstance(masks, list)
    out = None
    for mask in masks:
        m = mask['std'].hard
        assert m.dim() == 4
        m = m[0]  # only show the first image of the batch
        if out is None:
            out = m
        else:
            out += F.interpolate(m.unsqueeze(0),
                                 size=(out.shape[1], out.shape[2]), mode='nearest').squeeze(0)
    return out.squeeze(0).cpu().numpy()

def cost_per_layer(meta):
    cost, total = torch.tensor(.0).cuda(), torch.tensor(.0).cuda()
    percs = []
    cost = []
    total = []

    for i, mask in enumerate(meta['masks']):
        m_dil = mask['dilate']
        m = mask['std']

        assert  m_dil.hard.dim() == 4
        assert  m_dil.hard.shape[1] == 1
        assert  m.hard.dim() == 4
        assert  m.hard.shape[1] == 1

        c = m_dil.hard.sum(3).sum(2).sum(1) * m_dil.flops_per_position \
            + m.hard.sum(3).sum(2).sum(1) * m.flops_per_position
        c = c.cpu()
        t = torch.tensor(m_dil.hard[0,0].numel()).repeat(m_dil.hard.shape[0]) * m_dil.flops_per_position \
            + torch.tensor(m.hard[0,0].numel()).repeat(m.hard.shape[0]) * m.flops_per_position
        cost.append(c)
        total.append(t)
        percs.append(c/t)
    
    return torch.stack(percs).transpose(0,1), torch.stack(cost).transpose(0,1), torch.stack(total).transpose(0,1)