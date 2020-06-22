import torch.nn.functional as F

META_KEYS = ('masks', 'masks_dilate', 'flops_per_position', 'flops_per_position_dilate')

def add_meta(meta):
    for k in META_KEYS:
        meta[k] = []

def get_output_meta(meta):
    return {k:meta[k] for k in META_KEYS}

def apply_mask(x, mask):
    assert mask.shape[0] == x.shape[0]
    assert mask.shape[2:4] == x.shape[2:4], (mask.shape, x.shape)
    return mask.float().expand_as(x) * x

def ponder_cost_map(masks):
    """ takes in the mask list and returns a 2D image of ponder cost """
    assert isinstance(masks, list)
    out = None
    for mask in masks:
        assert mask.dim() == 4
        mask = mask[0]  # only show the first image of the batch
        if out is None:
            out = mask
        else:
            out += F.interpolate(mask.unsqueeze(0),
                                 size=(out.shape[1], out.shape[2]), mode='nearest').squeeze(0)
    return out.squeeze(0).cpu().numpy()
