import math
import matplotlib.pyplot as plt
import utils.utils as utils
import dynconv

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
unnormalize = utils.UnNormalize(mean, std)

def plot_image(input):
    ''' shows the first image of a 4D pytorch batch '''
    assert input.dim() == 4
    plt.figure('Image')
    im = unnormalize(input[0]).cpu().numpy().transpose(1,2,0)
    plt.imshow(im)


def plot_ponder_cost(masks):
    ''' plots ponder cost
    argument masks is a list with masks as returned by the network '''
    assert isinstance(masks, list)
    plt.figure('Ponder Cost')
    ponder_cost = dynconv.ponder_cost_map(masks)
    plt.imshow(ponder_cost, vmin=0, vmax=len(masks))
    plt.colorbar()

def plot_masks(masks):
    ''' plots individual masks as subplots 
    argument masks is a list with masks as returned by the network '''
    nb_mask = len(masks)
    WIDTH = 4
    HEIGHT = math.ceil(nb_mask / 4)
    f, axarr = plt.subplots(HEIGHT, WIDTH)

    for i, mask in enumerate(masks):
        x = i % WIDTH
        y = i // WIDTH

        m = mask['std'].hard[0].cpu().numpy().squeeze(0)

        assert m.ndim == 2
        axarr[y,x].imshow(m, vmin=0, vmax=1)
        axarr[y,x].axis('off')
    
    for j in range(i+1, WIDTH*HEIGHT):
        x = j % WIDTH
        y = j // WIDTH
        f.delaxes(axarr[y,x])

def showKey():
    ''' 
    shows a plot, closable by pressing a key 
    '''
    plt.draw()
    plt.pause(1)
    input("<Hit Enter To Close>")
    plt.clf()
    plt.cla()
    plt.close('all')