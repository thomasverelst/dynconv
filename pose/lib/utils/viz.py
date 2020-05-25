import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms


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

def frame2mpl(im, denormalize=False):
    ''' 
    transforms a given image/matrix into something matplotlib understands
    which is a numpy matrix with dimensions W*H*3 for coloror W*H for greyscale  
    '''
    if torch.is_tensor(im):
        im = im.cpu().numpy()

    if len(im.shape) == 2:
        return im
    assert isinstance(im, np.ndarray), type(im)
    assert len(im.shape) == 3

    # if needed, change to WHC
    if not (im.shape[2] == 1 or im.shape[2] == 3) and (im.shape[0] == 1 or im.shape[0] == 3):
        im = im.transpose(1,2,0)
    assert im.shape[2] == 1 or im.shape[2] == 3, im.shape

    # if greyscale image, just return WH channels
    if im.shape[2] == 1:
        return im[:,:,0]
    
    # standard denormalizatoin
    if denormalize:
        im = torch.from_numpy(im)
        unnormalize = UnNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        im = unnormalize(im.permute(2,0,1)).permute(1,2,0).numpy()
        im = np.clip(im, 0, 1)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    assert im.shape[2] == 3
    return im

def add_skeleton(image, points, joints_vis, thres=0.3):
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170,0,255],[255,0,255]]
    colors = np.asarray(colors)/255

    # 0: right ankle
    # 1: right knee
    # 2: right hip
    # 3: left hip
    # 4: left knee
    # 5: left ankle
    # 6: belly
    # 7: center
    # 8: chin
    # 9: head top
    # 10: right wrist
    # 11: right elbow
    # 12: right shoudler
    # 13: left shoulder
    # 14: left elbow
    # 15:  left wrist

    mapping = ((0,1), (1,2), (2,6), (3,6), (3,4),(4,5),(6,7),(7,8),(8,9),(10,11),(11,12),(12,7),(13,7),(15,14),(14,13))

    image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    stickwidth = 4
    x = points[:, 0]
    y = points[:, 1]

    for i, m in enumerate(mapping):
        j1, j2 = m
        if joints_vis[j1][0] < thres:
            continue
        if joints_vis[j2][0] < thres:
            continue

        x1 = (x[j1])
        y1 = (y[j1])
        x2 = (x[j2])
        y2 = (y[j2])

        cv2.line(image, (x1, y1), (x2, y2), colors[i], stickwidth)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.unnormalize = transforms.Normalize((-self.mean / self.std).tolist(), (1.0 / self.std).tolist())

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        assert tensor.shape[0] == 3
        return self.unnormalize(tensor)
