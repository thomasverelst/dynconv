# import importlib
# import numpy as np
# import torch
# from easydict import EasyDict as edict

# #### CUDA
# def cudafy(cfg, var):
#     '''
#     moves tensor to cuda
#     '''
#     return var.to(cfg.device, non_blocking=True)

# def cudafy_list(cfg, var_list):
#     ''' 
#     moves list of tensors to cuda
#     '''
#     return map(lambda var:cudafy(cfg, var), var_list)

# def importmod(pathname):
#     ''' 
#     imports a file from the given pathname
#     pathname is relative to executing file, but with . instead of normal /
#     '''
#     assert '/' not in pathname
#     t = pathname.split('.')
#     path, name = '.'.join(t[:-1]), t[-1]
#     module = importlib.import_module(path)
#     return eval('module.{}'.format(name))

# def isset(x):
#     '''
#     checks if variable is not None or empty
#     ! does not check whether it really exists/was set
#     TODO should maybe also check for len(x) == 0 and x is not False ?
#     '''
#     return x is not None and not x == ''

# def dict_merge(source, destination):
#     ''' 
#     recursively merges the dict source and dict destination
#     source will overwrite destination if key exists in both
#     '''
#     assert isinstance(source, dict), (type(source), source)
#     assert isinstance(destination, dict), (type(destination), destination)
    
#     for key, value in source.items():
#         if isinstance(value, dict):
#             # get node or create one
#             node = destination.setdefault(key, {})
#             if not isinstance(node, dict): # needed when original is for example float or int and being overwirten with dict
#                 destination[key] = {}
#             dict_merge(edict(value), destination[key])
#         else:
#             destination[key] = value
#     return destination

# ## Checks
# def isTensor(a):
#     ''' check if it is a pytorch tensor '''
#     return torch.is_tensor(a)

# def isNumpy(a):
#     ''' check if it is a numpy array '''
#     return isinstance(a, np.ndarray)

# # # Checkers
# # def can_convert_float(value):
# #   try:
# #     float(value)
# #     return True
# #   except ValueError:
# #     return False

# # def can_convert_int(value):
# #   try:
# #     int(value)
# #     return True
# #   except ValueError:
# #     return False

# # def parsestring(value):
# #     if can_convert_int(value):
# #         return int(value)
# #     if can_convert_float(value):
# #         return float(value)\




# def get_transform(center, scale, res, rot=0):
#     # Generate transformation matrix

#     if isinstance(scale, float) or len(scale) < 2:
#         h0 = 200 * scale
#         h1 = 200 * scale
#     else:
#         h0 = 200 * scale[0]
#         h1 = 200 * scale[1]

#     t = np.zeros((3, 3))
#     t[0, 0] = float(res[0]) / h0   
#     t[1, 1] = float(res[1]) / h1
#     t[0, 2] = res[0] * (-float(center[0]) / h0 + .5)
#     t[1, 2] = res[1] * (-float(center[1]) / h1+ .5)
#     t[2, 2] = 1
#     if not rot == 0:
#         rot = -rot # To match direction of rotation from cropping
#         rot_mat = np.zeros((3,3))
#         rot_rad = rot * np.pi / 180
#         sn,cs = np.sin(rot_rad), np.cos(rot_rad)
#         rot_mat[0,:2] = [cs, -sn]
#         rot_mat[1,:2] = [sn, cs]
#         rot_mat[2,2] = 1
#         # Need to rotate around center
#         t_mat = np.eye(3)
#         t_mat[0,2] = -res[0]/2
#         t_mat[1,2] = -res[1]/2
#         t_inv = t_mat.copy()
#         t_inv[:2,2] *= -1
#         t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
#     return t

# import os.path
# import matplotlib.pyplot as plt
# import cv2
# def save_figure(img, folder, name, vmin=None, vmax=None):
#     if not folder or folder == '':
#         return
#     if torch.is_tensor(img):
#         img = img.cpu().numpy()

#     if not os.path.exists(os.path.join('fig',folder)):
#         os.makedirs(os.path.join('fig',folder))
#     print('saving ', os.path.join('fig',folder, name))
#     if '.' not in name:
#         plt.imsave(os.path.join('fig',folder, name+'.png'), img, vmin=vmin, vmax=vmax)
#         # plt.imsave(os.path.join('fig',folder, name+'.eps'), img)
#         plt.imsave(os.path.join('fig',folder, name+'.svg'), img, vmin=vmin, vmax=vmax)

#         if img.shape[0] < 224 or img.shape[1] < 224:
#             fx = 224/img.shape[0]
#             im_resize = cv2.resize(img, dsize=(0,0), fx=fx, fy=fx, interpolation=cv2.INTER_NEAREST)
#             plt.imsave(os.path.join('fig',folder, name+'_large.svg'), im_resize, vmin=vmin, vmax=vmax)
#             plt.imsave(os.path.join('fig',folder, name+'_large.png'), im_resize, vmin=vmin, vmax=vmax)

#     else:
#         plt.imsave(os.path.join('fig',folder, name), img)

