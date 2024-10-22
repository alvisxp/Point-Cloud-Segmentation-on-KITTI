import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .pointnet import PointNetSeg, feature_transform_reguliarzer

def load_pointnet(model_name, num_classes, fn_pth):
    if model_name == 'pointnet':
        model = PointNetSeg(num_classes, input_dims = 4, feature_transform=True)
    else:
        model = PointNet2SemSeg(num_classes, feature_dims = 1)

    torch.backends.cudnn.benchmark = True
    model = torch.nn.DataParallel(model)

    assert fn_pth is not None,'No pretrain model'
    if not torch.cuda.is_available():
        print('=> cuda not available')
        checkpoint = torch.load(fn_pth, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(fn_pth)
        model.cuda()
    
    model.load_state_dict(checkpoint)
    model.eval()
    return model
