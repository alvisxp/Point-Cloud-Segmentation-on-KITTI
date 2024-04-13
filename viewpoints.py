import open3d
import argparse
import os
import json
import cv2
import yaml
import colorsys
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import my_log as log

from model.pointnet import PointNetSeg, feature_transform_reguliarzer
from model.utils import load_pointnet

from train_eval import parse_args
from data_utils.SemKITTI_Loader import pcd_normalize
from data_utils.kitti_utils import Semantic_KITTI_Utils

KITTI_ROOT = os.environ['KITTI_ROOT']

class Window_Manager():
    def __init__(self):
        self.param = open3d.io.read_pinhole_camera_parameters('config/ego_view.json')
        self.vis = open3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(width=800, height=800, left=100)
        self.vis.register_key_callback(32, lambda vis: exit())
        self.vis.get_render_option().load_from_json('config/render_option.json')
        self.pcd = open3d.geometry.PointCloud()
    
    def update(self, pts_3d, colors):
        self.pcd.points = open3d.utility.Vector3dVector(pts_3d)
        self.pcd.colors = open3d.utility.Vector3dVector(colors/255)
        self.vis.remove_geometry(self.pcd)
        self.vis.add_geometry(self.pcd)
        self.vis.get_view_control().convert_from_pinhole_camera_parameters(self.param)
        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

    #def capture_screen(self,fn):
      #  self.vis.capture_screen_image(fn, True)

def vis(args):
    part = args.sequence
    args.subset ='inview'
    args.model_name = 'pointnet'

    kitti_utils = Semantic_KITTI_Utils(KITTI_ROOT, subset=args.subset)

    vis_handle = Window_Manager()
    if args.model_name == 'pointnet':
        args.pretrain = 'experiment/pointnet/pointnet_026.pth'
    else:
        args.pretrain = 'experiment/pointnet/pointnet_026.pth'

    model = load_pointnet(args.model_name, kitti_utils.num_classes, args.pretrain)

    for index in range(0, kitti_utils.get_max_index(part)):
        point_cloud, label = kitti_utils.get(part, index, load_image=True)
        length = point_cloud.shape[0]
        npoints = 20000
        choice = np.random.choice(length, npoints, replace=True)
        point_cloud = point_cloud[choice]
        label = label[choice]

        pts_3d = point_cloud[:,:3]
        pcd = pcd_normalize(point_cloud)

        with log.Tick():
            points = torch.from_numpy(pcd).unsqueeze(0).transpose(2, 1).cuda()
            
            with torch.no_grad():
                if args.model_name == 'pointnet':
                    logits, _ = model(points)
                else:
                    logits = model(points)
                pred = logits[0].argmax(-1).cpu().numpy()

        pts_2d = kitti_utils.torch_project_3d_to_2d(pts_3d)

        vis_handle.update(pts_3d, kitti_utils.colors[pred])
        sem_img = kitti_utils.draw_2d_points(pts_2d, kitti_utils.colors_bgr[pred])

        cv2.imshow('Semantic Seg', sem_img)
        cv2.imshow('RGB Image', cv2.cvtColor(kitti_utils.frame,cv2.COLOR_BGR2RGB))
        if 32 == cv2.waitKey(1):
            break

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    vis(args)
