#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os.path as ops
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import torch
from torch.autograd import Variable
from PIL import Image, ImageOps
import cv2
import json
import numbers
import random
import warnings
from scipy.optimize import fsolve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torch.utils.data.dataloader import default_collate
warnings.simplefilter('ignore', np.RankWarning)


class LaneDataset(Dataset):
    """Dataset with labeled lanes"""
    def __init__(self, dataset_base_dir, dataset_info_file, args):
        """

        :param dataset_info_file: json file list
        """

        self.h_org = args.org_h
        self.w_org = args.org_w
        self.h_crop = args.crop_size
        self.h_net = args.resize
        self.w_net = args.resize*2
        self.h_ipm = args.ipm_h
        self.w_ipm = args.ipm_w
        self.num_y_anchor = args.num_y_anchor
        self.y_ref = args.y_ref
        self.x_ratio = float(self.w_net) / float(self.w_org)
        self.y_ratio = float(self.h_net) / float(self.h_org - self.h_crop)
        self.top_view_region = args.top_view_region
        # initialize the homography between image and IPM, this need to be exactly the same as network
        org_img_size = [args.org_h, args.org_w]
        resize_img_size = [args.resize, 2*args.resize]
        pitch = np.pi / 180 * args.pitch
        M, M_inv = self._Init_Projective_tranform(args.top_view_region, org_img_size, args.crop_size, resize_img_size,
                                                  pitch, args.cam_height, args.K)
        self.M = M

        self.n_samples = 0
        self._label_image_path, self._label_lane_pts_all, self._label_lane_types_all = self._init_dataset(dataset_base_dir, dataset_info_file)
        self._random_dataset()

    def _init_dataset(self, dataset_base_dir, dataset_info_file):
        """
        :param dataset_info_file:
        :return: image paths, labels in normalized net input coordinates

        data processing:
        ground truth labels map are scaled wrt network input sizes
        """

        # load image path, and lane pts
        label_image_path = []
        gt_lane_pts_all = []
        gt_lane_types_all = []

        for json_file_path in dataset_info_file:
            assert ops.exists(json_file_path), '{:s} not exist'.format(json_file_path)

            # src_dir = ops.split(json_file_path)[0]

            with open(json_file_path, 'r') as file:
                for line in file:
                    info_dict = json.loads(line)

                    image_path = ops.join(dataset_base_dir, info_dict['raw_file'])
                    assert ops.exists(image_path), '{:s} not exist'.format(image_path)

                    label_image_path.append(image_path)

                    gt_lane_pts = info_dict['lanes']
                    # rescale to net input
                    for i, lane in enumerate(gt_lane_pts):
                        # A GT lane can be either 2D or 3D
                        # if a GT lane is 3D, the height is intact from 3D GT, so keep it intact here too
                        lane = np.array(lane)
                        lane[:, 0] *= self.x_ratio
                        lane[:, 1] = (lane[:, 1] - self.h_crop) * self.y_ratio
                        # normalize 2D attributes
                        lane[:, 0] /= self.w_net
                        lane[:, 1] /= self.h_net
                        gt_lane_pts[i] = lane
                    gt_lane_types = info_dict['lane_types']

                    gt_lane_pts_all.append(gt_lane_pts)
                    gt_lane_types_all.append(np.array(gt_lane_types))

                    self.n_samples += 1
        label_image_path = np.array(label_image_path)
        return label_image_path, gt_lane_pts_all, gt_lane_types_all

    def _random_dataset(self):
        """

        :return:
        """

        random_idx = np.random.permutation(self._label_image_path.shape[0])
        self._label_image_path = self._label_image_path[random_idx]
        self._label_lane_types_all = [self._label_lane_types_all[i] for i in random_idx]
        self._label_lane_pts_all = [self._label_lane_pts_all[i] for i in random_idx]
        self._next_batch_loop_count = 0

    # compute normalized transformation matrix for a top-view region boundaries
    def _Init_Projective_tranform(self, top_view_region, org_img_size, crop_y, resize_img_size, pitch, cam_height, K):
        """
            Compute the normalized transformation (M_inv) such that image region corresponding to top_view region maps to
            the top view image's 4 corners
            Ground coordinates: x-right, y-forward, z-up

        :param top_view_region: a 4 X 2 list of (X, Y) indicating the top-view region corners in order:
                                top-left, top-right, bottom-left, bottom-right
        :param org_img_size: the size of original image size: (h, w)
        :param crop_y: pixels croped from original img
        :param resize_img_size: the size of image as network input: (h, w)
        :param pitch: camera pitch angle wrt ground plane
        :param cam_height: camera height wrt ground plane in meters
        :param K: camera intrinsic parameters
        :return: M_inv: the normalized transformation from image to IPM image
        """

        # transform top-view region to original image region
        R_g2c = np.array([[1, 0, 0],
                          [0, np.cos(-np.pi / 2 - pitch), -np.sin(-np.pi / 2 - pitch)],
                          [0, np.sin(-np.pi / 2 - pitch), np.cos(-np.pi / 2 - pitch)]])
        H_g2c = np.matmul(K, np.concatenate(
            [R_g2c[:, 0:1], R_g2c[:, 1:2], np.matmul(R_g2c, np.array([[0], [0], [-cam_height]]))], 1))
        X = np.concatenate([top_view_region, np.ones([4, 1])], 1)
        img_region = np.matmul(X, H_g2c.T)
        border_org = np.divide(img_region[:, :2], img_region[:, 2:3])

        # transform original image region to network input region
        ratio_x = resize_img_size[1] / org_img_size[1]
        ratio_y = resize_img_size[0] / (org_img_size[0] - crop_y)
        H_c = np.array([[ratio_x, 0, 0],
                        [0, ratio_y, -ratio_y * crop_y]])
        border_net = np.matmul(np.concatenate([border_org, np.ones([4, 1])], 1), H_c.T)
        border_net[:, 0] = border_net[:, 0] / resize_img_size[1]
        border_net[:, 1] = border_net[:, 1] / resize_img_size[0]
        border_net = np.float32(border_net)

        # compute the normalized transformation
        dst = np.float32([[0, 0], [0, 1], [1, 0], [1, 1]])
        M = cv2.getPerspectiveTransform(border_net, dst)
        M_inv = cv2.getPerspectiveTransform(dst, border_net)
        return M, M_inv

    def __len__(self):
        """
        Conventional len method
        """
        return self.n_samples

    def __getitem__(self, idx):
        """
        Args: idx (int): Index in list to load image
        """
        img_name = self._label_image_path[idx]
        gt_lane_2D = self._label_lane_pts_all[idx]
        with open(img_name, 'rb') as f:
            image = (Image.open(f).convert('RGB'))

        image = F.crop(image, self.h_crop, 0, self.h_org-self.h_crop, self.w_org)
        image = F.resize(image, size=(self.h_net, self.w_net), interpolation=Image.BILINEAR)
        gt_anchor = np.zeros([self.w_ipm/8, 3, 2*self.num_y_anchor+1], dtype=np.float32)

        # TODO: project lane to ground
        #  assign gt tensor anchor values based on association at Y_ref
        #  in what scope should the GT values should be
        image, gt_anchor = self.totensor(image).float(), (self.totensor(gt_anchor)).float()

        return image, gt_anchor


# unit test
if __name__ == '__main__':
    print('done')
