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
        self.x_ratio = float(self.w_net) / float(self.w_org)
        self.y_ratio = float(self.h_net) / float(self.h_org - self.h_crop)
        self.anchor_y_steps = args.anchor_y_steps
        self.num_y_steps = len(self.anchor_y_steps)
        self.top_view_region = args.top_view_region
        # initialize the homography between image and IPM, this need to be exactly the same as network
        self.pitch = np.pi / 180 * args.pitch
        # M, M_inv = compute_normalized_homography(self.top_view_region, [self.h_org, self.w_org], self.h_crop,
        #                                     [self.h_net, self.w_net], self.pitch, args.cam_height, args.K)
        H_g2c, H_c2g = compute_homograpthy(self.pitch, args.cam_height, args.K)
        self.H_c2g = H_c2g
        # compute y_ref in ground
        self.y_ref = args.y_ref
        # compute x_steps
        x_min = self.top_view_region[0, 0]
        x_max = self.top_view_region[1, 0]
        self.anchor_x_steps = np.linspace(x_min, x_max, args.w_ipm/8, endpoint=True)

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
        gt_lanes_2D = self._label_lane_pts_all[idx]
        with open(img_name, 'rb') as f:
            image = (Image.open(f).convert('RGB'))

        image = F.crop(image, self.h_crop, 0, self.h_org-self.h_crop, self.w_org)
        image = F.resize(image, size=(self.h_net, self.w_net), interpolation=Image.BILINEAR)
        gt_anchor = np.zeros([self.w_ipm / 8, 3, 2 * self.num_y_steps + 1], dtype=np.float32)

        for i in range(len(gt_lanes_2D)):
            gt_lane_2D = gt_lanes_2D[i]
            # project to ground coordinates
            gt_lane_grd = homogenous_transformation(self.H_c2g, gt_lane_2D[:, 0], gt_lane_2D[:, 1])
            gt_lane_3D = np.zeros_like(gt_lane_2D, dtype=np.float32)
            gt_lane_3D[:2, :] = gt_lane_grd
            if gt_lane_2D.shape[1] > 2:
                gt_lane_3D[2, :] = gt_lane_2D[2, :]

            # ignore GT does not pass y_ref
            if gt_lane_3D[0, 1] > self.y_ref or gt_lane_3D[-1, 1] < self.y_ref:
                continue

            # resample at y steps
            x_values, z_values = resample_3D_laneline(gt_lane_3D, self.anchor_x_steps)

            # decide association at r_ref
            min_idx = np.amin((self.anchor_x_steps - x_values[1])**2)

            # TODO: now only save results for laneline, need to include centerline later/ separate datasets
            gt_anchor[min_idx, 2, 0: self.num_y_steps] = x_values - self.anchor_x_steps[min_idx]
            gt_anchor[min_idx, 2, self.num_y_steps:2*self.num_y_steps] = z_values
            gt_anchor[min_idx, 2, -1] = 1.0

        image, gt_anchor = self.totensor(image).float(), (self.totensor(gt_anchor)).float()

        return image, gt_anchor

def resample_3D_laneline(gt_lane_3D, y_steps):
    """
        suppose to interpolate x, z values even there associated y's are beyond ground-truth scope
    :param gt_lane_3D:
    :param y_steps:
    :return:
    """
    j = 0
    x_values = np.zeros_like(y_steps)
    z_values = np.zeros_like(y_steps)
    for i, y_grid in enumerate(y_steps):
        while gt_lane_3D[j, 1] < y_grid and j < gt_lane_3D.shape[0]:
            j += 1
        y1 = gt_lane_3D[j, 1]
        x1 = gt_lane_3D[j, 0]
        z1 = gt_lane_3D[j, 2]
        if (j-1) >= 0:
            y2 = gt_lane_3D[j-1, 1]
            x2 = gt_lane_3D[j-1, 0]
            z2 = gt_lane_3D[j-1, 2]
        elif (j+1) < gt_lane_3D.shape[0]:
            y2 = gt_lane_3D[j+1, 1]
            x2 = gt_lane_3D[j+1, 0]
            z2 = gt_lane_3D[j+1, 2]
        else:
            continue

        if y2 == y1:
            continue
        x_values[i] = (x1 * (y2 - y_grid) + x2 * (y_grid - y1)) / (y2 - y1)
        z_values[i] = (z1 * (y2 - y_grid) + z2 * (y_grid - y1)) / (y2 - y1)
    return x_values, z_values


# compute normalized transformation matrix for a top-view region boundaries
def compute_normalized_homography(top_view_region, org_img_size, crop_y, resize_img_size, pitch, cam_height, K):
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

def compute_homograpthy(pitch, cam_height, K):
    # transform top-view region to original image region
    R_g2c = np.array([[1, 0, 0],
                      [0, np.cos(-np.pi / 2 - pitch), -np.sin(-np.pi / 2 - pitch)],
                      [0, np.sin(-np.pi / 2 - pitch), np.cos(-np.pi / 2 - pitch)]])
    H_g2c = np.matmul(K, np.concatenate(
        [R_g2c[:, 0:1], R_g2c[:, 1:2], np.matmul(R_g2c, np.array([[0], [0], [-cam_height]]))], 1))
    H_c2g = np.invert(H_g2c)
    return H_g2c, H_c2g

def homogenous_transformation(Matrix, x, y):
    """
    Helper function to transform coordionates defined by transformation matrix

    Args:
            Matrix (multi dim - array): Transformation matrix
            x (array): original x coordinates
            y (array): original y coordinates
    """
    ones = np.ones((1,len(y)))
    coordinates = np.vstack((x, y, ones))
    trans = np.matmul(Matrix, coordinates)

    x_vals = trans[0,:]/trans[2,:]
    y_vals = trans[1,:]/trans[2,:]
    return x_vals, y_vals


# unit test
if __name__ == '__main__':
    print('done')
