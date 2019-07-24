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
    """
    Dataset with labeled lanes
        This implementation considers:
        w/o laneline 3D attributes
        w/o centerline annotations
        default considers 3D laneline, including centerlines
    """
    def __init__(self, dataset_base_dir, json_file_path, args):
        """

        :param dataset_info_file: json file list
        """
        self.totensor = transforms.ToTensor()
        self.no_3d = args.no_3d
        self.no_centerline = args.no_centerline

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
        H_g2c, H_c2g = compute_homograpthy(self.pitch, args.cam_height, args.K)
        self.H_c2g = H_c2g
        # compute y_ref in ground
        self.y_ref = args.y_ref
        # compute x_steps
        x_min = self.top_view_region[0, 0]
        x_max = self.top_view_region[1, 0]
        self.anchor_x_steps = np.linspace(x_min, x_max, args.ipm_w/8, endpoint=True)

        # self._label_image_path, self._label_lane_pts_all, self._label_lane_types_all = \
        #     self._init_dataset(dataset_base_dir, json_file_path)
        self._label_image_path, self._label_lane_pts_all = \
            self._init_dataset_tusimple(dataset_base_dir, json_file_path)
        self.n_samples = self._label_image_path.shape[0]
        # self._random_dataset()

    def _init_dataset(self, dataset_base_dir, json_file_path):
        """
        :param dataset_info_file:
        :return: image paths, labels in normalized net input coordinates

        data processing:
        ground truth labels map are scaled wrt network input sizes
        """

        # load image path, and lane pts
        label_image_path = []
        gt_lane_pts_all = []

        assert ops.exists(json_file_path), '{:s} not exist'.format(json_file_path)

        # src_dir = ops.split(json_file_path)[0]

        with open(json_file_path, 'r') as file:
            for line in file:
                info_dict = json.loads(line)

                image_path = ops.join(dataset_base_dir, info_dict['raw_file'])
                assert ops.exists(image_path), '{:s} not exist'.format(image_path)

                label_image_path.append(image_path)

                gt_lane_pts = info_dict['lanes']
                for i, lane in enumerate(gt_lane_pts):
                    # A GT lane can be either 2D or 3D
                    # if a GT lane is 3D, the height is intact from 3D GT, so keep it intact here too
                    lane = np.array(lane)
                    # # rescale to net input
                    # lane[:, 0] *= self.x_ratio
                    # lane[:, 1] = (lane[:, 1] - self.h_crop) * self.y_ratio
                    gt_lane_pts[i] = lane
                gt_lane_pts_all.append(gt_lane_pts)

                # TODO: implement centerline case when avaliable

        label_image_path = np.array(label_image_path)
        return label_image_path, gt_lane_pts_all

    def _init_dataset_tusimple(self, dataset_base_dir, json_file_path):
        """
        :param json_file_path:
        :return: image paths, labels in normalized net input coordinates

        data processing:
        ground truth labels map are scaled wrt network input sizes
        """

        # load image path, and lane pts
        label_image_path = []
        gt_lane_pts_all = []

        assert ops.exists(json_file_path), '{:s} not exist'.format(json_file_path)

        # src_dir = ops.split(json_file_path)[0]

        with open(json_file_path, 'r') as file:
            for line in file:
                info_dict = json.loads(line)

                image_path = ops.join(dataset_base_dir, info_dict['raw_file'])
                assert ops.exists(image_path), '{:s} not exist'.format(image_path)

                label_image_path.append(image_path)

                gt_lane_pts_X = info_dict['lanes']
                gt_y_steps = np.array(info_dict['h_samples'])
                gt_lane_pts = []

                for i, lane_x in enumerate(gt_lane_pts_X):
                    lane = np.zeros([gt_y_steps.shape[0], 2])

                    lane_x = np.array(lane_x)
                    lane[:, 0] = lane_x
                    lane[:, 1] = gt_y_steps
                    # remove invalid samples
                    lane = lane[lane_x >= 0, :]

                    if lane.shape[0] < 2:
                        continue

                    # # rescale to net input
                    # lane[:, 0] *= self.x_ratio
                    # lane[:, 1] = (lane[:, 1] - self.h_crop) * self.y_ratio

                    gt_lane_pts.append(lane)

                gt_lane_pts_all.append(gt_lane_pts)
        label_image_path = np.array(label_image_path)
        return label_image_path, gt_lane_pts_all

    # def _random_dataset(self):
    #     """
    #
    #     :return:
    #     """
    #
    #     random_idx = np.random.permutation(self._label_image_path.shape[0])
    #     self._label_image_path = self._label_image_path[random_idx]
    #     self._label_lane_types_all = [self._label_lane_types_all[i] for i in random_idx]
    #     self._label_lane_pts_all = [self._label_lane_pts_all[i] for i in random_idx]
    #     self._next_batch_loop_count = 0

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
        with open(img_name, 'rb') as f:
            image = (Image.open(f).convert('RGB'))

        # image preprocess with crop and resize
        image = F.crop(image, self.h_crop, 0, self.h_org-self.h_crop, self.w_org)
        image = F.resize(image, size=(self.h_net, self.w_net), interpolation=Image.BILINEAR)

        if self.no_3d:
            num_types = 1
        else:
            num_types = 3

        if self.no_centerline:
            dim_anchor = self.num_y_steps + 1
        else:
            dim_anchor = 2*self.num_y_steps + 1
        gt_anchor = np.zeros([np.int32(self.w_ipm / 8), num_types, dim_anchor], dtype=np.float32)

        # cv2.imshow('image', np.asarray(image))
        # cv2.waitKey(50)
        gt_lanes_2d = self._label_lane_pts_all[idx]
        for i in range(len(gt_lanes_2d)):
            gt_lane_2d = gt_lanes_2d[i]
            # project to ground coordinates
            gt_lane_grd_x, gt_lane_grd_y = homogenous_transformation(self.H_c2g, gt_lane_2d[:, 0], gt_lane_2d[:, 1])
            gt_lane_3d = np.zeros_like(gt_lane_2d, dtype=np.float32)
            gt_lane_3d[:, 0] = gt_lane_grd_x
            gt_lane_3d[:, 1] = gt_lane_grd_y
            if gt_lane_2d.shape[1] > 2:
                gt_lane_3d[:, 2] = gt_lane_2d[:, 2]

            # remove points with y out of range
            gt_lane_3d = gt_lane_3d[np.logical_and(gt_lane_3d[:, 1] > 0, gt_lane_3d[:, 1] < 100), ...]
            if gt_lane_3d.shape[0] < 2:
                continue

            # ignore GT does not pass y_ref
            if gt_lane_3d[0, 1] > self.y_ref or gt_lane_3d[-1, 1] < self.y_ref:
                continue

            # resample ground-truth laneline at anchor y steps
            x_values, z_values = resample_3d_laneline(gt_lane_3d, self.anchor_y_steps)

            # decide association at r_ref
            min_idx = np.argmin((self.anchor_x_steps - x_values[1])**2)
            # assign anchor tensor values
            gt_anchor[min_idx, 0, 0: self.num_y_steps] = x_values - self.anchor_x_steps[min_idx]
            if not self.no_3d:
                gt_anchor[min_idx, 0, self.num_y_steps:2*self.num_y_steps] = z_values
            gt_anchor[min_idx, 0, -1] = 1.0

        # TODO: implement centerlines case when avaliable

        image = self.totensor(image).float()
        gt_anchor = gt_anchor.reshape([np.int32(self.w_ipm / 8), -1])
        gt_anchor = torch.from_numpy(gt_anchor)
        return image, gt_anchor


def resample_3d_laneline(gt_lane_3d, y_steps):
    """
        suppose to interpolate x, z values even there associated y's are beyond ground-truth scope
    :param gt_lane_3d:
    :param y_steps:
    :return:
    """
    assert(gt_lane_3d.shape[0] >= 2)

    x_values = np.zeros_like(y_steps)
    z_values = np.zeros_like(y_steps)

    if gt_lane_3d.shape[1] < 3:
        gt_lane_3d = np.concatenate([gt_lane_3d, np.zeros([gt_lane_3d.shape[0], 1])], axis=1)

    y1 = gt_lane_3d[0, 1]
    x1 = gt_lane_3d[0, 0]
    z1 = gt_lane_3d[0, 2]
    y2 = gt_lane_3d[1, 1]
    x2 = gt_lane_3d[1, 0]
    z2 = gt_lane_3d[1, 2]

    j = 0
    for i, y_grid in enumerate(y_steps):
        while j < gt_lane_3d.shape[0] and gt_lane_3d[j, 1] < y_grid:
            j += 1

        # for the y_grid beyond ground-truth y range, keep using the previous two to interpolate in a line
        if j < gt_lane_3d.shape[0]:
            y1 = gt_lane_3d[j, 1]
            x1 = gt_lane_3d[j, 0]
            z1 = gt_lane_3d[j, 2]
            if (j-1) >= 0:
                y2 = gt_lane_3d[j - 1, 1]
                x2 = gt_lane_3d[j - 1, 0]
                z2 = gt_lane_3d[j - 1, 2]
            elif (j+1) < gt_lane_3d.shape[0]:
                y2 = gt_lane_3d[j + 1, 1]
                x2 = gt_lane_3d[j + 1, 0]
                z2 = gt_lane_3d[j + 1, 2]
            else:
                continue

        if y2 == y1:
            continue
        # interpolate x value and z value at anchor grid
        x_values[i] = (x1 * (y2 - y_grid) + x2 * (y_grid - y1)) / (y2 - y1)
        z_values[i] = (z1 * (y2 - y_grid) + z2 * (y_grid - y1)) / (y2 - y1)
    return x_values, z_values


def compute_homograpthy(pitch, cam_height, K):
    # transform top-view region to original image region
    R_g2c = np.array([[1, 0, 0],
                      [0, np.cos(np.pi / 2 + pitch), -np.sin(np.pi / 2 + pitch)],
                      [0, np.sin(np.pi / 2 + pitch), np.cos(np.pi / 2 + pitch)]])
    H_g2c = np.matmul(K, np.concatenate(
        [R_g2c[:, 0:2], np.matmul(R_g2c.transpose(), np.array([[0], [0], [-cam_height]]))], 1))
    H_c2g = np.linalg.inv(H_g2c)
    return H_g2c, H_c2g


def homography_crop_resize(org_img_size, crop_y, resize_img_size):
    """
        compute the homography matrix transform original image to cropped and resized image
    :param org_img_size: [org_h, org_w]
    :param crop_y:
    :param resize_img_size: [resize_h, resize_w]
    :return:
    """
    # transform original image region to network input region
    ratio_x = resize_img_size[1] / org_img_size[1]
    ratio_y = resize_img_size[0] / (org_img_size[0] - crop_y)
    H_c = np.array([[ratio_x, 0, 0],
                    [0, ratio_y, -ratio_y*crop_y]])
    return H_c


def homogenous_transformation(Matrix, x, y):
    """
    Helper function to transform coordinates defined by transformation matrix

    Args:
            Matrix (multi dim - array): Transformation matrix
            x (array): original x coordinates
            y (array): original y coordinates
    """
    ones = np.ones((1, len(y)))
    coordinates = np.vstack((x, y, ones))
    trans = np.matmul(Matrix, coordinates)

    x_vals = trans[0,:]/trans[2, :]
    y_vals = trans[1,:]/trans[2, :]
    return x_vals, y_vals

# TODO: A series of data augmentation functions can be implemented here


def get_loader(dataset_base_dir, json_file_path, args):
    """
        create dataset from ground-truth
        return a batch sampler based ont the dataset
    """

    transformed_dataset = LaneDataset(dataset_base_dir, json_file_path, args)
    train_idx = range(transformed_dataset.n_samples)
    train_idx = train_idx[0:len(train_idx)//args.batch_size*args.batch_size]
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    train_loader = DataLoader(transformed_dataset,
                              batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=args.nworkers, pin_memory=True) #, collate_fn=my_collate)

    return train_loader


# unit test
if __name__ == '__main__':
    from tools.utils import define_args

    parser = define_args()
    args = parser.parse_args()
    args.top_view_region = np.array([[-20, 100], [20, 100], [-20, 5], [20, 5]])
    args.anchor_y_steps = np.array([5, 20, 40, 60, 80, 100])
    args.num_y_anchor = len(args.anchor_y_steps)
    args.K = np.array([[720, 0, 640],
                       [0, 720, 360],
                       [0, 0, 1]])
    # current test only considers no_centerline, no 3d case
    args.no_centerline = True
    args.no_3d = True

    x_min = args.top_view_region[0, 0]
    x_max = args.top_view_region[1, 0]
    anchor_x_steps = np.linspace(x_min, x_max, args.ipm_w/8, endpoint=True)
    anchor_y_steps = args.anchor_y_steps
    pitch = np.pi / 180 * args.pitch

    H_g2c, H_c2g = compute_homograpthy(pitch, args.cam_height, args.K)
    H_crop = homography_crop_resize([args.org_h, args.org_w], args.crop_size, [args.resize, 2*args.resize])

    # prepare datasets and loaders
    dataset_base_dir = '/media/yuliangguo/NewVolume2TB/Datasets/TuSimple/labeled/'
    json_file_path = ops.join(dataset_base_dir, 'label_data_0601.json')
    test_loader = get_loader(dataset_base_dir, json_file_path, args)

    # get a batch from loader
    for batch_ndx, (image_tensor, gt_tensor) in enumerate(test_loader):
        print('batch id: {:d}, image tensor shape:'.format(batch_ndx))
        print(image_tensor.shape)
        print('batch id: {:d}, gt tensor shape:'.format(batch_ndx))
        print(gt_tensor.shape)

        images = image_tensor.numpy()
        gt_anchors = gt_tensor.numpy()
        for i in range(args.batch_size):
            img = images[i, :, :, :].transpose([1, 2, 0])[...,::-1]*255
            img = img.astype(np.uint8)
            gt_anchor = gt_anchors[i, :, :]
            for j in range(gt_anchor.shape[0]):
                if gt_anchor[j, -1] > 0:
                    x_offsets = gt_anchor[j, :-1]
                    x_3d = x_offsets + anchor_x_steps[j]
                    x_2d, y_2d = homogenous_transformation(H_g2c, x_3d, anchor_y_steps)
                    pts_2d = np.matmul(H_crop, np.vstack([x_2d, y_2d, np.ones_like(x_2d)]))
                    x_2d = pts_2d[0, :].astype(np.int)
                    y_2d = pts_2d[1, :].astype(np.int)
                    for k in range(1, x_2d.shape[0]):
                        img = cv2.line(img, (x_2d[k-1], y_2d[k-1]), (x_2d[k], y_2d[k]), [0, 0, 255], 2)
            cv2.imshow('2D gt check', img)
            cv2.waitKey()

    # recover lanelines from gt_tensor and visualize the result

    print('done')
