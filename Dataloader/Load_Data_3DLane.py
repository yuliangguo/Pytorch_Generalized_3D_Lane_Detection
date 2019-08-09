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
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torch.utils.data.dataloader import default_collate
from tools.utils import homogenous_transformation, homograpthy_g2c, homography_crop_resize, nms_1d, tusimple_config
warnings.simplefilter('ignore', np.RankWarning)
matplotlib.use('Agg')


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
        # define image pre-processor
        self.totensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(args.vgg_mean, args.vgg_std)

        # dataset parameters
        self.no_3d = args.no_3d
        self.no_centerline = args.no_centerline

        self.h_org = args.org_h
        self.w_org = args.org_w
        self.h_crop = args.crop_size

        # parameters related to service network
        self.h_net = args.resize_h
        self.w_net = args.resize_w
        self.h_ipm = args.ipm_h
        self.w_ipm = args.ipm_w
        self.x_ratio = float(self.w_net) / float(self.w_org)
        self.y_ratio = float(self.h_net) / float(self.h_org - self.h_crop)
        self.top_view_region = args.top_view_region
        self.y_ref = args.y_ref

        self.K = args.K
        if args.fix_cam:
            self.fix_cam = True
            # compute the homography between image and IPM, and crop transformation
            self.cam_height = args.cam_height
            self.cam_pitch = np.pi / 180 * args.pitch
            self.H_g2c,  self.H_c2g = homograpthy_g2c(self.cam_pitch, args.cam_height, args.K)
            self.H_crop = homography_crop_resize([args.org_h, args.org_w], args.crop_size, [args.resize_h, args.resize_w])
        else:
            self.fix_cam = False

        # compute anchor steps
        x_min = self.top_view_region[0, 0]
        x_max = self.top_view_region[1, 0]
        self.anchor_x_steps = np.linspace(x_min, x_max, np.int(args.ipm_w/8), endpoint=True)
        self.anchor_y_steps = args.anchor_y_steps
        self.num_y_steps = len(self.anchor_y_steps)

        # parse ground-truth file
        if args.dataset_name is 'tusimple':
            self._label_image_path,\
                self._label_laneline_pts_all = self._init_dataset_tusimple(dataset_base_dir, json_file_path)
        else:  # assume loading apollo sim 3D lane
            self._label_image_path, self._label_laneline_pts_all, \
                self._label_centerline_pts_all, self._label_cam_height_all,\
                self._label_cam_pitch_all = self._init_dataset_3D(dataset_base_dir, json_file_path)
        self.n_samples = self._label_image_path.shape[0]

    def _init_dataset_3D(self, dataset_base_dir, json_file_path):
        """
        :param dataset_info_file:
        :return: image paths, labels in normalized net input coordinates

        data processing:
        ground truth labels map are scaled wrt network input sizes
        """

        # load image path, and lane pts
        label_image_path = []
        gt_laneline_pts_all = []
        gt_centerline_pts_all = []
        gt_cam_height_all = []
        gt_cam_pitch_all = []

        assert ops.exists(json_file_path), '{:s} not exist'.format(json_file_path)

        with open(json_file_path, 'r') as file:
            for line in file:
                info_dict = json.loads(line)

                image_path = ops.join(dataset_base_dir, info_dict['raw_file'])
                assert ops.exists(image_path), '{:s} not exist'.format(image_path)

                label_image_path.append(image_path)

                gt_lane_pts = info_dict['laneLines']
                for i, lane in enumerate(gt_lane_pts):
                    # A GT lane can be either 2D or 3D
                    # if a GT lane is 3D, the height is intact from 3D GT, so keep it intact here too
                    lane = np.array(lane)
                    gt_lane_pts[i] = lane
                gt_laneline_pts_all.append(gt_lane_pts)

                if not self.no_centerline:
                    gt_lane_pts = info_dict['centerLines']
                    for i, lane in enumerate(gt_lane_pts):
                        # A GT lane can be either 2D or 3D
                        # if a GT lane is 3D, the height is intact from 3D GT, so keep it intact here too
                        lane = np.array(lane)
                        gt_lane_pts[i] = lane
                    gt_centerline_pts_all.append(gt_lane_pts)

                if not self.fix_cam:
                    gt_cam_height = info_dict['cam_height']
                    gt_cam_height_all.append(gt_cam_height)
                    gt_cam_pitch = info_dict['cam_pitch']
                    gt_cam_pitch_all.append(gt_cam_pitch)

        label_image_path = np.array(label_image_path)
        gt_cam_height_all = np.array(gt_cam_height_all)
        gt_cam_pitch_all = np.array(gt_cam_pitch_all)

        return label_image_path, gt_laneline_pts_all, gt_centerline_pts_all, gt_cam_height_all, gt_cam_pitch_all

    def _init_dataset_tusimple(self, dataset_base_dir, json_file_path):
        """
        :param json_file_path:
        :return: image paths, labels in normalized net input coordinates

        data processing:
        ground truth labels map are scaled wrt network input sizes
        """

        # load image path, and lane pts
        label_image_path = []
        gt_laneline_pts_all = []

        assert ops.exists(json_file_path), '{:s} not exist'.format(json_file_path)

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
                    lane = np.zeros([gt_y_steps.shape[0], 2], dtype=np.float32)

                    lane_x = np.array(lane_x)
                    lane[:, 0] = lane_x
                    lane[:, 1] = gt_y_steps
                    # remove invalid samples
                    lane = lane[lane_x >= 0, :]

                    if lane.shape[0] < 2:
                        continue

                    gt_lane_pts.append(lane)
                gt_laneline_pts_all.append(gt_lane_pts)
        label_image_path = np.array(label_image_path)
        return label_image_path, gt_laneline_pts_all

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
        gt_lanes = self._label_laneline_pts_all[idx]
        for i in range(len(gt_lanes)):
            if self.no_3d:  # For ground-truth in 2D image coordinates (TuSimple)
                gt_lane_2d = gt_lanes[i]
                # project to ground coordinates
                gt_lane_grd_x, gt_lane_grd_y = homogenous_transformation(self.H_c2g, gt_lane_2d[:, 0], gt_lane_2d[:, 1])
                gt_lane_3d = np.zeros_like(gt_lane_2d, dtype=np.float32)
                gt_lane_3d[:, 0] = gt_lane_grd_x
                gt_lane_3d[:, 1] = gt_lane_grd_y
            else:  # For ground-truth in ground coordinates (Apollo Sim)
                gt_lane_3d = gt_lanes[i]

            # remove points with y out of range
            gt_lane_3d = gt_lane_3d[np.logical_and(gt_lane_3d[:, 1] > 0, gt_lane_3d[:, 1] < 100), ...]
            if gt_lane_3d.shape[0] < 2:
                continue

            # reverse the order of 3d pints to make the first point the closest
            gt_lane_3d = gt_lane_3d[::-1, :]

            # ignore GT does not pass y_ref
            if gt_lane_3d[0, 1] > self.y_ref or gt_lane_3d[-1, 1] < self.y_ref:
                continue

            # resample ground-truth laneline at anchor y steps
            x_values, z_values = resample_laneline_in_y(gt_lane_3d, self.anchor_y_steps)

            # decide association at r_ref
            lane_id = np.argmin((self.anchor_x_steps - x_values[1])**2)

            # assign anchor tensor values
            gt_anchor[lane_id, 0, 0: self.num_y_steps] = x_values - self.anchor_x_steps[lane_id]
            if not self.no_3d:
                gt_anchor[lane_id, 0, self.num_y_steps:2*self.num_y_steps] = z_values
            gt_anchor[lane_id, 0, -1] = 1.0

        # fetch centerlines when available
        if not self.no_centerline:
            gt_lanes = self._label_laneline_pts_all[idx]
            for i in range(len(gt_lanes)):
                if self.no_3d:  # For ground-truth in 2D image coordinates (TuSimple)
                    gt_lane_2d = gt_lanes[i]
                    # project to ground coordinates
                    gt_lane_grd_x, gt_lane_grd_y = homogenous_transformation(self.H_c2g, gt_lane_2d[:, 0],
                                                                             gt_lane_2d[:, 1])
                    gt_lane_3d = np.zeros_like(gt_lane_2d, dtype=np.float32)
                    gt_lane_3d[:, 0] = gt_lane_grd_x
                    gt_lane_3d[:, 1] = gt_lane_grd_y
                else:  # For ground-truth in ground coordinates (Apollo Sim)
                    gt_lane_3d = gt_lanes[i]

                # remove points with y out of range
                gt_lane_3d = gt_lane_3d[np.logical_and(gt_lane_3d[:, 1] > 0, gt_lane_3d[:, 1] < 100), ...]
                if gt_lane_3d.shape[0] < 2:
                    continue

                # TODO: decide whether to reverse the order later
                # reverse the order of 3d pints to make the first point the closest
                gt_lane_3d = gt_lane_3d[::-1, :]

                # ignore GT does not pass y_ref
                if gt_lane_3d[0, 1] > self.y_ref or gt_lane_3d[-1, 1] < self.y_ref:
                    continue

                # resample ground-truth laneline at anchor y steps
                x_values, z_values = resample_laneline_in_y(gt_lane_3d, self.anchor_y_steps)

                # decide association at r_ref
                lane_id = np.argmin((self.anchor_x_steps - x_values[1]) ** 2)

                # assign anchor tensor values
                if gt_anchor[lane_id, 1, -1] > 0:  # the case one spliting lane has been assigned
                    gt_anchor[lane_id, 2, 0: self.num_y_steps] = x_values - self.anchor_x_steps[lane_id]
                    if not self.no_3d:
                        gt_anchor[lane_id, 2, self.num_y_steps:2*self.num_y_steps] = z_values
                    gt_anchor[lane_id, 2, -1] = 1.0
                else:
                    gt_anchor[lane_id, 1, 0: self.num_y_steps] = x_values - self.anchor_x_steps[lane_id]
                    if not self.no_3d:
                        gt_anchor[lane_id, 1, self.num_y_steps:2*self.num_y_steps] = z_values
                    gt_anchor[lane_id, 1, -1] = 1.0

        # fetch camera height and pitch
        if not self.fix_cam:
            gt_cam_height = self._label_cam_height_all[idx]
            gt_cam_pitch = self._label_cam_pitch_all[idx]
        else:
            gt_cam_height = self.cam_height
            gt_cam_pitch = self.cam_pitch

        image = self.totensor(image).float()
        image = self.normalize(image)
        gt_anchor = gt_anchor.reshape([np.int32(self.w_ipm / 8), -1])
        gt_anchor = torch.from_numpy(gt_anchor)
        gt_cam_height = torch.tensor(gt_cam_height, dtype=torch.float32)
        gt_cam_pitch = torch.tensor(gt_cam_pitch, dtype=torch.float32)
        return image, gt_anchor, idx, gt_cam_height, gt_cam_pitch

    def proj_trainsforms(self, idx):
        if not self.fix_cam:
            H_g2c, H_c2g = homograpthy_g2c(self._label_cam_pitch_all[idx],
                                           self._label_cam_height_all[idx], self.K)
            H_crop = homography_crop_resize([self.h_org, self.w_org], self.h_crop, [self.h_net, self.w_net])
            return H_g2c, H_c2g, H_crop
        else:
            return self.H_g2c, self.H_c2g, self.H_crop


def resample_laneline_in_y(input_lane, y_steps):
    """
        Interpolate x, z values at each anchor grid, including those beyond the range of input lnae y range
    :param input_lane: N x 2 or N x 3 ndarray, one row for a point (x, y, z-optional).
                       It requires y values of input lane in ascending order
    :param y_steps: a vector of steps in y
    :return:
    """

    # at least two points are included
    assert(input_lane.shape[0] >= 2)

    x_values = np.zeros_like(y_steps, dtype=np.float32)
    z_values = np.zeros_like(y_steps, dtype=np.float32)

    if input_lane.shape[1] < 3:
        input_lane = np.concatenate([input_lane, np.zeros([input_lane.shape[0], 1], dtype=np.float32)], axis=1)

    j = 0
    for i, y_grid in enumerate(y_steps):
        # search the next input point further than current y grid
        while j < input_lane.shape[0] and input_lane[j, 1] < y_grid:
            j += 1

        # locate the two input points for interpolating x, z values at current y grid
        if j < input_lane.shape[0]:
            y1 = input_lane[j, 1]
            x1 = input_lane[j, 0]
            z1 = input_lane[j, 2]
            if (j-1) >= 0:
                y2 = input_lane[j - 1, 1]
                x2 = input_lane[j - 1, 0]
                z2 = input_lane[j - 1, 2]
            elif (j+1) < input_lane.shape[0]:  # for a y grid closer than the closest ground-truth
                y2 = input_lane[j + 1, 1]
                x2 = input_lane[j + 1, 0]
                z2 = input_lane[j + 1, 2]
            else:  # only a single ground-truth point existing
                continue
        else:  # for the y_grid farther than the farthest ground-truth y range,
            y1 = input_lane[-1, 1]
            x1 = input_lane[-1, 0]
            z1 = input_lane[-1, 2]
            y2 = input_lane[-2, 1]
            x2 = input_lane[-2, 0]
            z2 = input_lane[-2, 2]

        # interpolate x value and z value at anchor grid
        x_values[i] = (x1 * (y2 - y_grid) + x2 * (y_grid - y1)) / (y2 - y1)
        z_values[i] = (z1 * (y2 - y_grid) + z2 * (y_grid - y1)) / (y2 - y1)
    return x_values, z_values


# TODO: A series of data augmentation functions can be implemented here


def get_loader(transformed_dataset, args):
    """
        create dataset from ground-truth
        return a batch sampler based ont the dataset
    """

    # transformed_dataset = LaneDataset(dataset_base_dir, json_file_path, args)
    sample_idx = range(transformed_dataset.n_samples)
    sample_idx = sample_idx[0:len(sample_idx)//args.batch_size*args.batch_size]
    data_sampler = torch.utils.data.sampler.SubsetRandomSampler(sample_idx)
    data_loader = DataLoader(transformed_dataset,
                             batch_size=args.batch_size, sampler=data_sampler,
                             num_workers=args.nworkers, pin_memory=True)

    return data_loader


def compute_tusimple_lanes(pred_anchor, h_samples, H_g2c, anchor_x_steps, anchor_y_steps, x_min, x_max, prob_th=0.5):
    """
        convert anchor lanes to image lanes in tusimple format
    :return: x values at h_samples in image coordinates
    """
    lanes_out = []

    # apply nms to output lanes
    pred_anchor[:, -1] = nms_1d(pred_anchor[:, -1])

    # need to resample network lane results at h_samples
    for j in range(pred_anchor.shape[0]):
        if pred_anchor[j, -1] > prob_th:
            x_offsets = pred_anchor[j, :-1]
            x_3d = x_offsets + anchor_x_steps[j]
            # compute x, y in original image coordinates
            x_2d, y_2d = homogenous_transformation(H_g2c, x_3d, anchor_y_steps)
            # reverse the order such that y_2d is ascending
            x_2d = x_2d[::-1]
            y_2d = y_2d[::-1]
            # resample at h_samples
            x_values, z_values = resample_laneline_in_y(np.vstack([x_2d, y_2d]).T, h_samples)
            # assign out-of-range x values to be -2
            x_values = x_values.astype(np.int)
            x_values[np.where(np.logical_or(x_values < x_min, x_values >= x_max))] = -2
            # assign far side y values to be -2
            x_values[np.where(h_samples < y_2d[0])] = -2

            lanes_out.append(x_values.data.tolist())
    return lanes_out


# unit test
if __name__ == '__main__':
    from tools.utils import define_args

    parser = define_args()
    args = parser.parse_args()

    args.dataset_name = 'tusimple'
    args.data_dir = ops.join('../data', args.dataset_name)
    args.dataset_dir = '/media/yuliangguo/NewVolume2TB/Datasets/TuSimple/labeled/'

    # load configuration for certain dataset
    if args.dataset_name is 'tusimple':
        tusimple_config(args)

    # set 3D ground area for visualization
    vis_border_3d = np.array([[-1.75, 100.], [1.75, 100.], [-1.75, 5.], [1.75, 5.]])
    print('visual area border:')
    print(vis_border_3d)

    # load data
    test_dataset = LaneDataset(args.dataset_dir, ops.join(args.data_dir, 'val.json'), args)
    test_loader = get_loader(test_dataset, args)
    anchor_x_steps = test_dataset.anchor_x_steps

    # get a batch of data/label pairs from loader
    for batch_ndx, (image_tensor, gt_tensor, idx, gt_cam_height, gt_cam_pitch) in enumerate(test_loader):
        print('batch id: {:d}, image tensor shape:'.format(batch_ndx))
        print(image_tensor.shape)
        print('batch id: {:d}, gt tensor shape:'.format(batch_ndx))
        print(gt_tensor.shape)

        # convert to BGR and numpy for visualization in opencv
        images = image_tensor.permute(0, 2, 3, 1).data.cpu().numpy()
        gt_anchors = gt_tensor.numpy()
        for i in range(args.batch_size):
            img = images[i]
            img = img * np.array(args.vgg_std).astype(np.float32)
            img = img + np.array(args.vgg_mean).astype(np.float32)
            if img.min() < 0. or img.max() > 1.0:
                print('found an invalid normalized sample')
            img = np.clip(img, 0, 1)

            H_g2c, H_c2g, H_crop = test_dataset.proj_trainsforms(idx[i])
            M = np.matmul(H_crop, H_g2c)
            # visualize visual border for confirming calibration
            x_2d, y_2d = homogenous_transformation(M, vis_border_3d[:, 0], vis_border_3d[:, 1])
            x_2d = x_2d.astype(np.int)
            y_2d = y_2d.astype(np.int)
            img = cv2.line(img, (x_2d[0], y_2d[0]), (x_2d[1], y_2d[1]), [0, 0, 1], 2)
            img = cv2.line(img, (x_2d[2], y_2d[2]), (x_2d[3], y_2d[3]), [0, 0, 1], 2)
            img = cv2.line(img, (x_2d[0], y_2d[0]), (x_2d[2], y_2d[2]), [0, 0, 1], 2)
            img = cv2.line(img, (x_2d[1], y_2d[1]), (x_2d[3], y_2d[3]), [0, 0, 1], 2)

            # visualize ground-truth anchor lanelines by projecting them on the image
            gt_anchor = gt_anchors[i, :, :]
            for j in range(gt_anchor.shape[0]):
                if gt_anchor[j, -1] > 0:
                    x_offsets = gt_anchor[j, :-1]
                    x_3d = x_offsets + anchor_x_steps[j]
                    # x_3d[:] = anchor_x_steps[j]
                    x_2d, y_2d = homogenous_transformation(H_g2c, x_3d, args.anchor_y_steps)
                    pts_2d = np.matmul(H_crop, np.vstack([x_2d, y_2d, np.ones_like(x_2d)]))
                    x_2d = pts_2d[0, :].astype(np.int)
                    y_2d = pts_2d[1, :].astype(np.int)
                    for k in range(1, x_2d.shape[0]):
                        img = cv2.line(img, (x_2d[k-1], y_2d[k-1]), (x_2d[k], y_2d[k]), [1, 0, 0], 2)
            # convert image to BGR for opencv imshow
            cv2.imshow('2D gt check', np.flip(img, axis=2))
            cv2.waitKey(500)
            print('image: {:d} in batch: {:d}'.format(i, batch_ndx))

    print('done')
