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
from tools.utils import homographic_transformation, projective_transformation, homograpthy_g2im, projection_g2im,\
    homography_crop_resize, nms_1d, tusimple_config, sim3d_config, Visualizer
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
    def __init__(self, dataset_base_dir, json_file_path, args, data_aug=False):
        """

        :param dataset_info_file: json file list
        """
        # define image pre-processor
        self.totensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(args.vgg_mean, args.vgg_std)
        self.data_aug = data_aug

        # dataset parameters
        self.dataset_name = args.dataset_name
        self.no_3d = args.no_3d
        self.no_centerline = args.no_centerline

        self.h_org = args.org_h
        self.w_org = args.org_w
        self.h_crop = args.crop_y

        # parameters related to service network
        self.h_net = args.resize_h
        self.w_net = args.resize_w
        self.ipm_h = args.ipm_h
        self.ipm_w = args.ipm_w
        # self.x_ratio = float(self.w_net) / float(self.w_org)
        # self.y_ratio = float(self.h_net) / float(self.h_org - self.h_crop)
        self.top_view_region = args.top_view_region
        self.y_ref = args.y_ref

        self.K = args.K
        self.H_crop = homography_crop_resize([args.org_h, args.org_w], args.crop_y, [args.resize_h, args.resize_w])
        # transformation from ipm to ground region
        self.H_ipm2g = cv2.getPerspectiveTransform(np.float32([[0, 0],
                                                               [self.ipm_w-1, 0],
                                                               [0, self.ipm_h-1],
                                                               [self.ipm_w-1, self.ipm_h-1]]),
                                                   np.float32(args.top_view_region))
        # self.H_g2ipm = np.linalg.inv(H_ipm2g)

        if args.fix_cam:
            self.fix_cam = True
            # compute the homography between image and IPM, and crop transformation
            self.cam_height = args.cam_height
            self.cam_pitch = np.pi / 180 * args.pitch
            self.H_g2im = homograpthy_g2im(self.cam_pitch, args.cam_height, args.K)
            self.H_im2g = np.linalg.inv(self.H_g2im)
            self.H_im2ipm = np.linalg.inv(np.matmul(self.H_crop, np.matmul(self.H_g2im, self.H_ipm2g)))
        else:
            self.fix_cam = False

        # compute anchor steps
        x_min = self.top_view_region[0, 0]
        x_max = self.top_view_region[1, 0]
        self.anchor_x_steps = np.linspace(x_min, x_max, np.int(args.ipm_w/8), endpoint=True)
        self.anchor_y_steps = args.anchor_y_steps
        self.num_y_steps = len(self.anchor_y_steps)
        if self.no_centerline:
            self.num_types = 1
        else:
            self.num_types = 3

        if self.no_3d:
            self.anchor_dim = self.num_y_steps + 1
        else:
            self.anchor_dim = 2*self.num_y_steps + 1

        # parse ground-truth file
        if self.dataset_name is 'tusimple':
            self._label_image_path,\
                self._label_laneline_all, \
                self._laneline_ass_ids, \
                self._x_off_std = self.init_dataset_tusimple(dataset_base_dir, json_file_path)
        elif self.dataset_name is 'sim3d':  # assume loading apollo sim 3D lane
            self._label_image_path, \
                self._label_laneline_all, \
                self._label_centerline_all, \
                self._label_cam_height_all, \
                self._label_cam_pitch_all, \
                self._laneline_ass_ids, \
                self._centerline_ass_ids, \
                self._x_off_std, \
                self._z_std = self.init_dataset_3D(dataset_base_dir, json_file_path)
        self.n_samples = self._label_image_path.shape[0]

        # normalize label values
        self.normalize_lane_label()

    def __len__(self):
        """
        Conventional len method
        """
        return self.n_samples

    def __getitem__(self, idx):
        """
        Args: idx (int): Index in list to load image
        """

        # fetch camera height and pitch
        if not self.fix_cam:
            gt_cam_height = self._label_cam_height_all[idx]
            gt_cam_pitch = self._label_cam_pitch_all[idx]
            # H_g2im = homograpthy_g2im(gt_cam_pitch, gt_cam_height, self.K)
            # H_im2g = np.linalg.inv(H_g2im)
        else:
            gt_cam_height = self.cam_height
            gt_cam_pitch = self.cam_pitch
            # H_im2g = self.H_im2g

        img_name = self._label_image_path[idx]

        with open(img_name, 'rb') as f:
            image = (Image.open(f).convert('RGB'))

        # image preprocess with crop and resize
        image = F.crop(image, self.h_crop, 0, self.h_org-self.h_crop, self.w_org)
        image = F.resize(image, size=(self.h_net, self.w_net), interpolation=Image.BILINEAR)

        gt_anchor = np.zeros([np.int32(self.ipm_w / 8), self.num_types, self.anchor_dim], dtype=np.float32)

        gt_lanes = self._label_laneline_all[idx]
        for i in range(len(gt_lanes)):
            # # convert gt label to anchor label
            # ass_id, x_off_values, z_values = self.convert_label_to_anchor(gt_lanes[i], H_im2g)

            # if ass_id >= 0:
            ass_id = self._laneline_ass_ids[idx][i]
            x_off_values = gt_lanes[i][:, 0]
            z_values = gt_lanes[i][:, 1]
            # assign anchor tensor values
            gt_anchor[ass_id, 0, 0: self.num_y_steps] = x_off_values
            if not self.no_3d:
                gt_anchor[ass_id, 0, self.num_y_steps:2 * self.num_y_steps] = z_values
            gt_anchor[ass_id, 0, -1] = 1.0

        # fetch centerlines when available
        if not self.no_centerline:
            gt_lanes = self._label_centerline_all[idx]
            for i in range(len(gt_lanes)):
                # # convert gt label to anchor label
                # ass_id, x_off_values, z_values = self.convert_label_to_anchor(gt_lanes[i], H_im2g)

                # if ass_id >= 0:
                ass_id = self._centerline_ass_ids[idx][i]
                x_off_values = gt_lanes[i][:, 0]
                z_values = gt_lanes[i][:, 1]

                # assign anchor tensor values
                # if ass_id >= 0:
                if gt_anchor[ass_id, 1, -1] > 0:  # the case one spliting lane has been assigned
                    gt_anchor[ass_id, 2, 0: self.num_y_steps] = x_off_values
                    if not self.no_3d:
                        gt_anchor[ass_id, 2, self.num_y_steps:2*self.num_y_steps] = z_values
                    gt_anchor[ass_id, 2, -1] = 1.0
                else:
                    gt_anchor[ass_id, 1, 0: self.num_y_steps] = x_off_values
                    if not self.no_3d:
                        gt_anchor[ass_id, 1, self.num_y_steps:2*self.num_y_steps] = z_values
                    gt_anchor[ass_id, 1, -1] = 1.0

        if self.data_aug:
            img_rot, aug_mat = data_aug_rotate(image)
            image = Image.fromarray(img_rot)
        image = self.totensor(image).float()
        image = self.normalize(image)
        gt_anchor = gt_anchor.reshape([np.int32(self.ipm_w / 8), -1])
        gt_anchor = torch.from_numpy(gt_anchor)
        gt_cam_height = torch.tensor(gt_cam_height, dtype=torch.float32)
        gt_cam_pitch = torch.tensor(gt_cam_pitch, dtype=torch.float32)
        if self.data_aug:
            aug_mat = torch.from_numpy(aug_mat.astype(np.float32))
            return image, gt_anchor, idx, gt_cam_height, gt_cam_pitch, aug_mat
        return image, gt_anchor, idx, gt_cam_height, gt_cam_pitch

    def init_dataset_3D(self, dataset_base_dir, json_file_path):
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

        # convert labeled laneline to anchor format
        gt_laneline_ass_ids = []
        gt_centerline_ass_ids = []
        lane_x_off_all = []
        lane_z_all = []
        for idx in range(len(gt_laneline_pts_all)):

            # fetch camera height and pitch
            if not self.fix_cam:
                gt_cam_height = gt_cam_height_all[idx]
                gt_cam_pitch = gt_cam_pitch_all[idx]
                H_g2im = homograpthy_g2im(gt_cam_pitch, gt_cam_height, self.K)
                H_im2g = np.linalg.inv(H_g2im)
            else:
                H_im2g = self.H_im2g

            gt_lanes = gt_laneline_pts_all[idx]
            gt_anchors = []
            ass_ids = []
            for i in range(len(gt_lanes)):
                # convert gt label to anchor label
                ass_id, x_off_values, z_values = self.convert_label_to_anchor(gt_lanes[i], H_im2g)
                if ass_id >= 0:
                    gt_anchors.append(np.vstack([x_off_values, z_values]).T)
                    ass_ids.append(ass_id)
                    lane_x_off_all.append(x_off_values)
                    lane_z_all.append(z_values)
            gt_laneline_ass_ids.append(ass_ids)
            gt_laneline_pts_all[idx] = gt_anchors

            if not self.no_centerline:
                gt_lanes = gt_centerline_pts_all[idx]
                gt_anchors = []
                ass_ids = []
                for i in range(len(gt_lanes)):
                    # convert gt label to anchor label
                    ass_id, x_off_values, z_values = self.convert_label_to_anchor(gt_lanes[i], H_im2g)
                    if ass_id >= 0:
                        gt_anchors.append(np.vstack([x_off_values, z_values]).T)
                        ass_ids.append(ass_id)
                        lane_x_off_all.append(x_off_values)
                        lane_z_all.append(z_values)
                gt_centerline_ass_ids.append(ass_ids)
                gt_centerline_pts_all[idx] = gt_anchors

        lane_x_off_all = np.array(lane_x_off_all)
        lane_z_all = np.array(lane_z_all)
        lane_x_off_std = np.std(lane_x_off_all, axis=0)
        lane_z_std = np.std(lane_z_all, axis=0)
        return label_image_path, gt_laneline_pts_all, gt_centerline_pts_all, gt_cam_height_all, gt_cam_pitch_all, gt_laneline_ass_ids, gt_centerline_ass_ids, lane_x_off_std, lane_z_std

    def init_dataset_tusimple(self, dataset_base_dir, json_file_path):
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

        # convert labeled laneline to anchor format
        H_im2g = self.H_im2g
        gt_laneline_ass_ids = []
        lane_x_off_all = []
        for idx in range(len(gt_laneline_pts_all)):
            gt_lanes = gt_laneline_pts_all[idx]
            gt_anchors = []
            ass_ids = []
            for i in range(len(gt_lanes)):
                # convert gt label to anchor label
                ass_id, x_off_values, z_values = self.convert_label_to_anchor(gt_lanes[i], H_im2g)
                if ass_id >= 0:
                    gt_anchors.append(np.vstack([x_off_values, z_values]).T)
                    ass_ids.append(ass_id)
                    lane_x_off_all.append(x_off_values)
            gt_laneline_ass_ids.append(ass_ids)
            gt_laneline_pts_all[idx] = gt_anchors

        lane_x_off_all = np.array(lane_x_off_all)
        lane_x_off_std = np.std(lane_x_off_all, axis=0)

        return label_image_path, gt_laneline_pts_all, gt_laneline_ass_ids, lane_x_off_std

    def set_x_off_std(self, x_off_std):
        self._x_off_std = x_off_std

    def set_z_std(self, z_std):
        self._z_std = z_std

    def normalize_lane_label(self):
        for lanes in self._label_laneline_all:
            for lane in lanes:
                lane[:, 0] = np.divide(lane[:, 0], self._x_off_std)
                if not self.no_3d:
                    lane[:, 1] = np.divide(lane[:, 1], self._z_std)

        if not self.no_centerline:
            for lanes in self._label_centerline_all:
                for lane in lanes:
                    lane[:, 0] = np.divide(lane[:, 0], self._x_off_std)
                    if not self.no_3d:
                        lane[:, 1] = np.divide(lane[:, 1], self._z_std)

    def convert_label_to_anchor(self, laneline_gt, H_im2g):
        if self.no_3d:  # For ground-truth in 2D image coordinates (TuSimple)
            gt_lane_2d = laneline_gt
            # project to ground coordinates
            gt_lane_grd_x, gt_lane_grd_y = homographic_transformation(H_im2g, gt_lane_2d[:, 0], gt_lane_2d[:, 1])
            gt_lane_3d = np.zeros_like(gt_lane_2d, dtype=np.float32)
            gt_lane_3d[:, 0] = gt_lane_grd_x
            gt_lane_3d[:, 1] = gt_lane_grd_y
        else:  # For ground-truth in ground coordinates (Apollo Sim)
            gt_lane_3d = laneline_gt

        # remove points with y out of range
        # 3D label will miss super long straight-line with only two points
        # 2D dataset requires this to rule out those points projected to ground, but out of meaningful range
        if self.dataset_name is 'tusimple':
            gt_lane_3d = gt_lane_3d[np.logical_and(gt_lane_3d[:, 1] > 0, gt_lane_3d[:, 1] < 100), ...]
            if gt_lane_3d.shape[0] < 2:
                return -1, np.array([]), np.array([])

        if self.dataset_name is 'tusimple':
            # reverse the order of 3d pints to make the first point the closest
            gt_lane_3d = gt_lane_3d[::-1, :]

        # ignore GT does not pass y_ref
        if gt_lane_3d[0, 1] > self.y_ref or gt_lane_3d[-1, 1] < self.y_ref:
            return -1, np.array([]), np.array([])

        # resample ground-truth laneline at anchor y steps
        x_values, z_values = resample_laneline_in_y(gt_lane_3d, self.anchor_y_steps)

        # decide association at r_ref
        ass_id = np.argmin((self.anchor_x_steps - x_values[1]) ** 2)
        # compute offset values
        x_off_values = x_values - self.anchor_x_steps[ass_id]

        return ass_id, x_off_values, z_values

    def transform_mats(self, idx):
        if not self.fix_cam:
            H_g2im = homograpthy_g2im(self._label_cam_pitch_all[idx],
                                      self._label_cam_height_all[idx], self.K)
            P_g2im = projection_g2im(self._label_cam_pitch_all[idx],
                                     self._label_cam_height_all[idx], self.K)

            H_im2ipm = np.linalg.inv(np.matmul(self.H_crop, np.matmul(H_g2im, self.H_ipm2g)))
            if self.no_3d:
                return H_g2im, self.H_crop, H_im2ipm
            else:
                return P_g2im, self.H_crop, H_im2ipm
        else:
            return self.H_g2im, self.H_crop, self.H_im2ipm


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


"""
    Data Augmentation: 
        idea 1:
            when initializing dataset, all labels will be prepared in 3D which do not need to be changed in image augmenting
            Image data augmentation would change the spatial transform matrix integrated in the network, provide 
            the transformation matrix related to random cropping, scaling and rotation
        idea 2:
            Introduce random sampling of cam_h, cam_pitch and their associated transformed image
            img2 = [R2[:, 0:2], T2] [R1[:, 0:2], T1]^-1 img1
            output augmented hcam, pitch, and img2 and untouched 3D anchor label value, Before forward pass, update spatial
            transform in network. However, However, image rotation is not considered, additional cropping is still needed
"""


def data_aug_rotate(img):
    # assume img in PIL image format
    rot = random.uniform(-np.pi/18, np.pi/18)
    # rot = random.uniform(-10, 10)
    center_x = img.width / 2
    center_y = img.height / 2
    rot_mat = cv2.getRotationMatrix2D((center_x, center_y), rot, 1.0)
    img_rot = np.array(img)
    img_rot = cv2.warpAffine(img_rot, rot_mat, (img.width, img.height), flags=cv2.INTER_LINEAR)
    # img_rot = img.rotate(rot)
    # rot = rot / 180 * np.pi
    rot_mat = np.vstack([rot_mat, [0, 0, 1]])
    return img_rot, rot_mat


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


def compute_tusimple_lanes(pred_anchor, h_samples, H_g2im, anchor_x_steps, anchor_y_steps, x_min, x_max, prob_th=0.5):
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
            x_2d, y_2d = homographic_transformation(H_g2im, x_3d, anchor_y_steps)
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


def compute_sim3d_lanes(pred_anchor, anchor_dim, anchor_x_steps, anchor_y_steps, prob_th=0.5):
    lanelines_out = []
    centerlines_out = []
    num_y_steps = anchor_y_steps.shape[0]

    # apply nms to output lanes probabilities
    # consider w/o centerline cases
    pred_anchor[:, anchor_dim - 1] = nms_1d(pred_anchor[:, anchor_dim - 1])
    pred_anchor[:, 2 * anchor_dim - 1] = nms_1d(pred_anchor[:, 2 * anchor_dim - 1])
    pred_anchor[:, 3 * anchor_dim - 1] = nms_1d(pred_anchor[:, 3 * anchor_dim - 1])

    for j in range(pred_anchor.shape[0]):
        # draw laneline
        if pred_anchor[j, anchor_dim - 1] > prob_th:
            x_offsets = pred_anchor[j, :num_y_steps]
            x_g = x_offsets + anchor_x_steps[j]
            z_g = pred_anchor[j, num_y_steps:anchor_dim - 1]
            line = np.vstack([x_g, anchor_y_steps, z_g]).T
            lanelines_out.append(line.data.tolist())

        # draw centerline
        if pred_anchor[j, 2 * anchor_dim - 1] > prob_th:
            x_offsets = pred_anchor[j, anchor_dim:anchor_dim + num_y_steps]
            x_g = x_offsets + anchor_x_steps[j]
            z_g = pred_anchor[j, anchor_dim + num_y_steps:2 * anchor_dim - 1]
            line = np.vstack([x_g, anchor_y_steps, z_g]).T
            centerlines_out.append(line.data.tolist())

        # draw the additional centerline for the merging case
        if pred_anchor[j, 3 * anchor_dim - 1] > prob_th:
            x_offsets = pred_anchor[j, 2 * anchor_dim:2 * anchor_dim + num_y_steps]
            x_g = x_offsets + anchor_x_steps[j]
            z_g = pred_anchor[j, 2 * anchor_dim + num_y_steps:3 * anchor_dim - 1]
            line = np.vstack([x_g, anchor_y_steps, z_g]).T
            centerlines_out.append(line.data.tolist())

    return lanelines_out, centerlines_out


def unormalize_lane_anchor(anchor, dataset):
    num_y_steps = dataset.num_y_steps
    anchor_dim = dataset.anchor_dim
    for i in range(dataset.num_types):
        anchor[:, i * anchor_dim : i * anchor_dim + num_y_steps] = \
            np.multiply(anchor[:, i * anchor_dim: i * anchor_dim + num_y_steps], dataset._x_off_std)
        if not dataset.no_3d:
            anchor[:, i * anchor_dim + num_y_steps: (i+1) * anchor_dim-1] = \
                np.multiply(anchor[:, i * anchor_dim + num_y_steps: (i+1) * anchor_dim-1], dataset._z_std)


# unit test
if __name__ == '__main__':
    import sys
    from tools.utils import define_args

    parser = define_args()
    args = parser.parse_args()

    # dataset_name 'tusimple' or 'sim3d'
    args.dataset_name = 'sim3d'
    args.dataset_dir = '/home/yuliangguo/Datasets/Apollo_Sim_3D_Lane/'
    # args.dataset_name = 'tusimple'
    # args.dataset_dir = '/home/yuliangguo/Datasets/tusimple/'
    args.data_dir = ops.join('data', args.dataset_name)

    # load configuration for certain dataset
    if args.dataset_name is 'tusimple':
        tusimple_config(args)
    elif args.dataset_name is 'sim3d':
        sim3d_config(args)
    else:
        print('Not using a supported dataset')
        sys.exit()

    # set 3D ground area for visualization
    vis_border_3d = np.array([[-1.75, 100.], [1.75, 100.], [-1.75, 5.], [1.75, 5.]])
    print('visual area border:')
    print(vis_border_3d)

    # load data
    dataset = LaneDataset(args.dataset_dir, ops.join(args.data_dir, 'val.json'), args, data_aug=True)
    loader = get_loader(dataset, args)
    anchor_x_steps = dataset.anchor_x_steps

    # initialize visualizer
    visualizer = Visualizer(args)

    # get a batch of data/label pairs from loader
    for batch_ndx, (image_tensor, gt_tensor, idx, gt_cam_height, gt_cam_pitch, aug_mat) in enumerate(loader):
        print('batch id: {:d}, image tensor shape:'.format(batch_ndx))
        print(image_tensor.shape)
        print('batch id: {:d}, gt tensor shape:'.format(batch_ndx))
        print(gt_tensor.shape)

        # convert to BGR and numpy for visualization in opencv
        images = image_tensor.permute(0, 2, 3, 1).data.cpu().numpy()
        gt_anchors = gt_tensor.numpy()
        idx = idx.numpy()
        gt_cam_height = gt_cam_height.numpy()
        gt_cam_pitch = gt_cam_pitch.numpy()
        aug_mat = aug_mat.numpy()
        for i in range(args.batch_size):
            img = images[i]
            img = img * np.array(args.vgg_std).astype(np.float32)
            img = img + np.array(args.vgg_mean).astype(np.float32)
            if img.min() < 0. or img.max() > 1.0:
                print('found an invalid normalized sample')
            img = np.clip(img, 0, 1)

            if args.no_3d:
                H_g2im, H_crop, H_im2ipm = dataset.transform_mats(idx[i])
                M = np.matmul(H_crop, H_g2im)
                # update transformation with image augmentation
                M = np.matmul(aug_mat[i], M)
                x_2d, y_2d = homographic_transformation(M, vis_border_3d[:, 0], vis_border_3d[:, 1])
            else:
                P_g2im, H_crop, H_im2ipm = dataset.transform_mats(idx[i])
                M = np.matmul(H_crop, P_g2im)
                # update transformation with image augmentation
                M = np.matmul(aug_mat[i], M)
                x_2d, y_2d = projective_transformation(M, vis_border_3d[:, 0],
                                                       vis_border_3d[:, 1], np.zeros(vis_border_3d.shape[0]))
            # update transformation with image augmentation
            H_im2ipm = np.matmul(H_im2ipm, np.linalg.inv(aug_mat[i]))
            im_ipm = cv2.warpPerspective(img, H_im2ipm, (args.ipm_w, args.ipm_h))
            im_ipm = np.clip(im_ipm, 0, 1)

            # draw visual border on image to confirm calibration
            x_2d = x_2d.astype(np.int)
            y_2d = y_2d.astype(np.int)
            img = cv2.line(img, (x_2d[0], y_2d[0]), (x_2d[1], y_2d[1]), [1, 0, 0], 2)
            img = cv2.line(img, (x_2d[2], y_2d[2]), (x_2d[3], y_2d[3]), [1, 0, 0], 2)
            img = cv2.line(img, (x_2d[0], y_2d[0]), (x_2d[2], y_2d[2]), [1, 0, 0], 2)
            img = cv2.line(img, (x_2d[1], y_2d[1]), (x_2d[3], y_2d[3]), [1, 0, 0], 2)
            gt_anchor = gt_anchors[i, :, :]

            # un-normalize
            unormalize_lane_anchor(gt_anchor, dataset)

            # visualize ground-truth anchor lanelines by projecting them on the image
            img = visualizer.draw_on_img(img, gt_anchor, M, 'laneline', color=[0, 0, 1])
            if not args.no_centerline:
                img = visualizer.draw_on_img(img, gt_anchor, M, 'centerline', color=[0, 1, 0])

            cv2.putText(img, 'camara pitch: {:.3f}'.format(gt_cam_pitch[i]/np.pi*180),
                        (5, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 0, 1), thickness=2)
            cv2.putText(img, 'camara height: {:.3f}'.format(gt_cam_height[i]),
                        (5, 60), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 0, 1), thickness=2)

            # visualize on ipm
            im_ipm = visualizer.draw_on_ipm(im_ipm, gt_anchor, 'laneline', color=[0, 0, 1])
            if not args.no_centerline:
                im_ipm = visualizer.draw_on_ipm(im_ipm, gt_anchor, 'centerline', color=[0, 1, 0])

            # convert image to BGR for opencv imshow
            cv2.imshow('image gt check', np.flip(img, axis=2))
            cv2.imshow('ipm gt check', np.flip(im_ipm, axis=2))
            cv2.waitKey()
            print('image: {:d} in batch: {:d}'.format(i, batch_ndx))

    print('done')
