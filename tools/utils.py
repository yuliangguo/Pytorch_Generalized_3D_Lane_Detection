# -*- coding: utf-8 -*-

import argparse
import errno
import os
import sys

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.init as init
import torch.optim
from PIL import Image
from torch.optim import lr_scheduler
import os.path as ops

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
plt.rcParams['figure.figsize'] = (35, 30)


def define_args():
    parser = argparse.ArgumentParser(description='Lane_detection_all_objectives')
    # Paths settings
    parser.add_argument('--dataset_name', type=str, help='the dataset name to be used in saving model names')
    parser.add_argument('--data_dir', type=str, help='The path saving train.json and val.json files')
    parser.add_argument('--dataset_dir', type=str, help='The path saving actual data')
    parser.add_argument('--save_path', type=str, default='data/', help='directory to save output')
    # Dataset settings
    parser.add_argument('--org_h', type=int, default=720, help='height of the original image')
    parser.add_argument('--org_w', type=int, default=1280, help='width of the original image')
    parser.add_argument('--crop_size', type=int, default=0, help='crop from image')
    parser.add_argument('--cam_height', type=float, default=1.55, help='height of camera in meters')
    parser.add_argument('--pitch', type=float, default=3, help='pitch angle of camera to ground in centi degree')
    parser.add_argument('--fix_cam', type=str2bool, nargs='?', const=True, default=False, help='directory to save output')
    parser.add_argument('--no_3d', action='store_true', help='if a dataset include laneline 3D attributes')
    parser.add_argument('--no_centerline', action='store_true', help='if a dataset include centerline annotations')
    # 3DLaneNet settings
    parser.add_argument('--mod', type=str, default='3DLaneNet', help='model to train')
    parser.add_argument("--pretrained", type=str2bool, nargs='?', const=True, default=True, help="use pretrained vgg model")
    parser.add_argument("--batch_norm", type=str2bool, nargs='?', const=True, default=True, help="apply batch norm")
    parser.add_argument('--ipm_h', type=int, default=208, help='height of inverse projective map (IPM)')
    parser.add_argument('--ipm_w', type=int, default=128, help='width of inverse projective map (IPM)')
    parser.add_argument('--resize_h', type=int, default=360, help='height of the original image')
    parser.add_argument('--resize_w', type=int, default=480, help='width of the original image')
    parser.add_argument('--y_ref', type=float, default=20.0, help='the reference Y distance in meters from where lane association is determined')
    parser.add_argument('--prob_th', type=float, default=0.5, help='probability threshold for selecting output lanes')
    # General model settings
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--nepochs', type=int, default=350, help='total numbers of epochs')
    parser.add_argument('--learning_rate', type=float, default=5*1e-4, help='learning rate')
    parser.add_argument('--no_cuda', action='store_true', help='if gpu available')
    parser.add_argument('--nworkers', type=int, default=8, help='num of threads')
    parser.add_argument('--no_dropout', action='store_true', help='no dropout in network')
    parser.add_argument('--pretrain_epochs', type=int, default=20, help='Number of epochs to perform segmentation pretraining')
    parser.add_argument('--channels_in', type=int, default=3, help='num channels of input image')
    parser.add_argument('--flip_on', action='store_true', help='Random flip input images on?')
    parser.add_argument('--test_mode', action='store_true', help='prevents loading latest saved model')
    parser.add_argument('--start_epoch', type=int, default=0, help='prevents loading latest saved model')
    parser.add_argument('--evaluate', action='store_true', help='only perform evaluation')
    parser.add_argument('--resume', type=str, default='', help='resume latest saved run')
    parser.add_argument('--vgg_mean', type=float, default=[0.485, 0.456, 0.406], help='Mean of rgb used in pretrained model on ImageNet')
    parser.add_argument('--vgg_std', type=float, default=[0.229, 0.224, 0.225], help='Std of rgb used in pretrained model on ImageNet')
    # Optimizer settings
    parser.add_argument('--optimizer', type=str, default='adam', help='adam or sgd')
    parser.add_argument('--weight_init', type=str, default='kaiming', help='normal, xavier, kaiming, orhtogonal weights initialisation')
    parser.add_argument('--weight_decay', type=float, default=0, help='L2 weight decay/regularisation on?')
    parser.add_argument('--lr_decay', action='store_true', help='decay learning rate with rule')
    parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=400, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--lr_policy', default=None, help='learning rate policy: lambda|step|plateau')
    parser.add_argument('--lr_decay_iters', type=int, default=30, help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--clip_grad_norm', type=int, default=0, help='performs gradient clipping')
    # CUDNN usage
    parser.add_argument("--cudnn", type=str2bool, nargs='?', const=True, default=True, help="cudnn optimization active")
    # Tensorboard settings
    parser.add_argument("--no_tb", type=str2bool, nargs='?', const=True, default=True, help="Use tensorboard logging by tensorflow")
    # Print settings
    parser.add_argument('--print_freq', type=int, default=500, help='padding')
    parser.add_argument('--save_freq', type=int, default=500, help='padding')
    # Skip batch
    parser.add_argument('--list', type=int, nargs='+', default=[954, 2789], help='Images you want to skip')

    return parser


def tusimple_config(args):

    # set dataset parameters
    args.save_path = ops.join(args.save_path, args.dataset_name)
    args.org_h = 720
    args.org_w = 1280
    args.crop_size = 80
    args.no_centerline = True
    args.no_3d = True
    args.fix_cam = True

    # set camera parameters for the test dataset
    args.K = np.array([[1000, 0, 640],
                       [0, 1000, 400],
                       [0, 0, 1]])
    args.cam_height = 1.6
    args.pitch = 9

    # specify model settings
    """
    paper presented params:
        args.top_view_region = np.array([[-10, 85], [10, 85], [-10, 5], [10, 5]])
        args.anchor_y_steps = np.array([5, 20, 40, 60, 80, 100])
    """
    args.top_view_region = np.array([[-10, 81], [10, 81], [-10, 1], [10, 1]])
    args.anchor_y_steps = np.array([2, 3, 5, 10, 15, 20, 30, 40, 60, 80])
    args.num_y_steps = len(args.anchor_y_steps)

    # initialize with pre-trained vgg weights: paper suggested true
    args.pretrained = False
    # apply batch norm in network
    args.batch_norm = True


def apollo_sim_config(args):

    # set dataset parameters
    args.save_path = ops.join(args.save_path, args.dataset_name)
    args.org_h = 1080
    args.org_w = 1920
    args.crop_size = 0
    args.no_centerline = False
    args.no_3d = False
    args.fix_cam = False

    # set camera parameters for the test dataset
    args.K = np.array([[2015., 0., 960.],
                       [0., 2015., 540.],
                       [0., 0., 1.]])

    # specify model settings
    """
    paper presented params:
        args.top_view_region = np.array([[-10, 85], [10, 85], [-10, 5], [10, 5]])
        args.anchor_y_steps = np.array([5, 20, 40, 60, 80, 100])
    """
    args.top_view_region = np.array([[-10, 81], [10, 81], [-10, 1], [10, 1]])
    args.anchor_y_steps = np.array([2, 3, 5, 10, 15, 20, 30, 40, 60, 80])
    args.num_y_steps = len(args.anchor_y_steps)

    # initialize with pre-trained vgg weights: paper suggested true
    args.pretrained = False
    # apply batch norm in network
    args.batch_norm = True


# TODO: decide whetehr put visualizer functions under dataset class
# for 3D, ipm view needs no change, but projection to image view needs to apply projective transformation
class Visualizer:
    def __init__(self, args):
        self.no_3d = args.no_3d
        self.no_centerline = args.no_centerline
        self.vgg_mean = args.vgg_mean
        self.vgg_std = args.vgg_std
        self.save_path = args.save_path
        self.ipm_w = args.ipm_w
        self.ipm_h = args.ipm_h
        self.num_y_steps = args.num_y_steps

        if args.no_3d:
            self.anchor_dim = args.num_y_steps + 1
        else:
            self.anchor_dim = 2 * args.num_y_steps + 1

        x_min = args.top_view_region[0, 0]
        x_max = args.top_view_region[1, 0]
        self.anchor_x_steps = np.linspace(x_min, x_max, np.int(args.ipm_w / 8), endpoint=True)
        self.anchor_y_steps = args.anchor_y_steps

        # transformation from ipm to ground region
        M_ipm2g = cv2.getPerspectiveTransform(np.float32([[0, 0],
                                                          [self.ipm_w-1, 0],
                                                          [0, self.ipm_h-1],
                                                          [self.ipm_w-1, self.ipm_h-1]]),
                                              np.float32(args.top_view_region))
        # M_im2g = np.matmul(H_ipm2g, self.M_im2ipm)
        # self.M_g2im = np.linalg.inv(M_im2g)
        self.M_g2ipm = np.linalg.inv(M_ipm2g)

        # probability threshold for choosing visualize lanes
        self.prob_th = args.prob_th

    def draw_on_img(self, img, lane_anchor, P_g2im, draw_type='laneline', color=[0, 0, 1]):
        """
        :param img: image in numpy array, each pixel in [0, 1] range
        :param lane_anchor: lane anchor in N X C numpy ndarray, dimension in agree with dataloader
        :param P_g2im: projection from ground 3D coordinates to image 2D coordinates
        :param draw_type: 'laneline' or 'centerline' deciding which to draw
        :param color: [r, g, b] color for line,  each range in [0, 1]
        :return:
        """

        for j in range(lane_anchor.shape[0]):
            # draw laneline
            if draw_type is 'laneline' and lane_anchor[j, self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, :self.num_y_steps]
                x_3d = x_offsets + self.anchor_x_steps[j]
                if self.no_3d:
                    x_2d, y_2d = homographic_transformation(P_g2im, x_3d, self.anchor_y_steps)
                else:
                    z_3d = lane_anchor[j, self.num_y_steps:self.anchor_dim - 1]
                    x_2d, y_2d = projective_transformation(P_g2im, x_3d, self.anchor_y_steps, z_3d)
                x_2d = x_2d.astype(np.int)
                y_2d = y_2d.astype(np.int)
                for k in range(1, x_2d.shape[0]):
                    img = cv2.line(img, (x_2d[k - 1], y_2d[k - 1]), (x_2d[k], y_2d[k]), color, 2)

            # draw centerline
            if draw_type is 'centerline' and lane_anchor[j, 2 * self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, self.anchor_dim:self.anchor_dim + self.num_y_steps]
                x_3d = x_offsets + self.anchor_x_steps[j]
                if self.no_3d:
                    x_2d, y_2d = homographic_transformation(P_g2im, x_3d, self.anchor_y_steps)
                else:
                    z_3d = lane_anchor[j, self.anchor_dim + self.num_y_steps:2 * self.anchor_dim - 1]
                    x_2d, y_2d = projective_transformation(P_g2im, x_3d, self.anchor_y_steps, z_3d)
                x_2d = x_2d.astype(np.int)
                y_2d = y_2d.astype(np.int)
                for k in range(1, x_2d.shape[0]):
                    img = cv2.line(img, (x_2d[k - 1], y_2d[k - 1]), (x_2d[k], y_2d[k]), color, 2)

            # draw the additional centerline for the merging case
            if draw_type is 'centerline' and  lane_anchor[j, 3 * self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, 2 * self.anchor_dim:2 * self.anchor_dim + self.num_y_steps]
                x_3d = x_offsets + self.anchor_x_steps[j]
                if self.no_3d:
                    x_2d, y_2d = homographic_transformation(P_g2im, x_3d, self.anchor_y_steps)
                else:
                    z_3d = lane_anchor[j, 2 * self.anchor_dim + self.num_y_steps:3 * self.anchor_dim - 1]
                    x_2d, y_2d = projective_transformation(P_g2im, x_3d, self.anchor_y_steps, z_3d)
                x_2d = x_2d.astype(np.int)
                y_2d = y_2d.astype(np.int)
                for k in range(1, x_2d.shape[0]):
                    img = cv2.line(img, (x_2d[k - 1], y_2d[k - 1]), (x_2d[k], y_2d[k]), color, 2)
        return img

    def draw_on_ipm(self, im_ipm, lane_anchor, draw_type='laneline', color=[0, 0, 1]):
        for j in range(lane_anchor.shape[0]):
            # draw laneline
            if draw_type is 'laneline' and lane_anchor[j, self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, :self.num_y_steps]
                x_g = x_offsets + self.anchor_x_steps[j]

                # compute lanelines in ipm view
                x_ipm, y_ipm = homographic_transformation(self.M_g2ipm, x_g, self.anchor_y_steps)
                x_ipm = x_ipm.astype(np.int)
                y_ipm = y_ipm.astype(np.int)
                for k in range(1, x_g.shape[0]):
                    im_ipm = cv2.line(im_ipm, (x_ipm[k - 1], y_ipm[k - 1]),
                                      (x_ipm[k], y_ipm[k]), color, 1)

            # draw centerline
            if draw_type is 'centerline' and lane_anchor[j, 2 * self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, self.anchor_dim:self.anchor_dim + self.num_y_steps]
                x_g = x_offsets + self.anchor_x_steps[j]

                # compute lanelines in ipm view
                x_ipm, y_ipm = homographic_transformation(self.M_g2ipm, x_g, self.anchor_y_steps)
                x_ipm = x_ipm.astype(np.int)
                y_ipm = y_ipm.astype(np.int)
                for k in range(1, x_g.shape[0]):
                    im_ipm = cv2.line(im_ipm, (x_ipm[k - 1], y_ipm[k - 1]),
                                      (x_ipm[k], y_ipm[k]), color, 1)

            # draw the additional centerline for the merging case
            if draw_type is 'centerline' and lane_anchor[j, 3 * self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, 2 * self.anchor_dim:2 * self.anchor_dim + self.num_y_steps]
                x_g = x_offsets + self.anchor_x_steps[j]

                # compute lanelines in ipm view
                x_ipm, y_ipm = homographic_transformation(self.M_g2ipm, x_g, self.anchor_y_steps)
                x_ipm = x_ipm.astype(np.int)
                y_ipm = y_ipm.astype(np.int)
                for k in range(1, x_g.shape[0]):
                    im_ipm = cv2.line(im_ipm, (x_ipm[k - 1], y_ipm[k - 1]),
                                      (x_ipm[k], y_ipm[k]), color, 1)
        return im_ipm

    def save_result(self, train_or_val, epoch, batch_i, idx, images, gt, pred, dataset, evaluate=False):
        for i in range(idx.shape[0]):
            # during training, only visualize the first sample of this batch
            if i > 0 and not evaluate:
                break
            im = images.permute(0, 2, 3, 1).data.cpu().numpy()[i]
            # the vgg_std and vgg_mean are for images in [0, 1] range
            im = im * np.array(self.vgg_std)
            im = im + np.array(self.vgg_mean)
            im = np.clip(im, 0, 1)

            gt_anchors = gt.data.cpu().numpy()[i]
            pred_anchors = pred.data.cpu().numpy()[i]
            # apply nms to avoid output directly neighbored lanes
            pred_anchors[:, -1] = nms_1d(pred_anchors[:, -1])

            if self.no_3d:
                H_g2im, H_crop, H_im2ipm = dataset.proj_trainsforms(idx[i])
                M_gt = np.matmul(H_crop, H_g2im)
            else:
                P_g2im, H_crop, H_im2ipm = dataset.proj_trainsforms(idx[i])
                M_gt = np.matmul(H_crop, P_g2im)

            im_ipm = cv2.warpPerspective(im, H_im2ipm, (self.ipm_w, self.ipm_h))
            im_ipm = np.clip(im_ipm, 0, 1)

            # draw lanes on image
            im_laneline = im.copy()
            im_laneline = self.draw_on_img(im_laneline, gt_anchors, M_gt, 'laneline', [0, 0, 1])
            # TODO need to use predicted pitch and height to compute M
            im_laneline = self.draw_on_img(im_laneline, pred_anchors, M_gt, 'laneline', [1, 0, 0])
            if not dataset.no_centerline:
                im_centerline = im.copy()
                im_centerline = self.draw_on_img(im_centerline, gt_anchors, M_gt, 'centerline', [0, 0, 1])
                # TODO need to use predicted pitch and height to compute M
                im_centerline = self.draw_on_img(im_centerline, pred_anchors, M_gt, 'centerline', [1, 0, 0])

            # draw lanes on ipm
            ipm_laneline = im_ipm.copy()
            ipm_laneline = self.draw_on_ipm(ipm_laneline, gt_anchors, 'laneline', [0, 0, 1])
            ipm_laneline = self.draw_on_ipm(ipm_laneline, pred_anchors, 'laneline', [1, 0, 0])
            if not dataset.no_centerline:
                ipm_centerline = im_ipm.copy()
                ipm_centerline = self.draw_on_ipm(ipm_centerline, gt_anchors, 'centerline', [0, 0, 1])
                ipm_centerline = self.draw_on_ipm(ipm_centerline, pred_anchors, 'centerline', [1, 0, 0])

            if self.no_centerline:
                fig = plt.figure()
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
                ax1.imshow(im_laneline)
                ax2.imshow(ipm_laneline)
            else:
                fig = plt.figure()
                ax1 = fig.add_subplot(221)
                ax2 = fig.add_subplot(222)
                ax3 = fig.add_subplot(223)
                ax4 = fig.add_subplot(224)
                ax1.imshow(im_laneline)
                ax2.imshow(ipm_laneline)
                ax3.imshow(im_centerline)
                ax4.imshow(ipm_centerline)

            if evaluate:
                fig.savefig(self.save_path + '/example/eval_vis/infer_{}'.format(idx[i]))
            else:
                fig.savefig(self.save_path + '/example/{}/epoch-{}_batch-{}_idx-{}'.format(train_or_val,
                                                                                           epoch, batch_i, idx[i]))
            plt.clf()
            plt.close(fig)


# compute normalized transformation matrix for a top-view region boundaries
def init_projective_transform(top_view_region, org_img_size, crop_y, resize_img_size, cam_pitch, cam_height, K):
    """
        Compute the normalized transformation (M_inv) such that image region corresponding to top_view region maps to
        the top view image's 4 corners
        Ground coordinates: x-right, y-forward, z-up
        The purpose of applying normalized transformation is for invariance in scale change

    :param top_view_region: a 4 X 2 list of (X, Y) indicating the top-view region corners in order:
                            top-left, top-right, bottom-left, bottom-right
    :param batch_size: number of samples for each batch
    :param org_img_size: the size of original image size: (h, w)
    :param crop_y: pixels croped from original img
    :param resize_img_size: the size of image as network input: (h, w)
    :param cam_pitch: camera pitch angle wrt ground plane
    :param cam_height: camera height wrt ground plane in meters
    :param K: camera intrinsic parameters
    :return: M: the normalized transformation from image to IPM image
    """

    # compute homography transformation from ground to image
    H_g2im = homograpthy_g2im(cam_pitch, cam_height, K)

    # transform original image region to network input region
    H_c = homography_crop_resize(org_img_size, crop_y, resize_img_size)

    # compute top-view corners' coordinates in image
    P = np.matmul(H_c, H_g2im)
    x_2d, y_2d = homographic_transformation(P, top_view_region[:, 0], top_view_region[:, 1])
    border_net = np.concatenate([x_2d.reshape(-1, 1), y_2d.reshape(-1, 1)], axis=1)

    # compute the normalized transformation
    border_net[:, 0] = border_net[:, 0] / resize_img_size[1]
    border_net[:, 1] = border_net[:, 1] / resize_img_size[0]
    border_net = np.float32(border_net)
    dst = np.float32([[0, 0], [1, 0], [0, 1], [1, 1]])
    # img to ipm
    M_im2g_norm = cv2.getPerspectiveTransform(border_net, dst)
    # ipm to im
    M_g2im_norm = cv2.getPerspectiveTransform(dst, border_net)
    return M_im2g_norm, M_g2im_norm


def homograpthy_g2im(cam_pitch, cam_height, K):
    # transform top-view region to original image region
    R_g2c = np.array([[1, 0, 0],
                      [0, np.cos(np.pi / 2 + cam_pitch), -np.sin(np.pi / 2 + cam_pitch)],
                      [0, np.sin(np.pi / 2 + cam_pitch), np.cos(np.pi / 2 + cam_pitch)]])
    H_g2im = np.matmul(K, np.concatenate([R_g2c[:, 0:2], [[0], [cam_height], [0]]], 1))
    return H_g2im


def projection_g2im(cam_pitch, cam_height, K):
    P_g2c = np.array([[1,                             0,                              0,          0],
                         [0, np.cos(np.pi / 2 + cam_pitch), -np.sin(np.pi / 2 + cam_pitch), cam_height],
                         [0, np.sin(np.pi / 2 + cam_pitch),  np.cos(np.pi / 2 + cam_pitch),          0]])
    P_g2im = np.matmul(K, P_g2c)
    return P_g2im


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
                    [0, ratio_y, -ratio_y*crop_y],
                    [0, 0, 1]])
    return H_c


def homographic_transformation(Matrix, x, y):
    """
    Helper function to transform coordinates defined by transformation matrix

    Args:
            Matrix (multi dim - array): 3x3 homography matrix
            x (array): original x coordinates
            y (array): original y coordinates
    """
    ones = np.ones((1, len(y)))
    coordinates = np.vstack((x, y, ones))
    trans = np.matmul(Matrix, coordinates)

    x_vals = trans[0, :]/trans[2, :]
    y_vals = trans[1, :]/trans[2, :]
    return x_vals, y_vals


def projective_transformation(Matrix, x, y, z):
    """
    Helper function to transform coordinates defined by transformation matrix

    Args:
            Matrix (multi dim - array): 3x4 projection matrix
            x (array): original x coordinates
            y (array): original y coordinates
            z (array): original z coordinates
    """
    ones = np.ones((1, len(z)))
    coordinates = np.vstack((x, y, z, ones))
    trans = np.matmul(Matrix, coordinates)

    x_vals = trans[0, :]/trans[2, :]
    y_vals = trans[1, :]/trans[2, :]
    return x_vals, y_vals


def nms_1d(v):
    """

    :param v: a 1D numpy array
    :return:
    """
    v_out = v.copy()
    len = v.shape[0]
    if len < 2:
        return v
    for i in range(len):
        if i is not 0 and v[i - 1] > v[i]:
            v_out[i] = 0.
        elif i is not len-1 and v[i+1] > v[i]:
            v_out[i] = 0.
    return v_out


def first_run(save_path):
    txt_file = os.path.join(save_path,'first_run.txt')
    if not os.path.exists(txt_file):
        open(txt_file, 'w').close()
    else:
        saved_epoch = open(txt_file).read()
        if saved_epoch is None:
            print('You forgot to delete [first run file]')
            return '' 
        return saved_epoch
    return ''


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


# trick from stackoverflow
def str2bool(argument):
    if argument.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif argument.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Wrong argument in argparse, should be a boolean')


class Logger(object):
    """
    Source https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        self.fpath = fpath
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def define_optim(optim, params, lr, weight_decay):
    if optim == 'adam':
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optim == 'sgd':
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise KeyError("The requested optimizer: {} is not implemented".format(optim))
    return optimizer


def define_scheduler(optimizer, args):
    if args.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - args.niter) / float(args.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=args.lr_decay_iters, gamma=args.gamma)
    elif args.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                   factor=args.gamma,
                                                   threshold=0.0001,
                                                   patience=args.lr_decay_iters)
    elif args.lr_policy == 'none':
        scheduler = None
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


def define_init_weights(model, init_w='normal', activation='relu'):
    print('Init weights in network with [{}]'.format(init_w))
    if init_w == 'normal':
        model.apply(weights_init_normal)
    elif init_w == 'xavier':
        model.apply(weights_init_xavier)
    elif init_w == 'kaiming':
        model.apply(weights_init_kaiming)
    elif init_w == 'orthogonal':
        model.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [{}] is not implemented'.format(init_w))


def weights_init_normal(m):
    classname = m.__class__.__name__
#    print(classname)
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
#    print(classname)
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
        init.orthogonal(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
