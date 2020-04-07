"""
Description: Visualization code to draw predicted lane-lines and center-lines in three views: image, virtual top,
             3D ego-car. Respectively, lane-lines are shown in the top row, center-lines are drawn in the bottom.
Author: Yuliang Guo (33yuliangguo@gmail.com)
Date: March, 2020
"""

import numpy as np
import cv2
import os
import os.path as ops
import math
import ujson as json
import matplotlib
from tools.utils import *
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (35, 30)
plt.rcParams.update({'font.size': 25})
plt.rcParams.update({'font.weight': 'semibold'})

min_y = 0
max_y = 80

colors = [[1, 0, 0],  # red
          [0, 1, 0],  # green
          [0, 0, 1],  # blue
          [1, 0, 1],  # purple
          [0, 1, 1],  # cyan
          [1, 0.7, 0]]  # orange


class lane_visualizer(object):
    def __init__(self, args):
        self.dataset_dir = args.dataset_dir
        self.K = args.K
        self.no_centerline = args.no_centerline
        """
            this visualizer use higher resolution than network input for better look
        """
        self.resize_h = args.org_h
        self.resize_w = args.org_w
        # self.resize_h = args.resize_h
        # self.resize_w = args.resize_w
        self.ipm_w = 2*args.ipm_w
        self.ipm_h = 2*args.ipm_h
        self.H_crop = homography_crop_resize([args.org_h, args.org_w], args.crop_y, [self.resize_h, self.resize_w])
        # transformation from ipm to ground region
        self.H_ipm2g = cv2.getPerspectiveTransform(np.float32([[0, 0],
                                                              [self.ipm_w-1, 0],
                                                              [0, self.ipm_h-1],
                                                              [self.ipm_w-1, self.ipm_h-1]]),
                                                   np.float32(args.top_view_region))
        self.H_g2ipm = np.linalg.inv(self.H_ipm2g)

        self.x_min = args.top_view_region[0, 0]
        self.x_max = args.top_view_region[1, 0]
        # self.y_samples = np.linspace(args.anchor_y_steps[0], args.anchor_y_steps[-1], num=100, endpoint=False)
        self.y_samples = np.linspace(min_y, max_y, num=100, endpoint=False)

    def visualize_lanes(self, pred_lanes, raw_file, gt_cam_height, gt_cam_pitch, ax1, ax2, ax3):
        P_g2im = projection_g2im(gt_cam_pitch, gt_cam_height, self.K)
        # P_gt = P_g2im
        P_gt = np.matmul(self.H_crop, P_g2im)
        H_g2im = homograpthy_g2im(gt_cam_pitch, gt_cam_height, self.K)
        H_im2ipm = np.linalg.inv(np.matmul(self.H_crop, np.matmul(H_g2im, self.H_ipm2g)))

        img = cv2.imread(ops.join(self.dataset_dir, raw_file))
        img = cv2.warpPerspective(img, self.H_crop, (self.resize_w, self.resize_h))
        img = img.astype(np.float) / 255
        im_ipm = cv2.warpPerspective(img, H_im2ipm, (self.ipm_w, self.ipm_h))
        im_ipm = np.clip(im_ipm, 0, 1)

        cnt_pred = len(pred_lanes)
        pred_visibility_mat = np.zeros((cnt_pred, 100))
        for i in range(cnt_pred):
            # # ATTENTION: ensure y mono increase before interpolation: but it can reduce size
            # pred_lanes[i] = make_lane_y_mono_inc(np.array(pred_lanes[i]))
            # pred_lane = prune_3d_lane_by_range(np.array(pred_lanes[i]), self.x_min, self.x_max)
            min_y = np.min(np.array(pred_lanes[i])[:, 1])
            max_y = np.max(np.array(pred_lanes[i])[:, 1])
            x_values, z_values, visibility_vec = resample_laneline_in_y(np.array(pred_lanes[i]), self.y_samples, out_vis=True)
            pred_lanes[i] = np.vstack([x_values, z_values]).T
            pred_visibility_mat[i, :] = np.logical_and(x_values >= self.x_min,
                                                       np.logical_and(x_values <= self.x_max,
                                                                      np.logical_and(self.y_samples >= min_y,
                                                                                     self.y_samples <= max_y)))
            pred_visibility_mat[i, :] = np.logical_and(pred_visibility_mat[i, :], visibility_vec)

        # draw lanes in multiple color
        for i in range(cnt_pred):
            x_values = pred_lanes[i][:, 0]
            z_values = pred_lanes[i][:, 1]
            # if 'gflat' in pred_file or 'ext' in pred_file:
            x_ipm_values, y_ipm_values = transform_lane_g2gflat(gt_cam_height, x_values, self.y_samples, z_values)
            # remove those points with z_values > gt_cam_height, this is only for visualization on top-view
            x_ipm_values = x_ipm_values[np.where(z_values < gt_cam_height)]
            y_ipm_values = y_ipm_values[np.where(z_values < gt_cam_height)]
            # else:  # mean to visualize original anchor's preparation
            #     x_ipm_values = x_values
            #     y_ipm_values = self.y_samples
            x_ipm_values, y_ipm_values = homographic_transformation(self.H_g2ipm, x_ipm_values, y_ipm_values)
            x_ipm_values = x_ipm_values.astype(np.int)
            y_ipm_values = y_ipm_values.astype(np.int)
            x_2d, y_2d = projective_transformation(P_gt, x_values, self.y_samples, z_values)
            x_2d = x_2d.astype(np.int)
            y_2d = y_2d.astype(np.int)

            color = colors[np.mod(i, len(colors))]
            # draw on image
            for k in range(1, x_2d.shape[0]):
                # only draw the visible portion
                if pred_visibility_mat[i, k - 1] and pred_visibility_mat[i, k]:
                    img = cv2.line(img, (x_2d[k - 1], y_2d[k - 1]), (x_2d[k], y_2d[k]), color[-1::-1], 10)

            # draw on ipm
            for k in range(1, x_ipm_values.shape[0]):
                # only draw the visible portion
                if pred_visibility_mat[i, k - 1] and pred_visibility_mat[i, k]:
                    im_ipm = cv2.line(im_ipm, (x_ipm_values[k - 1], y_ipm_values[k - 1]), (x_ipm_values[k], y_ipm_values[k]), color[-1::-1], 3)

            # draw in 3d
            ax3.plot(x_values[np.where(pred_visibility_mat[i, :])],
                     self.y_samples[np.where(pred_visibility_mat[i, :])],
                     z_values[np.where(pred_visibility_mat[i, :])], color=color, linewidth=5)
        ax1.imshow(img[:, :, [2, 1, 0]])
        ax2.imshow(im_ipm[:, :, [2, 1, 0]])


if __name__ == '__main__':
    parser = define_args()
    args = parser.parse_args()

    # dataset_name: 'standard' / 'rare_subset' / 'illus_chg'
    args.dataset_name = 'illus_chg'
    args.dataset_dir = '/media/yuliangguo/DATA1/Datasets/Apollo_Sim_3D_Lane_Release/'

    # model name: 'Gen_LaneNet_ext' / '3D_LaneNet'
    model_name = 'Gen_LaneNet_ext'

    # load configuration for certain dataset
    sim3d_config(args)
    args.top_view_region = np.array([[-10, max_y], [10, max_y], [-10, 3], [10, 3]])
    vs = lane_visualizer(args)

    pred_file = '../data_splits/' + args.dataset_name + '/' + model_name + '/test_pred_file.json'
    gt_file = '../data_splits/' + args.dataset_name + '/test.json'

    save_path = pred_file[:pred_file.rfind('/')]
    save_path += '/example/test_vis_pred'
    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except OSError as e:
            print(e.message)

    pred_lines = open(pred_file).readlines()
    json_pred = [json.loads(line) for line in pred_lines]
    # except BaseException as e:
    #     raise Exception('Fail to load json file of the prediction.')
    json_gt = [json.loads(line) for line in open(gt_file).readlines()]
    if len(json_gt) != len(json_pred):
        raise Exception('We do not get the predictions of all the test tasks')
    gts = {l['raw_file']: l for l in json_gt}

    for i, pred in enumerate(json_pred):
        raw_file = pred['raw_file']

        pred_lanelines = pred['laneLines']
        pred_centerlines = pred['centerLines']

        if raw_file not in gts:
            continue
        gt = gts[raw_file]
        gt_cam_height = gt['cam_height']
        gt_cam_pitch = gt['cam_pitch']

        fig = plt.figure()
        ax1 = fig.add_subplot(231)
        ax2 = fig.add_subplot(232)
        ax3 = fig.add_subplot(233, projection='3d')
        ax4 = fig.add_subplot(234)
        ax5 = fig.add_subplot(235)
        ax6 = fig.add_subplot(236, projection='3d')

        # draw lanes
        vs.visualize_lanes(pred_lanelines, raw_file, gt_cam_height, gt_cam_pitch, ax1, ax2, ax3)
        vs.visualize_lanes(pred_centerlines, raw_file, gt_cam_height, gt_cam_pitch, ax4, ax5, ax6)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        # ax2.set_xlabel('x axis')
        # ax2.set_ylabel('y axis')
        # ax2.set_zlabel('z axis')
        bottom, top = ax3.get_zlim()
        left, right = ax3.get_xlim()
        ax3.set_zlim(min(bottom, -0.1), max(top, 0.1))
        ax3.set_xlim(left, right)
        ax3.set_ylim(min_y, max_y)
        ax3.locator_params(nbins=5, axis='x')
        ax3.locator_params(nbins=5, axis='z')
        ax3.tick_params(pad=18)

        ax4.set_xticks([])
        ax4.set_yticks([])
        ax5.set_xticks([])
        ax5.set_yticks([])
        # ax4.set_xlabel('x axis')
        # ax4.set_ylabel('y axis')
        # ax4.set_zlabel('z axis')
        bottom, top = ax6.get_zlim()
        left, right = ax6.get_xlim()
        ax6.set_zlim(min(bottom, -0.1), max(top, 0.1))
        ax6.set_xlim(left, right)
        ax6.set_ylim(min_y, max_y)
        ax6.locator_params(nbins=5, axis='x')
        ax6.locator_params(nbins=5, axis='z')
        ax6.tick_params(pad=18)

        fig.subplots_adjust(wspace=0, hspace=0.01)
        fig.savefig(ops.join(save_path, raw_file.replace("/", "_")))
        plt.close(fig)
        print('processed sample: {}  {}'.format(i, raw_file))