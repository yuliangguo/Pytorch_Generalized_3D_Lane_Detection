import numpy as np
from scipy import ndimage
import cv2
import os
import os.path as ops
import math
import ujson as json
from tools.utils import define_args, homography_im2ipm_norm,\
    homographic_transformation, projective_transformation,\
    homograpthy_g2im, projection_g2im, homography_crop_resize,\
    tusimple_config, sim3d_config

color = [[0, 0, 255],  # red
         [0, 255, 0],  # green
         [255, 0, 255],  # purple
         [255, 255, 0]]  # cyan

class LaneEval(object):
    def __init__(self, args):
        self.pt_th = args.pt_th
        self.min_num_pixels = args.min_num_pixels
        self.dataset_dir = args.dataset_dir
        self.top_view_region = args.top_view_region
        self.org_h = args.org_h
        self.org_w = args.org_w
        self.crop_y = args.crop_size
        self.resize_h = args.resize_h
        self.resize_w = args.resize_w
        self.K = args.K
        self.no_centerline = args.no_centerline

        # use ipm keeping the aspect ratio of top-view region
        top_view_w = args.top_view_region[1, 0] - args.top_view_region[0, 0]
        top_view_h = args.top_view_region[0, 1] - args.top_view_region[2, 1]
        self.ipm_w = (top_view_w * args.pixel_per_meter).astype(np.int)
        self.ipm_h = (top_view_h * args.pixel_per_meter).astype(np.int)

        # pixel distance threshold in IPM for matching lanes
        self.dist_th = args.dist_th * args.pixel_per_meter

        # scale up transforms the normalized homographic transformation from network input image to ipm image
        self.S_ipm = np.array([[self.ipm_w, 0, 0],
                              [0, self.ipm_h, 0],
                              [0, 0, 1]], dtype=np.float)
        self.S_im = np.array([[args.resize_w, 0, 0],
                             [0, args.resize_h, 0],
                             [0, 0, 1]], dtype=np.float)

        # transformation from ipm to ground region
        self.M_ipm2g = cv2.getPerspectiveTransform(np.float32([[0, 0],
                                                              [self.ipm_w-1, 0],
                                                              [0, self.ipm_h-1],
                                                              [self.ipm_w-1, self.ipm_h-1]]),
                                                   np.float32(args.top_view_region))
        self.M_g2ipm = np.linalg.inv(self.M_ipm2g)

        self.H_crop = homography_crop_resize([self.org_h, self.org_w], self.crop_y, [self.resize_h, self.resize_w])

    def bench(self, pred_lanes, gt_lanes, raw_file, gt_cam_height, gt_cam_pitch, vis):
        """
            Matching predicted lanes and ground-truth lanes in their IPM projection, ignoring z attributes.
            x error, y_error, and z error are all considered, although the matching does not rely on z
            The input of prediction and ground-truth lanes are in ground coordinate, x-right, y-forward, z-up
            The fundamental assumption is: 1. there are no two points from different lanes with identical x, y
                                              but different z's
                                           2. there are no two points from a single lane having identical x, y
                                              but different z's
            If the interest area is within the current drivable road, the above assumptions are almost always valid.

        :param pred_lanes: N X 2 or N X 3 lists depending on 2D or 3D
        :param gt_lanes: N X 2 or N X 3 lists depending on 2D or 3D
        :param raw_file: file path rooted in dataset folder
        :param gt_cam_height: camera height given in ground-truth data
        :param gt_cam_pitch: camera pitch given in ground-truth data
        :return:
        """
        r_pixel, p_pixel = 0., 0.
        r_lane, p_lane = 0., 0.
        cnt_gt = len(gt_lanes)
        cnt_pred = len(pred_lanes)

        # TODO: vary confidence to compute all stats in vectors, aiming to generate PR curve

        # compute curve to curve distance


        # solve bipartite matching based on adjacency matrix


        #

        return r_pixel, p_pixel, cnt_gt_pixel, cnt_pred_pixel, r_lane, p_lane, cnt_gt_out, cnt_pred_out, vis_ipm

    def bench_one_submit(self, pred_file, gt_file, vis=False):
        if vis:
            save_path = pred_file[:pred_file.rfind('/')]
            save_path += '/example/eval_vis'
            if vis and not os.path.exists(save_path):
                try:
                    os.makedirs(save_path)
                except OSError as e:
                    print(e.message)
        try:
            json_pred = [json.loads(line) for line in open(pred_file).readlines()]
        except BaseException as e:
            raise Exception('Fail to load json file of the prediction.')
        json_gt = [json.loads(line) for line in open(gt_file).readlines()]
        if len(json_gt) != len(json_pred):
            raise Exception('We do not get the predictions of all the test tasks')
        gts = {l['raw_file']: l for l in json_gt}

        laneline_stats = np.zeros(8)
        centerline_stats = np.zeros(8)
        for i, pred in enumerate(json_pred):
            if 'raw_file' not in pred or 'laneLines' not in pred:
                raise Exception('raw_file or lanelines not in some predictions.')
            raw_file = pred['raw_file']
            pred_lanelines = pred['laneLines']
            if raw_file not in gts:
                raise Exception('Some raw_file from your predictions do not exist in the test tasks.')
            gt = gts[raw_file]
            gt_cam_height = gt['cam_height']
            gt_cam_pitch = gt['cam_pitch']

            # evaluate lanelines
            gt_lanelines = gt['laneLines']

            # TODO sample gt_laneline and pred_laneline in y
            y_samples =

            # N to N matching of lanelines
            r_pixel, p_pixel, gt_pixel, pred_pixel,\
                r_lane, p_lane, cnt_gt, cnt_pred, vis_map = self.bench(pred_lanelines,
                                                                       gt_lanelines,
                                                                       raw_file,
                                                                       gt_cam_height,
                                                                       gt_cam_pitch,
                                                                       vis)
            # if math.isnan(p_pixel) or math.isnan(r_pixel) or math.isnan(p_lane) or math.isnan(r_pixel):
            #     break
            laneline_stats += np.array([r_pixel, p_pixel, gt_pixel, pred_pixel, r_lane, p_lane, cnt_gt, cnt_pred])
            # save visualize map
            if vis:
                # img_name = raw_file.split('/')[-1]
                img_name = raw_file.replace("/", "_")
                cv2.imwrite(save_path + '/laneline_' + img_name, vis_map)

            # evaluate centerlines
            if not self.no_centerline:
                pred_centerlines = pred['centerLines']
                gt_centerlines = gt['centerLines']

                # N to N matching of lanelines
                r_pixel, p_pixel, gt_pixel, pred_pixel,\
                    r_lane, p_lane, cnt_gt, cnt_pred, vis_map = self.bench(pred_centerlines,
                                                                           gt_centerlines,
                                                                           raw_file,
                                                                           gt_cam_height,
                                                                           gt_cam_pitch,
                                                                           vis)
                centerline_stats += np.array([r_pixel, p_pixel, gt_pixel, pred_pixel, r_lane, p_lane, cnt_gt, cnt_pred])
                # save visualize map
                if vis:
                    # img_name = raw_file.split('/')[-1]
                    img_name = raw_file.replace("/", "_")
                    cv2.imwrite(save_path + '/centerline_' + img_name, vis_map)

            # print('processed sample: {}'.format(i))

        output_stats = []
        r_pixel, p_pixel, gt_pixel, pred_pixel,\
            r_lane, p_lane, cnt_gt, cnt_pred = [stat for stat in laneline_stats]

        R_pixel = r_pixel / gt_pixel
        P_pixel = p_pixel / pred_pixel
        F_pixel = 2 * R_pixel * P_pixel / (R_pixel + P_pixel + 1e-6)
        R_lane = r_lane / cnt_gt
        P_lane = p_lane / cnt_pred
        F_lane = 2 * R_lane * P_lane / (R_lane + P_lane + 1e-6)

        output_stats.append(F_pixel)
        output_stats.append(F_lane)

        if not self.no_centerline:
            r_pixel, p_pixel, gt_pixel, pred_pixel, \
            r_lane, p_lane, cnt_gt, cnt_pred = [stat for stat in centerline_stats]

            R_pixel = r_pixel / gt_pixel
            P_pixel = p_pixel / pred_pixel
            F_pixel = 2 * R_pixel * P_pixel / (R_pixel + P_pixel + 1e-6)
            R_lane = r_lane / cnt_gt
            P_lane = p_lane / cnt_pred
            F_lane = 2 * R_lane * P_lane / (R_lane + P_lane + 1e-6)

            output_stats.append(F_pixel)
            output_stats.append(F_lane)

        # TODO: generate PR curve, output max_F or AP (not appropriate) as key indicator


        # TODO: for the case of max_F, compute std of x error and z error for close range (0-30 meters) and far range (30 - 80 meters) separately

        return output_stats


if __name__ == '__main__':

    parser = define_args()
    args = parser.parse_args()

    args.dataset_name = 'sim3d'
    args.data_dir = ops.join('../data', args.dataset_name)
    args.dataset_dir = '/home/yuliangguo/Datasets/Apollo_Sim_3D_Lane/'

    # load configuration for certain dataset
    sim3d_config(args)

    args.pixel_per_meter = 10.
    args.dist_th = 1.5
    args.pt_th = 0.5
    args.min_num_pixels = 10
    evaluator = LaneEval(args)

    pred_file = '../data/sim3d/val.json'
    gt_file = '../data/sim3d/val.json'

    # try:
    print(evaluator.bench_one_submit(pred_file, gt_file, vis=True))