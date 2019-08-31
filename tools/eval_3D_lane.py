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
from Dataloader.Load_Data_3DLane import resample_laneline_in_y
from tools.MinCostFlow import SolveMinCostFlow

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

        dist_th = 1.5
        y_samples = np.linspace(1, 81, num=100, endpoint=False)
        close_range_idx = np.int((30 - 1) / 0.8)

        r_lane, p_lane = 0., 0.
        x_error_close = []
        x_error_far = []
        z_error_close = []
        z_error_far = []
        cnt_gt = len(gt_lanes)
        cnt_pred = len(pred_lanes)

        # resample gt and pred at y_samples
        for i in range(cnt_gt):
            x_values, z_values = resample_laneline_in_y(np.array(gt_lanes[i]), y_samples)
            gt_lanes[i] = np.vstack([x_values, z_values]).T

        for i in range(cnt_pred):
            x_values, z_values = resample_laneline_in_y(np.array(pred_lanes[i]), y_samples)
            pred_lanes[i] = np.vstack([x_values, z_values]).T

        # TODO: vary confidence to compute all stats in vectors, aiming to generate PR curve
        # TODO: it is necessary to visualize here? as the visualization has been done in testing

        adj_mat = np.zeros((cnt_gt, cnt_pred), dtype=np.int)
        cost_mat = np.zeros((cnt_gt, cnt_pred), dtype=np.int)
        cost_mat.fill(1000)
        x_dist_mat_close = np.zeros((cnt_gt, cnt_pred), dtype=np.float)
        x_dist_mat_close.fill(1000.)
        x_dist_mat_far = np.zeros((cnt_gt, cnt_pred), dtype=np.float)
        x_dist_mat_far.fill(1000.)
        z_dist_mat_close = np.zeros((cnt_gt, cnt_pred), dtype=np.float)
        z_dist_mat_close.fill(1000.)
        z_dist_mat_far = np.zeros((cnt_gt, cnt_pred), dtype=np.float)
        z_dist_mat_far.fill(1000.)
        # compute curve to curve distance
        for i in range(cnt_gt):
            for j in range(cnt_pred):
                x_dist = np.abs(gt_lanes[i][:, 0] - pred_lanes[j][:, 0])
                z_dist = np.abs(gt_lanes[i][:, 1] - pred_lanes[j][:, 1])
                euclidean_dist = np.sqrt(x_dist**2 + z_dist**2)
                # if np.average(euclidean_dist) < 2*dist_th: # don't prune here to encourage finding perfect match
                adj_mat[i, j] = 1
                x_dist_mat_close[i, j] = np.average(x_dist[:close_range_idx])
                x_dist_mat_far[i, j] = np.average(x_dist[close_range_idx:])
                z_dist_mat_close[i, j] = np.average(z_dist[:close_range_idx])
                z_dist_mat_far[i, j] = np.average(z_dist[close_range_idx:])
                # ATTENTION: use sum and int here to meet the requirements of min cost flow optimization
                cost_mat[i, j] = np.sum(euclidean_dist).astype(np.int)

        # solve bipartite matching vis min cost flow solver
        match_results = SolveMinCostFlow(adj_mat, cost_mat)
        match_results = np.array(match_results)

        if len(match_results) is not 0:
            # only match with avg cost < dist_th is consider valid
            r_lane = np.sum(match_results[:, 2] < dist_th * y_samples.shape[0])
            p_lane = np.sum(match_results[:, 2] < dist_th * y_samples.shape[0])
            for i in range(len(match_results)):
                x_error_close.append(x_dist_mat_close[match_results[i, 0], match_results[i, 1]])
                x_error_far.append(x_dist_mat_far[match_results[i, 0], match_results[i, 1]])
                z_error_close.append(z_dist_mat_close[match_results[i, 0], match_results[i, 1]])
                z_error_far.append(z_dist_mat_far[match_results[i, 0], match_results[i, 1]])

            x_error_close = np.average(np.array(x_error_close))
            x_error_far = np.average(np.array(x_error_far))
            z_error_close = np.average(np.array(z_error_close))
            z_error_far = np.average(np.array(z_error_far))
        else:
            x_error_close = 1000
            x_error_far = 1000
            z_error_close = 1000
            z_error_far = 1000
        return r_lane, p_lane, cnt_gt, cnt_pred, x_error_close, x_error_far, z_error_close, z_error_far

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
            pred_lines = open(pred_file).readlines()
            json_pred = [json.loads(line) for line in pred_lines]
        except BaseException as e:
            raise Exception('Fail to load json file of the prediction.')
        json_gt = [json.loads(line) for line in open(gt_file).readlines()]
        if len(json_gt) != len(json_pred):
            raise Exception('We do not get the predictions of all the test tasks')
        gts = {l['raw_file']: l for l in json_gt}

        laneline_stats = []
        laneline_x_error_close = []
        laneline_x_error_far = []
        laneline_z_error_close = []
        laneline_z_error_far = []
        centerline_stats = []
        centerline_x_error_close = []
        centerline_x_error_far = []
        centerline_z_error_close = []
        centerline_z_error_far = []
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

            # N to N matching of lanelines
            r_lane, p_lane, cnt_gt, cnt_pred, \
                x_error_close, x_error_far, \
                z_error_close, z_error_far = self.bench(pred_lanelines,
                                                        gt_lanelines,
                                                        raw_file,
                                                        gt_cam_height,
                                                        gt_cam_pitch,
                                                        vis)
            # if math.isnan(p_pixel) or math.isnan(r_pixel) or math.isnan(p_lane) or math.isnan(r_pixel):
            #     break
            laneline_stats.append(np.array([r_lane, p_lane, cnt_gt, cnt_pred]))
            # consider x_error z_error only for the matched lanes
            if r_lane > 0 and p_lane > 0:
                laneline_x_error_close.append(x_error_close)
                laneline_x_error_far.append(x_error_far)
                laneline_z_error_close.append(z_error_close)
                laneline_z_error_far.append(z_error_far)

            # save visualize map
            # if vis:
            #     # img_name = raw_file.split('/')[-1]
            #     img_name = raw_file.replace("/", "_")
            #     cv2.imwrite(save_path + '/laneline_' + img_name, vis_map)

            # evaluate centerlines
            if not self.no_centerline:
                pred_centerlines = pred['centerLines']
                gt_centerlines = gt['centerLines']

                # N to N matching of lanelines
                r_lane, p_lane, cnt_gt, cnt_pred, \
                    x_error_close, x_error_far, \
                    z_error_close, z_error_far = self.bench(pred_centerlines,
                                                            gt_centerlines,
                                                            raw_file,
                                                            gt_cam_height,
                                                            gt_cam_pitch,
                                                            vis)
                centerline_stats.append(np.array([r_lane, p_lane, cnt_gt, cnt_pred]))
                # consider x_error z_error only for the matched lanes
                if r_lane > 0 and p_lane > 0:
                    centerline_x_error_close.append(x_error_close)
                    centerline_x_error_far.append(x_error_far)
                    centerline_z_error_close.append(z_error_close)
                    centerline_z_error_far.append(z_error_far)
                # # save visualize map
                # if vis:
                #     # img_name = raw_file.split('/')[-1]
                #     img_name = raw_file.replace("/", "_")
                #     cv2.imwrite(save_path + '/centerline_' + img_name, vis_map)

            # print('processed sample: {}'.format(i))

        output_stats = []
        laneline_stats = np.array(laneline_stats)
        laneline_x_error_close = np.array(laneline_x_error_close)
        laneline_x_error_far = np.array(laneline_x_error_far)
        laneline_z_error_close = np.array(laneline_z_error_close)
        laneline_z_error_far = np.array(laneline_z_error_far)

        R_lane = np.sum(laneline_stats[:, 0]) / (np.sum(laneline_stats[:, 2]) + 1e-6)
        P_lane = np.sum(laneline_stats[:, 1]) / (np.sum(laneline_stats[:, 3]) + 1e-6)
        F_lane = 2 * R_lane * P_lane / (R_lane + P_lane + 1e-6)
        x_error_close_avg = np.average(laneline_x_error_close)
        x_error_far_avg = np.average(laneline_x_error_far)
        z_error_close_avg = np.average(laneline_z_error_close)
        z_error_far_avg = np.average(laneline_z_error_far)

        output_stats.append(F_lane)
        output_stats.append(R_lane)
        output_stats.append(P_lane)
        output_stats.append(x_error_close_avg)
        output_stats.append(x_error_far_avg)
        output_stats.append(z_error_close_avg)
        output_stats.append(z_error_far_avg)

        if not self.no_centerline:
            centerline_stats = np.array(centerline_stats)
            centerline_x_error_close = np.array(centerline_x_error_close)
            centerline_x_error_far = np.array(centerline_x_error_far)
            centerline_z_error_close = np.array(centerline_z_error_close)
            centerline_z_error_far = np.array(centerline_z_error_far)

            R_lane = np.sum(centerline_stats[:, 0]) / (np.sum(centerline_stats[:, 2]) + 1e-6)
            P_lane = np.sum(centerline_stats[:, 1]) / (np.sum(centerline_stats[:, 3]) + 1e-6)
            F_lane = 2 * R_lane * P_lane / (R_lane + P_lane + 1e-6)
            x_error_close_avg = np.average(centerline_x_error_close)
            x_error_far_avg = np.average(centerline_x_error_far)
            z_error_close_avg = np.average(centerline_z_error_close)
            z_error_far_avg = np.average(centerline_z_error_far)

            output_stats.append(F_lane)
            output_stats.append(R_lane)
            output_stats.append(P_lane)
            output_stats.append(x_error_close_avg)
            output_stats.append(x_error_far_avg)
            output_stats.append(z_error_close_avg)
            output_stats.append(z_error_far_avg)

        # TODO: generate PR curve, output max_F or AP (not appropriate) as key indicator

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

    pred_file = '../data/sim3d/Model_3DLaneNet_opt_adam_lr_0.0005_batch_8_360X480_pretrain_False_batchnorm_True/val_pred_file.json'
    gt_file = '../data/sim3d/val.json'

    # try:
    eval_stats = evaluator.bench_one_submit(pred_file, gt_file, vis=True)

    print("===> Evaluation on validation set: \n"
          "laneline F-measure {:.8} \n"
          "laneline Recall  {:.8} \n"
          "laneline Precision  {:.8} \n"
          "laneline x error (close)  {:.8} m\n"
          "laneline x error (far)  {:.8} m\n"
          "laneline z error (close)  {:.8} m\n"
          "laneline z error (far)  {:.8} m\n\n"
          "centerline F-measure {:.8} \n"
          "centerline Recall  {:.8} \n"
          "centerline Precision  {:.8} \n"
          "centerline x error (close)  {:.8} m\n"
          "centerline x error (far)  {:.8} m\n"
          "centerline z error (close)  {:.8} m\n"
          "centerline z error (far)  {:.8} m\n".format(eval_stats[0], eval_stats[1],
                                                       eval_stats[2], eval_stats[3],
                                                       eval_stats[3], eval_stats[5],
                                                       eval_stats[6], eval_stats[7],
                                                       eval_stats[8], eval_stats[9],
                                                       eval_stats[10], eval_stats[11],
                                                       eval_stats[12], eval_stats[13]))
