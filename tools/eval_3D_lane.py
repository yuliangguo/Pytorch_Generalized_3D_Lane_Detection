import numpy as np
import cv2
import os
import os.path as ops
import math
import ujson as json
import matplotlib
from tools.utils import define_args, homography_im2ipm_norm,\
    homographic_transformation, projective_transformation,\
    homograpthy_g2im, projection_g2im, homography_crop_resize,\
    tusimple_config, sim3d_config, resample_laneline_in_y
from tools.MinCostFlow import SolveMinCostFlow
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (35, 30)

color = [[0, 0, 255],  # red
         [0, 255, 0],  # green
         [255, 0, 255],  # purple
         [255, 255, 0]]  # cyan


class LaneEval(object):
    def __init__(self, args):
        self.dataset_dir = args.dataset_dir
        self.K = args.K
        self.no_centerline = args.no_centerline

        self.resize_h = args.resize_h
        self.resize_w = args.resize_w
        self.H_crop = homography_crop_resize([args.org_h, args.org_w], args.crop_y, [args.resize_h, args.resize_w])

    def bench(self, pred_lanes, gt_lanes, raw_file, gt_cam_height, gt_cam_pitch, vis, ax1, ax2):
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

        gt_visibility_mat = np.zeros((cnt_gt, 100))
        pred_visibility_mat = np.zeros((cnt_pred, 100))
        # resample gt and pred at y_samples
        for i in range(cnt_gt):
            min_y = np.min(np.array(gt_lanes[i])[:, 1])
            max_y = np.max(np.array(gt_lanes[i])[:, 1])
            gt_visibility_mat[i, :] = np.logical_and(y_samples >= min_y, y_samples <= max_y)
            x_values, z_values = resample_laneline_in_y(np.array(gt_lanes[i]), y_samples)
            gt_lanes[i] = np.vstack([x_values, z_values]).T

        for i in range(cnt_pred):
            # # ATTENTION: ensure y mono increase before interpolation: but it can reduce size
            # pred_lanes[i] = make_lane_y_mono_inc(np.array(pred_lanes[i]))
            min_y = np.min(np.array(pred_lanes[i])[:, 1])
            max_y = np.max(np.array(pred_lanes[i])[:, 1])
            pred_visibility_mat[i, :] = np.logical_and(y_samples >= min_y, y_samples <= max_y)
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

                # apply visibility to penalize different partial matching accordingly
                euclidean_dist[np.logical_or(gt_visibility_mat[i, :] == 0, pred_visibility_mat[j, :] == 0)] = dist_th

                # if np.average(euclidean_dist) < 2*dist_th: # don't prune here to encourage finding perfect match
                adj_mat[i, j] = 1
                x_dist_mat_close[i, j] = np.average(x_dist[:close_range_idx])
                x_dist_mat_far[i, j] = np.average(x_dist[close_range_idx:])
                z_dist_mat_close[i, j] = np.average(z_dist[:close_range_idx])
                z_dist_mat_far[i, j] = np.average(z_dist[close_range_idx:])
                # ATTENTION: use sum and int here to meet the requirements of min cost flow optimization (int type)
                cost_mat[i, j] = np.sum(euclidean_dist).astype(np.int)

        # solve bipartite matching vis min cost flow solver
        match_results = SolveMinCostFlow(adj_mat, cost_mat)
        match_results = np.array(match_results)

        # only a match with avg cost < dist_th is consider valid one
        match_gt_ids = []
        match_pred_ids = []
        if match_results.shape[0] > 0 and np.sum(match_results[:, 2] < dist_th * y_samples.shape[0]) > 0:
            r_lane = 0
            p_lane = 0
            for i in range(len(match_results)):
                if match_results[i, 2] < dist_th * y_samples.shape[0]:
                    r_lane += 1
                    p_lane += 1
                    match_gt_ids.append(match_results[i, 0])
                    match_pred_ids.append(match_results[i, 1])
                    x_error_close.append(x_dist_mat_close[match_results[i, 0], match_results[i, 1]])
                    x_error_far.append(x_dist_mat_far[match_results[i, 0], match_results[i, 1]])
                    z_error_close.append(z_dist_mat_close[match_results[i, 0], match_results[i, 1]])
                    z_error_far.append(z_dist_mat_far[match_results[i, 0], match_results[i, 1]])

        # visualize lanelines and matching results both in image and 3D
        if vis:
            P_g2im = projection_g2im(gt_cam_pitch, gt_cam_height, self.K)
            P_gt = np.matmul(self.H_crop, P_g2im)
            img = cv2.imread(ops.join(self.dataset_dir, raw_file))
            img = cv2.warpPerspective(img, self.H_crop, (self.resize_w, self.resize_h))
            img = img.astype(np.float)/255

            for i in range(cnt_gt):
                x_values = gt_lanes[i][:, 0]
                z_values = gt_lanes[i][:, 1]
                x_2d, y_2d = projective_transformation(P_gt, x_values, y_samples, z_values)
                x_2d = x_2d.astype(np.int)
                y_2d = y_2d.astype(np.int)

                if i in match_gt_ids:
                    color = [0, 0, 1]
                else:
                    color = [0, 1, 1]
                for k in range(1, x_2d.shape[0]):
                    img = cv2.line(img, (x_2d[k - 1], y_2d[k - 1]), (x_2d[k], y_2d[k]), color[-1::-1], 2)
                ax1.imshow(img[:, :, [2, 1, 0]])
                ax2.plot(x_values, y_samples, z_values, color=color)

            for i in range(cnt_pred):
                x_values = pred_lanes[i][:, 0]
                z_values = pred_lanes[i][:, 1]
                x_2d, y_2d = projective_transformation(P_gt, x_values, y_samples, z_values)
                x_2d = x_2d.astype(np.int)
                y_2d = y_2d.astype(np.int)

                if i in match_gt_ids:
                    color = [1, 0, 0]
                else:
                    color = [1, 0, 1]
                for k in range(1, x_2d.shape[0]):
                    img = cv2.line(img, (x_2d[k - 1], y_2d[k - 1]), (x_2d[k], y_2d[k]), color[-1::-1], 2)
                ax1.imshow(img[:, :, [2, 1, 0]])
                ax2.plot(x_values, y_samples, z_values, color=color)

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
        # try:
        pred_lines = open(pred_file).readlines()
        json_pred = [json.loads(line) for line in pred_lines]
        # except BaseException as e:
        #     raise Exception('Fail to load json file of the prediction.')
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

            if vis:
                fig = plt.figure()
                ax1 = fig.add_subplot(221)
                ax2 = fig.add_subplot(222, projection='3d')
                ax3 = fig.add_subplot(223)
                ax4 = fig.add_subplot(224, projection='3d')
            else:
                ax1 = 0
                ax2 = 0
                ax3 = 0
                ax4 = 0

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
                                                        vis, ax1, ax2)
            laneline_stats.append(np.array([r_lane, p_lane, cnt_gt, cnt_pred]))
            # consider x_error z_error only for the matched lanes
            if r_lane > 0 and p_lane > 0:
                laneline_x_error_close.extend(x_error_close)
                laneline_x_error_far.extend(x_error_far)
                laneline_z_error_close.extend(z_error_close)
                laneline_z_error_far.extend(z_error_far)

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
                                                            vis, ax3, ax4)
                centerline_stats.append(np.array([r_lane, p_lane, cnt_gt, cnt_pred]))
                # consider x_error z_error only for the matched lanes
                if r_lane > 0 and p_lane > 0:
                    centerline_x_error_close.extend(x_error_close)
                    centerline_x_error_far.extend(x_error_far)
                    centerline_z_error_close.extend(z_error_close)
                    centerline_z_error_far.extend(z_error_far)

            if vis:
                ax2.set_xlabel('x axis')
                ax2.set_ylabel('y axis')
                ax2.set_zlabel('z axis')
                bottom, top = ax2.get_zlim()
                ax2.set_zlim(min(bottom, -1), max(top, 1))
                ax2.set_xlim(-20, 20)
                ax2.set_ylim(0, 100)

                ax4.set_xlabel('x axis')
                ax4.set_ylabel('y axis')
                ax4.set_zlabel('z axis')
                bottom, top = ax4.get_zlim()
                ax4.set_zlim(min(bottom, -1), max(top, 1))
                ax4.set_xlim(-20, 20)
                ax4.set_ylim(0, 100)

                fig.savefig(ops.join(save_path, raw_file.replace("/", "_")))
                plt.close(fig)
                print('processed sample: {}  {}'.format(i, raw_file))


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
    vis = True
    parser = define_args()
    args = parser.parse_args()

    args.dataset_name = 'sim3d_0906'
    args.dataset_dir = '/home/yuliangguo/Datasets/Apollo_Sim_3D_Lane_0906/'

    # load configuration for certain dataset
    sim3d_config(args)
    evaluator = LaneEval(args)

    pred_file = '../data/sim3d_0906/Model_3DLaneNet_new_opt_adam_lr_0.0005_batch_8_360X480_pretrain_False_batchnorm_True_predcam_False/val_pred_file.json'
    gt_file = '../data/sim3d_0906/val.json'

    # try:
    eval_stats = evaluator.bench_one_submit(pred_file, gt_file, vis=vis)

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
