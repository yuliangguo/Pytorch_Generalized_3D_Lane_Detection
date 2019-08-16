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
    tusimple_config, apollo_sim_config

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
        # if cnt_gt == 0 or cnt_pred == 0:
        #     return 0, 0, np.sum(cnt_gt_list), np.sum(cnt_pred_list), 0, 0, vis_ipm

        gt_label_ipm = np.zeros((self.ipm_h, self.ipm_w), dtype=np.int8)
        pred_label_ipm = np.zeros((self.ipm_h, self.ipm_w), dtype=np.int8)

        # draw IPM label maps
        cnt_gt_list = np.zeros(cnt_gt + 1, dtype=np.int32)
        cnt_pred_list = np.zeros(cnt_pred + 1, dtype=np.int32)

        for i, gt_lane in enumerate(gt_lanes):
            # convert lane points from ground-coordinates to ipm coordinates
            gt_lane = np.array(gt_lane)
            x_ipm, y_ipm = homographic_transformation(self.M_g2ipm, gt_lane[:, 0], gt_lane[:, 1])
            x_ipm = x_ipm.astype(np.int)
            y_ipm = y_ipm.astype(np.int)
            for j in range(1, len(gt_lane)):
                gt_label_ipm = cv2.line(gt_label_ipm, (x_ipm[j - 1], y_ipm[j - 1]),
                                        (x_ipm[j], y_ipm[j]), color=i + 1)
            cnt_gt_list[i + 1] = np.sum(np.sum(gt_label_ipm == (i + 1)))

        for i, pred_lane in enumerate(pred_lanes):
            # convert lane points from ground-coordinates to ipm coordinates
            pred_lane = np.array(pred_lane)
            # TODO: remove this debug line in training, need to check what was not matched right
            # pred_lane[:, 0] = pred_lane[:, 0] + 0.1

            x_ipm, y_ipm = homographic_transformation(self.M_g2ipm, pred_lane[:, 0], pred_lane[:, 1])
            x_ipm = x_ipm.astype(np.int)
            y_ipm = y_ipm.astype(np.int)
            for j in range(1, len(pred_lane)):
                pred_label_ipm = cv2.line(pred_label_ipm, (x_ipm[j - 1], y_ipm[j - 1]),
                                        (x_ipm[j], y_ipm[j]), color=i + 1)
            cnt_pred_list[i + 1] = np.sum(np.sum(pred_label_ipm == (i + 1)))

        gt_binary_ipm = np.ones((self.ipm_h, self.ipm_w), dtype=np.int8)
        gt_binary_ipm[np.where(gt_label_ipm > 0)] = 0

        pred_binary_ipm = np.ones((self.ipm_h, self.ipm_w), dtype=np.int8)
        pred_binary_ipm[np.where(pred_label_ipm > 0)] = 0

        # compute distance transform map for ground-truth
        gt_dst_map, gt_index_map = ndimage.distance_transform_edt(gt_binary_ipm, return_indices=True)
        # compute distance transform map for prediction
        pred_dst_map, pred_index_map = ndimage.distance_transform_edt(pred_binary_ipm, return_indices=True)
        gt_dst_map = gt_dst_map < self.dist_th
        pred_dst_map = pred_dst_map < self.dist_th

        # find recall-related GT lanelines from matching prediction to ground-truth DT map
        pixel_r_list = np.zeros(cnt_gt + 1, dtype=np.int32)
        pixel_r_coords_list = [0 for _ in range(cnt_gt + 1)]
        for l in range(1, cnt_pred + 1):
            pred_pixels = np.logical_and(pred_label_ipm == l, gt_dst_map)
            y, x = np.where(pred_pixels > 0)
            yy, xx = gt_index_map[:, np.array(y), np.array(x)]
            match_gt_labels = gt_label_ipm[yy, xx]
            if len(match_gt_labels) == 0:
                continue
            label_cnt = np.bincount(match_gt_labels)
            max_id = np.argmax(label_cnt)

            # find number of matched gt pixel using the prediction DT map
            gt_pixels = np.logical_and(gt_label_ipm == max_id, pred_dst_map)
            y1, x1 = np.where(gt_pixels > 0)
            yy1, xx1 = pred_index_map[:, np.array(y1), np.array(x1)]
            match_pred_labels = pred_label_ipm[yy1, xx1]
            match_pred_indices = np.where(match_pred_labels == l)[0]
            cnt_r = len(match_pred_indices)
            y1 = y1[match_pred_indices]
            x1 = x1[match_pred_indices]

            if cnt_r > pixel_r_list[max_id]:
                pixel_r_list[max_id] = cnt_r
                pixel_r_coords_list[max_id] = [y1, x1]

        cnt_gt_pixel = np.sum(cnt_gt_list)
        r_pixel += np.sum(pixel_r_list)
        r_lane += len(np.where(pixel_r_list[np.where(cnt_gt_list > self.min_num_pixels)].astype(np.float32) /
                               cnt_gt_list[np.where(cnt_gt_list > self.min_num_pixels)].astype(np.float32) > self.pt_th)[0])

        # compute precision from matching ground truth to prediction DT map
        pixel_p_list = np.zeros(cnt_pred + 1, dtype=np.int32)
        pixel_p_coords_list = [0 for _ in range(cnt_pred + 1)]
        for l in range(1, cnt_gt + 1):
            gt_pixels = np.logical_and(gt_label_ipm == l, pred_dst_map)
            y, x = np.where(gt_pixels > 0)
            yy, xx = pred_index_map[:, np.array(y), np.array(x)]
            match_pred_labels = pred_label_ipm[yy, xx]
            if len(match_pred_labels) == 0:
                continue
            label_cnt = np.bincount(match_pred_labels)
            max_id = np.argmax(label_cnt)

            # find number of matched pred pixel using the ground-truth DT map
            pred_pixels = np.logical_and(pred_label_ipm == max_id, gt_dst_map)
            y1, x1 = np.where(pred_pixels > 0)
            yy1, xx1 = gt_index_map[:, np.array(y1), np.array(x1)]
            match_gt_labels = gt_label_ipm[yy1, xx1]
            match_gt_indices = np.where(match_gt_labels == l)[0]
            cnt_p = len(match_gt_indices)
            y1 = y1[match_gt_indices]
            x1 = x1[match_gt_indices]

            if cnt_p > pixel_p_list[max_id]:
                pixel_p_list[max_id] = cnt_p
                pixel_p_coords_list[max_id] = [y1, x1]

        cnt_pred_pixel = np.sum(cnt_pred_list)
        p_pixel += np.sum(pixel_p_list)
        p_lane += len(np.where(pixel_p_list[np.where(cnt_pred_list > self.min_num_pixels)].astype(np.float32) /
                               cnt_pred_list[np.where(cnt_pred_list > self.min_num_pixels)].astype(np.float32) > self.pt_th)[0])

        # those lanes out of the top-view range will not be counted
        cnt_gt_out = sum(cnt_gt_list > 0)
        cnt_pred_out = sum(cnt_pred_list > 0)

        # visualization
        # TODO: implement visualization on image when necessary
        if vis:
            # compute homography between normalized coordinates of network input image and ipm image
            H_im2ipm_norm, H_ipm2im_norm = homography_im2ipm_norm(self.top_view_region, [self.org_h, self.org_w],
                                                                  self.crop_y, [self.resize_h, self.resize_w],
                                                                  gt_cam_pitch, gt_cam_height, self.K)
            M_im2ipm = np.matmul(np.matmul(self.S_ipm, H_im2ipm_norm), np.linalg.inv(self.S_im))

            img = cv2.imread(ops.join(self.dataset_dir, raw_file))
            img = cv2.warpPerspective(img, self.H_crop, (self.resize_w, self.resize_h))
            im_ipm = cv2.warpPerspective(img, M_im2ipm, (self.ipm_w, self.ipm_h))
            vis_ipm = np.clip(im_ipm, 0, 255)
            # gray_map = cv2.cvtColor(im_ipm, cv2.COLOR_BGR2GRAY)
            # vis_ipm = np.repeat(gray_map[:, :, np.newaxis], 3, axis=2)

            # visualize pred lane pixels
            vis_ipm[..., 0][np.where(pred_label_ipm > 0)] = color[0][0]
            vis_ipm[..., 1][np.where(pred_label_ipm > 0)] = color[0][1]
            vis_ipm[..., 2][np.where(pred_label_ipm > 0)] = color[0][2]

            # visualize gt lane pixels
            vis_ipm[..., 0][np.where(gt_label_ipm > 0)] = color[1][0]
            vis_ipm[..., 1][np.where(gt_label_ipm > 0)] = color[1][1]
            vis_ipm[..., 2][np.where(gt_label_ipm > 0)] = color[1][2]

            # visualize matched gt pixels
            for i in range(1, cnt_gt + 1):
                if pixel_r_list[i] > 0:
                    vis_ipm[pixel_r_coords_list[i][0], pixel_r_coords_list[i][1], 0] = 255

            # visualize matched pred pixels
            for i in range(1, cnt_pred + 1):
                if pixel_p_list[i] > 0:
                    vis_ipm[pixel_p_coords_list[i][0], pixel_p_coords_list[i][1], 0] = 255
        else:
            vis_ipm = np.zeros((self.ipm_h, self.ipm_w), dtype=np.int8)

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

            print('processed sample: {}'.format(i))

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

        return output_stats


if __name__ == '__main__':

    parser = define_args()
    args = parser.parse_args()

    args.dataset_name = 'apollosim'
    args.data_dir = ops.join('../data', args.dataset_name)
    args.dataset_dir = '/media/yuliangguo/NewVolume2TB/Datasets/Apollo_Sim_lane/'

    # load configuration for certain dataset
    if args.dataset_name is 'tusimple':
        tusimple_config(args)
    elif args.dataset_name is 'apollosim':
        apollo_sim_config(args)

    args.pixel_per_meter = 10.
    args.dist_th = 1.5
    args.pt_th = 0.5
    args.min_num_pixels = 10

    pred_file = '/home/yuliangguo/Projects/3DLaneNet/data/apollosim/val.json'
    gt_file = '/home/yuliangguo/Projects/3DLaneNet/data/apollosim/val.json'

    evaluator = LaneEval(args)
    # try:
    print(evaluator.bench_one_submit(pred_file, gt_file, vis=True))
    # except Exception as e:
    #     print(e.message)
    #     sys.exit(e.message)
