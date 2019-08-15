import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import ndimage
import cv2
import os
import math
import ujson as json

color = [[0, 0, 255],  # red
         [0, 255, 0],  # green
         [255, 0, 255],  # purple
         [255, 255, 0]]  # cyan
# parameters need to be consistent with Ground-truth and prediction
# crop image from raw image
# crop_height = 768
crop_height = 1080
crop_width = 1920
# starting_height = 312
starting_height = 0
# eval image size
# eval_height = 720
# eval_width = 1280
eval_height = 1080
eval_width = 1920


class LaneEval(object):
    lr = LinearRegression()
    pixel_thresh = 20
    pt_thresh = 0.85

    @staticmethod
    def bench(pred, gt, raw_file):
        r_pixel, p_pixel = 0., 0.
        r_lane, p_lane = 0., 0.

        img = cv2.imread(raw_file)
        crop_img = img[starting_height:starting_height + crop_height, 0:0 + crop_width]
        visualize_map = cv2.resize(crop_img, (eval_width, eval_height))
        gray_map = cv2.cvtColor(visualize_map, cv2.COLOR_BGR2GRAY)
        visualize_map[..., 0] = gray_map
        visualize_map[..., 1] = gray_map
        visualize_map[..., 2] = gray_map

        # visualize_map = np.ones((eval_height, eval_width, 3), dtype=np.uint8) * 255
        gt_label_map = np.zeros((eval_height, eval_width), dtype=np.int8)
        pred_label_map = np.zeros((eval_height, eval_width), dtype=np.int8)
        cnt_gt = len(gt)
        cnt_pred = len(pred)
        # if cnt_gt == 0 or cnt_pred == 0:
        #     return 0, 0, 0, 0, 0, 0, visualize_map

        cnt_gt_list = np.zeros(cnt_gt + 1, dtype=np.int32)
        cnt_pred_list = np.zeros(cnt_pred + 1, dtype=np.int32)

        for i, gt_lane in enumerate(gt):
            for j in range(1, len(gt_lane)):
                cv2.line(gt_label_map, (gt_lane[j - 1][0], gt_lane[j - 1][1]),
                         (gt_lane[j][0], gt_lane[j][1]), color=i + 1)
            cnt_gt_list[i + 1] = np.sum(np.sum(gt_label_map == (i + 1)))

        for i, pred_lane in enumerate(pred):
            for j in range(1, len(pred_lane)):
                cv2.line(pred_label_map, (pred_lane[j - 1][0], pred_lane[j - 1][1]),
                         (pred_lane[j][0], pred_lane[j][1]), color=i + 1)
            cnt_pred_list[i + 1] = np.sum(np.sum(pred_label_map == (i + 1)))

        gt_binary_map = np.ones((eval_height, eval_width), dtype=np.int8)
        gt_binary_map[np.where(gt_label_map > 0)] = 0
        visualize_map[..., 0][np.where(gt_label_map > 0)] = color[1][0]
        visualize_map[..., 1][np.where(gt_label_map > 0)] = color[1][1]
        visualize_map[..., 2][np.where(gt_label_map > 0)] = color[1][2]

        pred_binary_map = np.ones((eval_height, eval_width), dtype=np.int8)
        pred_binary_map[np.where(pred_label_map > 0)] = 0
        visualize_map[..., 0][np.where(pred_label_map > 0)] = color[0][0]
        visualize_map[..., 1][np.where(pred_label_map > 0)] = color[0][1]
        visualize_map[..., 2][np.where(pred_label_map > 0)] = color[0][2]

        if cnt_gt == 0 or cnt_pred == 0:
            return 0, 0, np.sum(cnt_gt_list), np.sum(cnt_pred_list), 0, 0, visualize_map

        # distance transform for ground-truth
        gt_dst_map, gt_index_map = ndimage.distance_transform_edt(gt_binary_map, return_indices=True)
        # distance transform for prediction
        pred_dst_map, pred_index_map = ndimage.distance_transform_edt(pred_binary_map, return_indices=True)
        gt_dst_map = gt_dst_map < LaneEval.pixel_thresh
        pred_dst_map = pred_dst_map < LaneEval.pixel_thresh

        # find recall-related GT lanes from matching prediction to ground-truth DT map
        pixel_r_list = np.zeros(cnt_gt + 1, dtype=np.int32)
        pixel_r_coords_list = [0 for _ in range(cnt_gt + 1)]
        for l in range(1, cnt_pred + 1):
            pred_pixels = np.logical_and(pred_label_map == l, gt_dst_map)
            y, x = np.where(pred_pixels > 0)
            yy, xx = gt_index_map[:, np.array(y), np.array(x)]
            match_gt_labels = gt_label_map[yy, xx]
            if len(match_gt_labels) == 0:
                continue
            label_cnt = np.bincount(match_gt_labels)
            max_id = np.argmax(label_cnt)

            # find number of matched gt pixel using the prediction DT map
            gt_pixels = np.logical_and(gt_label_map == max_id, pred_dst_map)
            y1, x1 = np.where(gt_pixels > 0)
            yy1, xx1 = pred_index_map[:, np.array(y1), np.array(x1)]
            match_pred_labels = pred_label_map[yy1, xx1]
            match_pred_indices = np.where(match_pred_labels == l)[0]
            cnt_r = len(match_pred_indices)
            y1 = y1[match_pred_indices]
            x1 = x1[match_pred_indices]

            if cnt_r > pixel_r_list[max_id]:
                pixel_r_list[max_id] = cnt_r
                pixel_r_coords_list[max_id] = [y1, x1]

        # visualize
        for i in range(1, cnt_gt + 1):
            if pixel_r_list[i] > 0:
                visualize_map[pixel_r_coords_list[i][0], pixel_r_coords_list[i][1], 0] = 255

        # pixel_r_list = pixel_r_list[1:].astype(np.float32) / cnt_gt_list[1:].astype(np.float32)
        cnt_gt_pixel = np.sum(cnt_gt_list)
        r_pixel += np.sum(pixel_r_list)
        r_lane += len(
            np.where(pixel_r_list[1:].astype(np.float32) / cnt_gt_list[1:].astype(np.float32) > LaneEval.pt_thresh)[0])

        # compute precision from matching ground truth to prediction DT map
        pixel_p_list = np.zeros(cnt_pred + 1, dtype=np.int32)
        pixel_p_coords_list = [0 for _ in range(cnt_pred + 1)]
        for l in range(1, cnt_gt + 1):
            gt_pixels = np.logical_and(gt_label_map == l, pred_dst_map)
            y, x = np.where(gt_pixels > 0)
            yy, xx = pred_index_map[:, np.array(y), np.array(x)]
            match_pred_labels = pred_label_map[yy, xx]
            if len(match_pred_labels) == 0:
                continue
            label_cnt = np.bincount(match_pred_labels)
            max_id = np.argmax(label_cnt)

            # find number of matched pred pixel using the ground-truth DT map
            pred_pixels = np.logical_and(pred_label_map == max_id, gt_dst_map)
            y1, x1 = np.where(pred_pixels > 0)
            yy1, xx1 = gt_index_map[:, np.array(y1), np.array(x1)]
            match_gt_labels = gt_label_map[yy1, xx1]
            match_gt_indices = np.where(match_gt_labels == l)[0]
            cnt_p = len(match_gt_indices)
            y1 = y1[match_gt_indices]
            x1 = x1[match_gt_indices]

            if cnt_p > pixel_p_list[max_id]:
                pixel_p_list[max_id] = cnt_p
                pixel_p_coords_list[max_id] = [y1, x1]

        # visualize
        for i in range(1, cnt_pred + 1):
            if pixel_p_list[i] > 0:
                visualize_map[pixel_p_coords_list[i][0], pixel_p_coords_list[i][1], 0] = 255

        # pixel_p_list = pixel_p_list[1:].astype(np.float32) / cnt_pred_list[1:].astype(np.float32)
        cnt_pred_pixel = np.sum(cnt_pred_list)
        p_pixel += np.sum(pixel_p_list)
        p_lane += len(
            np.where(pixel_p_list[1:].astype(np.float32) / cnt_pred_list[1:].astype(np.float32) > LaneEval.pt_thresh)[
                0])

        # angles = [LaneEval.get_angle(np.array(x_gts), np.array(y_samples)) for x_gts in gt]
        # threshs = [LaneEval.pixel_thresh / np.cos(angle) for angle in angles]

        return r_pixel, p_pixel, cnt_gt_pixel, cnt_pred_pixel, r_lane, p_lane, visualize_map

    @staticmethod
    def bench_id_match(pred_types, gt_types, pred, gt, raw_file):
        r_pixel, p_pixel = 0., 0.
        r_lane, p_lane = 0., 0.

        img = cv2.imread(raw_file)
        crop_img = img[starting_height:starting_height + crop_height, 0:0 + crop_width]
        visualize_map = cv2.resize(crop_img, (eval_width, eval_height))
        gray_map = cv2.cvtColor(visualize_map, cv2.COLOR_BGR2GRAY)
        visualize_map[..., 0] = gray_map
        visualize_map[..., 1] = gray_map
        visualize_map[..., 2] = gray_map

        # visualize_map = np.zeros((eval_height, eval_width, 3), dtype=np.uint8)
        gt_label_map = np.zeros((eval_height, eval_width), dtype=np.int8)
        pred_label_map = np.zeros((eval_height, eval_width), dtype=np.int8)
        gt_type_label_map = np.zeros((eval_height, eval_width), dtype=np.int8)
        pred_type_label_map = np.zeros((eval_height, eval_width), dtype=np.int8)
        cnt_gt = len(gt)
        cnt_pred = len(pred)
        # if cnt_gt == 0 or cnt_pred == 0:
        #     return 0, 0, 0, 0, 0, 0, visualize_map

        cnt_gt_list = np.zeros(cnt_gt + 1, dtype=np.int32)
        cnt_pred_list = np.zeros(cnt_pred + 1, dtype=np.int32)
        gt_label_to_type = np.zeros(cnt_gt + 1, dtype=np.int32)
        pred_label_to_type = np.zeros(cnt_pred + 1, dtype=np.int32)

        for i, gt_lane in enumerate(gt):
            for j in range(1, len(gt_lane)):
                cv2.line(gt_label_map, (gt_lane[j - 1][0], gt_lane[j - 1][1]),
                         (gt_lane[j][0], gt_lane[j][1]), color=i + 1)
                cv2.line(gt_type_label_map, (gt_lane[j - 1][0], gt_lane[j - 1][1]),
                         (gt_lane[j][0], gt_lane[j][1]), color=gt_types[i])
            cnt_gt_list[i + 1] = np.sum(np.sum(gt_label_map == (i + 1)))
            gt_label_to_type[i + 1] = gt_types[i]

        for i, pred_lane in enumerate(pred):
            for j in range(1, len(pred_lane)):
                cv2.line(pred_label_map, (pred_lane[j - 1][0], pred_lane[j - 1][1]),
                         (pred_lane[j][0], pred_lane[j][1]), color=i + 1)
                cv2.line(pred_type_label_map, (pred_lane[j - 1][0], pred_lane[j - 1][1]),
                         (pred_lane[j][0], pred_lane[j][1]), color=pred_types[i])
            cnt_pred_list[i + 1] = np.sum(np.sum(pred_label_map == (i + 1)))
            pred_label_to_type[i + 1] = pred_types[i]

        gt_binary_map = np.ones((eval_height, eval_width), dtype=np.int8)
        gt_binary_map[np.where(gt_label_map > 0)] = 0
        visualize_map[..., 0][np.where(gt_label_map > 0)] = color[1][0]
        visualize_map[..., 1][np.where(gt_label_map > 0)] = color[1][1]
        visualize_map[..., 2][np.where(gt_label_map > 0)] = color[1][2]

        pred_binary_map = np.ones((eval_height, eval_width), dtype=np.int8)
        pred_binary_map[np.where(pred_label_map > 0)] = 0
        visualize_map[..., 0][np.where(pred_label_map > 0)] = color[0][0]
        visualize_map[..., 1][np.where(pred_label_map > 0)] = color[0][1]
        visualize_map[..., 2][np.where(pred_label_map > 0)] = color[0][2]

        if cnt_gt == 0 or cnt_pred == 0:
            return 0, 0, np.sum(cnt_gt_list), np.sum(cnt_pred_list), 0, 0, visualize_map

        # distance transform for ground-truth
        gt_dst_map, gt_index_map = ndimage.distance_transform_edt(gt_binary_map, return_indices=True)
        # distance transform for prediction
        pred_dst_map, pred_index_map = ndimage.distance_transform_edt(pred_binary_map, return_indices=True)
        gt_dst_map = gt_dst_map < LaneEval.pixel_thresh
        pred_dst_map = pred_dst_map < LaneEval.pixel_thresh

        # find recall-related GT lanes from matching prediction to ground-truth DT map
        pixel_r_list = np.zeros(cnt_gt + 1, dtype=np.int32)
        pixel_r_coords_list = [0 for _ in range(cnt_gt + 1)]
        for l in range(1, cnt_pred + 1):
            pred_pixels = np.logical_and(pred_label_map == l, gt_dst_map)
            y, x = np.where(pred_pixels > 0)
            yy, xx = gt_index_map[:, np.array(y), np.array(x)]
            match_gt_labels = gt_label_map[yy, xx]
            match_gt_type_labels = gt_type_label_map[yy, xx]
            match_gt_labels = match_gt_labels[np.where(match_gt_type_labels == pred_label_to_type[l])]
            if len(match_gt_labels) == 0:
                continue
            label_cnt = np.bincount(match_gt_labels)
            max_id = np.argmax(label_cnt)

            # find number of matched gt pixel using the prediction DT map, also sharing the same type label
            gt_pixels = np.logical_and(gt_label_map == max_id, pred_dst_map)
            y1, x1 = np.where(gt_pixels > 0)
            yy1, xx1 = pred_index_map[:, np.array(y1), np.array(x1)]
            match_pred_labels = pred_label_map[yy1, xx1]
            match_pred_indices = np.where(match_pred_labels == l)[0]
            cnt_r = len(match_pred_indices)
            y1 = y1[match_pred_indices]
            x1 = x1[match_pred_indices]

            if cnt_r > pixel_r_list[max_id]:
                pixel_r_list[max_id] = cnt_r
                pixel_r_coords_list[max_id] = [y1, x1]

        # visualize
        for i in range(1, cnt_gt + 1):
            if pixel_r_list[i] > 0:
                visualize_map[pixel_r_coords_list[i][0], pixel_r_coords_list[i][1], 0] = 255

        # pixel_r_list = pixel_r_list[1:].astype(np.float32) / cnt_gt_list[1:].astype(np.float32)
        cnt_gt_pixel = np.sum(cnt_gt_list)
        r_pixel += np.sum(pixel_r_list)
        r_lane += len(
            np.where(pixel_r_list[1:].astype(np.float32) / cnt_gt_list[1:].astype(np.float32) > LaneEval.pt_thresh)[0])

        # compute precision from matching ground truth to prediction DT map, also sharing the same type label
        pixel_p_list = np.zeros(cnt_pred + 1, dtype=np.int32)
        pixel_p_coords_list = [0 for _ in range(cnt_pred + 1)]
        for l in range(1, cnt_gt + 1):
            gt_pixels = np.logical_and(gt_label_map == l, pred_dst_map)
            # gt_pixels = np.logical_and(gt_pixels, pred_type_label_map == gt_label_to_type[l])
            y, x = np.where(gt_pixels > 0)
            yy, xx = pred_index_map[:, np.array(y), np.array(x)]
            match_pred_labels = pred_label_map[yy, xx]
            match_pred_type_labels = pred_type_label_map[yy, xx]
            match_pred_labels = match_pred_labels[np.where(match_pred_type_labels == gt_label_to_type[l])]
            if len(match_pred_labels) == 0:
                continue
            label_cnt = np.bincount(match_pred_labels)
            max_id = np.argmax(label_cnt)

            # find number of matched pred pixel using the ground-truth DT map
            pred_pixels = np.logical_and(pred_label_map == max_id, gt_dst_map)
            y1, x1 = np.where(pred_pixels > 0)
            yy1, xx1 = gt_index_map[:, np.array(y1), np.array(x1)]
            match_gt_labels = gt_label_map[yy1, xx1]
            match_gt_indices = np.where(match_gt_labels == l)[0]
            cnt_p = len(match_gt_indices)
            y1 = y1[match_gt_indices]
            x1 = x1[match_gt_indices]

            if cnt_p > pixel_p_list[max_id]:
                pixel_p_list[max_id] = cnt_p
                pixel_p_coords_list[max_id] = [y1, x1]

        # visualize
        for i in range(1, cnt_pred + 1):
            if pixel_p_list[i] > 0:
                visualize_map[pixel_p_coords_list[i][0], pixel_p_coords_list[i][1], 0] = 255

        # pixel_p_list = pixel_p_list[1:].astype(np.float32) / cnt_pred_list[1:].astype(np.float32)
        cnt_pred_pixel = np.sum(cnt_pred_list)
        p_pixel += np.sum(pixel_p_list)
        p_lane += len(
            np.where(pixel_p_list[1:].astype(np.float32) / cnt_pred_list[1:].astype(np.float32) > LaneEval.pt_thresh)[
                0])

        # angles = [LaneEval.get_angle(np.array(x_gts), np.array(y_samples)) for x_gts in gt]
        # threshs = [LaneEval.pixel_thresh / np.cos(angle) for angle in angles]

        return r_pixel, p_pixel, cnt_gt_pixel, cnt_pred_pixel, r_lane, p_lane, visualize_map

    @staticmethod
    def bench_one_submit(pred_file, gt_file, vis_folder, output_eval_file):
        try:
            json_pred = [json.loads(line) for line in open(pred_file).readlines()]
        except BaseException as e:
            raise Exception('Fail to load json file of the prediction.')
        json_gt = [json.loads(line) for line in open(gt_file).readlines()]
        if len(json_gt) != len(json_pred):
            raise Exception('We do not get the predictions of all the test tasks')
        gts = {l['raw_file']: l for l in json_gt}

        if not os.path.exists(vis_folder):
            os.mkdir(vis_folder)

        P_pixel, R_pixel, GT_pixel, PRED_pixel, P_lane, R_lane = 0., 0., 0., 0., 0., 0.
        P_pixel_IDCorr, R_pixel_IDCorr, GT_pixel_IDCorr, PRED_pixel_IDCorr, P_lane_IDCorr, R_lane_IDCorr = 0., 0., 0., 0., 0., 0.
        num_gt_lane, num_pred_lane, num = 0., 0., 0.

        for i, pred in enumerate(json_pred):
            if 'raw_file' not in pred or 'lanes' not in pred or 'run_time' not in pred:
                raise Exception('raw_file or lanes or run_time not in some predictions.')
            raw_file = pred['raw_file']
            pred_lanes = pred['lanes']
            run_time = pred['run_time']
            if raw_file not in gts:
                print
                raw_file
                raise Exception('Some raw_file from your predictions do not exist in the test tasks.')
            gt = gts[raw_file]
            gt_lanes = gt['lanes']
            print
            len(gt_lanes), len(pred_lanes)

            # N to N matching of lanes
            r_pixel, p_pixel, gt_pixel, pred_pixel, r_lane, p_lane, vis_map = LaneEval.bench(pred_lanes, gt_lanes,
                                                                                             raw_file)
            # print p_pixel, r_pixel, fp_lane, fn_lane
            R_pixel += r_pixel
            P_pixel += p_pixel
            GT_pixel += gt_pixel
            PRED_pixel += pred_pixel
            R_lane += r_lane
            P_lane += p_lane

            # label-related matching of lanes
            pred_lane_types = pred['lane_types']
            gt_lane_types = gt['lane_types']
            r_pixel_IDCorr, p_pixel_IDCorr, gt_pixel_IDCorr, pred_pixel_IDCorr, r_lane_IDCorr, p_lane_IDCorr, vis_map_IDCorr = LaneEval.bench_id_match(
                pred_lane_types, gt_lane_types, pred_lanes, gt_lanes, raw_file)
            # print p_pixel_IDCorr, r_pixel_IDCorr, fP_lane_IDCorr, fn_lane_l
            R_pixel_IDCorr += r_pixel_IDCorr
            P_pixel_IDCorr += p_pixel_IDCorr
            GT_pixel_IDCorr += gt_pixel_IDCorr
            PRED_pixel_IDCorr += pred_pixel_IDCorr
            R_lane_IDCorr += r_lane_IDCorr
            P_lane_IDCorr += p_lane_IDCorr

            # save visualize map
            if i % 10 == 0:
                img_name = raw_file.split('/')[-1]
                cv2.imwrite(vis_folder + img_name, vis_map)
            # cv2.imshow('resize', vis_map)
            # cv2.waitKey()

            # accumulate lane counts
            num_gt_lane += len(gt_lanes)
            num_pred_lane += len(pred_lanes)

            print
            'processed sample: ' + str(num)
            num += 1
            # print r_pixel / max(len(gt_lanes), 0.000001), p_pixel / max(len(pred_lanes), 0.000001), r_lane / max(len(gt_lanes), 0.000001), p_lane / max(len(pred_lanes), 0.000001)
            # print r_pixel_IDCorr / max(len(gt_lanes), 0.000001), p_pixel_IDCorr / max(len(pred_lanes), 0.000001), r_lane_IDCorr / max(len(gt_lanes), 0.000001), p_lane_IDCorr / max(len(pred_lanes), 0.000001)
            print
            r_pixel / max(gt_pixel, 0.000001), p_pixel / max(pred_pixel, 0.000001), r_lane / max(len(gt_lanes),
                                                                                                 0.000001), p_lane / max(
                len(pred_lanes), 0.000001)
            print
            r_pixel_IDCorr / max(gt_pixel_IDCorr, 0.000001), p_pixel_IDCorr / max(pred_pixel_IDCorr,
                                                                                  0.000001), r_lane_IDCorr / max(
                len(gt_lanes), 0.000001), p_lane_IDCorr / max(len(pred_lanes), 0.000001)

            if math.isnan(p_pixel) or math.isnan(r_pixel) or math.isnan(p_lane) or math.isnan(r_pixel):
                break
            if math.isnan(p_pixel_IDCorr) or math.isnan(r_pixel_IDCorr) or math.isnan(p_lane_IDCorr) or math.isnan(
                    r_pixel_IDCorr):
                break

        R_pixel /= GT_pixel
        P_pixel /= PRED_pixel
        F_pixel = 2 * R_pixel * P_pixel / (R_pixel + P_pixel)
        R_lane /= num_gt_lane
        P_lane /= num_pred_lane
        F_lane = 2 * R_lane * P_lane / (R_lane + P_lane)
        R_pixel_IDCorr /= GT_pixel_IDCorr
        P_pixel_IDCorr /= PRED_pixel_IDCorr
        F_pixel_l = 2 * R_pixel_IDCorr * P_pixel_IDCorr / (R_pixel_IDCorr + P_pixel_IDCorr)
        R_lane_IDCorr /= num_gt_lane
        P_lane_IDCorr /= num_pred_lane
        F_lane_l = 2 * R_lane_IDCorr * P_lane_IDCorr / (R_lane_IDCorr + P_lane_IDCorr)

        with open(output_eval_file, "w") as outfile:
            outfile.write(json.dumps([
                {'name': 'R_pixel', 'value': R_pixel, 'order': 'desc'},
                {'name': 'P_pixel', 'value': P_pixel, 'order': 'desc'},
                {'name': 'F_pixel', 'value': F_pixel, 'order': 'desc'},
                {'name': 'R_lane', 'value': R_lane, 'order': 'asc'},
                {'name': 'P_lane', 'value': P_lane, 'order': 'asc'},
                {'name': 'F_lane', 'value': F_lane, 'order': 'asc'},
                {'name': 'R_pixel_IDCorr', 'value': R_pixel_IDCorr, 'order': 'desc'},
                {'name': 'P_pixel_IDCorr', 'value': P_pixel_IDCorr, 'order': 'desc'},
                {'name': 'F_pixel_l', 'value': F_pixel_l, 'order': 'desc'},
                {'name': 'R_lane_IDCorr', 'value': R_lane_IDCorr, 'order': 'asc'},
                {'name': 'P_lane_IDCorr', 'value': P_lane_IDCorr, 'order': 'asc'},
                {'name': 'F_lane_l', 'value': F_lane_l, 'order': 'asc'}
            ]))
            outfile.write("\n")
            outfile.close()
            return


if __name__ == '__main__':
    import sys

    pred_file = '/home/xiaoshuliu/DarkSCNN_LaneDetection/Data/baidu_all_data_runs/run_06022019/prediction_dense_cubic_ransac_new.json'
    gt_file = '/home/xiaoshuliu/DarkSCNN_LaneDetection/Data/baidu_all_data_runs/run_06022019/test_label_dense.json'
    vis_folder = '/home/xiaoshuliu/DarkSCNN_LaneDetection/Data/baidu_all_data_runs/run_06022019/eval_vis_cubic_ransac_new/'
    output_eval_file = '/home/xiaoshuliu/DarkSCNN_LaneDetection/Data/baidu_all_data_runs/run_06022019/test_eval_result_cubic_ransac_new.json'

    try:
        print
        LaneEval.bench_one_submit(pred_file, gt_file, vis_folder, output_eval_file)
    except Exception as e:
        print
        e.message
        sys.exit(e.message)
