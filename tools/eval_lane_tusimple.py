# -*- coding: utf-8 -*-
# code for TuSimple evaluation (based on the code provided by TuSimple)
"""
This code is originated from tusimple lane detection dataset
"""


import os
import os.path as ops
import numpy as np
from sklearn.linear_model import LinearRegression
import ujson as json
import cv2

global dataset_base
dataset_base = '/home/yuliangguo/Datasets/tusimple'


class LaneEval(object):
    lr = LinearRegression()
    pixel_thresh = 20
    pt_thresh = 0.85

    @staticmethod
    def get_angle(xs, y_samples):
        xs, ys = xs[xs >= 0], y_samples[xs >= 0]
        if len(xs) > 1:
            LaneEval.lr.fit(ys[:, None], xs)
            k = LaneEval.lr.coef_[0]
            theta = np.arctan(k)
        else:
            theta = 0
        return theta

    @staticmethod
    def line_accuracy(pred, gt, thresh):
        pred = np.array([p if p >= 0 else -100 for p in pred])
        gt = np.array([g if g >= 0 else -100 for g in gt])
        return np.sum(np.where(np.abs(pred - gt) < thresh, 1., 0.)) / len(gt)

    @staticmethod
    def bench(pred, gt, y_samples, running_time):
        if any(len(p) != len(y_samples) for p in pred):
            raise Exception('Format of lanes error.')
        if running_time > 200 or len(gt) + 2 < len(pred):
            return 0., 0., 1.
        angles = [LaneEval.get_angle(np.array(x_gts), np.array(y_samples)) for x_gts in gt]
        threshs = [LaneEval.pixel_thresh / np.cos(angle) for angle in angles]
        line_accs = []
        fp, fn = 0., 0.
        matched = 0.
        for x_gts, thresh in zip(gt, threshs):
            accs = [LaneEval.line_accuracy(np.array(x_preds), np.array(x_gts), thresh) for x_preds in pred]
            max_acc = np.max(accs) if len(accs) > 0 else 0.
            if max_acc < LaneEval.pt_thresh:
                fn += 1
            else:
                matched += 1
            line_accs.append(max_acc)
        fp = len(pred) - matched
        if len(gt) > 4 and fn > 0:
            fn -= 1
        s = sum(line_accs)
        if len(gt) > 4:
            s -= min(line_accs)
        return s / max(min(4.0, len(gt)), 1.), fp / len(pred) if len(pred) > 0 else 0., fn / max(min(len(gt), 4.) , 1.)

    @staticmethod
    def bench_one_submit(pred_file, gt_file, vis=False):
        # ################### debug ###################
        save_path = pred_file[:pred_file.rfind('/')]
        save_path += '/example/eval_vis'
        if vis and not os.path.exists(save_path):
            try:
                os.makedirs(save_path)
            except OSError as e:
                print(e.message)
        # #############################################

        try:
            json_pred = [json.loads(line) for line in open(pred_file).readlines()]
        except BaseException as e:
            raise Exception('Fail to load json file of the prediction.')
        json_gt = [json.loads(line) for line in open(gt_file).readlines()]
        # if len(json_gt) != len(json_pred):
        #     raise Exception('We do not get the predictions of all the test tasks')
        gts = {l['raw_file']: l for l in json_gt}
        gts_id = {l['raw_file']: i for i, l in enumerate(json_gt)}
        accuracy, fp, fn = 0., 0., 0.
        for pred in json_pred:
            # if 'raw_file' not in pred or 'lanes' not in pred or 'run_time' not in pred:
            if 'raw_file' not in pred or 'lanes' not in pred:
                # raise Exception('raw_file or lanes or run_time not in some predictions.')
                continue
            raw_file = pred['raw_file']
            pred_lanes = pred['lanes']
            run_time = pred['run_time']
            if raw_file not in gts:
                raise Exception('Some raw_file from your predictions do not exist in the test tasks.')
            gt = gts[raw_file]
            gt_id = gts_id[raw_file]
            gt_lanes = gt['lanes']
            y_samples = gt['h_samples']
            try:
                a, p, n = LaneEval.bench(pred_lanes, gt_lanes, y_samples, run_time)
            except BaseException as e:
                raise Exception('Format of lanes error.')
            accuracy += a
            fp += p
            fn += n
            # ################### debug ###################
            if vis:
                print('Accuracy: {:3f}'.format(a))
                draw_lanes(raw_file, y_samples, gt_lanes, pred_lanes, save_path, gt_id)
            # #############################################
        num = len(gts)
        # the first return parameter is the default ranking parameter
#        return json.dumps([
#            {'name': 'Accuracy', 'value': accuracy / num, 'order': 'desc'},
#            {'name': 'FP', 'value': fp / num, 'order': 'asc'},
#            {'name': 'FN', 'value': fn / num, 'order': 'asc'}
#        ])
        return [accuracy / num, fp / num, fn / num]


def draw_lanes(raw_file, y_samples, gt_lanes, pred_lanes, save_path, gt_id):
    # if gt_id is not 190:
    #     return

    img = cv2.imread(ops.join(dataset_base, raw_file))

    gt_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
    pred_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in pred_lanes]
    img_vis = img.copy()

    for lane in gt_lanes_vis:
        cv2.polylines(img_vis, np.int32([lane]), isClosed=False, color=(255, 0, 0), thickness=5)
    for lane in pred_lanes_vis:
        cv2.polylines(img_vis, np.int32([lane]), isClosed=False, color=(0, 0, 255), thickness=2)

    # cv2.imshow("resize", img_vis)
    # cv2.waitKey()

    save_file = ops.join(save_path, 'eval_'+str(gt_id)+'.jpg')
    cv2.imwrite(save_file, img_vis)


if __name__ == '__main__':
    import sys
    try:
        if len(sys.argv) != 3:
            raise Exception('Invalid input arguments')
        print(LaneEval.bench_one_submit(sys.argv[1], sys.argv[2], vis=True))
    except Exception as e:
        print(e.message)
        sys.exit(e.message)
