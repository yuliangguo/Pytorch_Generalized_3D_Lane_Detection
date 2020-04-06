"""
Batch test code for 3D-GeoNet with new anchor extension. It predicts 3D lanes from ground-truth segmentation.

Author: Yuliang Guo (33yuliangguo@gmail.com)
Date: March, 2020
"""

import numpy as np
import torch
import torch.optim
import glob
from tqdm import tqdm
from dataloader.Load_Data_3DLane_ext import *
from networks import GeoNet3D_ext
from tools.utils import *
from tools import eval_3D_lane


def deploy(args, loader, dataset, model, vs_saver, test_gt_file, lane_pred_file, vis=False, epoch=0):

    # model deploy mode
    model.eval()

    # read ground-truth lanes for later evaluation
    test_set_labels = [json.loads(line) for line in open(test_gt_file).readlines()]

    # Only forward pass
    with torch.no_grad():
        with open(lane_pred_file, 'w') as jsonFile:
            # Start validation loop
            for i, (input, seg_maps, gt, idx, gt_hcam, gt_pitch) in tqdm(enumerate(loader)):
                if not args.no_cuda:
                    input, gt = input.cuda(non_blocking=True), gt.cuda(non_blocking=True)
                    input = input.float()
                    seg_maps = seg_maps.cuda(non_blocking=True)

                if not args.fix_cam and not args.pred_cam:
                    model.update_projection(args, gt_hcam, gt_pitch)
                # Evaluate model
                try:
                    output_net, pred_hcam, pred_pitch = model(seg_maps)
                except RuntimeError as e:
                    print("Batch with idx {} skipped due to singular matrix".format(idx.numpy()))
                    print(e)
                    continue

                gt = gt.data.cpu().numpy()
                output_net = output_net.data.cpu().numpy()
                pred_pitch = pred_pitch.data.cpu().numpy().flatten()
                pred_hcam = pred_hcam.data.cpu().numpy().flatten()

                # unormalize lane outputs
                num_el = input.size(0)
                for j in range(num_el):
                    unormalize_lane_anchor(output_net[j], dataset)
                    unormalize_lane_anchor(gt[j], dataset)

                if vis:
                    # Plot curves in two views
                    vs_saver.save_result_new(dataset, args.vis_folder, epoch, i, idx,
                                             input, gt, output_net, pred_pitch, pred_hcam, evaluate=vis)

                # write results and evaluate
                for j in range(num_el):
                    im_id = idx[j]
                    H_g2im, P_g2im, H_crop, H_im2ipm = dataset.transform_mats(idx[j])
                    json_line = test_set_labels[im_id]
                    lane_anchors = output_net[j]
                    # convert to json output format
                    lanelines_pred, centerlines_pred, lanelines_prob, centerlines_prob =\
                        compute_3d_lanes_all_prob(lane_anchors, dataset.anchor_dim,
                                                  dataset.anchor_x_steps, args.anchor_y_steps, pred_hcam[j])
                    json_line["laneLines"] = lanelines_pred
                    json_line["centerLines"] = centerlines_pred
                    json_line["laneLines_prob"] = lanelines_prob
                    json_line["centerLines_prob"] = centerlines_prob
                    json.dump(json_line, jsonFile)
                    jsonFile.write('\n')
        # evaluation at varying thresholds
        eval_stats_pr = evaluator.bench_one_submit_varying_probs(lane_pred_file, test_gt_file)
        max_f_prob = eval_stats_pr['max_F_prob_th']

        # evaluate at the point with max F-measure. Additional eval of position error.
        eval_stats = evaluator.bench_one_submit(lane_pred_file, test_gt_file, prob_th=max_f_prob)

        print("Metrics: AP, F-score, x error (close), x error (far), z error (close), z error (far)")
        print(
            "Laneline:  {:.3}  {:.3}  {:.3}  {:.3}  {:.3}  {:.3}".format(eval_stats_pr['laneline_AP'], eval_stats[0],
                                                                         eval_stats[3], eval_stats[4],
                                                                         eval_stats[5], eval_stats[6]))
        print("Centerline:  {:.3}  {:.3}  {:.3}  {:.3}  {:.3}  {:.3}".format(eval_stats_pr['centerline_AP'], eval_stats[7],
                                                                             eval_stats[10], eval_stats[11],
                                                                             eval_stats[12], eval_stats[13]))
        return eval_stats


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = define_args()
    args = parser.parse_args()

    # manual settings
    args.dataset_dir = '/media/yuliangguo/DATA1/Datasets/Apollo_Sim_3D_Lane_Release/'  # raw data dir
    args.dataset_name = 'illus_chg'  # choose a data split 'standard' / 'rare_subset' / 'illus_chg'
    args.mod = '3D_GeoNet_ext'  # model name
    test_name = 'test'  # test set name
    vis = False  # choose to visualize lanes per image

    # generate relative paths
    args.data_dir = ops.join('data_splits', args.dataset_name)
    args.save_path = os.path.join(ops.join('data_splits', args.dataset_name), args.mod)
    args.vis_folder = test_name + '_vis'
    if vis:
        mkdir_if_missing(os.path.join(args.save_path, 'example/' + args.vis_folder))
    test_gt_file = ops.join(args.data_dir, test_name + '.json')
    lane_pred_file = ops.join(args.save_path, test_name + '_pred_file.json')

    # load configuration for certain dataset
    sim3d_config(args)
    args.y_ref = 5
    # define evaluator
    evaluator = eval_3D_lane.LaneEval(args)
    args.prob_th = 0.5

    # Check GPU availability
    if not args.no_cuda and not torch.cuda.is_available():
        raise Exception("No gpu available for usage")
    torch.backends.cudnn.benchmark = args.cudnn

    # Define network
    model = GeoNet3D_ext.Net(args)
    define_init_weights(model, args.weight_init)
    if not args.no_cuda:
        # Load model on gpu before passing params to optimizer
        model = model.cuda()

    # load trained model for testing
    best_file_name = glob.glob(os.path.join(args.save_path, 'model_best*'))[0]
    if os.path.isfile(best_file_name):
        sys.stdout = Logger(os.path.join(args.save_path, 'Evaluate.txt'))
        print("=> loading checkpoint '{}'".format(best_file_name))
        checkpoint = torch.load(best_file_name)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(best_file_name))

    # Data loader
    test_dataset = LaneDataset(args.dataset_dir, test_gt_file, args)
    # assign std of test dataset to be consistent with train dataset
    with open(ops.join(args.data_dir, 'geo_anchor_std.json')) as f:
        anchor_std = json.load(f)
    test_dataset.set_x_off_std(anchor_std['x_off_std'])
    if not args.no_3d:
        test_dataset.set_z_std(anchor_std['z_std'])
    # normalization anchor locations
    test_dataset.normalize_lane_label()
    test_loader = get_loader(test_dataset, args)

    # initialize visualizer
    vs_saver = Visualizer(args, args.vis_folder)

    # deploy the model
    eval_stats = deploy(args, test_loader, test_dataset, model, vs_saver, test_gt_file, lane_pred_file)
