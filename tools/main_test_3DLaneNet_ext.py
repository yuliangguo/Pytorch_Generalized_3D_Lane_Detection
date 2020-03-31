#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.optim
import glob
from tqdm import tqdm
from Dataloader.Load_Data_3DLane_ext import *
from Networks import LaneNet3D_ext, GeoNet3D_ext
from tools.utils import *
from tools import eval_3D_lane


def main():

    # Check GPU availability
    if not args.no_cuda and not torch.cuda.is_available():
        raise Exception("No gpu available for usage")
    torch.backends.cudnn.benchmark = args.cudnn

    # Dataloader for training and test set
    train_dataset = LaneDataset(args.dataset_dir, ops.join(args.data_dir, 'train.json'), args, data_aug=True)
    train_dataset.normalize_lane_label()
    test_dataset = LaneDataset(args.test_dataset_dir, test_gt_file, args)
    test_dataset.set_x_off_std(train_dataset._x_off_std)
    if not args.no_3d:
        test_dataset.set_z_std(train_dataset._z_std)
    # need to perform normalization after reset std
    test_dataset.normalize_lane_label()
    test_loader = get_loader(test_dataset, args)

    global test_set_labels
    test_set_labels = [json.loads(line) for line in open(test_gt_file).readlines()]
    global anchor_x_steps
    anchor_x_steps = test_dataset.anchor_x_steps

    # Define network
    model = LaneNet3D_ext.Net(args, debug=True)
    define_init_weights(model, args.weight_init)

    if not args.no_cuda:
        # Load model on gpu before passing params to optimizer
        model = model.cuda()

    global anchor_dim
    if args.no_3d:
        anchor_dim = args.num_y_steps + 1
    else:
        anchor_dim = 2 * args.num_y_steps + 1

    # initialize visual saver
    vs_saver = Visualizer(args, vis_folder)

    # load trained model for testing
    best_test_name = glob.glob(os.path.join(args.save_path, 'model_best*'))[0]
    if os.path.isfile(best_test_name):
        sys.stdout = Logger(os.path.join(args.save_path, 'Evaluate.txt'))
        print("=> loading checkpoint '{}'".format(best_test_name))
        checkpoint = torch.load(best_test_name)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(best_test_name))
    mkdir_if_missing(os.path.join(args.save_path, 'example/' + vis_folder))
    eval_stats = deploy(test_loader, test_dataset, model, vs_saver, test_gt_file)


def deploy(loader, dataset, model, vs_saver, test_gt_file, epoch=0):

    # Evaluate model
    model.eval()

    # Only forward pass, hence no gradients needed
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
                    output_net, pred_hcam, pred_pitch,\
                        x1_feat, x2_feat, x3_feat, x4_feat,\
                        x1_proj, x2_proj, x3_proj, x4_proj, top_2, top_3, top_4 = model(input)
                except RuntimeError as e:
                    print("Batch with idx {} skipped due to singular matrix".format(idx.numpy()))
                    print(e)
                    continue

                gt = gt.data.cpu().numpy()
                output_net = output_net.data.cpu().numpy()
                pred_pitch = pred_pitch.data.cpu().numpy().flatten()
                pred_hcam = pred_hcam.data.cpu().numpy().flatten()
                im = input.permute(0, 2, 3, 1).data.cpu().numpy()[0]
                im = im * np.array(args.vgg_std)
                im = im + np.array(args.vgg_mean)

                if vis_feat:
                    x1_feat = x1_feat.squeeze(0).data.cpu().numpy()
                    x2_feat = x2_feat.squeeze(0).data.cpu().numpy()
                    x3_feat = x3_feat.squeeze(0).data.cpu().numpy()
                    x4_feat = x4_feat.squeeze(0).data.cpu().numpy()
                    x1_proj = x1_proj.squeeze(0).data.cpu().numpy()
                    x2_proj = x2_proj.squeeze(0).data.cpu().numpy()
                    x3_proj = x3_proj.squeeze(0).data.cpu().numpy()
                    x4_proj = x4_proj.squeeze(0).data.cpu().numpy()
                    top_2 = top_2.squeeze(0).data.cpu().numpy()
                    top_3 = top_3.squeeze(0).data.cpu().numpy()
                    top_4 = top_4.squeeze(0).data.cpu().numpy()
                    # compute attention map for visualization
                    x1_feat = np.sum(np.square(x1_feat), axis=0)
                    x2_feat = np.sum(np.square(x2_feat), axis=0)
                    x3_feat = np.sum(np.square(x3_feat), axis=0)
                    x4_feat = np.sum(np.square(x4_feat), axis=0)
                    x1_proj = np.sum(np.square(x1_proj), axis=0)
                    x2_proj = np.sum(np.square(x2_proj), axis=0)
                    x3_proj = np.sum(np.square(x3_proj), axis=0)
                    x4_proj = np.sum(np.square(x4_proj), axis=0)
                    top_2 = np.sum(np.square(top_2), axis=0)
                    top_3 = np.sum(np.square(top_3), axis=0)
                    top_4 = np.sum(np.square(top_4), axis=0)

                    x1_feat = x1_feat/np.max(x1_feat)
                    x2_feat = x2_feat/np.max(x2_feat)
                    x3_feat = x3_feat/np.max(x3_feat)
                    x4_feat = x4_feat/np.max(x4_feat)
                    x1_proj = x1_proj/np.max(x1_proj)
                    x2_proj = x2_proj/np.max(x2_proj)
                    x3_proj = x3_proj/np.max(x3_proj)
                    x4_proj = x4_proj/np.max(x4_proj)
                    top_2 = top_2/np.max(top_2)
                    top_3 = top_3/np.max(top_3)
                    top_4 = top_4/np.max(top_4)

                    # visualize features
                    fig = plt.figure()

                    plt.subplot(341)
                    plt.title('img feat1', fontsize=30)
                    plt.imshow(x1_feat)
                    plt.subplot(342)
                    plt.title('img feat2', fontsize=30)
                    plt.imshow(x2_feat)
                    plt.subplot(343)
                    plt.title('img feat3', fontsize=30)
                    plt.imshow(x3_feat)
                    plt.subplot(344)
                    plt.title('img feat4', fontsize=30)
                    plt.imshow(x4_feat)

                    plt.subplot(345)
                    plt.title('proj feat1', fontsize=30)
                    plt.imshow(x1_proj)
                    plt.subplot(346)
                    plt.title('proj feat2', fontsize=30)
                    plt.imshow(x2_proj)
                    plt.subplot(347)
                    plt.title('proj feat3', fontsize=30)
                    plt.imshow(x3_proj)
                    plt.subplot(348)
                    plt.title('proj feat4', fontsize=30)
                    plt.imshow(x4_proj)

                    plt.subplot(349)
                    plt.title('image', fontsize=30)
                    plt.imshow(im)
                    plt.subplot(3, 4, 10)
                    plt.title('top feat2', fontsize=30)
                    plt.imshow(top_2)
                    plt.subplot(3, 4, 11)
                    plt.title('top feat3', fontsize=30)
                    plt.imshow(top_3)
                    plt.subplot(3, 4, 12)
                    plt.title('top feat4', fontsize=30)
                    plt.imshow(top_4)

                    plt.savefig(args.save_path + '/example/' + vis_folder + '/features_{}'.format(idx[0]))
                    plt.close(fig)

                # unormalize lane outputs
                num_el = input.size(0)
                for j in range(num_el):
                    unormalize_lane_anchor(output_net[j], dataset)
                    unormalize_lane_anchor(gt[j], dataset)

                # Plot curves in two views
                vs_saver.save_result_new(dataset, 'valid', epoch, i, idx,
                                         input, gt, output_net, pred_pitch, pred_hcam, evaluate=False)

                # write results and evaluate
                for j in range(num_el):
                    im_id = idx[j]
                    H_g2im, P_g2im, H_crop, H_im2ipm = dataset.transform_mats(idx[j])
                    json_line = test_set_labels[im_id]
                    lane_anchors = output_net[j]
                    # convert to json output format
                    lanelines_pred, centerlines_pred, lanelines_prob, centerlines_prob =\
                        compute_3d_lanes_all_prob(lane_anchors, dataset.anchor_dim,
                                                  anchor_x_steps, args.anchor_y_steps, pred_hcam[j])
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
        print("Centerline:  {:.3}  {:.3}  {:.3}  {:.3}  {:.3}  {:.3}".format(eval_stats_pr['centerline_AP'],
                                                                             eval_stats[7],
                                                                             eval_stats[10], eval_stats[11],
                                                                             eval_stats[12], eval_stats[13]))
        return eval_stats


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    global args
    parser = define_args()
    args = parser.parse_args()

    global vis_feat
    vis_feat = False

    # dataset_name: 'standard' / 'rare_subset' / 'illus_chg'
    args.dataset_name = 'rare_subset'
    args.dataset_dir = '/media/yuliangguo/DATA1/Datasets/Apollo_Sim_3D_Lane_Release/'
    args.test_dataset_dir = '/media/yuliangguo/DATA1/Datasets/Apollo_Sim_3D_Lane_Release/'
    args.data_dir = ops.join('data_splits', args.dataset_name)
    args.save_path = ops.join('data_splits', args.dataset_name)

    # load configuration for certain dataset
    global evaluator
    sim3d_config(args)
    # define evaluator
    evaluator = eval_3D_lane.LaneEval(args)
    args.prob_th = 0.5

    # define the network model
    args.mod = '3D_LaneNet_ext'

    # settings for save and visualize
    args.save_path = os.path.join(args.save_path, args.mod)
    global vis_folder
    global test_gt_file
    global lane_pred_file

    test_name = 'test'
    vis_folder = test_name + '_vis'
    test_gt_file = ops.join(args.data_dir, test_name + '.json')
    lane_pred_file = ops.join(args.save_path, test_name + '_pred_file.json')

    # run the test
    main()
