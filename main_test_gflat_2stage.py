#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.optim
import torch.nn as nn
import cv2
import copy

import os
import os.path as ops
import glob
import time
import sys
import shutil
import json
import matplotlib.pyplot as plt
import pdb
import torch.nn.functional as F
from tqdm import tqdm
from Dataloader.Load_Data_3DLane_gflat import LaneDataset, get_loader, compute_tusimple_lanes, compute_sim3d_lanes, unormalize_lane_anchor
from Networks import LaneNet3D_gflat, LaneNet3D_gflat_GeoOnly, erfnet
from tools.utils import define_args, first_run, tusimple_config, sim3d_config,\
                        mkdir_if_missing, Logger, define_init_weights,\
                        define_scheduler, define_optim, AverageMeter, Visualizer
from tools import eval_lane_tusimple, eval_3D_lane


def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
    own_state = model.state_dict()
    ckpt_name = []
    cnt = 0
    for name, param in state_dict.items():
        # TODO: why the trained model do not have modules in name?
        if name[7:] not in list(own_state.keys()) or 'output_conv' in name:
            ckpt_name.append(name)
            # continue
        own_state[name[7:]].copy_(param)
        cnt += 1
    print('#reused param: {}'.format(cnt))
    return model


def deploy(loader1, dataset1, dataset2, model1, model2, vs_saver1, vs_saver2, test_gt_file, epoch=0):

    # Evaluate model
    model2.eval()

    # Only forward pass, hence no gradients needed
    with torch.no_grad():
        with open(lane_pred_file, 'w') as jsonFile:
            # Start validation loop
            for i, (input, _, gt, idx, gt_hcam, gt_pitch) in tqdm(enumerate(loader1)):
                if not args1.no_cuda:
                    input, gt = input.cuda(non_blocking=True), gt.cuda(non_blocking=True)
                    input = input.float()
                input = input.contiguous()
                input = torch.autograd.Variable(input)

                # if not args1.fix_cam and not args1.pred_cam:
                # ATTENTION: here requires to update with test dataset args
                model2.update_projection(args1, gt_hcam, gt_pitch)

                # Evaluate model
                try:
                    output1 = model1(input, no_lane_exist=True)
                    output1 = F.softmax(output1, dim=1)
                    output1 = output1[:, 1, :, :].unsqueeze_(1)
                    output_net, pred_hcam, pred_pitch, x_proj, x_feat = model2(output1)
                except RuntimeError as e:
                    print("Batch with idx {} skipped due to singular matrix".format(idx.numpy()))
                    print(e)
                    continue

                gt = gt.data.cpu().numpy()
                output_net = output_net.data.cpu().numpy()
                output1 = output1.data.cpu().numpy()
                x_proj = x_proj.data.cpu().numpy()
                x_feat = x_feat.data.cpu().numpy()
                # pred_pitch = pred_pitch.data.cpu().numpy().flatten()
                pred_hcam = pred_hcam.data.cpu().numpy().flatten()
                gt_hcam = gt_hcam.data.cpu().numpy().flatten()

                # unormalize lane outputs
                num_el = input.size(0)
                for j in range(num_el):
                    unormalize_lane_anchor(gt[j], dataset1)
                    unormalize_lane_anchor(output_net[j], dataset2)

                input = input.permute(0, 2, 3, 1).data.cpu().numpy()
                # visualize and write results
                for j in range(num_el):
                    im_id = idx[j]
                    H_g2im, P_g2im, H_crop, H_im2ipm = dataset1.transform_mats(idx[j])

                    if vis_feat:
                        """
                            use two visual savers to satisfy both dimension setups for gt and output_net
                        """
                        img1 = input[j]
                        img1 = img1 * np.array(args1.vgg_std)
                        img1 = img1 + np.array(args1.vgg_mean)
                        img1 = np.clip(img1, 0, 1)
                        im_ipm1 = cv2.warpPerspective(img1, H_im2ipm, (args1.ipm_w, args1.ipm_h))
                        im_ipm1 = np.clip(im_ipm1, 0, 1)

                        # visualize on image
                        M1 = np.matmul(H_crop, H_g2im)
                        M2 = np.matmul(H_crop, P_g2im)
                        if args1.no_3d:
                            img1 = vs_saver1.draw_on_img_new(img1, gt[j], M1, 'laneline', color=[0, 0, 1])
                        else:
                            img1 = vs_saver1.draw_on_img_new(img1, gt[j], M2, 'laneline', color=[0, 0, 1])
                        img1 = vs_saver2.draw_on_img_new(img1, output_net[j], M2, 'laneline', color=[1, 0, 0])

                        # visualize on ipm
                        im_ipm1 = vs_saver1.draw_on_ipm_new(im_ipm1, gt[j], 'laneline', color=[0, 0, 1])
                        im_ipm1 = vs_saver2.draw_on_ipm_new(im_ipm1, output_net[j], 'laneline', color=[1, 0, 0])

                        fig = plt.figure()
                        ax1 = fig.add_subplot(231)
                        ax2 = fig.add_subplot(232)
                        ax3 = fig.add_subplot(233, projection='3d')
                        ax4 = fig.add_subplot(234)
                        ax5 = fig.add_subplot(235)
                        ax6 = fig.add_subplot(236)
                        ax1.imshow(img1)
                        ax2.imshow(im_ipm1)
                        vs_saver1.draw_3d_curves_new(ax3, gt[j], gt_hcam[j], 'laneline', [0, 0, 1])
                        vs_saver2.draw_3d_curves_new(ax3, output_net[j], pred_hcam[j], 'laneline', [1, 0, 0])
                        ax3.set_xlabel('x axis')
                        ax3.set_ylabel('y axis')
                        ax3.set_zlabel('z axis')
                        bottom, top = ax3.get_zlim()
                        ax3.set_zlim(min(bottom, -1), max(top, 1))
                        ax3.set_xlim(-20, 20)
                        ax3.set_ylim(0, 100)
                        # visualize features
                        pred = output1[j, 0, :, :]
                        ax4.imshow(pred)
                        x_proj_i = x_proj[j, 0, :, :]
                        ax5.imshow(x_proj_i)
                        x_feat_i = x_feat[j, :, :, :]
                        x_feat_i = np.sum(np.square(x_feat_i), axis=0)
                        x_feat_i = x_feat_i / np.max(x_feat_i)
                        ax6.imshow(x_feat_i)

                        fig.savefig(vs_saver2.save_path + '/example/' + vs_saver2.vis_folder + '/infer_{}'.format(idx[j]))
                        plt.clf()
                        plt.close(fig)

                    """
                        save results in test dataset format
                    """
                    json_line = test_set_labels[im_id]
                    lane_anchors = output_net[j]
                    # convert to json output format
                    if 'tusimple' in args1.dataset_name:
                        h_samples = json_line["h_samples"]
                        lane_anchor_short = np.concatenate((lane_anchors[:, 0:args1.num_y_steps],
                                                            lane_anchors[:, 3*args1.num_y_steps].reshape(-1, 1)), axis=1)
                        lanes_pred = compute_tusimple_lanes(lane_anchor_short, h_samples, H_g2im,
                                                            dataset1.anchor_x_steps, args1.anchor_y_steps, 0, args1.org_w, args1.prob_th)
                        json_line["lanes"] = lanes_pred
                        json_line["run_time"] = 0
                        json.dump(json_line, jsonFile)
                        jsonFile.write('\n')
                    elif 'sim3d' in args1.dataset_name:
                        lanelines_pred, centerlines_pred = compute_sim3d_lanes(lane_anchors, dataset1.anchor_dim,
                                                                               dataset1.anchor_x_steps,
                                                                               args1.anchor_y_steps,
                                                                               pred_hcam[j], args1.prob_th)
                        json_line["laneLines"] = lanelines_pred
                        json_line["centerLines"] = centerlines_pred
                        json.dump(json_line, jsonFile)
                        jsonFile.write('\n')
        eval_stats = evaluator.bench_one_submit(lane_pred_file, test_gt_file)

        if 'tusimple' in args1.dataset_name:
            print("===> Evaluation accuracy on validation set is {:.8}".format(eval_stats[0]))
        elif 'sim3d' in args1.dataset_name:
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
                                                               eval_stats[4], eval_stats[5],
                                                               eval_stats[6], eval_stats[7],
                                                               eval_stats[8], eval_stats[9],
                                                               eval_stats[10], eval_stats[11],
                                                               eval_stats[12], eval_stats[13]))

        return eval_stats


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    global vis_feat
    vis_feat = False

    global args1, args2
    parser = define_args()
    args1 = parser.parse_args()
    args2 = parser.parse_args()

    # dataset_name 'tusimple' or 'sim3d'
    args1.dataset_name = 'tusimple'
    args1.dataset_dir = '/home/yuliangguo/Datasets/tusimple/'
    # args1.dataset_name = 'sim3d_0924'
    # args1.dataset_dir = '/home/yuliangguo/Datasets/Apollo_Sim_3D_Lane_0924/'
    args2.dataset_name = 'sim3d_0924'
    args2.dataset_dir = '/home/yuliangguo/Datasets/Apollo_Sim_3D_Lane_0924/'

    args1.data_dir = ops.join('data', args1.dataset_name)
    args2.data_dir = ops.join('data', args2.dataset_name)

    # define trained Geo model
    geo_model_dir = 'Model_3DLaneNet_gflat_2stage_crit_loss_gflat_opt_adam_lr_0.0005_batch_8_360X480_pretrain_False_batchnorm_True_predcam_False'
    # use two sets of configurations for different datasets
    sim3d_config(args2)
    args2.save_path = os.path.join(args2.save_path, geo_model_dir)

    global evaluator
    # define evaluator
    if 'tusimple' in args1.dataset_name:
        tusimple_config(args1)
        args1.crop_y = 0
        evaluator = eval_lane_tusimple.LaneEval
        # define pretrained feat model
        pretrained_feat_model = 'pretrained/erfnet_model_tusimple.tar'
        vis_folder = 'test_vis_tusimple'
        test_gt_file = ops.join(args1.data_dir, 'test.json')
        lane_pred_file = ops.join(args2.save_path, 'test_pred_file_tusimple.json')
    elif 'sim3d' in args1.dataset_name:
        sim3d_config(args1)
        evaluator = eval_3D_lane.LaneEval(args1)
        # define pretrained feat model
        pretrained_feat_model = 'pretrained/erfnet_model_sim3d.tar'
        vis_folder = 'test2_vis_sim3d'
        test_gt_file = ops.join(args1.data_dir, 'test2.json')
        lane_pred_file = ops.join(args2.save_path, 'test2_pred_file_sim3d.json')

    # define the network model
    args1.mod = '3DLaneNet_gflat_2stage'
    args2.mod = '3DLaneNet_gflat_2stage'

    """   run the test   """
    # Check GPU availability
    if not args1.no_cuda and not torch.cuda.is_available():
        raise Exception("No gpu available for usage")
    torch.backends.cudnn.benchmark = args1.cudnn

    # Dataloader for training and test set
    train_dataset = LaneDataset(args2.dataset_dir, ops.join(args2.data_dir, 'train.json'), args2, data_aug=True)
    train_dataset.normalize_lane_label()

    test_dataset = LaneDataset(args1.dataset_dir, test_gt_file, args1)
    test_dataset.normalize_lane_label()
    test_loader = get_loader(test_dataset, args1)

    global test_set_labels
    test_set_labels = [json.loads(line) for line in open(test_gt_file).readlines()]

    # Define network
    model1 = erfnet.ERFNet(2)
    model2 = LaneNet3D_gflat_GeoOnly.Net(args2, debug=True)
    define_init_weights(model2, args2.weight_init)

    if not args1.no_cuda:
        # Load model on gpu before passing params to optimizer
        model1 = model1.cuda()
        model2 = model2.cuda()

    # load in vgg pretrained weights
    checkpoint = torch.load(pretrained_feat_model)
    model1 = load_my_state_dict(model1, checkpoint['state_dict'])
    model1.eval()  # do not back propagate to model1

    # initialize visual saver
    vs_saver1 = Visualizer(args1, vis_folder)
    vs_saver2 = Visualizer(args2, vis_folder)

    # load trained model for testing
    best_file_name = glob.glob(os.path.join(args2.save_path, 'model_best*'))[0]
    if os.path.isfile(best_file_name):
        sys.stdout = Logger(os.path.join(args1.save_path, 'Evaluate.txt'))
        print("=> loading checkpoint '{}'".format(best_file_name))
        checkpoint = torch.load(best_file_name)
        model2.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(best_file_name))
    mkdir_if_missing(os.path.join(args2.save_path, 'example/' + vis_folder))
    eval_stats = deploy(test_loader, test_dataset, train_dataset, model1, model2, vs_saver1, vs_saver2, test_gt_file)
