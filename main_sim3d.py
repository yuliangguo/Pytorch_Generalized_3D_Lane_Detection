#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.optim
import torch.nn as nn

import os
import os.path as ops
import glob
import time
import sys
import shutil
import json
import pdb
from tqdm import tqdm
from tensorboardX import SummaryWriter

from Dataloader.Load_Data_3DLane import LaneDataset, get_loader, compute_tusimple_lanes, compute_sim3d_lanes
from Networks.Loss_crit import Laneline_3D_loss
from Networks.LaneNet3D import Net
from tools.utils import define_args, first_run, tusimple_config, sim3d_config,\
                        mkdir_if_missing, Logger, define_init_weights,\
                        define_scheduler, define_optim, AverageMeter, Visualizer
from tools import eval_lane_tusimple, eval_3D_lane


def train_net():

    # Check GPU availability
    if not args.no_cuda and not torch.cuda.is_available():
        raise Exception("No gpu available for usage")
    torch.backends.cudnn.benchmark = args.cudnn

    # Define save path
    save_id = 'Model_{}_opt_{}_lr_{}_batch_{}_{}X{}_pretrain_{}_batchnorm_{}' \
              .format(args.mod,
                      args.optimizer,
                      args.learning_rate,
                      args.batch_size,
                      args.resize_h,
                      args.resize_w,
                      args.pretrained,
                      args.batch_norm)

    # Dataloader for training and validation set
    val_gt_file = ops.join(args.data_dir, 'val.json')
    train_dataset = LaneDataset(args.dataset_dir, ops.join(args.data_dir, 'train.json'), args)
    train_loader = get_loader(train_dataset, args)
    valid_dataset = LaneDataset(args.dataset_dir, val_gt_file, args)
    valid_loader = get_loader(valid_dataset, args)

    # extract valid set labels for evaluation later
    global valid_set_labels
    valid_set_labels = [json.loads(line) for line in open(val_gt_file).readlines()]
    global anchor_x_steps
    anchor_x_steps = valid_dataset.anchor_x_steps

    # Define network
    model = Net(args)
    define_init_weights(model, args.weight_init)

    # load in vgg pretrained weights on ImageNet
    if args.pretrained:
        model.load_pretrained_vgg(args.batch_norm)
        print('vgg weights pretrained on ImageNet loaded!')

    if not args.no_cuda:
        # Load model on gpu before passing params to optimizer
        model = model.cuda()

    # Define optimizer and scheduler
    optimizer = define_optim(args.optimizer, model.parameters(),
                             args.learning_rate, args.weight_decay)
    scheduler = define_scheduler(optimizer, args)

    if args.no_centerline:
        num_lane_type = 1
    else:
        num_lane_type = 3

    global anchor_dim
    if args.no_3d:
        anchor_dim = args.num_y_steps + 1
    else:
        anchor_dim = 2 * args.num_y_steps + 1

    # Define loss criteria
    criterion = Laneline_3D_loss(num_lane_type, anchor_dim, args.pred_cam)

    if not args.no_cuda:
        criterion = criterion.cuda()

    # Name
    global crit_string
    crit_string = '3Dlaneline loss'

    # Logging setup
    best_epoch = 0
    lowest_loss = np.inf
    log_file_name = 'log_train_start_0.txt'
    args.save_path = os.path.join(args.save_path, save_id)
    mkdir_if_missing(args.save_path)
    mkdir_if_missing(os.path.join(args.save_path, 'example/'))
    mkdir_if_missing(os.path.join(args.save_path, 'example/train'))
    mkdir_if_missing(os.path.join(args.save_path, 'example/valid'))

    # Tensorboard writer
    if not args.no_tb:
        global writer
        writer = SummaryWriter(os.path.join(args.save_path, 'Tensorboard/'))

    # initialize visual saver
    vs_saver = Visualizer(args)

    # Train, evaluate or resume
    args.resume = first_run(args.save_path)
    if args.resume and not args.test_mode and not args.evaluate:
        path = os.path.join(args.save_path, 'checkpoint_model_epoch_{}.pth.tar'.format(
            int(args.resume)))
        if os.path.isfile(path):
            log_file_name = 'log_train_start_{}.txt'.format(args.resume)
            # Redirect stdout
            sys.stdout = Logger(os.path.join(args.save_path, log_file_name))
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(path)
            args.start_epoch = checkpoint['epoch']
            lowest_loss = checkpoint['loss']
            best_epoch = checkpoint['best epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            log_file_name = 'log_train_start_0.txt'
            # Redirect stdout
            sys.stdout = Logger(os.path.join(args.save_path, log_file_name))
            print("=> no checkpoint found at '{}'".format(path))

    # Only evaluate
    elif args.evaluate:
        best_file_name = glob.glob(os.path.join(args.save_path, 'model_best*'))[0]
        if os.path.isfile(best_file_name):
            sys.stdout = Logger(os.path.join(args.save_path, 'Evaluate.txt'))
            print("=> loading checkpoint '{}'".format(best_file_name))
            checkpoint = torch.load(best_file_name)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(best_file_name))
        mkdir_if_missing(os.path.join(args.save_path, 'example/eval_vis'))
        loss_valid, eval_stats = validate(valid_loader, valid_dataset, model, criterion, vs_saver, val_gt_file)
        return

    # Start training from clean slate
    else:
        # Redirect stdout
        sys.stdout = Logger(os.path.join(args.save_path, log_file_name))

    # INIT MODEL
    print(40*"="+"\nArgs:{}\n".format(args)+40*"=")
    print("Init model: '{}'".format(args.mod))
    print("Number of parameters in model {} is {:.3f}M".format(
        args.mod, sum(tensor.numel() for tensor in model.parameters())/1e6))

    # Start training and validation for nepochs
    for epoch in range(args.start_epoch, args.nepochs):
        print("\n => Start train set for EPOCH {}".format(epoch + 1))
        # Adjust learning rate
        if args.lr_policy is not None and args.lr_policy != 'plateau':
            scheduler.step()
            lr = optimizer.param_groups[0]['lr']
            print('lr is set to {}'.format(lr))

        # Define container objects to keep track of multiple losses/metrics
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        # Specify operation modules
        model.train()

        # compute timing
        end = time.time()

        # Start training loop
        for i, (input, gt, idx, gt_hcam, gt_pitch) in tqdm(enumerate(train_loader)):

            # Time dataloader
            data_time.update(time.time() - end)

            # Put inputs on gpu if possible
            if not args.no_cuda:
                input, gt = input.cuda(non_blocking=True), gt.cuda(non_blocking=True)
                input = input.float()

            if not args.fix_cam and not args.pred_cam:
                model.update_projection(args, gt_hcam, gt_pitch)

            # Run model
            optimizer.zero_grad()
            # Evaluate model
            try:
                output_net, pred_hcam, pred_pitch = model(input)
            except RuntimeError as e:
                print("Batch with idx {} skipped due to singular matrix".format(idx.numpy()))
                print(e)
                continue

            # Compute losses on
            gt = gt.cuda(non_blocking=True)
            if args.fix_cam:
                loss = criterion(output_net, gt)
            else:
                loss = criterion(output_net, gt, pred_hcam, gt_hcam, pred_pitch, gt_pitch)

            losses.update(loss.item(), input.size(0))

            # Clip gradients (usefull for instabilities or mistakes in ground truth)
            if args.clip_grad_norm != 0:
                nn.utils.clip_grad_norm(model.parameters(), args.clip_grad_norm)

            # Setup backward pass
            loss.backward()
            optimizer.step()

            # Time trainig iteration
            batch_time.update(time.time() - end)
            end = time.time()

            # Print info
            if (i + 1) % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.8f} ({loss.avg:.8f})'.format(
                       epoch+1, i+1, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses))

            # Plot curves in two views
            if (i + 1) % args.save_freq == 0:
                vs_saver.save_result('train', epoch, i, idx,
                                     input, gt, output_net, pred_pitch, pred_hcam, train_dataset)

        losses_valid, eval_stats = validate(valid_loader, valid_dataset, model, criterion, vs_saver, val_gt_file, epoch)

        print("===> Average {}-loss on training set is {:.8f}".format(crit_string, losses.avg))
        print("===> Average {}-loss on validation set is {:.8f}".format(crit_string, losses_valid))
        if args.dataset_name is 'tusimple':
            print("===> Evaluation accuracy: {:3f}".format(eval_stats[0]))
        elif args.dataset_name is 'sim3d':
            print("===> Evaluation laneline F (pixel): {:3f}".format(eval_stats[0]))
            print("===> Evaluation laneline F (lane): {:3f}".format(eval_stats[1]))
            print("===> Evaluation centerline F (pixel): {:3f}".format(eval_stats[2]))
            print("===> Evaluation centerline F (lane): {:3f}".format(eval_stats[3]))

        print("===> Last best {}-loss was {:.8f} in epoch {}".format(crit_string, lowest_loss, best_epoch))

        if not args.no_tb:
            writer.add_scalars('3D-Lane-Loss', {'Training': losses.avg}, epoch)
            writer.add_scalars('3D-Lane-Loss', {'Validation': losses_valid}, epoch)
            if args.dataset_name is 'tusimple':
                writer.add_scalars('Evaluation', {'Accuracy': eval_stats[0]}, epoch)
            elif args.dataset_name is 'sim3d':
                writer.add_scalars('Evaluation', {'laneline F (pixel)': eval_stats[0]}, epoch)
                writer.add_scalars('Evaluation', {'laneline F (lane)': eval_stats[1]}, epoch)
                writer.add_scalars('Evaluation', {'centerline F (pixel)': eval_stats[2]}, epoch)
                writer.add_scalars('Evaluation', {'centerline F (lane)': eval_stats[3]}, epoch)
        total_score = losses.avg

        # Adjust learning_rate if loss plateaued
        if args.lr_policy == 'plateau':
            scheduler.step(total_score)
            lr = optimizer.param_groups[0]['lr']
            print('LR plateaued, hence is set to {}'.format(lr))

        # File to keep latest epoch
        with open(os.path.join(args.save_path, 'first_run.txt'), 'w') as f:
            f.write(str(epoch))
        # Save model
        to_save = False
        if total_score < lowest_loss:
            to_save = True
            best_epoch = epoch+1
            lowest_loss = total_score
        save_checkpoint({
            'epoch': epoch + 1,
            'best epoch': best_epoch,
            'arch': args.mod,
            'state_dict': model.state_dict(),
            'loss': lowest_loss,
            'optimizer': optimizer.state_dict()}, to_save, epoch)
    if not args.no_tb:
        writer.close()


def validate(loader, dataset, model, criterion, vs_saver, val_gt_file, epoch=0):

    # Define container to keep track of metric and loss
    losses = AverageMeter()
    lane_pred_file = ops.join(args.save_path, 'val_pred_file.json')

    # Evaluate model
    model.eval()

    # Only forward pass, hence no gradients needed
    with torch.no_grad():
        with open(lane_pred_file, 'w') as jsonFile:
            # Start validation loop
            for i, (input, gt, idx, gt_hcam, gt_pitch) in tqdm(enumerate(loader)):
                if not args.no_cuda:
                    input, gt = input.cuda(non_blocking=True), gt.cuda(non_blocking=True)
                    input = input.float()

                if not args.fix_cam and not args.pred_cam:
                    model.update_projection(args, gt_hcam, gt_pitch)
                # Evaluate model
                try:
                    output_net, pred_hcam, pred_pitch = model(input)
                except RuntimeError as e:
                    print("Batch with idx {} skipped due to singular matrix".format(idx.numpy()))
                    print(e)
                    continue

                # Compute losses on parameters or segmentation
                gt = gt.cuda(non_blocking=True)
                loss = criterion(output_net, gt, pred_hcam, gt_hcam, pred_pitch, gt_pitch)
                losses.update(loss.item(), input.size(0))

                # Print info
                if (i + 1) % args.print_freq == 0:
                        print('Test: [{0}/{1}]\t'
                              'Loss {loss.val:.8f} ({loss.avg:.8f})'.format(
                               i+1, len(loader), loss=losses))

                # Plot curves in two views
                if (i + 1) % args.save_freq == 0 or args.evaluate:
                    vs_saver.save_result('valid', epoch, i, idx,
                                         input, gt, output_net, pred_pitch, pred_hcam,
                                         dataset, args.evaluate)

                # write results and evaluate
                output_net = output_net.data.cpu().numpy()
                num_el = input.size(0)

                for j in range(num_el):
                    im_id = idx[j]
                    H_g2im, H_crop, H_im2ipm = dataset.proj_trainsforms(idx[j])
                    json_line = valid_set_labels[im_id]
                    if args.dataset_name is 'tusimple':
                        h_samples = json_line["h_samples"]
                        lanes_pred = compute_tusimple_lanes(output_net[j], h_samples, H_g2im,
                                                            anchor_x_steps, args.anchor_y_steps, 0, args.org_w)
                        json_line["lanes"] = lanes_pred
                        json_line["run_time"] = 0
                        json.dump(json_line, jsonFile)
                        jsonFile.write('\n')
                    elif args.dataset_name is 'sim3d':
                        lanelines_pred, centerlines_pred = compute_sim3d_lanes(output_net[j], anchor_dim,
                                                                               anchor_x_steps, args.anchor_y_steps)
                        json_line["laneLines"] = lanelines_pred
                        json_line["centerLines"] = centerlines_pred
                        json.dump(json_line, jsonFile)
                        jsonFile.write('\n')
        eval_stats = evaluator.bench_one_submit(lane_pred_file, val_gt_file)

        if args.evaluate:
            print("===> Average {}-loss on validation set is {:.8}".format(crit_string, losses.avg))
            if args.dataset_name is 'tusimple':
                print("===> Evaluation accuracy on validation set is {:.8}".format(eval_stats[0]))
            elif args.dataset_name is 'sim3d':
                print("===> Evaluation on validation set: \n"
                      "laneline F (pixel) {:.8} \n"
                      "laneline F (lane) {:.8} \n"
                      "centerline F (pixel) {:.8} \n"
                      "centerline F (lane) {:.8} \n".format(eval_stats[0], eval_stats[1], eval_stats[2], eval_stats[3]))

        return losses.avg, eval_stats


def save_checkpoint(state, to_copy, epoch):
    filepath = os.path.join(args.save_path, 'checkpoint_model_epoch_{}.pth.tar'.format(epoch))
    torch.save(state, filepath)
    if to_copy:
        if epoch > 0:
            lst = glob.glob(os.path.join(args.save_path, 'model_best*'))
            if len(lst) != 0:
                os.remove(lst[0])
        shutil.copyfile(filepath, os.path.join(args.save_path, 
            'model_best_epoch_{}.pth.tar'.format(epoch)))
        print("Best model copied")
    if epoch > 0:
        prev_checkpoint_filename = os.path.join(args.save_path, 
                'checkpoint_model_epoch_{}.pth.tar'.format(epoch-1))
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    global args
    parser = define_args()
    args = parser.parse_args()

    args.dataset_name = 'sim3d'
    args.data_dir = ops.join('data', args.dataset_name)
    args.dataset_dir = '/home/yuliangguo/Datasets/Apollo_Sim_3D_Lane/'

    # load configuration for certain dataset
    global evaluator
    if args.dataset_name is 'tusimple':
        tusimple_config(args)
        # define evaluator
        evaluator = eval_lane_tusimple.LaneEval
    elif args.dataset_name is 'sim3d':
        sim3d_config(args)
        # define evaluator
        args.pixel_per_meter = 10.
        args.dist_th = 1.5
        args.pt_th = 0.5
        args.min_num_pixels = 10
        evaluator = eval_3D_lane.LaneEval(args)

    # for the case only running evaluation
    args.evaluate = True

    # settings for save and visualize
    args.nworkers = 0
    args.no_tb = False
    args.print_freq = 40
    args.save_freq = 40

    # run the training
    train_net()
