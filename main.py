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

from Dataloader.Load_Data_3DLane import get_loader, compute_homograpthy, homography_crop_resize
from Networks.Loss_crit import Laneline_3D_loss, Laneline_3D_loss_fix_cam
from Networks.LaneNet3D import Net, init_projective_transform
from tools.utils import define_args, first_run,\
                           mkdir_if_missing, Logger, define_init_weights,\
                           define_scheduler, define_optim, AverageMeter, save_vis_result_2d


def train_net():

    # Check GPU availability
    if not args.no_cuda and not torch.cuda.is_available():
        raise Exception("No gpu available for usage")
    torch.backends.cudnn.benchmark = args.cudnn

    # Define save path
    save_id = 'Model_{}_opt_{}_lr_{}_batch_{}_pretrain{}' \
              .format(args.mod,
                      args.optimizer,
                      args.learning_rate,
                      args.batch_size,
                      args.pretrained)

    # compute homography matrix
    pitch = np.pi / 180 * args.pitch
    M, M_inv = init_projective_transform(args.top_view_region, [args.org_h, args.org_w],
                                         args.crop_size, [args.resize_h, args.resize_w], pitch, args.cam_height, args.K)

    # Dataloader for training and validation set
    train_loader = get_loader(args.dataset_dir, ops.join(args.data_dir, 'train.json'), args)
    valid_loader = get_loader(args.dataset_dir, ops.join(args.data_dir, 'val.json'), args)

    # Define network
    model = Net(args)
    define_init_weights(model, args.weight_init)

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

    if args.no_3d:
        anchor_dim = args.num_y_anchor + 1
    else:
        anchor_dim = 2 * args.num_y_anchor + 1

    # Define loss criteria for multiple tasks
    if args.fix_cam:
        criterion = Laneline_3D_loss_fix_cam(num_lane_type, anchor_dim)
    else:
        criterion = Laneline_3D_loss(num_lane_type, anchor_dim)

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
        validate(valid_loader, model, criterion, M)
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
        # avg_area = AverageMeter()
        # exact_area = AverageMeter()

        # Specify operation modules
        # TODO: check this later
        model.train()

        # compute timing
        end = time.time()

        # Start training loop
        for i, (input, gt, idx) in tqdm(enumerate(train_loader)):

            # Time dataloader
            data_time.update(time.time() - end)

            # Put inputs on gpu if possible
            if not args.no_cuda:
                input, gt = input.cuda(non_blocking=True), gt.cuda(non_blocking=True)
                input = input.float()

            # Run model
            try:
                output_net = model(input)
            except RuntimeError as e:
                print("Batch with idx {} skipped due to singular matrix".format(idx.numpy()))
                print(e)
                continue

            # Compute losses on
            gt = gt.cuda(non_blocking=True)
            if args.fix_cam:
                loss = criterion(output_net, gt)
            else:
                # TODO: implement this when integrated online calibration network
                loss = criterion(output_net, gt, 0, 0, 0, 0)

            losses.update(loss.item(), input.size(0))

            # Clip gradients (usefull for instabilities or mistakes in ground truth)
            if args.clip_grad_norm != 0:
                nn.utils.clip_grad_norm(model.parameters(), args.clip_grad_norm)

            # Setup backward pass
            optimizer.zero_grad()
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
                save_vis_result_2d('train', M, gt, output_net, idx, i, input,
                                   args.ipm_h, args.ipm_w, args.save_path)

        losses_valid = validate(valid_loader, model, criterion, M, epoch)

        print("===> Average {}-loss on training set is {:.8f}".format(crit_string, losses.avg))
        print("===> Average {}-loss on validation set is {:.8f}".format(crit_string, losses_valid))

        print("===> Last best {}-loss was {:.8f} in epoch {}".format(
            crit_string, lowest_loss, best_epoch))

        if not args.no_tb:
            writer.add_scalars('3D-Lane-Loss', {'Training': losses.avg}, epoch)
            writer.add_scalars('3D-Lane-Loss', {'Validation': losses_valid}, epoch)

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


def validate(loader, model, criterion, M, epoch=0):

    # Define container to keep track of metric and loss
    losses = AverageMeter()

    # Evaluate model
    model.eval()

    # Only forward pass, hence no gradients needed
    with torch.no_grad():
        
        # Start validation loop
        for i, (input, gt, idx) in tqdm(enumerate(loader)):
            if not args.no_cuda:
                input, gt = input.cuda(non_blocking=True), gt.cuda(non_blocking=True)
                input = input.float()

            # Evaluate model
            try:
                output_net, outputs_line = model(input)
            except RuntimeError as e:
                print("Batch with idx {} skipped due to singular matrix".format(idx.numpy()))
                print(e)
                continue

            # Compute losses on parameters or segmentation
            gt = gt.cuda(non_blocking=True)
            loss = criterion(output_net, gt)
            losses.update(loss.item(), input.size(0))

            # Print info
            if (i + 1) % args.print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                          'Loss {loss.val:.8f} ({loss.avg:.8f})'.format(
                           i+1, len(loader), loss=losses))

            # Plot curves in two views
            if (i + 1) % args.save_freq == 0:
                save_vis_result_2d('valid', M, gt, output_net, idx, i, input,
                                   args.ipm_h, args.ipm_w, args.save_path)

        if args.evaluate:
            print("===> Average {}-loss on validation set is {:.8}".format(crit_string, 
                                                                           losses.avg))
        return losses.avg


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
    global args
    parser = define_args()
    args = parser.parse_args()

    # set dataset parameters
    args.dataset_name = 'tusimple'
    args.data_dir = ops.join('data', args.dataset_name)
    args.dataset_dir = '/media/yuliangguo/NewVolume2TB/Datasets/TuSimple/labeled'
    args.save_path = ops.join(args.save_path, args.dataset_name)
    args.no_centerline = True
    args.no_3d = True

    # set camera parameters for the test dataset
    args.fix_cam = True
    args.K = np.array([[1000, 0, 640],
                       [0, 1000, 400],
                       [0, 0, 1]])
    args.cam_height = 1.6
    args.pitch = 9

    # set ipm and anchor parameters
    args.top_view_region = np.array([[-20, 100], [20, 100], [-20, 5], [20, 5]])
    args.anchor_y_steps = np.array([5, 20, 40, 60, 80, 100])
    args.num_y_anchor = len(args.anchor_y_steps)

    # seems some system bug only allows 0 nworker
    args.nworkers = 0

    train_net()
