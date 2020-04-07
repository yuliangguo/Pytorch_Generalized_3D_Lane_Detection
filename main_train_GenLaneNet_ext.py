"""
The training code for 'Gen-LaneNet' which is a two-stage framework composed of segmentation subnetwork (erfnet)
and 3D lane prediction subnetwork (3D-GeoNet). A new lane anchor is integrated in the 3D-GeoNet. The architecture and
new anchor design are based on:

    "Gen-laneNet: a generalized and scalable approach for 3D lane detection", Y.Guo, etal., arxiv 2020

The training of Gen-LaneNet is based on a pretrained ERFNet saved in ./pretrained folder. The training is on a
synthetic dataset for 3D lane detection proposed in the above paper.

Author: Yuliang Guo (33yuliangguo@gmail.com)
Date: March, 2020
"""

import numpy as np
import torch
import torch.optim
import torch.nn as nn
import glob
import time
import shutil
import torch.nn.functional as F
from tqdm import tqdm
from tensorboardX import SummaryWriter
from dataloader.Load_Data_3DLane_ext import *
from networks import Loss_crit, GeoNet3D_ext, erfnet
from tools.utils import *
from tools import eval_3D_lane


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


def train_net():

    # Check GPU availability
    if not args.no_cuda and not torch.cuda.is_available():
        raise Exception("No gpu available for usage")
    torch.backends.cudnn.benchmark = args.cudnn

    # Define save path
    # save_id = 'Model_{}_crit_{}_opt_{}_lr_{}_batch_{}_{}X{}_pretrain_{}_batchnorm_{}_predcam_{}' \
    #           .format(args.mod,
    #                   crit_string,
    #                   args.optimizer,
    #                   args.learning_rate,
    #                   args.batch_size,
    #                   args.resize_h,
    #                   args.resize_w,
    #                   args.pretrained,
    #                   args.batch_norm,
    #                   args.pred_cam)
    save_id = args.mod
    args.save_path = os.path.join(args.save_path, save_id)
    mkdir_if_missing(args.save_path)
    mkdir_if_missing(os.path.join(args.save_path, 'example/'))
    mkdir_if_missing(os.path.join(args.save_path, 'example/train'))
    mkdir_if_missing(os.path.join(args.save_path, 'example/valid'))

    # dataloader for training and validation set
    val_gt_file = ops.join(args.data_dir, 'test.json')
    train_dataset = LaneDataset(args.dataset_dir, ops.join(args.data_dir, 'train.json'), args, data_aug=True, save_std=True)
    train_dataset.normalize_lane_label()
    train_loader = get_loader(train_dataset, args)
    valid_dataset = LaneDataset(args.dataset_dir, val_gt_file, args)
    # assign std of valid dataset to be consistent with train dataset
    valid_dataset.set_x_off_std(train_dataset._x_off_std)
    if not args.no_3d:
        valid_dataset.set_z_std(train_dataset._z_std)
    valid_dataset.normalize_lane_label()
    valid_loader = get_loader(valid_dataset, args)

    # extract valid set labels for evaluation later
    global valid_set_labels
    valid_set_labels = [json.loads(line) for line in open(val_gt_file).readlines()]

    # Define network
    model1 = erfnet.ERFNet(args.num_class)
    model2 = GeoNet3D_ext.Net(args, input_dim=args.num_class - 1)
    define_init_weights(model2, args.weight_init)

    if not args.no_cuda:
        # Load model on gpu before passing params to optimizer
        model1 = model1.cuda()
        model2 = model2.cuda()

    # load in vgg pretrained weights
    checkpoint = torch.load(args.pretrained_feat_model)
    # args.start_epoch = checkpoint['epoch']
    model1 = load_my_state_dict(model1, checkpoint['state_dict'])
    model1.eval()  # do not back propagate to model1

    # Define optimizer and scheduler
    optimizer = define_optim(args.optimizer, model2.parameters(),
                             args.learning_rate, args.weight_decay)
    scheduler = define_scheduler(optimizer, args)

    # Define loss criteria
    if crit_string == 'loss_gflat_3D':
        criterion = Loss_crit.Laneline_loss_gflat_3D(args.batch_size, train_dataset.num_types,
                                                     train_dataset.anchor_x_steps, train_dataset.anchor_y_steps,
                                                     train_dataset._x_off_std, train_dataset._y_off_std,
                                                     train_dataset._z_std, args.pred_cam, args.no_cuda)
    else:
        criterion = Loss_crit.Laneline_loss_gflat(train_dataset.num_types, args.num_y_steps, args.pred_cam)

    if not args.no_cuda:
        criterion = criterion.cuda()

    # Logging setup
    best_epoch = 0
    lowest_loss = np.inf
    log_file_name = 'log_train_start_0.txt'

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
            model2.load_state_dict(checkpoint['state_dict'])
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
            model2.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(best_file_name))
        mkdir_if_missing(os.path.join(args.save_path, 'example/val_vis'))
        losses_valid, eval_stats = validate(valid_loader, valid_dataset, model1, model2, criterion, vs_saver, val_gt_file)
        return

    # Start training from clean slate
    else:
        # Redirect stdout
        sys.stdout = Logger(os.path.join(args.save_path, log_file_name))

    # INIT MODEL
    print(40*"="+"\nArgs:{}\n".format(args)+40*"=")
    print("Init model: '{}'".format(args.mod))
    print("Number of parameters in model {} is {:.3f}M".format(
        args.mod, sum(tensor.numel() for tensor in model2.parameters())/1e6))

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
        model2.train()

        # compute timing
        end = time.time()

        # Start training loop
        for i, (input, seg_maps, gt, idx, gt_hcam, gt_pitch, aug_mat) in tqdm(enumerate(train_loader)):

            # Time dataloader
            data_time.update(time.time() - end)

            # Put inputs on gpu if possible
            if not args.no_cuda:
                input, gt = input.cuda(non_blocking=True), gt.cuda(non_blocking=True)
                seg_maps = seg_maps.cuda(non_blocking=True)
                gt_hcam = gt_hcam.cuda()
                gt_pitch = gt_pitch.cuda()
            input = input.contiguous().float()

            if not args.fix_cam and not args.pred_cam:
                model2.update_projection(args, gt_hcam, gt_pitch)

            # update transformation for data augmentation (only for training)
            model2.update_projection_for_data_aug(aug_mat)

            # Run model
            optimizer.zero_grad()
            # Inference model
            try:
                output1 = model1(input, no_lane_exist=True)
                with torch.no_grad():
                    # output1 = F.softmax(output1, dim=1)
                    output1 = output1.softmax(dim=1)
                    output1 = output1 / torch.max(torch.max(output1, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
                # pred = output1.data.cpu().numpy()[0, 1:, :, :]
                # pred = np.max(pred, axis=0)
                # cv2.imshow('check probmap', pred)
                # cv2.waitKey()
                output1 = output1[:, 1:, :, :]
                output_net, pred_hcam, pred_pitch = model2(output1)
            except RuntimeError as e:
                print("Batch with idx {} skipped due to inference error".format(idx.numpy()))
                print(e)
                continue

            # Compute losses on
            loss = criterion(output_net, gt, pred_hcam, gt_hcam, pred_pitch, gt_pitch)
            losses.update(loss.item(), input.size(0))

            # Clip gradients (usefull for instabilities or mistakes in ground truth)
            if args.clip_grad_norm != 0:
                nn.utils.clip_grad_norm(model2.parameters(), args.clip_grad_norm)

            # Setup backward pass
            loss.backward()
            optimizer.step()

            # Time trainig iteration
            batch_time.update(time.time() - end)
            end = time.time()

            pred_pitch = pred_pitch.data.cpu().numpy().flatten()
            pred_hcam = pred_hcam.data.cpu().numpy().flatten()
            aug_mat = aug_mat.data.cpu().numpy()
            output_net = output_net.data.cpu().numpy()
            gt = gt.data.cpu().numpy()

            # unormalize lane outputs
            num_el = input.size(0)
            for j in range(num_el):
                unormalize_lane_anchor(output_net[j], train_dataset)
                unormalize_lane_anchor(gt[j], train_dataset)

            # Print info
            if (i + 1) % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.8f} ({loss.avg:.8f})'.format(
                       epoch+1, i+1, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses))

            # Plot curves in two views
            if (i + 1) % args.save_freq == 0:
                vs_saver.save_result_new(train_dataset, 'train', epoch, i, idx,
                                         input, gt, output_net, pred_pitch, pred_hcam, aug_mat)

        losses_valid, eval_stats = validate(valid_loader, valid_dataset, model1, model2, criterion, vs_saver, val_gt_file, epoch)

        print("===> Average {}-loss on training set is {:.8f}".format(crit_string, losses.avg))
        print("===> Average {}-loss on validation set is {:.8f}".format(crit_string, losses_valid))
        print("===> Evaluation laneline F-measure: {:3f}".format(eval_stats[0]))
        print("===> Evaluation laneline Recall: {:3f}".format(eval_stats[1]))
        print("===> Evaluation laneline Precision: {:3f}".format(eval_stats[2]))
        print("===> Evaluation centerline F-measure: {:3f}".format(eval_stats[7]))
        print("===> Evaluation centerline Recall: {:3f}".format(eval_stats[8]))
        print("===> Evaluation centerline Precision: {:3f}".format(eval_stats[9]))

        print("===> Last best {}-loss was {:.8f} in epoch {}".format(crit_string, lowest_loss, best_epoch))

        if not args.no_tb:
            writer.add_scalars('3D-Lane-Loss', {'Training': losses.avg}, epoch)
            writer.add_scalars('3D-Lane-Loss', {'Validation': losses_valid}, epoch)
            writer.add_scalars('Evaluation', {'laneline F-measure': eval_stats[0]}, epoch)
            writer.add_scalars('Evaluation', {'centerline F-measure': eval_stats[7]}, epoch)
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
            'state_dict': model2.state_dict(),
            'loss': lowest_loss,
            'optimizer': optimizer.state_dict()}, to_save, epoch)
    if not args.no_tb:
        writer.close()


def validate(loader, dataset, model1, model2, criterion, vs_saver, val_gt_file, epoch=0):

    # Define container to keep track of metric and loss
    losses = AverageMeter()
    lane_pred_file = ops.join(args.save_path, 'test_pred_file.json')

    # Evaluate model
    model2.eval()

    # Only forward pass, hence no gradients needed
    with torch.no_grad():
        with open(lane_pred_file, 'w') as jsonFile:
            # Start validation loop
            for i, (input, seg_maps, gt, idx, gt_hcam, gt_pitch) in tqdm(enumerate(loader)):
                if not args.no_cuda:
                    input, gt = input.cuda(non_blocking=True), gt.cuda(non_blocking=True)
                    seg_maps = seg_maps.cuda(non_blocking=True)
                    gt_hcam = gt_hcam.cuda()
                    gt_pitch = gt_pitch.cuda()
                input = input.contiguous().float()

                if not args.fix_cam and not args.pred_cam:
                    model2.update_projection(args, gt_hcam, gt_pitch)
                # Inference model
                try:
                    output1 = model1(input, no_lane_exist=True)
                    # output1 = F.softmax(output1, dim=1)
                    output1 = output1.softmax(dim=1)
                    output1 = output1 / torch.max(torch.max(output1, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
                    output1 = output1[:, 1:, :, :]
                    output_net, pred_hcam, pred_pitch = model2(output1)
                except RuntimeError as e:
                    print("Batch with idx {} skipped due to inference error".format(idx.numpy()))
                    print(e)
                    continue

                # Compute losses on parameters or segmentation
                loss = criterion(output_net, gt, pred_hcam, gt_hcam, pred_pitch, gt_pitch)
                losses.update(loss.item(), input.size(0))

                pred_pitch = pred_pitch.data.cpu().numpy().flatten()
                pred_hcam = pred_hcam.data.cpu().numpy().flatten()
                output_net = output_net.data.cpu().numpy()
                gt = gt.data.cpu().numpy()

                # unormalize lane outputs
                num_el = input.size(0)
                for j in range(num_el):
                    unormalize_lane_anchor(output_net[j], dataset)
                    unormalize_lane_anchor(gt[j], dataset)

                # Print info
                if (i + 1) % args.print_freq == 0:
                        print('Test: [{0}/{1}]\t'
                              'Loss {loss.val:.8f} ({loss.avg:.8f})'.format(
                               i+1, len(loader), loss=losses))

                # Plot curves in two views
                if (i + 1) % args.save_freq == 0 or args.evaluate:
                    vs_saver.save_result_new(dataset, 'valid', epoch, i, idx,
                                             input, gt, output_net, pred_pitch, pred_hcam, evaluate=args.evaluate)

                # write results and evaluate
                for j in range(num_el):
                    im_id = idx[j]
                    H_g2im, P_g2im, H_crop, H_im2ipm = dataset.transform_mats(idx[j])
                    json_line = valid_set_labels[im_id]
                    lane_anchors = output_net[j]
                    # convert to json output format
                    # P_g2gflat = np.matmul(np.linalg.inv(H_g2im), P_g2im)
                    lanelines_pred, centerlines_pred, lanelines_prob, centerlines_prob = \
                        compute_3d_lanes_all_prob(lane_anchors, dataset.anchor_dim,
                                                  dataset.anchor_x_steps, args.anchor_y_steps, pred_hcam[j])
                    json_line["laneLines"] = lanelines_pred
                    json_line["centerLines"] = centerlines_pred
                    json_line["laneLines_prob"] = lanelines_prob
                    json_line["centerLines_prob"] = centerlines_prob
                    json.dump(json_line, jsonFile)
                    jsonFile.write('\n')
        eval_stats = evaluator.bench_one_submit(lane_pred_file, val_gt_file)

        if args.evaluate:
            print("===> Average {}-loss on validation set is {:.8}".format(crit_string, losses.avg))
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

    # dataset_name: 'standard' / 'rare_subset' / 'illus_chg'
    args.dataset_name = 'illus_chg'
    args.dataset_dir = '/media/yuliangguo/DATA1/Datasets/Apollo_Sim_3D_Lane_Release/'
    args.data_dir = ops.join('data_splits', args.dataset_name)
    args.save_path = ops.join('data_splits', args.dataset_name)

    # load configuration for certain dataset
    global evaluator
    sim3d_config(args)
    # define evaluator
    evaluator = eval_3D_lane.LaneEval(args)
    args.prob_th = 0.5

    # define the network model
    args.num_class = 2  # 1 background + n lane labels
    args.pretrained_feat_model = 'pretrained/erfnet_model_sim3d.tar'
    args.mod = 'Gen_LaneNet_ext'
    args.y_ref = 5  # new anchor prefer closer range gt assign
    global crit_string
    crit_string = 'loss_gflat'

    # for the case only running evaluation
    args.evaluate = False

    # settings for save and visualize
    args.print_freq = 50
    args.save_freq = 50

    # run the training
    train_net()
