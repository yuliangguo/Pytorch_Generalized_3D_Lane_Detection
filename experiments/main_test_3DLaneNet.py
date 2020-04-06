"""
Batch test code for 3D-LaneNet. It predicts 3D lanes from a single image.

Author: Yuliang Guo (33yuliangguo@gmail.com)
Date: March, 2020
"""

import numpy as np
import torch
import torch.optim
import glob
from tqdm import tqdm
from dataloader.Load_Data_3DLane import *
from networks import LaneNet3D
from tools.utils import *
from tools import eval_lane_tusimple, eval_3D_lane


def visualize_features(args, vis_folder, im, idx, x1_feat, x2_feat, x3_feat, x4_feat,
                       x1_proj, x2_proj, x3_proj, x4_proj, top_2, top_3, top_4):
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

    x1_feat = x1_feat / np.max(x1_feat)
    x2_feat = x2_feat / np.max(x2_feat)
    x3_feat = x3_feat / np.max(x3_feat)
    x4_feat = x4_feat / np.max(x4_feat)
    x1_proj = x1_proj / np.max(x1_proj)
    x2_proj = x2_proj / np.max(x2_proj)
    x3_proj = x3_proj / np.max(x3_proj)
    x4_proj = x4_proj / np.max(x4_proj)
    top_2 = top_2 / np.max(top_2)
    top_3 = top_3 / np.max(top_3)
    top_4 = top_4 / np.max(top_4)

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

    plt.savefig(args.save_path + '/example/' + vis_folder + '/features_{}'.format(idx))
    plt.close(fig)


def deploy(args, loader, dataset, model, vs_saver, test_gt_file, lane_pred_file, vis=False, vis_feat=False, epoch=0):

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
                    if vis_feat:
                        output_net, pred_hcam, pred_pitch,\
                            x1_feat, x2_feat, x3_feat, x4_feat,\
                            x1_proj, x2_proj, x3_proj, x4_proj, top_2, top_3, top_4 = model(input)
                    else:
                        output_net, pred_hcam, pred_pitch = model(input)
                except RuntimeError as e:
                    print("Batch with idx {} skipped due to singular matrix".format(idx.numpy()))
                    print(e)
                    continue

                gt = gt.data.cpu().numpy()
                output_net = output_net.data.cpu().numpy()
                pred_pitch = pred_pitch.data.cpu().numpy().flatten()
                pred_hcam = pred_hcam.data.cpu().numpy().flatten()

                if vis_feat:
                    # only visualize features for the first sample in a batch
                    im = input.permute(0, 2, 3, 1).data.cpu().numpy()[0]
                    im = im * np.array(args.vgg_std)
                    im = im + np.array(args.vgg_mean)
                    visualize_features(args, args.vis_folder, im, idx[0], x1_feat, x2_feat, x3_feat, x4_feat,
                                       x1_proj, x2_proj, x3_proj, x4_proj, top_2, top_3, top_4)

                # unormalize lane outputs
                num_el = input.size(0)
                for j in range(num_el):
                    unormalize_lane_anchor(output_net[j], dataset)
                    unormalize_lane_anchor(gt[j], dataset)

                if vis:
                    # Plot curves in two views
                    vs_saver.save_result(dataset, args.vis_folder, epoch, i, idx,
                                         input, gt, output_net, pred_pitch, pred_hcam, evaluate=vis)

                # write results and evaluate
                for j in range(num_el):
                    im_id = idx[j]
                    H_g2im, P_g2im, H_crop, H_im2ipm = dataset.transform_mats(idx[j])
                    json_line = test_set_labels[im_id]
                    lane_anchors = output_net[j]
                    # convert to json output format
                    if 'tusimple' in args.dataset_name:
                        h_samples = json_line["h_samples"]
                        lanes_pred = compute_2d_lanes(lane_anchors, h_samples, H_g2im,
                                                      dataset.anchor_x_steps, args.anchor_y_steps, 0, args.org_w, args.prob_th)
                        json_line["lanes"] = lanes_pred
                        json_line["run_time"] = 0
                        json.dump(json_line, jsonFile)
                        jsonFile.write('\n')
                    else:
                        lanelines_pred, centerlines_pred, lanelines_prob, centerlines_prob = \
                            compute_3d_lanes_all_prob(lane_anchors, dataset.anchor_dim,
                                                      dataset.anchor_x_steps, args.anchor_y_steps)
                        json_line["laneLines"] = lanelines_pred
                        json_line["centerLines"] = centerlines_pred
                        json_line["laneLines_prob"] = lanelines_prob
                        json_line["centerLines_prob"] = centerlines_prob
                        json.dump(json_line, jsonFile)
                        jsonFile.write('\n')

        if 'tusimple' in args.dataset_name:
            eval_stats = evaluator.bench_one_submit(lane_pred_file, test_gt_file)
            print("===> Evaluation accuracy on validation set is {:.8}".format(eval_stats[0]))
        else:
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
    args.mod = '3D_LaneNet'  # model name
    test_name = 'test'  # test set name
    vis = False  # choose to visualize lanes per image
    vis_feat = False  # choose to visualize features from selected key layers

    # generate relative paths
    args.data_dir = ops.join('data_splits', args.dataset_name)
    args.save_path = os.path.join(ops.join('data_splits', args.dataset_name), args.mod)
    args.vis_folder = test_name + '_vis'
    if vis or vis_feat:
        mkdir_if_missing(os.path.join(args.save_path, 'example/' + args.vis_folder))
    test_gt_file = ops.join(args.data_dir, test_name + '.json')
    lane_pred_file = ops.join(args.save_path, test_name + '_pred_file.json')

    # load configuration for certain dataset
    global evaluator
    if 'tusimple' in args.dataset_name:
        tusimple_config(args)
        # define evaluator
        evaluator = eval_lane_tusimple.LaneEval
    else:
        sim3d_config(args)
        # define evaluator
        evaluator = eval_3D_lane.LaneEval(args)
    args.prob_th = 0.5

    # Check GPU availability
    if not args.no_cuda and not torch.cuda.is_available():
        raise Exception("No gpu available for usage")
    torch.backends.cudnn.benchmark = args.cudnn

    # Define network
    model = LaneNet3D.Net(args, debug=vis_feat)
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
    with open(ops.join(args.data_dir, 'anchor_std.json')) as f:
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
    eval_stats = deploy(args, test_loader, test_dataset, model, vs_saver, test_gt_file, lane_pred_file, vis, vis_feat)
