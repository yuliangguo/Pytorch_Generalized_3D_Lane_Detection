"""
Batch test code for Gen-LaneNet. It predicts 3D lanes from a single image.

Author: Yuliang Guo (33yuliangguo@gmail.com)
Date: March, 2020
"""

import numpy as np
import torch
import torch.optim
import glob
from tqdm import tqdm
from dataloader.Load_Data_3DLane import *
from networks import GeoNet3D, erfnet
from tools.utils import *
from tools import eval_3D_lane


# def visualize_features(input, output_seg, output_geo, gt, idx, gt_hcam, pred_hcam, x_proj, x_feat):
#     input = input.permute(0, 2, 3, 1).data.cpu().numpy()
#     img1 = input[idx]
#     img1 = img1 * np.array(args.vgg_std)
#     img1 = img1 + np.array(args.vgg_mean)
#     img1 = np.clip(img1, 0, 1)
#
#     im_ipm1 = cv2.warpPerspective(img1, H_im2ipm, (args.ipm_w, args.ipm_h))
#     im_ipm1 = np.clip(im_ipm1, 0, 1)
#
#     # visualize on image
#     M1 = np.matmul(H_crop, H_g2im)
#     M2 = np.matmul(H_crop, P_g2im)
#     if args.no_3d:
#         img1 = vs_saver1.draw_on_img(img1, gt[idx], M1, 'laneline', color=[0, 0, 1])
#     else:
#         img1 = vs_saver1.draw_on_img(img1, gt[idx], M2, 'laneline', color=[0, 0, 1])
#     img1 = vs_saver2.draw_on_img(img1, output_geo[idx], M2, 'laneline', color=[1, 0, 0])
#
#     # visualize on ipm
#     im_ipm1 = vs_saver1.draw_on_ipm(im_ipm1, gt[idx], 'laneline', color=[0, 0, 1])
#     im_ipm1 = vs_saver2.draw_on_ipm(im_ipm1, output_geo[idx], 'laneline', color=[1, 0, 0])
#
#     fig = plt.figure()
#     ax1 = fig.add_subplot(231)
#     ax2 = fig.add_subplot(232)
#     ax3 = fig.add_subplot(233, projection='3d')
#     ax4 = fig.add_subplot(234)
#     ax5 = fig.add_subplot(235)
#     ax6 = fig.add_subplot(236)
#     ax1.imshow(img1)
#     ax2.imshow(im_ipm1)
#     vs_saver1.draw_3d_curves(ax3, gt[idx], gt_hcam[idx], 'laneline', [0, 0, 1])
#     vs_saver2.draw_3d_curves(ax3, output_geo[idx], pred_hcam[idx], 'laneline', [1, 0, 0])
#     ax3.set_xlabel('x axis')
#     ax3.set_ylabel('y axis')
#     ax3.set_zlabel('z axis')
#     bottom, top = ax3.get_zlim()
#     ax3.set_zlim(min(bottom, -1), max(top, 1))
#     ax3.set_xlim(-20, 20)
#     ax3.set_ylim(0, 100)
#     # visualize features
#     pred = np.max(output_seg[idx, :, :, :], axis=0)
#     ax4.imshow(pred)
#     x_proj_i = np.max(x_proj[idx, :, :, :], axis=0)
#     ax5.imshow(x_proj_i)
#     x_feat_i = x_feat[idx, :, :, :]
#     x_feat_i = np.sum(np.square(x_feat_i), axis=0)
#     x_feat_i = x_feat_i / np.max(x_feat_i)
#     ax6.imshow(x_feat_i)
#
#     fig.savefig(vs_saver2.save_path + '/example/' + vs_saver2.vis_folder + '/infer_{}'.format(idx[idx]))
#     plt.clf()
#     plt.close(fig)


def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
    own_state = model.state_dict()
    ckpt_name = []
    cnt = 0
    for name, param in state_dict.items():
        if name[7:] not in list(own_state.keys()) or 'output_conv' in name:
            ckpt_name.append(name)
            # continue
        own_state[name[7:]].copy_(param)
        cnt += 1
    print('#reused param: {}'.format(cnt))
    return model


def deploy(args, loader, dataset, model_seg, model_geo, vs_saver, test_gt_file, vis=False, epoch=0):

    # model deploy mode
    model_geo.eval()

    # read ground-truth lanes for later evaluation
    test_set_labels = [json.loads(line) for line in open(test_gt_file).readlines()]

    # Only forward pass, hence no gradients needed
    with torch.no_grad():
        with open(lane_pred_file, 'w') as jsonFile:
            # Start validation loop
            for i, (input, _, gt, idx, gt_hcam, gt_pitch) in tqdm(enumerate(loader)):
                if not args.no_cuda:
                    input, gt = input.cuda(non_blocking=True), gt.cuda(non_blocking=True)
                    input = input.float()
                input = input.contiguous()
                input = torch.autograd.Variable(input)

                # if not args.fix_cam and not args.pred_cam:
                # ATTENTION: here requires to update with test dataset args
                model_geo.update_projection(args, gt_hcam, gt_pitch)

                # Evaluate model
                try:
                    output_seg = model_seg(input, no_lane_exist=True)
                    # output1 = F.softmax(output1, dim=1)
                    output_seg = output_seg.softmax(dim=1)
                    output_seg = output_seg / torch.max(torch.max(output_seg, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
                    output_seg = output_seg[:, 1:, :, :]
                    output_geo, pred_hcam, pred_pitch = model_geo(output_seg)
                except RuntimeError as e:
                    print("Batch with idx {} skipped due to singular matrix".format(idx.numpy()))
                    print(e)
                    continue

                gt = gt.data.cpu().numpy()
                output_seg = output_seg.data.cpu().numpy()
                output_geo = output_geo.data.cpu().numpy()
                # x_proj = x_proj.data.cpu().numpy()
                # x_feat = x_feat.data.cpu().numpy()
                pred_pitch = pred_pitch.data.cpu().numpy().flatten()
                pred_hcam = pred_hcam.data.cpu().numpy().flatten()
                # gt_hcam = gt_hcam.data.cpu().numpy().flatten()

                # unormalize lane outputs
                num_el = input.size(0)
                for j in range(num_el):
                    unormalize_lane_anchor(gt[j], dataset)
                    unormalize_lane_anchor(output_geo[j], dataset)

                if vis:
                    # Plot curves in two views
                    vs_saver.save_result(dataset, args.vis_folder, epoch, i, idx,
                                         input, gt, output_geo, pred_pitch, pred_hcam, evaluate=vis)

                # visualize and write results
                for j in range(num_el):
                    im_id = idx[j]
                    H_g2im, P_g2im, H_crop, H_im2ipm = dataset.transform_mats(idx[j])
                    """
                        save results in test dataset format
                    """
                    json_line = test_set_labels[im_id]
                    lane_anchors = output_geo[j]
                    # convert to json output format
                    lanelines_pred, centerlines_pred, lanelines_prob, centerlines_prob =\
                        compute_3d_lanes_all_prob(lane_anchors, dataset.anchor_dim,
                                                  dataset.anchor_x_steps, args.anchor_y_steps)
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
    args.mod = 'Gen_LaneNet'  # model name
    test_name = 'test'  # test set name
    num_class = 2
    pretrained_feat_model = 'pretrained/erfnet_model_sim3d.tar'
    vis = True  # choose to save visualization result

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
    # define evaluator
    evaluator = eval_3D_lane.LaneEval(args)
    args.prob_th = 0.5

    # Check GPU availability
    if not args.no_cuda and not torch.cuda.is_available():
        raise Exception("No gpu available for usage")
    torch.backends.cudnn.benchmark = args.cudnn

    # Define network
    model_seg = erfnet.ERFNet(num_class)
    model_geo = GeoNet3D.Net(args, input_dim=num_class - 1)
    define_init_weights(model_geo, args.weight_init)

    if not args.no_cuda:
        # Load model on gpu before passing params to optimizer
        model_seg = model_seg.cuda()
        model_geo = model_geo.cuda()

    # load segmentation model
    checkpoint = torch.load(pretrained_feat_model)
    model_seg = load_my_state_dict(model_seg, checkpoint['state_dict'])
    model_seg.eval()  # do not back propagate to model1

    # load geometry model
    best_test_name = glob.glob(os.path.join(args.save_path, 'model_best*'))[0]
    if os.path.isfile(best_test_name):
        sys.stdout = Logger(os.path.join(args.save_path, 'Evaluate.txt'))
        print("=> loading checkpoint '{}'".format(best_test_name))
        checkpoint = torch.load(best_test_name)
        model_geo.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(best_test_name))

    # Data loader
    test_dataset = LaneDataset(args.dataset_dir, test_gt_file, args)
    # assign std of valid dataset to be consistent with train dataset
    with open(ops.join(args.data_dir, 'anchor_std.json')) as f:
        anchor_std = json.load(f)
    test_dataset.set_x_off_std(anchor_std['x_off_std'])
    if not args.no_3d:
        test_dataset.set_z_std(anchor_std['z_std'])
    test_dataset.normalize_lane_label()
    test_loader = get_loader(test_dataset, args)

    # initialize visual saver
    vs_saver = Visualizer(args, args.vis_folder)

    mkdir_if_missing(os.path.join(args.save_path, 'example/' + args.vis_folder))
    eval_stats = deploy(args, test_loader, test_dataset, model_seg, model_geo, vs_saver, test_gt_file, vis)
