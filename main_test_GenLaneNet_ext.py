"""
Batch test code for Gen-LaneNet with new anchor extension. It predicts 3D lanes from a single image.

Author: Yuliang Guo (33yuliangguo@gmail.com)
Date: March, 2020
"""

import numpy as np
import torch
import torch.optim
import glob
from tqdm import tqdm
from dataloader.Load_Data_3DLane_ext import *
from networks import GeoNet3D_ext, erfnet
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
                    # output1 = F.softmax(output1, dim=1)
                    output1 = output1.softmax(dim=1)
                    output1 = output1 / torch.max(torch.max(output1, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
                    output1 = output1[:, 1:, :, :]
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
                        pred = np.max(output1[j, :, :, :], axis=0)
                        ax4.imshow(pred)
                        x_proj_i = np.max(x_proj[j, :, :, :], axis=0)
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
                    lanelines_pred, centerlines_pred, lanelines_prob, centerlines_prob =\
                        compute_3d_lanes_all_prob(lane_anchors, dataset1.anchor_dim,
                                                  dataset1.anchor_x_steps, args1.anchor_y_steps, pred_hcam[j])
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

    global vis_feat
    vis_feat = False

    global args1, args2
    parser = define_args()
    args1 = parser.parse_args()
    args2 = parser.parse_args()

    # dataset_name: 'standard' / 'rare_subset' / 'illus_chg'
    args1.dataset_name = 'rare_subset'
    args1.dataset_dir = '/media/yuliangguo/DATA1/Datasets/Apollo_Sim_3D_Lane_Release/'
    args2.dataset_name = 'rare_subset'
    args2.dataset_dir = '/media/yuliangguo/DATA1/Datasets/Apollo_Sim_3D_Lane_Release/'

    args1.data_dir = ops.join('data_splits', args1.dataset_name)
    args2.data_dir = ops.join('data_splits', args2.dataset_name)
    args1.save_path = ops.join('data_splits', args1.dataset_name)
    args2.save_path = ops.join('data_splits', args2.dataset_name)

    # define trained Geo model
    num_class = 2
    model_name = 'Gen_LaneNet_ext'
    # use two sets of configurations for different datasets
    sim3d_config(args2)
    args2.save_path = os.path.join(args2.save_path, model_name)

    global evaluator
    # define evaluator
    sim3d_config(args1)
    args1.save_path = os.path.join(args1.save_path, model_name)
    evaluator = eval_3D_lane.LaneEval(args1)
    # define pretrained feat model
    pretrained_feat_model = 'pretrained/erfnet_model_sim3d.tar'

    test_name = 'test'
    vis_folder = test_name + '_vis'
    test_gt_file = ops.join(args1.data_dir, test_name + '.json')
    lane_pred_file = ops.join(args2.save_path, test_name + '_pred_file.json')

    # define the network model
    args1.mod = model_name
    args2.mod = model_name
    args2.y_ref = 5

    """   run the test   """
    # Check GPU availability
    if not args1.no_cuda and not torch.cuda.is_available():
        raise Exception("No gpu available for usage")
    torch.backends.cudnn.benchmark = args1.cudnn

    # dataloader for training and test set, training set is for normalizing
    train_dataset = LaneDataset(args2.dataset_dir, ops.join(args2.data_dir, 'train.json'), args2, data_aug=True)
    train_dataset.normalize_lane_label()

    test_dataset = LaneDataset(args1.dataset_dir, test_gt_file, args1)
    test_dataset.normalize_lane_label()
    test_loader = get_loader(test_dataset, args1)

    global test_set_labels
    test_set_labels = [json.loads(line) for line in open(test_gt_file).readlines()]

    # Define network
    model1 = erfnet.ERFNet(num_class)
    model2 = GeoNet3D_ext.Net(args2, input_dim=num_class - 1, debug=True)
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
    best_test_name = glob.glob(os.path.join(args2.save_path, 'model_best*'))[0]
    if os.path.isfile(best_test_name):
        sys.stdout = Logger(os.path.join(args1.save_path, 'Evaluate.txt'))
        print("=> loading checkpoint '{}'".format(best_test_name))
        checkpoint = torch.load(best_test_name)
        model2.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(best_test_name))
    mkdir_if_missing(os.path.join(args2.save_path, 'example/' + vis_folder))
    eval_stats = deploy(test_loader, test_dataset, train_dataset, model1, model2, vs_saver1, vs_saver2, test_gt_file)

