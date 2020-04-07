"""
A demo for Gen-LaneNet with new anchor extension. It predicts 3D lanes from a single image.

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
from tools.visualize_pred import lane_visualizer


def unormalize_lane_anchor(anchor, num_y_steps, anchor_dim, x_off_std, z_std, num_types=3):
    for i in range(num_types):
        anchor[:, i*anchor_dim:i*anchor_dim + num_y_steps] = \
            np.multiply(anchor[:, i*anchor_dim: i*anchor_dim + num_y_steps], x_off_std)
        anchor[:, i*anchor_dim + num_y_steps: i*anchor_dim + 2*num_y_steps] = \
            np.multiply(anchor[:, i*anchor_dim + num_y_steps: i*anchor_dim + 2*num_y_steps], z_std)


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


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = define_args()
    args = parser.parse_args()

    # manual settings
    image_file = './example/0000101.jpg'
    cam_file = './example/0000101_cam.json'
    args.mod = 'Gen_LaneNet_ext'  # model name
    pretrained_feat_model = 'pretrained/erfnet_model_sim3d.tar'
    trained_geo_model = 'pretrained/gen_lanenet_geo_model.tar'
    anchor_std_file = 'pretrained/geo_anchor_std.json'

    # load configuration for the model
    sim3d_config(args)
    args.y_ref = 5
    args.batch_size = 1
    anchor_y_steps = args.anchor_y_steps
    num_y_steps = len(anchor_y_steps)
    anchor_dim = 3 * num_y_steps + 1
    x_min = args.top_view_region[0, 0]
    x_max = args.top_view_region[1, 0]
    anchor_x_steps = np.linspace(x_min, x_max, np.int(args.ipm_w / 8), endpoint=True)

    # Check GPU availability
    if not args.no_cuda and not torch.cuda.is_available():
        raise Exception("No gpu available for usage")
    torch.backends.cudnn.benchmark = args.cudnn

    # Define network
    model_seg = erfnet.ERFNet(2)  # 2-class model
    model_geo = GeoNet3D_ext.Net(args)
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
    if os.path.isfile(trained_geo_model):
        print("=> loading checkpoint '{}'".format(trained_geo_model))
        checkpoint = torch.load(trained_geo_model)
        model_geo.load_state_dict(checkpoint['state_dict'])
        model_geo.eval()
    else:
        print("=> no checkpoint found at '{}'".format(trained_geo_model))

    # load anchor std saved from training
    with open(anchor_std_file) as f:
        anchor_std = json.load(f)
    x_off_std = np.array(anchor_std['x_off_std'])
    z_std = np.array(anchor_std['z_std'])

    #  load image
    with open(image_file, 'rb') as f:
        image = (Image.open(f).convert('RGB'))
    # image preprocess
    w, h = image.size
    image = F.crop(image, args.crop_y, 0, args.org_h - args.crop_y, w)
    image = F.resize(image, size=(args.resize_h, args.resize_w), interpolation=Image.BILINEAR)
    image = transforms.ToTensor()(image).float()
    image = transforms.Normalize(args.vgg_mean, args.vgg_std)(image)
    image.unsqueeze_(0)
    image = torch.cat(list(torch.split(image, 1, dim=0)) * args.batch_size)

    if not args.no_cuda:
        image = image.cuda()
    # image = image.contiguous()
    # image = torch.autograd.Variable(image)

    # update camera setting os the model
    with open(cam_file) as f:
        cam_params = json.load(f)
    gt_pitch = torch.tensor([cam_params['cameraPitch']], dtype=torch.float32)
    gt_hcam = torch.tensor([cam_params['cameraHeight']], dtype=torch.float32)
    model_geo.update_projection(args, gt_hcam, gt_pitch)

    with torch.no_grad():
        # deploy model
        try:
            output_seg = model_seg(image, no_lane_exist=True)
            # output1 = F.softmax(output1, dim=1)
            output_seg = output_seg.softmax(dim=1)
            output_seg = output_seg / torch.max(torch.max(output_seg, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
            output_seg = output_seg[:, 1:, :, :]
            output_geo, pred_hcam, pred_pitch = model_geo(output_seg)
        except RuntimeError as e:
            print(e)

    output_geo = output_geo[0].data.cpu().numpy()

    # unormalize lane outputs
    unormalize_lane_anchor(output_geo, num_y_steps, anchor_dim, x_off_std, z_std, num_types=3)

    # compute 3D lanes from network output, geometric transformation is involved
    lanelines_pred, centerlines_pred, lanelines_prob, centerlines_prob = \
        compute_3d_lanes_all_prob(output_geo, anchor_dim, anchor_x_steps, anchor_y_steps, cam_params['cameraHeight'])

    # visualize predicted lanes
    # args.top_view_region = np.array([[-10, 80], [10, 80], [-10, 3], [10, 3]])
    vs = lane_visualizer(args)
    vs.dataset_dir = './'

    fig = plt.figure()
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233, projection='3d')
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236, projection='3d')

    # draw lanes
    vs.visualize_lanes(lanelines_pred, image_file, cam_params['cameraHeight'], cam_params['cameraPitch'], ax1, ax2, ax3)
    vs.visualize_lanes(centerlines_pred, image_file, cam_params['cameraHeight'], cam_params['cameraPitch'], ax4, ax5, ax6)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    bottom, top = ax3.get_zlim()
    left, right = ax3.get_xlim()
    ax3.set_zlim(min(bottom, -0.1), max(top, 0.1))
    ax3.set_xlim(left, right)
    ax3.set_ylim(0, 80)
    ax3.locator_params(nbins=5, axis='x')
    ax3.locator_params(nbins=5, axis='z')
    ax3.tick_params(pad=18)

    ax4.set_xticks([])
    ax4.set_yticks([])
    ax5.set_xticks([])
    ax5.set_yticks([])

    bottom, top = ax6.get_zlim()
    left, right = ax6.get_xlim()
    ax6.set_zlim(min(bottom, -0.1), max(top, 0.1))
    ax6.set_xlim(left, right)
    ax6.set_ylim(0, 80)
    ax6.locator_params(nbins=5, axis='x')
    ax6.locator_params(nbins=5, axis='z')
    ax6.tick_params(pad=18)

    fig.subplots_adjust(wspace=0, hspace=0.01)
    fig.savefig('test.png')
    plt.close(fig)


