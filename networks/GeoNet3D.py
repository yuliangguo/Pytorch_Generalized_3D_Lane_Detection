"""
3D-GeoNet: predict 3D lanes from segmentation input. Besides preprocess layers, the later layer used the same arch as
the lane prediction head of 3D-LaneNet.

Overall dimension of the output tensor would be: N * W * 3 *(3 * K + 1), where
    K          : number of y samples.
    (2 * K + 1): Each lane includes K attributes for X_g offset + K attributes for Z + 1 lane probability
    3          : Each anchor column include one laneline and two centerlines --> 3
    W          : Number of columns for the output tensor each corresponds to a IPM X_g location
    N          : batch size

Author: Yuliang Guo (33yuliangguo@gmail.com)
Date: March, 2020
"""

import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tools.utils import *


def make_layers(cfg, in_channels=3, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_one_layer(in_channels, out_channels, kernel_size=3, padding=1, stride=1, batch_norm=False):
    conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
    if batch_norm:
        layers = [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
    else:
        layers = [conv2d, nn.ReLU(inplace=True)]
    return layers


# initialize base_grid with different sizes can adapt to different sizes
class ProjectiveGridGenerator(nn.Module):
    def __init__(self, size_ipm, M, no_cuda):
        """

        :param size_ipm: size of ipm tensor NCHW
        :param im_h: height of image tensor
        :param im_w: width of image tensor
        :param M: normalized transformation matrix between image view and IPM
        :param no_cuda:
        """
        super().__init__()
        self.N, self.H, self.W = size_ipm
        # self.im_h = im_h
        # self.im_w = im_w
        linear_points_W = torch.linspace(0, 1 - 1/self.W, self.W)
        linear_points_H = torch.linspace(0, 1 - 1/self.H, self.H)

        # use M only to decide the type not value
        self.base_grid = M.new(self.N, self.H, self.W, 3)
        self.base_grid[:, :, :, 0] = torch.ger(
                torch.ones(self.H), linear_points_W).expand_as(self.base_grid[:, :, :, 0])
        self.base_grid[:, :, :, 1] = torch.ger(
                linear_points_H, torch.ones(self.W)).expand_as(self.base_grid[:, :, :, 1])
        self.base_grid[:, :, :, 2] = 1

        self.base_grid = Variable(self.base_grid)
        if not no_cuda:
            self.base_grid = self.base_grid.cuda()
            # self.im_h = self.im_h.cuda()
            # self.im_w = self.im_w.cuda()

    def forward(self, M):
        # compute the grid mapping based on the input transformation matrix M
        # if base_grid is top-view, M should be ipm-to-img homography transformation, and vice versa
        grid = torch.bmm(self.base_grid.view(self.N, self.H * self.W, 3), M.transpose(1, 2))
        grid = torch.div(grid[:, :, 0:2], grid[:, :, 2:]).reshape((self.N, self.H, self.W, 2))
        #
        """
        output grid to be used for grid_sample. 
            1. grid specifies the sampling pixel locations normalized by the input spatial dimensions.
            2. pixel locations need to be converted to the range (-1, 1)
        """
        grid = (grid - 0.5) * 2
        return grid


#  Lane Prediction Head: through a series of convolutions with no padding in the y dimension, the feature maps are
#  reduced in height, and finally the prediction layer size is N × 1 × 3 ·(2 · K + 1)
class LanePredictionHead(nn.Module):
    def __init__(self, num_lane_type, anchor_dim, batch_norm=False):
        super(LanePredictionHead, self).__init__()
        self.num_lane_type = num_lane_type
        self.anchor_dim = anchor_dim
        layers = []
        layers += make_one_layer(64, 64, kernel_size=3, padding=(0, 1), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=3, padding=(0, 1), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=3, padding=(0, 1), batch_norm=batch_norm)

        layers += make_one_layer(64, 64, kernel_size=5, padding=(0, 2), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=5, padding=(0, 2), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=5, padding=(0, 2), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=5, padding=(0, 2), batch_norm=batch_norm)
        self.features = nn.Sequential(*layers)

        # x suppose to be N X 64 X 4 X ipm_w/8, need to be reshaped to N X 256 X ipm_w/8 X 1
        # TODO: use large kernel_size in x or fc layer to estimate z with global parallelism
        dim_rt_layers = []
        dim_rt_layers += make_one_layer(256, 128, kernel_size=(5, 1), padding=(2, 0), batch_norm=batch_norm)
        dim_rt_layers += [nn.Conv2d(128, self.num_lane_type*self.anchor_dim, kernel_size=(5, 1), padding=(2, 0))]
        self.dim_rt = nn.Sequential(*dim_rt_layers)

    def forward(self, x):
        x = self.features(x)
        # x suppose to be N X 64 X 4 X ipm_w/8, reshape to N X 256 X ipm_w/8 X 1
        sizes = x.shape
        x = x.reshape(sizes[0], sizes[1]*sizes[2], sizes[3], 1)
        x = self.dim_rt(x)
        x = x.squeeze(-1).transpose(1, 2)
        # apply sigmoid to the probability terms to make it in (0, 1)
        for i in range(self.num_lane_type):
            x[:, :, (i+1)*self.anchor_dim-1] = torch.sigmoid(x[:, :, (i+1)*self.anchor_dim-1])
        return x


# The 3D-lanenet composed of image encode, top view pathway, and lane predication head
class Net(nn.Module):
    def __init__(self, args, input_dim=1, debug=False):
        super().__init__()

        self.no_cuda = args.no_cuda
        self.debug = debug
        self.pred_cam = args.pred_cam
        self.batch_size = args.batch_size
        if args.no_centerline:
            self.num_lane_type = 1
        else:
            self.num_lane_type = 3

        if args.no_3d:
            self.anchor_dim = args.num_y_steps + 1
        else:
            self.anchor_dim = 2*args.num_y_steps + 1

        # define required transformation matrices
        # define homographic transformation between image and ipm
        org_img_size = np.array([args.org_h, args.org_w])
        resize_img_size = np.array([args.resize_h, args.resize_w])
        cam_pitch = np.pi / 180 * args.pitch

        self.cam_height = torch.tensor(args.cam_height).unsqueeze_(0).expand([self.batch_size, 1]).type(torch.FloatTensor)
        self.cam_pitch = torch.tensor(cam_pitch).unsqueeze_(0).expand([self.batch_size, 1]).type(torch.FloatTensor)
        self.cam_height_default = torch.tensor(args.cam_height).unsqueeze_(0).expand(self.batch_size).type(torch.FloatTensor)
        self.cam_pitch_default = torch.tensor(cam_pitch).unsqueeze_(0).expand(self.batch_size).type(torch.FloatTensor)

        # image scale matrix
        self.S_im = torch.from_numpy(np.array([[args.resize_w,              0, 0],
                                               [            0,  args.resize_h, 0],
                                               [            0,              0, 1]], dtype=np.float32))
        self.S_im_inv = torch.from_numpy(np.array([[1/np.float(args.resize_w),                         0, 0],
                                                   [                        0, 1/np.float(args.resize_h), 0],
                                                   [                        0,                         0, 1]], dtype=np.float32))
        self.S_im_inv_batch = self.S_im_inv.unsqueeze_(0).expand([self.batch_size, 3, 3]).type(torch.FloatTensor)

        # image transform matrix
        H_c = homography_crop_resize(org_img_size, args.crop_y, resize_img_size)
        self.H_c = torch.from_numpy(H_c).unsqueeze_(0).expand([self.batch_size, 3, 3]).type(torch.FloatTensor)

        # camera intrinsic matrix
        self.K = torch.from_numpy(args.K).unsqueeze_(0).expand([self.batch_size, 3, 3]).type(torch.FloatTensor)

        # homograph ground to camera
        # H_g2cam = np.array([[1,                             0,               0],
        #                     [0, np.cos(np.pi / 2 + cam_pitch), args.cam_height],
        #                     [0, np.sin(np.pi / 2 + cam_pitch),               0]])
        H_g2cam = np.array([[1,                             0,               0],
                            [0, np.sin(-cam_pitch), args.cam_height],
                            [0, np.cos(-cam_pitch),               0]])
        self.H_g2cam = torch.from_numpy(H_g2cam).unsqueeze_(0).expand([self.batch_size, 3, 3]).type(torch.FloatTensor)

        # transform from ipm normalized coordinates to ground coordinates
        H_ipmnorm2g = homography_ipmnorm2g(args.top_view_region)
        self.H_ipmnorm2g = torch.from_numpy(H_ipmnorm2g).unsqueeze_(0).expand([self.batch_size, 3, 3]).type(torch.FloatTensor)

        # compute the tranformation from ipm norm coords to image norm coords
        M_ipm2im = torch.bmm(self.H_g2cam, self.H_ipmnorm2g)
        M_ipm2im = torch.bmm(self.K, M_ipm2im)
        M_ipm2im = torch.bmm(self.H_c, M_ipm2im)
        M_ipm2im = torch.bmm(self.S_im_inv_batch, M_ipm2im)
        M_ipm2im = torch.div(M_ipm2im,  M_ipm2im[:, 2, 2].reshape([self.batch_size, 1, 1]).expand([self.batch_size, 3, 3]))
        self.M_inv = M_ipm2im

        if not self.no_cuda:
            self.M_inv = self.M_inv.cuda()
            self.S_im = self.S_im.cuda()
            self.S_im_inv = self.S_im_inv.cuda()
            self.S_im_inv_batch = self.S_im_inv_batch.cuda()
            self.H_c = self.H_c.cuda()
            self.K = self.K.cuda()
            self.H_g2cam = self.H_g2cam.cuda()
            self.H_ipmnorm2g = self.H_ipmnorm2g.cuda()
            self.cam_height_default = self.cam_height_default.cuda()
            self.cam_pitch_default = self.cam_pitch_default.cuda()

        # Define network
        # the grid considers both src and dst grid normalized
        size_top = torch.Size([self.batch_size, np.int(args.ipm_h), np.int(args.ipm_w)])
        self.project_layer = ProjectiveGridGenerator(size_top, self.M_inv, args.no_cuda)

        # Conv layers to convert original resolution binary map to target resolution with high-dimension
        self.encoder = make_layers([8, 'M', 16, 'M', 32, 'M', 64], input_dim, batch_norm=args.batch_norm)

        self.lane_out = LanePredictionHead(self.num_lane_type, self.anchor_dim, args.batch_norm)

    def forward(self, input):
        # compute image features from multiple layers

        cam_height = self.cam_height
        cam_pitch = self.cam_pitch

        # spatial transfer image features to IPM features
        grid = self.project_layer(self.M_inv)
        x_proj = F.grid_sample(input, grid)

        # conv layers to convert original resolution binary map to target resolution with high-dimension
        x_feat = self.encoder(x_proj)

        # convert top-view features to anchor output
        out = self.lane_out(x_feat)

        if self.debug:
            return out, cam_height, cam_pitch, x_proj, x_feat

        return out, cam_height, cam_pitch

    def update_projection(self, args, cam_height, cam_pitch):
        """
            Update transformation matrix based on ground-truth cam_height and cam_pitch
            This function is "Mutually Exclusive" to the updates of M_inv from network prediction
        :param args:
        :param cam_height:
        :param cam_pitch:
        :return:
        """
        for i in range(self.batch_size):
            M, M_inv = homography_im2ipm_norm(args.top_view_region, np.array([args.org_h, args.org_w]),
                                              args.crop_y, np.array([args.resize_h, args.resize_w]),
                                              cam_pitch[i].data.cpu().numpy(), cam_height[i].data.cpu().numpy(), args.K)
            self.M_inv[i] = torch.from_numpy(M_inv).type(torch.FloatTensor)
        self.cam_height = cam_height
        self.cam_pitch = cam_pitch

    def update_projection_for_data_aug(self, aug_mats):
        """
            update transformation matrix when data augmentation have been applied, and the image augmentation matrix are provided
            Need to consider both the cases of 1. when using ground-truth cam_height, cam_pitch, update M_inv
                                               2. when cam_height, cam_pitch are online estimated, update H_c for later use
        """
        if not self.no_cuda:
            aug_mats = aug_mats.cuda()

        for i in range(aug_mats.shape[0]):
            # update H_c directly
            self.H_c[i] = torch.matmul(aug_mats[i], self.H_c[i])
            # augmentation need to be applied in unnormalized image coords for M_inv
            aug_mats[i] = torch.matmul(torch.matmul(self.S_im_inv, aug_mats[i]), self.S_im)
            self.M_inv[i] = torch.matmul(aug_mats[i], self.M_inv[i])


# unit test
if __name__ == '__main__':
    import os
    from PIL import Image
    from torchvision import transforms
    import torchvision.transforms.functional as F2
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    global args
    parser = define_args()
    args = parser.parse_args()

    # args.dataset_name = 'tusimple'
    # tusimple_config(args)
    args.dataset_name = 'sim3d'
    sim3d_config(args)
    args.pred_cam = True
    args.batch_size = 1

    # construct model
    model = Net(args)
    print(model)

    # initialize model weights
    define_init_weights(model, args.weight_init)

    # load in vgg pretrained weights on ImageNet
    if args.pretrained:
        model.load_pretrained_vgg(args.batch_norm)
        print('vgg weights pretrained on ImageNet loaded!')
    model = model.cuda()

    # prepare input
    image = torch.randn(1, 1, args.resize_h, args.resize_w)
    image = image.cuda()

    # test update of camera height and pitch
    cam_height = torch.tensor(1.65).unsqueeze_(0).expand([args.batch_size, 1]).type(torch.FloatTensor)
    cam_pitch = torch.tensor(0.1).unsqueeze_(0).expand([args.batch_size, 1]).type(torch.FloatTensor)
    # model.update_projection(args, cam_height, cam_pitch)

    # inference the model
    output_net, pred_height, pred_pitch = model(image)

    print(output_net.shape)
    print(pred_height)
    print(pred_pitch)
