"""
3D-LanNet: predict 3D lanes from a single image.
This arch is an unofficial implementation of:
    "3d-lanenet: end-to-end 3d multiple lane detection", N. Garnet, etal., ICCV 2019"

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
import torchvision.models as models
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


class VggEncoder(nn.Module):

    def __init__(self, batch_norm=False, init_weights=True):
        super(VggEncoder, self).__init__()
        if batch_norm:
            model_org = models.vgg16_bn()
            output_layers = [12, 22, 32, 42]
        else:
            model_org = models.vgg16()
            output_layers = [8, 15, 22, 29]
        self.features1 = nn.Sequential(
                    *list(model_org.features.children())[:output_layers[0]+1])
        self.features2 = nn.Sequential(
                    *list(model_org.features.children())[output_layers[0]+1:output_layers[1]+1])
        self.features3 = nn.Sequential(
                    *list(model_org.features.children())[output_layers[1]+1:output_layers[2]+1])
        self.features4 = nn.Sequential(
                    *list(model_org.features.children())[output_layers[2]+1:output_layers[3]+1])

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x1)
        x3 = self.features3(x2)
        x4 = self.features4(x3)
        return x1, x2, x3, x4

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # if m.bias is not None:
                #     nn.init.constant_(m.bias, 0)
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# Road plane prediction: estimate camera height and pitch angle
class RoadPlanePredHead(nn.Module):
    def __init__(self, im_h, im_w, batch_norm=False, init_weights=True):
        super().__init__()
        self.im_h = im_h
        self.im_w = im_w
        self.features1 = make_layers(['M', 256, 256, 256], 512, batch_norm)
        self.features2 = make_layers(['M', 128, 128, 128], 256, batch_norm)
        self.features3 = make_layers(['M', 64, 64, 64], 128, batch_norm)
        # fc layer
        self.fc = nn.Sequential(
            nn.Linear(64 * int(self.im_h/128) * int(self.im_w/128), 64),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(64, 2))

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x1)
        x3 = self.features3(x2)
        x3 = x3.reshape([-1, 64 * int(self.im_h/128) * int(self.im_w/128)])
        out = self.fc(x3)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # if m.bias is not None:
                #     nn.init.constant_(m.bias, 0)
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


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


# Sub-network corresponding to the top view pathway
class TopViewPathway(nn.Module):
    def __init__(self, batch_norm=False, init_weights=True):
        super(TopViewPathway, self).__init__()
        self.features1 = make_layers(['M', 128, 128, 128], 128, batch_norm)
        self.features2 = make_layers(['M', 256, 256, 256], 256, batch_norm)
        self.features3 = make_layers(['M', 256, 256, 256], 512, batch_norm)

        if init_weights:
            self._initialize_weights()

    def forward(self, a, b, c, d):
        x = self.features1(a)
        feat_1 = x
        x = torch.cat((x, b), 1)
        x = self.features2(x)
        feat_2 = x
        x = torch.cat((x, c), 1)
        x = self.features3(x)
        feat_3 = x
        x = torch.cat((x, d), 1)
        return x, feat_1, feat_2, feat_3

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # if m.bias is not None:
                #     nn.init.constant_(m.bias, 0)
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


#  Lane Prediction Head: through a series of convolutions with no padding in the y dimension, the feature maps are
#  reduced in height, and finally the prediction layer size is N × 1 × 3 ·(2 · K + 1)
class LanePredictionHead(nn.Module):
    def __init__(self, num_lane_type, anchor_dim, batch_norm=False):
        super(LanePredictionHead, self).__init__()
        self.num_lane_type = num_lane_type
        self.anchor_dim = anchor_dim
        layers = []
        layers += make_one_layer(512, 64, kernel_size=3, padding=(0, 1), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=3, padding=(0, 1), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=3, padding=(0, 1), batch_norm=batch_norm)

        layers += make_one_layer(64, 64, kernel_size=5, padding=(0, 2), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=5, padding=(0, 2), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=5, padding=(0, 2), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=5, padding=(0, 2), batch_norm=batch_norm)
        self.features = nn.Sequential(*layers)

        # x suppose to be N X 64 X 4 X ipm_w/8, need to be reshaped to N X 256 X ipm_w/8 X 1
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
    def __init__(self, args, debug=False):
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

        # compute the transformation from ipm norm coords to image norm coords
        M_ipm2im = torch.bmm(self.H_g2cam, self.H_ipmnorm2g)
        M_ipm2im = torch.bmm(self.K, M_ipm2im)
        M_ipm2im = torch.bmm(self.H_c, M_ipm2im)
        M_ipm2im = torch.bmm(self.S_im_inv_batch, M_ipm2im)
        M_ipm2im = torch.div(M_ipm2im,  M_ipm2im[:, 2, 2].reshape([self.batch_size, 1, 1]).expand([self.batch_size, 3, 3]))
        self.M_inv = M_ipm2im

        # M, M_inv = homography_im2ipm_norm(args.top_view_region, org_img_size,
        #                                   args.crop_y, resize_img_size, cam_pitch, args.cam_height, args.K)
        # # M = torch.from_numpy(M).unsqueeze_(0).expand([self.batch_size, 3, 3]).type(torch.FloatTensor)
        # M_inv = torch.from_numpy(M_inv).unsqueeze_(0).expand([self.batch_size, 3, 3]).type(torch.FloatTensor)
        #
        # self.M_inv = M_inv  # M_inv is the homography ipm2im in normalized coordinates

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
        self.im_encoder = VggEncoder(batch_norm=args.batch_norm)

        if self.pred_cam:
            self.road_plane_pred_head = RoadPlanePredHead(args.resize_h, args.resize_w, batch_norm=False)

        # the grid considers both src and dst grid normalized
        # resize_img_size = torch.from_numpy(resize_img_size).type(torch.FloatTensor)
        size_top1 = torch.Size([self.batch_size, args.ipm_h, args.ipm_w])
        self.project_layer1 = ProjectiveGridGenerator(size_top1, self.M_inv, args.no_cuda)
        size_top2 = torch.Size([self.batch_size, np.int(args.ipm_h / 2), np.int(args.ipm_w / 2)])
        self.project_layer2 = ProjectiveGridGenerator(size_top2, self.M_inv, args.no_cuda)
        size_top3 = torch.Size([self.batch_size, np.int(args.ipm_h / 4), np.int(args.ipm_w / 4)])
        self.project_layer3 = ProjectiveGridGenerator(size_top3, self.M_inv, args.no_cuda)
        size_top4 = torch.Size([self.batch_size, np.int(args.ipm_h / 8), np.int(args.ipm_w / 8)])
        self.project_layer4 = ProjectiveGridGenerator(size_top4, self.M_inv, args.no_cuda)

        self.dim_rt1 = nn.Sequential(*make_one_layer(256, 128, kernel_size=1, padding=0, batch_norm=args.batch_norm))
        self.dim_rt2 = nn.Sequential(*make_one_layer(512, 256, kernel_size=1, padding=0, batch_norm=args.batch_norm))
        self.dim_rt3 = nn.Sequential(*make_one_layer(512, 256, kernel_size=1, padding=0, batch_norm=args.batch_norm))

        self.top_pathway = TopViewPathway(args.batch_norm)
        self.lane_out = LanePredictionHead(self.num_lane_type, self.anchor_dim, args.batch_norm)

    def forward(self, input):
        # compute image features from multiple layers
        x1, x2, x3, x4 = self.im_encoder(input)

        if self.pred_cam:
            pred_cam = self.road_plane_pred_head(x4)
            cam_height = self.cam_height_default + pred_cam[:, 0]
            cam_pitch = self.cam_pitch_default + pred_cam[:, 1]
            # compute projection matrix based on predicted camera height and pitch
            # ATTENTION: need to implement in tensor format, how to prevent back-propagation?
            with torch.no_grad():
                self.H_g2cam[:, 1, 1] = torch.sin(-cam_pitch)
                self.H_g2cam[:, 2, 1] = torch.cos(-cam_pitch)
                self.H_g2cam[:, 1, 2] = cam_height
                M_ipm2im = torch.bmm(self.H_g2cam, self.H_ipmnorm2g)
                M_ipm2im = torch.bmm(self.K, M_ipm2im)
                M_ipm2im = torch.bmm(self.H_c, M_ipm2im)
                M_ipm2im = torch.bmm(self.S_im_inv_batch, M_ipm2im)
                M_ipm2im = torch.div(M_ipm2im,
                                     M_ipm2im[:, 2, 2].reshape([self.batch_size, 1, 1]).expand([self.batch_size, 3, 3]))
                self.M_inv = M_ipm2im
        else:
            cam_height = self.cam_height
            cam_pitch = self.cam_pitch

        # spatial transfer image features to IPM features
        grid1 = self.project_layer1(self.M_inv)
        grid2 = self.project_layer2(self.M_inv)
        grid3 = self.project_layer3(self.M_inv)
        grid4 = self.project_layer4(self.M_inv)

        x1_proj = F.grid_sample(x1, grid1)
        x2_proj = F.grid_sample(x2, grid2)
        x2_proj_out = x2_proj
        x2_proj = self.dim_rt1(x2_proj)
        x3_proj = F.grid_sample(x3, grid3)
        x3_proj_out = x3_proj
        x3_proj = self.dim_rt2(x3_proj)
        x4_proj = F.grid_sample(x4, grid4)
        x4_proj_out = x4_proj
        x4_proj = self.dim_rt3(x4_proj)

        # process features from top view
        x, top_2, top_3, top_4 = self.top_pathway(x1_proj, x2_proj, x3_proj, x4_proj)

        # convert top-view features to anchor output
        out = self.lane_out(x)

        if self.debug:
            return out, cam_height, cam_pitch, x1, x2, x3, x4, x1_proj, x2_proj_out, x3_proj_out, x4_proj_out, top_2, top_3, top_4

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

    def load_pretrained_vgg(self, batch_norm):
        if batch_norm:
            vgg = models.vgg16_bn(pretrained=True)
            output_layers = [12, 22, 32, 42]
        else:
            vgg = models.vgg16(pretrained=True)
            output_layers = [8, 15, 22, 29]

        layer_ids_list = [range(0, output_layers[0]+1),
                          range(output_layers[0]+1, output_layers[1]+1),
                          range(output_layers[1] + 1, output_layers[2] + 1),
                          range(output_layers[2]+1, output_layers[3]+1)]
        features_list = [self.im_encoder.features1,
                         self.im_encoder.features2,
                         self.im_encoder.features3,
                         self.im_encoder.features4]

        for j in range(4):
            layer_ids = layer_ids_list[j]
            features = features_list[j]
            for i, lid in enumerate(layer_ids):
                classname = features[i].__class__.__name__
                if classname.find('Conv') != -1:
                    features[i].weight.data.copy_(vgg.features[lid].weight.data)
                    features[i].bias.data.copy_(vgg.features[lid].bias.data)
                elif classname.find('BatchNorm2d') != -1:
                    features[i].weight.data.copy_(vgg.features[lid].weight.data)
                    features[i].bias.data.copy_(vgg.features[lid].bias.data)


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
    image = torch.randn(1, 3, args.resize_h, args.resize_w)
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
