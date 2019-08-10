import numpy as np
import os.path as ops
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
import torchvision.models as models
from tools.utils import define_args, define_init_weights, init_projective_transform, tusimple_config


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
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# initialize base_grid with different sizes can adapt to different sizes
class ProjectiveGridGenerator(nn.Module):
    def __init__(self, size_ipm, im_h, im_w, theta, no_cuda):
        super().__init__()
        self.N, self.C, self.H, self.W = size_ipm
        self.im_h = im_h
        self.im_w = im_w
        linear_points_W = torch.linspace(0, 1 - 1/self.W, self.W)
        linear_points_H = torch.linspace(0, 1 - 1/self.H, self.H)

        self.base_grid = theta.new(self.N, self.H, self.W, 3)
        self.base_grid[:, :, :, 0] = torch.ger(
                torch.ones(self.H), linear_points_W).expand_as(self.base_grid[:, :, :, 0])
        self.base_grid[:, :, :, 1] = torch.ger(
                linear_points_H, torch.ones(self.W)).expand_as(self.base_grid[:, :, :, 1])
        self.base_grid[:, :, :, 2] = 1

        self.base_grid = Variable(self.base_grid)
        if not no_cuda:
            self.base_grid = self.base_grid.cuda()
            self.im_h = self.im_h.cuda()
            self.im_w = self.im_w.cuda()

    def forward(self, theta):
        # if base_grid is top-view, should theta be top-to-img homograph, and vice versa
        grid = torch.bmm(self.base_grid.view(self.N, self.H * self.W, 3), theta.transpose(1, 2))
        grid = torch.div(grid[:, :, 0:2], grid[:, :, 2:]).reshape([self.N, self.H, self.W, 2])
        # output grid to be used for grid_sample. grid specifies the sampling pixel locations normalized by the
        # input spatial dimensions.
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
        x = torch.cat([x, b], 1)
        x = self.features2(x)
        x = torch.cat([x, c], 1)
        x = self.features3(x)
        x = torch.cat([x, d], 1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


#  Lane Prediction Head: through a series of convolutions with no padding in the y dimension, the feature maps are
#  reduced in height, and finally the prediction layer size is N × 1 × 3 ·(2 · K + 1)
class LanePredictionHead(nn.Module):
    def __init__(self, batch_norm=False, num_lane_type=3, anchor_dim=5):
        super(LanePredictionHead, self).__init__()
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

        # reshape is needed before executing later layers
        dim_rt_layers = []
        dim_rt_layers += make_one_layer(256, 64, kernel_size=1, padding=0, batch_norm=batch_norm)
        dim_rt_layers += [nn.Conv2d(64, num_lane_type*anchor_dim, kernel_size=1, padding=0)]
        self.dim_rt = nn.Sequential(*dim_rt_layers)

    def forward(self, x):
        x = self.features(x)
        # x suppose to be N X 64 X 4 X w_ipm/8, reshape to N X 256 X w_ipm/8 X 1
        sizes = x.shape
        x = x.reshape(sizes[0], sizes[1]*sizes[2], sizes[3], 1)
        x = self.dim_rt(x)
        x = x.squeeze(-1).transpose(1, 2)
        # apply sigmoid to the probability terms to make it in (0, 1)
        x[:, :, self.anchor_dim-1:self.anchor_dim:] = torch.sigmoid(x[:, :, self.anchor_dim-1:self.anchor_dim:])
        return x

# TODO: implement homography net

# The 3D-lanenet composed of image encode, top view pathway, and lane predication head
class Net(nn.Module):
    def __init__(self, args):
        super().__init__()

        # define homographic transformation between image and ipm

        org_img_size = np.array([args.org_h, args.org_w])
        resize_img_size = np.array([args.resize_h, args.resize_w])
        pitch = np.pi / 180 * args.pitch
        M, M_inv = init_projective_transform(args.top_view_region, org_img_size,
                                             args.crop_size, resize_img_size, pitch, args.cam_height, args.K)
        # M = torch.from_numpy(M).unsqueeze_(0).expand([args.batch_size, 3, 3]).type(torch.FloatTensor)
        M_inv = torch.from_numpy(M_inv).unsqueeze_(0).expand([args.batch_size, 3, 3]).type(torch.FloatTensor)
        # self.M = M
        self.M_inv = M_inv
        if not args.no_cuda:
            self.M_inv = self.M_inv.cuda()

        if args.no_centerline:
            self.num_lane_type = 1
        else:
            self.num_lane_type = 3

        if args.no_3d:
            self.anchor_dim = args.num_y_anchor + 1
        else:
            self.anchor_dim = 2*args.num_y_anchor + 1

        # Define network
        self.im_encoder = VggEncoder(batch_norm=args.batch_norm)

        # the grid considers both src and dst grid normalized
        resize_img_size = torch.from_numpy(resize_img_size).type(torch.FloatTensor)
        size_top1 = torch.Size([args.batch_size, 128, args.ipm_h, args.ipm_w])
        self.project_layer1 = ProjectiveGridGenerator(size_top1, resize_img_size[0]/2, resize_img_size[1]/2,
                                                      M_inv, args.no_cuda)
        size_top2 = torch.Size([args.batch_size, 128, np.int(args.ipm_h / 2), np.int(args.ipm_w / 2)])
        self.project_layer2 = ProjectiveGridGenerator(size_top2, resize_img_size[0]/4, resize_img_size[1]/4,
                                                      M_inv, args.no_cuda)
        size_top3 = torch.Size([args.batch_size, 128, np.int(args.ipm_h / 4), np.int(args.ipm_w / 4)])
        self.project_layer3 = ProjectiveGridGenerator(size_top3, resize_img_size[0]/8, resize_img_size[1]/8,
                                                      M_inv, args.no_cuda)
        size_top4 = torch.Size([args.batch_size, 128, np.int(args.ipm_h / 8), np.int(args.ipm_w / 8)])
        self.project_layer4 = ProjectiveGridGenerator(size_top4, resize_img_size[0]/16, resize_img_size[1]/16,
                                                      M_inv, args.no_cuda)

        self.dim_rt1 = nn.Sequential(*make_one_layer(256, 128, kernel_size=1, padding=0, batch_norm=args.batch_norm))
        self.dim_rt2 = nn.Sequential(*make_one_layer(512, 256, kernel_size=1, padding=0, batch_norm=args.batch_norm))
        self.dim_rt3 = nn.Sequential(*make_one_layer(512, 256, kernel_size=1, padding=0, batch_norm=args.batch_norm))

        self.top_pathway = TopViewPathway(args.batch_norm)
        self.lane_out = LanePredictionHead(args.batch_norm, self.num_lane_type, self.anchor_dim)

    def forward(self, input):
        x1, x2, x3, x4 = self.im_encoder(input)

        # this need to be computed in run time for enabling back propagation?
        grid1 = self.project_layer1(self.M_inv)
        grid2 = self.project_layer2(self.M_inv)
        grid3 = self.project_layer3(self.M_inv)
        grid4 = self.project_layer4(self.M_inv)

        x1_proj = F.grid_sample(x1, grid1)
        x2_proj = F.grid_sample(x2, grid2)
        x2_proj = self.dim_rt1(x2_proj)
        x3_proj = F.grid_sample(x3, grid3)
        x3_proj = self.dim_rt2(x3_proj)
        x4_proj = F.grid_sample(x4, grid4)
        x4_proj = self.dim_rt3(x4_proj)
        x = self.top_pathway(x1_proj, x2_proj, x3_proj, x4_proj)
        out = self.lane_out(x)
        return out

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
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    global args
    parser = define_args()
    args = parser.parse_args()

    args.dataset_name = 'tusimple'
    args.data_dir = ops.join('data', args.dataset_name)
    args.dataset_dir = '/home/yuliangguo/Datasets/tusimple/'

    # load configuration for certain dataset
    if args.dataset_name is 'tusimple':
        tusimple_config(args)

    # construct model
    model = Net(args)
    print(model)

    # initialize model weights
    define_init_weights(model, args.weight_init)

    # load in vgg pretrained weights on ImageNet
    if args.pretrained:
        model.load_pretrained_vgg(args.batch_norm)
        print('vgg weights pretrained on ImageNet loaded!')

    # put model on gpu
    model = model.cuda()
    # load input
    img_name = '../1.jpg'
    with open(img_name, 'rb') as f:
        image = (Image.open(f).convert('RGB'))
    w, h = image.size
    image = F2.crop(image, args.crop_size, 0, args.org_h - args.crop_size, w)
    image = F2.resize(image, size=(args.resize_h, args.resize_w), interpolation=Image.BILINEAR)
    image = transforms.ToTensor()(image).float()
    image.unsqueeze_(0)
    image = torch.cat(list(torch.split(image, 1, dim=0))*args.batch_size)
    image = image.cuda()
    output_net = model(image)

    print(output_net.shape)
