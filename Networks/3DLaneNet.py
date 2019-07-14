import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import ceil
import cv2
import Networks.BaseNet as basenet
import torchvision.models as models


# TODO: compute normalized transformation matrix from real homography?
# find a top view rect, and computes its corresponding image pixels, and their normalized coords in image
# define their dst coords in visualize image
# compute the normalized homograph
# all the grids in dst view image can interpolated afterwards
def Init_Projective_transform(nclasses, batch_size, resize):
    # M_orig: unnormalized Transformation matrix
    # M: normalized transformation matrix
    # M_inv: Inverted normalized transformation matrix --> Needed for grid sample
    # original aspect ratio: 720x1280 --> after 80 rows cropped: 640x1280 --> after resize: 256x512 (default) or resize x 2*resize (in general)
    size = torch.Size([batch_size, nclasses, resize, 2*resize])
    y_start = 0.3
    y_stop = 1
    xd1, xd2, xd3, xd4 = 0.45, 0.55, 0.45, 0.55
    src = np.float32([[0.45, y_start], [0.55, y_start], [0.1, y_stop], [0.9, y_stop]])
    dst = np.float32([[xd3, y_start], [xd4, y_start], [xd1, y_stop], [xd2, y_stop]])
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    M = torch.from_numpy(M).unsqueeze_(0).expand([batch_size, 3, 3]).type(torch.FloatTensor)
    M_inv = torch.from_numpy(M_inv).unsqueeze_(0).expand([batch_size, 3, 3]).type(torch.FloatTensor)
    return size, M, M_inv


def square_tensor(x):
    return x**2

# this assumes the src, dst have the save dimension
class ProjectiveGridGenerator(nn.Module):
    def __init__(self, size, theta, no_cuda):
        super().__init__()
        self.N, self.C, self.H, self.W = size
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

    def forward(self, theta):
        grid = torch.bmm(self.base_grid.view(self.N, self.H * self.W, 3), theta.transpose(1, 2))
        grid = torch.div(grid[:, :, 0:2], grid[:, :, 2:])
        return grid


# Sub-network corresponding to the top view pathway
class TopViewPathway(nn.Module):
    def __init__(self, init_weights=True):
        super(TopViewPathway, self).__init__()
        self.features1 = basenet.make_layers(['M', 128, 128, 128])
        self.features2 = basenet.make_layers(['M', 256, 256, 256])
        self.features3 = basenet.make_layers(['M', 256, 256, 256])

        if init_weights:
            self._initialize_weights()

    def forward(self, a, b, c, d):
        x = self.features1(a)
        x = torch.cat([x, b], -1)
        x = self.features1(x)
        x = torch.cat([x, c], -1)
        x = self.features1(x)
        x = torch.cat([x, d], -1)
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


# TODO: implement this
class LanePredictionHead(nn.Module):
    def __init__(self, batch_norm=False, init_weights=True):
        super(LanePredictionHead, self).__init__()
        # TODO: Through a series of convolutions with no padding in the y dimension, the feature maps are
        #  reduced in height, and finally the prediction layer size is
        #  N × 1 × 3 ·(2 · K + 1)
        layers = []
        cfg =[]
        in_channels = 3
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

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        return x


# The 3D-lanenet composed of image encode, top view pathway, and lane predication head
class Net(nn.Module):
    def __init__(self, args):
        super().__init__()

        # define sizes and perspective transformation
        # resize = args.resize
        # size = torch.Size([args.batch_size, args.nclasses, args.resize, 2*args.resize])

        #
        size, M, M_inv = Init_Projective_transform(args.nclasses, args.batch_size, args.resize)
        self.M = M

        # Define network
        self.im_encoder = basenet.vgg16_multi_out()
        # the grid considers both src and dst grid normalized
        size[2:] = size[2:]/2
        self.project_layer1 = ProjectiveGridGenerator(size, M, args.no_cuda)
        size[2:] = size[2:]/2
        self.project_layer2 = ProjectiveGridGenerator(size, M, args.no_cuda)
        size[2:] = size[2:]/2
        self.project_layer3 = ProjectiveGridGenerator(size, M, args.no_cuda)
        size[2:] = size[2:]/2
        self.project_layer4 = ProjectiveGridGenerator(size, M, args.no_cuda)

        self.top_pathway = TopViewPathway()
        self.lane_out = LanePredictionHead()

    def forward(self, input):
        x1, x2, x3, x4 = self.im_encoder(input)

        grid1 = self.project_layer1(self.M)
        grid2 = self.project_layer1(self.M)
        grid3 = self.project_layer1(self.M)
        grid4 = self.project_layer1(self.M)

        x1_proj = F.grid_sample(x1, grid1)
        # TODO: dimension reduction and resize
        x2_proj = F.grid_sample(x2, grid2)
        # TODO: dimension reduction and resize
        x3_proj = F.grid_sample(x3, grid3)
        # TODO: dimension reduction and resize
        x4_proj = F.grid_sample(x4, grid4)
        # TODO: dimension reduction and resize
        x = self.top_pathway(x1_proj, x2_proj, x3_proj, x4_proj)
        out = self.lane_out(x)
        return out


# TODO: implement unit test
if __name__ == '__main__':
    lanenet_3D = Net()
    print('done')
    # TODO, how to load in VGG to part of this network?