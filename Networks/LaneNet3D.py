import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
import torchvision.models as models
from tools.utils import define_args, define_init_weights


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


def Init_Projective_transform_test(nclasses, batch_size, resize):
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


# compute normalized transformation matrix for a top-view region boundaries
def Init_Projective_tranform(top_view_region, batch_size, org_img_size, crop_y, resize_img_size, pitch, cam_height, K):
    """
        Compute the normalized transformation (M_inv) such that image region corresponding to top_view region maps to
        the top view image's 4 corners
        Ground coordinates: x-right, y-forward, z-up
        The purpose of applying normalized transformation is for invariance in scale change

    :param top_view_region: a 4 X 2 list of (X, Y) indicating the top-view region corners in order:
                            top-left, top-right, bottom-left, bottom-right
    :param batch_size: number of samples for each batch
    :param org_img_size: the size of original image size: (h, w)
    :param crop_y: pixels croped from original img
    :param resize_img_size: the size of image as network input: (h, w)
    :param pitch: camera pitch angle wrt ground plane
    :param cam_height: camera height wrt ground plane in meters
    :param K: camera intrinsic parameters
    :return: M_inv: the normalized transformation from image to IPM image
    """

    # transform top-view region to original image region
    R_g2c = np.array([[1, 0, 0],
                      [0, np.cos(-np.pi/2 - pitch), -np.sin(-np.pi/2 - pitch)],
                      [0, np.sin(-np.pi/2 - pitch), np.cos(-np.pi/2 - pitch)]])
    H_g2c = np.matmul(K, np.concatenate(
                [R_g2c[:, 0:1], R_g2c[:, 1:2], np.matmul(R_g2c.transpose(), np.array([[0], [0], [-cam_height]]))], 1))
    X = np.concatenate([top_view_region, np.ones([4, 1])], 1)
    img_region = np.matmul(X, H_g2c.T)
    border_org = np.divide(img_region[:, :2], img_region[:, 2:3])

    # transform original image region to network input region
    ratio_x = resize_img_size[1] / org_img_size[1]
    ratio_y = resize_img_size[0] / (org_img_size[0] - crop_y)
    H_c = np.array([[ratio_x, 0, 0],
                    [0, ratio_y, -ratio_y*crop_y]])
    border_net = np.matmul(np.concatenate([border_org, np.ones([4, 1])], axis=1), H_c.T)
    border_net[:, 0] = border_net[:, 0] / resize_img_size[1]
    border_net[:, 1] = border_net[:, 1] / resize_img_size[0]
    border_net = np.float32(border_net)

    # compute the normalized transformation
    dst = np.float32([[0, 0], [0, 1], [1, 0], [1, 1]])
    M = cv2.getPerspectiveTransform(border_net, dst)
    M_inv = cv2.getPerspectiveTransform(dst, border_net)
    M = torch.from_numpy(M).unsqueeze_(0).expand([batch_size, 3, 3]).type(torch.FloatTensor)
    M_inv = torch.from_numpy(M_inv).unsqueeze_(0).expand([batch_size, 3, 3]).type(torch.FloatTensor)
    return M, M_inv


def square_tensor(x):
    return x**2


class VGG_encoder(nn.Module):

    def __init__(self, output_layers, pretrained=False, init_weights=True):
        super(VGG_encoder, self).__init__()
        if pretrained:
            init_weights = False
        model_org = models.vgg16(pretrained)
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
        # if base_grid is top-view, should theta be top-to-img homograph, and vice versa
        grid = torch.bmm(self.base_grid.view(self.N, self.H * self.W, 3), theta.transpose(1, 2))
        grid = torch.div(grid[:, :, 0:2], grid[:, :, 2:]).reshape([self.N, self.H, self.W, 2])
        return grid


# Sub-network corresponding to the top view pathway
class TopViewPathway(nn.Module):
    def __init__(self, init_weights=True):
        super(TopViewPathway, self).__init__()
        self.features1 = make_layers(['M', 128, 128, 128], 128)
        self.features2 = make_layers(['M', 256, 256, 256], 256)
        self.features3 = make_layers(['M', 256, 256, 256], 512)

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
    def __init__(self, num_y_anchor):
        super(LanePredictionHead, self).__init__()
        layers = []
        conv2d = nn.Conv2d(512, 64, kernel_size=3, padding=(0, 1))
        layers += [conv2d, nn.ReLU(inplace=True)]
        conv2d = nn.Conv2d(64, 64, kernel_size=3, padding=(0, 1))
        layers += [conv2d, nn.ReLU(inplace=True)]
        conv2d = nn.Conv2d(64, 64, kernel_size=3, padding=(0, 1))
        layers += [conv2d, nn.ReLU(inplace=True)]

        conv2d = nn.Conv2d(64, 64, kernel_size=5, padding=(0, 2))
        layers += [conv2d, nn.ReLU(inplace=True)]
        conv2d = nn.Conv2d(64, 64, kernel_size=5, padding=(0, 2))
        layers += [conv2d, nn.ReLU(inplace=True)]
        conv2d = nn.Conv2d(64, 64, kernel_size=5, padding=(0, 2))
        layers += [conv2d, nn.ReLU(inplace=True)]
        conv2d = nn.Conv2d(64, 64, kernel_size=5, padding=(0, 2))
        layers += [conv2d, nn.ReLU(inplace=True)]
        self.features = nn.Sequential(*layers)

        # reshape is needed before executing later layers
        self.dim_rt1 = nn.Conv2d(256, 64, kernel_size=1, padding=0)
        self.dim_rt2 = nn.Conv2d(64, 3*(2*num_y_anchor+1), kernel_size=1, padding=0)

    def forward(self, x):
        x = self.features(x)
        # x suppose to be N X 64 X 4 X 26, reshape to N X 256 X 26 X 1
        sizes = x.shape
        x = x.reshape(sizes[0], sizes[1]*sizes[2], sizes[3], 1)
        x = self.dim_rt1(x)
        x = self.dim_rt2(x)
        x = x.squeeze(-1).transpose(1, 2)
        return x

# TODO: implement homography net

# The 3D-lanenet composed of image encode, top view pathway, and lane predication head
class Net(nn.Module):
    def __init__(self, args):
        super().__init__()

        # define sizes and perspective transformation
        # resize = args.resize
        # size = torch.Size([args.batch_size, args.nclasses, args.resize, 2*args.resize])

        # M, M_inv = Init_Projective_transform(args.nclasses, args.batch_size, args.resize)
        org_img_size = [args.org_h, args.org_w]
        resize_img_size = [args.resize, 2*args.resize]
        pitch = np.pi / 180 * args.pitch
        M, M_inv = Init_Projective_tranform(args.top_view_region, args.batch_size, org_img_size,
                                            args.crop_size, resize_img_size, pitch, args.cam_height, args.K)
        # self.M = M
        self.M_inv = M_inv
        if not args.no_cuda:
            self.M_inv = self.M_inv.cuda()

        # Define network
        output_layers = [8, 15, 22, 29]
        self.im_encoder = VGG_encoder(output_layers, True)
        # the grid considers both src and dst grid normalized
        self.size_top1 = torch.Size([args.batch_size, 128, args.ipm_h, args.ipm_w])
        self.project_layer1 = ProjectiveGridGenerator(self.size_top1, M_inv, args.no_cuda)
        self.size_top2 = torch.Size([args.batch_size, 128, np.int(args.ipm_h / 2), np.int(args.ipm_w / 2)])
        self.project_layer2 = ProjectiveGridGenerator(self.size_top2, M_inv, args.no_cuda)
        self.size_top3 = torch.Size([args.batch_size, 128, np.int(args.ipm_h / 4), np.int(args.ipm_w / 4)])
        self.project_layer3 = ProjectiveGridGenerator(self.size_top3, M_inv, args.no_cuda)
        self.size_top4 = torch.Size([args.batch_size, 128, np.int(args.ipm_h / 8), np.int(args.ipm_w / 8)])
        self.project_layer4 = ProjectiveGridGenerator(self.size_top4, M_inv, args.no_cuda)

        self.dim_rt1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.dim_rt2 = nn.Conv2d(512, 256, kernel_size=1, padding=0)
        self.dim_rt3 = nn.Conv2d(512, 256, kernel_size=1, padding=0)

        self.top_pathway = TopViewPathway()
        self.lane_out = LanePredictionHead(args.num_y_anchor)

    def forward(self, input):
        x1, x2, x3, x4 = self.im_encoder(input)

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


# unit test
if __name__ == '__main__':
    from PIL import Image
    from torchvision import transforms, utils
    import torchvision.transforms.functional as F2

    global args
    parser = define_args()
    args = parser.parse_args()
    args.top_view_region = np.array([[-20, 100], [20, 100], [-20, 5], [20, 5]])
    args.anchor_y_steps = np.array([5, 20, 40, 60, 80, 100])
    args.num_y_anchor = len(args.anchor_y_steps)
    args.K = np.array([[720, 0, 640],
                       [0, 720, 360],
                       [0, 0, 1]])
    model = Net(args)
    print(model)

    # initialize model weights
    define_init_weights(model, args.weight_init)
    # put model on gpu
    model = model.cuda()
    # load input
    img_name = '/home/yuliangguo/Projects/3DLaneNet/1.jpg'
    with open(img_name, 'rb') as f:
        image = (Image.open(f).convert('RGB'))
    w, h = image.size
    image = F2.crop(image, h-640, 0, 640, w)
    image = F2.resize(image, size=(args.resize, 2*args.resize), interpolation=Image.BILINEAR)
    image = transforms.ToTensor()(image).float()
    image.unsqueeze_(0)
    image = torch.cat(list(torch.split(image, 1, dim=0))*args.batch_size)
    image = image.cuda()
    output_net = model(image)

    print(output_net.shape)
