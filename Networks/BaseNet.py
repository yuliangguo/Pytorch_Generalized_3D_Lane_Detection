import torch.nn as nn
import torch.utils.model_zoo as model_zoo

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG_muli_out(nn.Module):

    def __init__(self, cfg1, cfg2, cfg3, cfg4, init_weights=True):
        super(VGG_muli_out, self).__init__()
        self.features1 = make_layers(cfg1)
        self.features2 = make_layers(cfg2)
        self.features3 = make_layers(cfg3)
        self.features4 = make_layers(cfg4)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features1(x1)
        x3 = self.features1(x2)
        x4 = self.features1(x3)
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


def make_layers(cfg, batch_norm=False):
    layers = []
    # TODO: need to change in_channels
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
    return nn.Sequential(*layers)


def vgg16_multi_out(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    cfg1 = [64, 64, 'M', 128, 128]
    cfg2 = ['M', 256, 256, 256]
    cfg3 = ['M', 512, 512, 512]
    cfg4 = ['M', 512, 512, 512]
    model = VGG_muli_out(cfg1, cfg2, cfg3, cfg4, **kwargs)
    # TODO: loading might not directly work
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


# TODO: implement unit test
if __name__ == '__main__':
    vgg16 = vgg16_multi_out()
    print('done')