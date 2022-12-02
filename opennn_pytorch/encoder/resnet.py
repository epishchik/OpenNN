from torch import nn
from collections import OrderedDict


class ResidualBlock1(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        if outc == inc:
            self.conv1 = nn.Conv2d(inc,
                                   outc,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1),
                                   bias=False)
            self.downsample = None
        else:
            self.conv1 = nn.Conv2d(inc,
                                   outc,
                                   kernel_size=(3, 3),
                                   stride=(2, 2),
                                   padding=(1, 1),
                                   bias=False)
            self.downsample = Downsample(inc, outc)
        self.bn1 = nn.BatchNorm2d(outc)
        self.conv2 = nn.Conv2d(outc,
                               outc,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu = nn.ReLU()

    def forward(self, x):
        b1 = self.relu(self.bn1(self.conv1(x)))
        b2 = self.bn2(self.conv2(b1))
        if self.downsample is not None:
            x = self.downsample(x)
        return x + b2


class ResidualBlock2(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.conv1 = nn.Conv2d(inc,
                               outc,
                               kernel_size=(1, 1),
                               stride=(1, 1),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(outc)
        if inc == outc or inc == int(outc * 4):
            self.conv2 = nn.Conv2d(outc,
                                   outc,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1),
                                   bias=False)
            tmp_down = Downsample(inc, int(outc * 4), stride=(1, 1))
            self.downsample = tmp_down if inc == outc else None
        else:
            self.conv2 = nn.Conv2d(outc,
                                   outc,
                                   kernel_size=(3, 3),
                                   stride=(2, 2),
                                   padding=(1, 1),
                                   bias=False)
            self.downsample = Downsample(inc, int(outc * 4))
        self.bn2 = nn.BatchNorm2d(outc)
        self.conv3 = nn.Conv2d(outc,
                               int(outc * 4),
                               kernel_size=(1, 1),
                               stride=(1, 1),
                               bias=False)
        self.bn3 = nn.BatchNorm2d(int(outc * 4))
        self.relu = nn.ReLU()

    def forward(self, x):
        b1 = self.bn1(self.conv1(x))
        b2 = self.bn2(self.conv2(b1))
        b3 = self.relu(self.bn3(self.conv3(b2)))
        if self.downsample is not None:
            x = self.downsample(x)
        return x + b3


class Downsample(nn.Module):
    def __init__(self, inc, outc, stride=(2, 2)):
        super().__init__()
        self.conv = nn.Conv2d(inc,
                              outc,
                              kernel_size=(1, 1),
                              stride=stride,
                              bias=False)
        self.bn = nn.BatchNorm2d(outc)

    def forward(self, x):
        return self.bn(self.conv(x))


def _make_layer(block, num_inc_outc):
    blocks = []
    idx = 0

    for triple in num_inc_outc:
        for _ in range(triple[0]):
            blocks.append((f'block{idx}', block(*triple[1:])))
            idx += 1

    return nn.Sequential(OrderedDict(blocks))


class BaseResNet(nn.Module):
    def __init__(self, inc):
        super().__init__()

        self.conv1 = nn.Conv2d(inc,
                               64,
                               kernel_size=(7, 7),
                               stride=(2, 2),
                               padding=(3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    dilation=1,
                                    ceil_mode=False)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        conv = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        l1 = self.layer1(conv)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        avgpool = self.avgpool(l4)
        return avgpool

    def out_features(self):
        return self.features


class ResNet18(BaseResNet):
    def __init__(self, inc):
        super().__init__(inc)
        self.features = 512

        self.layer1 = _make_layer(ResidualBlock1, ((2, 64, 64),))

        self.layer2 = _make_layer(ResidualBlock1, ((1, 64, 128),
                                                   (1, 128, 128)))

        self.layer3 = _make_layer(ResidualBlock1, ((1, 128, 256),
                                                   (1, 256, 256)))

        self.layer4 = _make_layer(ResidualBlock1, ((1, 256, 512),
                                                   (1, 512, 512)))


class ResNet34(BaseResNet):
    def __init__(self, inc):
        super().__init__(inc)
        self.features = 512

        self.layer1 = _make_layer(ResidualBlock1, ((3, 64, 64),))

        self.layer2 = _make_layer(ResidualBlock1, ((1, 64, 128),
                                                   (3, 128, 128)))

        self.layer3 = _make_layer(ResidualBlock1, ((1, 128, 256),
                                                   (5, 256, 256)))

        self.layer4 = _make_layer(ResidualBlock1, ((1, 256, 512),
                                                   (2, 512, 512)))


class ResNet50(BaseResNet):
    def __init__(self, inc):
        super().__init__(inc)
        self.features = 2048

        self.layer1 = _make_layer(ResidualBlock2, ((1, 64, 64),
                                                   (2, 256, 64)))

        self.layer2 = _make_layer(ResidualBlock2, ((1, 256, 128),
                                                   (3, 512, 128)))

        self.layer3 = _make_layer(ResidualBlock2, ((1, 512, 256),
                                                   (5, 1024, 256)))

        self.layer4 = _make_layer(ResidualBlock2, ((1, 1024, 512),
                                                   (2, 2048, 512)))


class ResNet101(BaseResNet):
    def __init__(self, inc):
        super().__init__(inc)
        self.features = 2048

        self.layer1 = _make_layer(ResidualBlock2, ((1, 64, 64),
                                                   (2, 256, 64)))

        self.layer2 = _make_layer(ResidualBlock2, ((1, 256, 128),
                                                   (3, 512, 128)))

        self.layer3 = _make_layer(ResidualBlock2, ((1, 512, 256),
                                                   (22, 1024, 256)))

        self.layer4 = _make_layer(ResidualBlock2, ((1, 1024, 512),
                                                   (2, 2048, 512)))


class ResNet152(BaseResNet):
    def __init__(self, inc):
        super().__init__(inc)
        self.features = 2048

        self.layer1 = _make_layer(ResidualBlock2, ((1, 64, 64),
                                                   (2, 256, 64)))

        self.layer2 = _make_layer(ResidualBlock2, ((1, 256, 128),
                                                   (7, 512, 128)))

        self.layer3 = _make_layer(ResidualBlock2, ((1, 512, 256),
                                                   (35, 1024, 256)))

        self.layer4 = _make_layer(ResidualBlock2, ((1, 1024, 512),
                                                   (2, 2048, 512)))
