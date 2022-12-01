from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, inc, outc, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(inc, outc, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(outc)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DepthWiseSepConvBlock(nn.Module):
    def __init__(self, inc, outc, **kwargs):
        super().__init__()
        self.depthwise = nn.Conv2d(inc,
                                   inc,
                                   kernel_size=(3, 3),
                                   padding=(1, 1),
                                   groups=inc,
                                   bias=False,
                                   **kwargs)
        self.bn1 = nn.BatchNorm2d(inc)
        self.pointwise = nn.Conv2d(inc, outc, kernel_size=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu = nn.ReLU()

    def forward(self, x):
        depth = self.relu(self.bn1(self.depthwise(x)))
        point = self.relu(self.bn2(self.pointwise(depth)))
        return point


class MobileNet(nn.Module):
    def __init__(self, inc):
        super().__init__()
        self.features = 1024
        self.conv = ConvBlock(inc,
                              32,
                              kernel_size=(3, 3),
                              stride=(2, 2),
                              padding=(1, 1))
        self.convdws1 = DepthWiseSepConvBlock(32, 64, stride=(1, 1))
        self.convdws2 = DepthWiseSepConvBlock(64, 128, stride=(2, 2))
        self.convdws3 = DepthWiseSepConvBlock(128, 128, stride=(1, 1))
        self.convdws4 = DepthWiseSepConvBlock(128, 256, stride=(2, 2))
        self.convdws5 = DepthWiseSepConvBlock(256, 256, stride=(1, 1))
        self.convdws6 = DepthWiseSepConvBlock(256, 512, stride=(2, 2))
        self.convdws7 = DepthWiseSepConvBlock(512, 512, stride=(1, 1))
        self.convdws8 = DepthWiseSepConvBlock(512, 512, stride=(1, 1))
        self.convdws9 = DepthWiseSepConvBlock(512, 512, stride=(1, 1))
        self.convdws10 = DepthWiseSepConvBlock(512, 512, stride=(1, 1))
        self.convdws11 = DepthWiseSepConvBlock(512, 512, stride=(1, 1))
        self.convdws12 = DepthWiseSepConvBlock(512, 1024, stride=(2, 2))
        self.convdws13 = DepthWiseSepConvBlock(1024, 1024, stride=(1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        b1 = self.convdws4(self.convdws3(
            self.convdws2(self.convdws1(self.conv(x)))))
        b2 = self.convdws9(self.convdws8(
            self.convdws7(self.convdws6(self.convdws5(b1)))))
        b3 = self.avgpool(self.convdws13(
            self.convdws12(self.convdws11(self.convdws10(b2)))))
        return b3

    def out_features(self):
        return self.features
