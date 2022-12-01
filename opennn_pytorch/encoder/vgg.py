from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, inc, outc, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(inc, outc, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(outc)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class VGG11(nn.Module):
    def __init__(self, inc):
        super().__init__()
        self.features = 512
        self.conv1 = ConvBlock(inc, 64, kernel_size=(3, 3), padding=(1, 1))
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = ConvBlock(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(128, 256, kernel_size=(3, 3), padding=(1, 1))
        self.conv4 = ConvBlock(256, 256, kernel_size=(3, 3), padding=(1, 1))
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = ConvBlock(256, 512, kernel_size=(3, 3), padding=(1, 1))
        self.conv6 = ConvBlock(512, 512, kernel_size=(3, 3), padding=(1, 1))
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv7 = ConvBlock(512, 512, kernel_size=(3, 3), padding=(1, 1))
        self.conv8 = ConvBlock(512, 512, kernel_size=(3, 3), padding=(1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        b1 = self.maxpool1(self.conv1(x))
        b2 = self.maxpool2(self.conv2(b1))
        b3 = self.maxpool3(self.conv4(self.conv3(b2)))
        b4 = self.maxpool4(self.conv6(self.conv5(b3)))
        b5 = self.avgpool(self.conv8(self.conv7(b4)))
        return b5

    def out_features(self):
        return self.features


class VGG16(nn.Module):
    def __init__(self, inc):
        super().__init__()
        self.features = 512
        self.conv1 = ConvBlock(inc, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = ConvBlock(64, 64, kernel_size=(3, 3), padding=(1, 1))
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.conv4 = ConvBlock(128, 128, kernel_size=(3, 3), padding=(1, 1))
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = ConvBlock(128, 256, kernel_size=(3, 3), padding=(1, 1))
        self.conv6 = ConvBlock(256, 256, kernel_size=(3, 3), padding=(1, 1))
        self.conv7 = ConvBlock(256, 256, kernel_size=(1, 1))
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv8 = ConvBlock(256, 512, kernel_size=(3, 3), padding=(1, 1))
        self.conv9 = ConvBlock(512, 512, kernel_size=(3, 3), padding=(1, 1))
        self.conv10 = ConvBlock(512, 512, kernel_size=(1, 1))
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv11 = ConvBlock(512, 512, kernel_size=(3, 3), padding=(1, 1))
        self.conv12 = ConvBlock(512, 512, kernel_size=(3, 3), padding=(1, 1))
        self.conv13 = ConvBlock(512, 512, kernel_size=(1, 1), padding=(1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        b1 = self.maxpool1(self.conv2(self.conv1(x)))
        b2 = self.maxpool2(self.conv4(self.conv3(b1)))
        b3 = self.maxpool3(self.conv7(self.conv6(self.conv5(b2))))
        b4 = self.maxpool4(self.conv10(self.conv9(self.conv8(b3))))
        b5 = self.avgpool(self.conv13(self.conv12(self.conv11(b4))))
        return b5

    def out_features(self):
        return self.features


class VGG19(nn.Module):
    def __init__(self, inc):
        super().__init__()
        self.features = 512
        self.conv1 = ConvBlock(inc, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = ConvBlock(64, 64, kernel_size=(3, 3), padding=(1, 1))
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.conv4 = ConvBlock(128, 128, kernel_size=(3, 3), padding=(1, 1))
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = ConvBlock(128, 256, kernel_size=(3, 3), padding=(1, 1))
        self.conv6 = ConvBlock(256, 256, kernel_size=(3, 3), padding=(1, 1))
        self.conv7 = ConvBlock(256, 256, kernel_size=(3, 3), padding=(1, 1))
        self.conv8 = ConvBlock(256, 256, kernel_size=(3, 3), padding=(1, 1))
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv9 = ConvBlock(256, 512, kernel_size=(3, 3), padding=(1, 1))
        self.conv10 = ConvBlock(512, 512, kernel_size=(3, 3), padding=(1, 1))
        self.conv11 = ConvBlock(512, 512, kernel_size=(3, 3), padding=(1, 1))
        self.conv12 = ConvBlock(512, 512, kernel_size=(3, 3), padding=(1, 1))
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv13 = ConvBlock(512, 512, kernel_size=(3, 3), padding=(1, 1))
        self.conv14 = ConvBlock(512, 512, kernel_size=(3, 3), padding=(1, 1))
        self.conv15 = ConvBlock(512, 512, kernel_size=(1, 1), padding=(1, 1))
        self.conv16 = ConvBlock(512, 512, kernel_size=(1, 1), padding=(1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        b1 = self.maxpool1(self.conv2(self.conv1(x)))
        b2 = self.maxpool2(self.conv4(self.conv3(b1)))
        b3 = self.maxpool3(self.conv8(self.conv7(self.conv6(self.conv5(b2)))))
        b4 = self.maxpool4(self.conv12(self.conv11(
            self.conv10(self.conv9(b3)))))
        b5 = self.avgpool(self.conv16(self.conv15(
            self.conv14(self.conv13(b4)))))
        return b5

    def out_features(self):
        return self.features
