from torch import nn


class ResidualBlock1(nn.Module):
    '''
    Residual block class type 1.

    Methods
    -------
    forward(x)
        calculate features from input image x.
    '''
    def __init__(self, inc, outc):
        '''
        Create residual block type 1 layers.

        Parameters
        ----------
        inc : int
            number of input channels.

        outc : int
            number of output channels.
        '''
        super().__init__()
        if outc == inc:
            self.conv1 = nn.Conv2d(inc, outc, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.downsample = None
        else:
            self.conv1 = nn.Conv2d(inc, outc, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.downsample = Downsample(inc, outc)
        self.bn1 = nn.BatchNorm2d(outc)
        self.conv2 = nn.Conv2d(outc, outc, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu = nn.ReLU()

    def forward(self, x):
        '''
        Calculate features from input image x.

        Parameterts
        -----------
        x : torch.tensor
            input image.
        '''
        b1 = self.relu(self.bn1(self.conv1(x)))
        b2 = self.bn2(self.conv2(b1))
        if self.downsample is not None:
            x = self.downsample(x)
        return x + b2


class ResidualBlock2(nn.Module):
    '''
    Residual block class type 2.

    Methods
    -------
    forward(x)
        calculate features from input image x.
    '''
    def __init__(self, inc, outc):
        '''
        Create residual block type 2 layers.

        Parameters
        ----------
        inc : int
            number of input channels.

        outc : int
            number of output channels.
        '''
        super().__init__()
        self.conv1 = nn.Conv2d(inc, outc, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(outc)
        if inc == outc or inc == int(outc * 4):
            self.conv2 = nn.Conv2d(outc, outc, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.downsample = Downsample(inc, int(outc * 4), stride=(1, 1)) if inc == outc else None
        else:
            self.conv2 = nn.Conv2d(outc, outc, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.downsample = Downsample(inc, int(outc * 4))
        self.bn2 = nn.BatchNorm2d(outc)
        self.conv3 = nn.Conv2d(outc, int(outc * 4), kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(int(outc * 4))
        self.relu = nn.ReLU()

    def forward(self, x):
        '''
        Calculate features from input image x.

        Parameterts
        -----------
        x : torch.tensor
            input image.
        '''
        b1 = self.bn1(self.conv1(x))
        b2 = self.bn2(self.conv2(b1))
        b3 = self.relu(self.bn3(self.conv3(b2)))
        if self.downsample is not None:
            x = self.downsample(x)
        return x + b3


class Downsample(nn.Module):
    '''
    Downsample block class.

    Methods
    -------
    forward(x)
        calculate features from input image x.
    '''
    def __init__(self, inc, outc, stride=(2, 2)):
        '''
        Create downsample block layers.

        Parameters
        ----------
        inc : int
            number of input channels.

        outc : int
            number of output channels.

        stride: tuple(int, int), int, optional
            stride for convolution layer.
        '''
        super().__init__()
        self.conv = nn.Conv2d(inc, outc, kernel_size=(1, 1), stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(outc)

    def forward(self, x):
        '''
        Calculate features from input image x.

        Parameterts
        -----------
        x : torch.tensor
            input image.
        '''
        return self.bn(self.conv(x))


class Resnet18Features(nn.Module):
    '''
    Class used to calculate features using resnet18 architecture.
    https://arxiv.org/pdf/1512.03385.pdf

    Methods
    -------
    forward(x)
        calculate features from input image x.

    out_features()
        return number of output features, needed for decoders input channels.

    name()
        return name of encoder.
    '''
    def __init__(self, inc):
        '''
        Create resnet18 encoder layers.

        Parameters
        ----------
        inc : int
            number of input channels [1, 3, 4].
        '''
        super().__init__()
        self.features = 512
        self.conv1 = nn.Conv2d(inc, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.resblock11 = ResidualBlock1(64, 64)
        self.resblock12 = ResidualBlock1(64, 64)
        self.resblock21 = ResidualBlock1(64, 128)
        self.resblock22 = ResidualBlock1(128, 128)
        self.resblock31 = ResidualBlock1(128, 256)
        self.resblock32 = ResidualBlock1(256, 256)
        self.resblock41 = ResidualBlock1(256, 512)
        self.resblock42 = ResidualBlock1(512, 512)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        '''
        Calculate features from input image x.

        Parameterts
        -----------
        x : torch.tensor
            input image.
        '''
        pr = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        b1 = self.resblock12(self.resblock11(pr))
        b2 = self.resblock22(self.resblock21(b1))
        b3 = self.resblock32(self.resblock31(b2))
        b4 = self.resblock42(self.resblock41(b3))
        b5 = self.avgpool(b4)
        return b5

    def out_features(self):
        '''
        Return number of output features, needed for decoders input channels.
        '''
        return self.features

    def name(self):
        '''
        Return name of encoder.
        '''
        return 'resnet18'


class Resnet34Features(nn.Module):
    '''
    Class used to calculate features using resnet34 architecture.
    https://arxiv.org/pdf/1512.03385.pdf

    Methods
    -------
    forward(x)
        calculate features from input image x.

    out_features()
        return number of output features, needed for decoders input channels.

    name()
        return name of encoder.
    '''
    def __init__(self, inc):
        '''
        Create resnet34 encoder layers.

        Parameters
        ----------
        inc : int
            number of input channels [1, 3, 4].
        '''
        super().__init__()
        self.features = 512
        self.conv1 = nn.Conv2d(inc, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.resblock11 = ResidualBlock1(64, 64)
        self.resblock12 = ResidualBlock1(64, 64)
        self.resblock13 = ResidualBlock1(64, 64)
        self.resblock21 = ResidualBlock1(64, 128)
        self.resblock22 = ResidualBlock1(128, 128)
        self.resblock23 = ResidualBlock1(128, 128)
        self.resblock24 = ResidualBlock1(128, 128)
        self.resblock31 = ResidualBlock1(128, 256)
        self.resblock32 = ResidualBlock1(256, 256)
        self.resblock33 = ResidualBlock1(256, 256)
        self.resblock34 = ResidualBlock1(256, 256)
        self.resblock35 = ResidualBlock1(256, 256)
        self.resblock36 = ResidualBlock1(256, 256)
        self.resblock41 = ResidualBlock1(256, 512)
        self.resblock42 = ResidualBlock1(512, 512)
        self.resblock43 = ResidualBlock1(512, 512)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        '''
        Calculate features from input image x.

        Parameterts
        -----------
        x : torch.tensor
            input image.
        '''
        pr = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        b1 = self.resblock13(self.resblock12(self.resblock11(pr)))
        b2 = self.resblock24(self.resblock23(self.resblock22(self.resblock21(b1))))
        b3 = self.resblock36(self.resblock35(self.resblock34(self.resblock33(self.resblock32(self.resblock31(b2))))))
        b4 = self.resblock43(self.resblock42(self.resblock41(b3)))
        b5 = self.avgpool(b4)
        return b5

    def out_features(self):
        '''
        Return number of output features, needed for decoders input channels.
        '''
        return self.features

    def name(self):
        '''
        Return name of encoder.
        '''
        return 'resnet34'


class Resnet50Features(nn.Module):
    '''
    Class used to calculate features using resnet50 architecture.
    https://arxiv.org/pdf/1512.03385.pdf

    Methods
    -------
    forward(x)
        calculate features from input image x.

    out_features()
        return number of output features, needed for decoders input channels.

    name()
        return name of encoder.
    '''
    def __init__(self, inc):
        '''
        Create resnet50 encoder layers.

        Parameters
        ----------
        inc : int
            number of input channels [1, 3, 4].
        '''
        super().__init__()
        self.features = 2048
        self.conv1 = nn.Conv2d(inc, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.resblock11 = ResidualBlock2(64, 64)
        self.resblock12 = ResidualBlock2(256, 64)
        self.resblock13 = ResidualBlock2(256, 64)
        self.resblock21 = ResidualBlock2(256, 128)
        self.resblock22 = ResidualBlock2(512, 128)
        self.resblock23 = ResidualBlock2(512, 128)
        self.resblock24 = ResidualBlock2(512, 128)
        self.resblock31 = ResidualBlock2(512, 256)
        self.resblock32 = ResidualBlock2(1024, 256)
        self.resblock33 = ResidualBlock2(1024, 256)
        self.resblock34 = ResidualBlock2(1024, 256)
        self.resblock35 = ResidualBlock2(1024, 256)
        self.resblock36 = ResidualBlock2(1024, 256)
        self.resblock41 = ResidualBlock2(1024, 512)
        self.resblock42 = ResidualBlock2(2048, 512)
        self.resblock43 = ResidualBlock2(2048, 512)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        '''
        Calculate features from input image x.

        Parameterts
        -----------
        x : torch.tensor
            input image.
        '''
        pr = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        b1 = self.resblock13(self.resblock12(self.resblock11(pr)))
        b2 = self.resblock24(self.resblock23(self.resblock22(self.resblock21(b1))))
        b3 = self.resblock36(self.resblock35(self.resblock34(self.resblock33(self.resblock32(self.resblock31(b2))))))
        b4 = self.resblock43(self.resblock42(self.resblock41(b3)))
        b5 = self.avgpool(b4)
        return b5

    def out_features(self):
        '''
        Return number of output features, needed for decoders input channels.
        '''
        return self.features

    def name(self):
        '''
        Return name of encoder.
        '''
        return 'resnet50'


class Resnet101Features(nn.Module):
    '''
    Class used to calculate features using resnet101 architecture.
    https://arxiv.org/pdf/1512.03385.pdf

    Methods
    -------
    forward(x)
        calculate features from input image x.

    out_features()
        return number of output features, needed for decoders input channels.

    name()
        return name of encoder.
    '''
    def __init__(self, inc):
        '''
        Create resnet101 encoder layers.

        Parameters
        ----------
        inc : int
            number of input channels [1, 3, 4].
        '''
        super().__init__()
        self.features = 2048
        self.conv1 = nn.Conv2d(inc, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.resblock11 = ResidualBlock2(64, 64)
        self.resblock12 = ResidualBlock2(256, 64)
        self.resblock13 = ResidualBlock2(256, 64)
        self.resblock21 = ResidualBlock2(256, 128)
        self.resblock22 = ResidualBlock2(512, 128)
        self.resblock23 = ResidualBlock2(512, 128)
        self.resblock24 = ResidualBlock2(512, 128)
        self.resblock31 = ResidualBlock2(512, 256)
        self.resblock32 = ResidualBlock2(1024, 256)
        self.resblock33 = ResidualBlock2(1024, 256)
        self.resblock34 = ResidualBlock2(1024, 256)
        self.resblock35 = ResidualBlock2(1024, 256)
        self.resblock36 = ResidualBlock2(1024, 256)
        self.resblock37 = ResidualBlock2(1024, 256)
        self.resblock38 = ResidualBlock2(1024, 256)
        self.resblock39 = ResidualBlock2(1024, 256)
        self.resblock310 = ResidualBlock2(1024, 256)
        self.resblock311 = ResidualBlock2(1024, 256)
        self.resblock312 = ResidualBlock2(1024, 256)
        self.resblock313 = ResidualBlock2(1024, 256)
        self.resblock314 = ResidualBlock2(1024, 256)
        self.resblock315 = ResidualBlock2(1024, 256)
        self.resblock316 = ResidualBlock2(1024, 256)
        self.resblock317 = ResidualBlock2(1024, 256)
        self.resblock318 = ResidualBlock2(1024, 256)
        self.resblock319 = ResidualBlock2(1024, 256)
        self.resblock320 = ResidualBlock2(1024, 256)
        self.resblock321 = ResidualBlock2(1024, 256)
        self.resblock322 = ResidualBlock2(1024, 256)
        self.resblock323 = ResidualBlock2(1024, 256)
        self.resblock41 = ResidualBlock2(1024, 512)
        self.resblock42 = ResidualBlock2(2048, 512)
        self.resblock43 = ResidualBlock2(2048, 512)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        '''
        Calculate features from input image x.

        Parameterts
        -----------
        x : torch.tensor
            input image.
        '''
        pr = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        b1 = self.resblock13(self.resblock12(self.resblock11(pr)))
        b2 = self.resblock24(self.resblock23(self.resblock22(self.resblock21(b1))))
        b3 = self.resblock36(self.resblock35(self.resblock34(self.resblock33(self.resblock32(self.resblock31(b2))))))
        b4 = self.resblock312(self.resblock311(self.resblock310(self.resblock39(self.resblock38(self.resblock37(b3))))))
        b5 = self.resblock318(self.resblock317(self.resblock316(self.resblock315(self.resblock314(self.resblock313(b4))))))
        b6 = self.resblock323(self.resblock322(self.resblock321(self.resblock320(self.resblock319(b5)))))
        b7 = self.resblock43(self.resblock42(self.resblock41(b6)))
        b8 = self.avgpool(b7)
        return b8

    def out_features(self):
        '''
        Return number of output features, needed for decoders input channels.
        '''
        return self.features

    def name(self):
        '''
        Return name of encoder.
        '''
        return 'resnet101'


class Resnet152Features(nn.Module):
    '''
    Class used to calculate features using resnet152 architecture.
    https://arxiv.org/pdf/1512.03385.pdf

    Methods
    -------
    forward(x)
        calculate features from input image x.

    out_features()
        return number of output features, needed for decoders input channels.

    name()
        return name of encoder.
    '''
    def __init__(self, inc):
        '''
        Create resnet152 encoder layers.

        Parameters
        ----------
        inc : int
            number of input channels [1, 3, 4].
        '''
        super().__init__()
        self.features = 2048
        self.conv1 = nn.Conv2d(inc, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.resblock11 = ResidualBlock2(64, 64)
        self.resblock12 = ResidualBlock2(256, 64)
        self.resblock13 = ResidualBlock2(256, 64)
        self.resblock21 = ResidualBlock2(256, 128)
        self.resblock22 = ResidualBlock2(512, 128)
        self.resblock23 = ResidualBlock2(512, 128)
        self.resblock24 = ResidualBlock2(512, 128)
        self.resblock25 = ResidualBlock2(512, 128)
        self.resblock26 = ResidualBlock2(512, 128)
        self.resblock27 = ResidualBlock2(512, 128)
        self.resblock28 = ResidualBlock2(512, 128)
        self.resblock31 = ResidualBlock2(512, 256)
        self.resblock32 = ResidualBlock2(1024, 256)
        self.resblock33 = ResidualBlock2(1024, 256)
        self.resblock34 = ResidualBlock2(1024, 256)
        self.resblock35 = ResidualBlock2(1024, 256)
        self.resblock36 = ResidualBlock2(1024, 256)
        self.resblock37 = ResidualBlock2(1024, 256)
        self.resblock38 = ResidualBlock2(1024, 256)
        self.resblock39 = ResidualBlock2(1024, 256)
        self.resblock310 = ResidualBlock2(1024, 256)
        self.resblock311 = ResidualBlock2(1024, 256)
        self.resblock312 = ResidualBlock2(1024, 256)
        self.resblock313 = ResidualBlock2(1024, 256)
        self.resblock314 = ResidualBlock2(1024, 256)
        self.resblock315 = ResidualBlock2(1024, 256)
        self.resblock316 = ResidualBlock2(1024, 256)
        self.resblock317 = ResidualBlock2(1024, 256)
        self.resblock318 = ResidualBlock2(1024, 256)
        self.resblock319 = ResidualBlock2(1024, 256)
        self.resblock320 = ResidualBlock2(1024, 256)
        self.resblock321 = ResidualBlock2(1024, 256)
        self.resblock322 = ResidualBlock2(1024, 256)
        self.resblock323 = ResidualBlock2(1024, 256)
        self.resblock324 = ResidualBlock2(1024, 256)
        self.resblock325 = ResidualBlock2(1024, 256)
        self.resblock326 = ResidualBlock2(1024, 256)
        self.resblock327 = ResidualBlock2(1024, 256)
        self.resblock328 = ResidualBlock2(1024, 256)
        self.resblock329 = ResidualBlock2(1024, 256)
        self.resblock330 = ResidualBlock2(1024, 256)
        self.resblock331 = ResidualBlock2(1024, 256)
        self.resblock332 = ResidualBlock2(1024, 256)
        self.resblock333 = ResidualBlock2(1024, 256)
        self.resblock334 = ResidualBlock2(1024, 256)
        self.resblock335 = ResidualBlock2(1024, 256)
        self.resblock336 = ResidualBlock2(1024, 256)
        self.resblock41 = ResidualBlock2(1024, 512)
        self.resblock42 = ResidualBlock2(2048, 512)
        self.resblock43 = ResidualBlock2(2048, 512)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        '''
        Calculate features from input image x.

        Parameterts
        -----------
        x : torch.tensor
            input image.
        '''
        pr = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        b1 = self.resblock13(self.resblock12(self.resblock11(pr)))
        b2 = self.resblock28(self.resblock27(self.resblock26(self.resblock25(self.resblock24(self.resblock23(self.resblock22(self.resblock21(b1))))))))
        b3 = self.resblock36(self.resblock35(self.resblock34(self.resblock33(self.resblock32(self.resblock31(b2))))))
        b4 = self.resblock312(self.resblock311(self.resblock310(self.resblock39(self.resblock38(self.resblock37(b3))))))
        b5 = self.resblock318(self.resblock317(self.resblock316(self.resblock315(self.resblock314(self.resblock313(b4))))))
        b6 = self.resblock324(self.resblock323(self.resblock322(self.resblock321(self.resblock320(self.resblock319(b5))))))
        b7 = self.resblock330(self.resblock329(self.resblock328(self.resblock327(self.resblock326(self.resblock325(b6))))))
        b8 = self.resblock336(self.resblock335(self.resblock334(self.resblock333(self.resblock332(self.resblock331(b7))))))
        b9 = self.resblock43(self.resblock42(self.resblock41(b8)))
        b10 = self.avgpool(b9)
        return b10

    def out_features(self):
        '''
        Return number of output features, needed for decoders input channels.
        '''
        return self.features

    def name(self):
        '''
        Return name of encoder.
        '''
        return 'resnet152'
