from torch import nn


class ConvBlock(nn.Module):
    '''
    Conv block class.

    Methods
    -------
    forward(x)
        calculate features from input image x.
    '''
    def __init__(self, inc, outc, **kwargs):
        '''
        Create convolution block layers.

        Parameters
        ----------
        inc : int
            number of input channels.

        outc : int
            number of output channels.
        '''
        super().__init__()
        self.conv = nn.Conv2d(inc, outc, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(outc)
        self.relu = nn.ReLU()

    def forward(self, x):
        '''
        Calculate features from input image x.

        Parameterts
        -----------
        x : torch.tensor
            input image.
        '''
        return self.relu(self.bn(self.conv(x)))


class VGG11Features(nn.Module):
    '''
    Class used to calculate features using vgg-11 architecture.
    https://arxiv.org/pdf/1409.1556.pdf

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
        Create vgg-11 encoder layers.

        Parameters
        ----------
        inc : int
            number of input channels [1, 3, 4].
        '''
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
        '''
        Calculate features from input image x.

        Parameterts
        -----------
        x : torch.tensor
            input image.
        '''
        b1 = self.maxpool1(self.conv1(x))
        b2 = self.maxpool2(self.conv2(b1))
        b3 = self.maxpool3(self.conv4(self.conv3(b2)))
        b4 = self.maxpool4(self.conv6(self.conv5(b3)))
        b5 = self.avgpool(self.conv8(self.conv7(b4)))
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
        return 'vgg11'


class VGG16Features(nn.Module):
    '''
    Class used to calculate features using vgg-16 architecture.
    https://arxiv.org/pdf/1409.1556.pdf

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
        Create vgg-16 encoder layers.

        Parameters
        ----------
        inc : int
            number of input channels [1, 3, 4].
        '''
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
        '''
        Calculate features from input image x.

        Parameterts
        -----------
        x : torch.tensor
            input image.
        '''
        b1 = self.maxpool1(self.conv2(self.conv1(x)))
        b2 = self.maxpool2(self.conv4(self.conv3(b1)))
        b3 = self.maxpool3(self.conv7(self.conv6(self.conv5(b2))))
        b4 = self.maxpool4(self.conv10(self.conv9(self.conv8(b3))))
        b5 = self.avgpool(self.conv13(self.conv12(self.conv11(b4))))
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
        return 'vgg16'


class VGG19Features(nn.Module):
    '''
    Class used to calculate features using vgg-19 architecture.
    https://arxiv.org/pdf/1409.1556.pdf

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
        Create vgg-19 encoder layers.

        Parameters
        ----------
        inc : int
            number of input channels [1, 3, 4].
        '''
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
        '''
        Calculate features from input image x.

        Parameterts
        -----------
        x : torch.tensor
            input image.
        '''
        b1 = self.maxpool1(self.conv2(self.conv1(x)))
        b2 = self.maxpool2(self.conv4(self.conv3(b1)))
        b3 = self.maxpool3(self.conv8(self.conv7(self.conv6(self.conv5(b2)))))
        b4 = self.maxpool4(self.conv12(self.conv11(self.conv10(self.conv9(b3)))))
        b5 = self.avgpool(self.conv16(self.conv15(self.conv14(self.conv13(b4)))))
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
        return 'vgg19'
