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
        self.bn = nn.BatchNorm2d(outc, eps=0.001)
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
        self.features = 12800
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
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(5, 5))

    def forward(self, x):
        '''
        Calculate features from input image x.

        Parameterts
        -----------
        x : torch.tensor
            input image.
        '''
        b1 = self.maxpool2(self.conv2(self.maxpool1(self.conv1(x))))
        b2 = self.conv5(self.maxpool3(self.conv4(self.conv3(b1))))
        b3 = self.conv8(self.conv7(self.maxpool4(self.conv6(b2))))
        avg = self.avgpool(b3)
        return avg

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

# TODO добавить vgg-16 и vgg-19
