from torch import nn


class LenetFeatures(nn.Module):
    '''
    Class used to calculate features using lenet architecture.
    http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf

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
        Create lenet encoder layers.

        Parameters
        ----------
        inc : int
            number of input channels [1, 3, 4].
        '''
        super().__init__()
        self.features = 400
        self.conv1 = nn.Conv2d(inc, 6, kernel_size=(5, 5), padding='same')
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(5, 5))
        self.tanh = nn.Tanh()

    def forward(self, x):
        '''
        Calculate features from input image x.

        Parameterts
        -----------
        x : torch.tensor
            input image.
        '''
        b1 = self.maxpool(self.tanh(self.conv1(x)))
        b2 = self.avgpool(self.maxpool(self.tanh(self.conv2(b1))))
        return b2

    def out_features(self):
        '''
        Return number of output features, needed for decoders input channels.
        '''
        return self.features

    def name(self):
        '''
        Return name of encoder.
        '''
        return 'lenet'
