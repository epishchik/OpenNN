from torch import nn


class AlexNet(nn.Module):
    def __init__(self, inc):
        super().__init__()
        self.features = 256
        self.conv1 = nn.Conv2d(inc,
                               64,
                               kernel_size=(11, 11),
                               stride=(4, 4),
                               padding=(2, 2))
        self.conv2 = nn.Conv2d(64,
                               192,
                               kernel_size=(5, 5),
                               stride=(1, 1),
                               padding=(2, 2))
        self.conv3 = nn.Conv2d(192,
                               384,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1))
        self.conv4 = nn.Conv2d(384,
                               256,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1))
        self.conv5 = nn.Conv2d(256,
                               256,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=3,
                                    stride=2,
                                    padding=0,
                                    dilation=1,
                                    ceil_mode=False)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        b1 = self.maxpool(self.relu(self.conv1(x)))
        b2 = self.maxpool(self.relu(self.conv2(b1)))
        b3 = self.relu(self.conv3(b2))
        b4 = self.relu(self.conv4(b3))
        b5 = self.avgpool(self.relu(self.conv5(b4)))
        return b5

    def out_features(self):
        return self.features
