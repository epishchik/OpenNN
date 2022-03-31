from torch import nn


class LenetFeatures(nn.Module):
    def __init__(self, inc):
        super().__init__()
        self.features = 400
        self.conv1 = nn.Conv2d(inc, 6, kernel_size=(5, 5), padding='same')
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.maxpol = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.avgpol = nn.AdaptiveAvgPool2d(output_size=(5, 5))
        self.act = nn.Tanh()

    def forward(self, x):
        b1 = self.maxpol(self.act(self.conv1(x)))
        b2 = self.avgpol(self.maxpol(self.act(self.conv2(b1))))
        return b2

    def out_features(self):
        return self.features
