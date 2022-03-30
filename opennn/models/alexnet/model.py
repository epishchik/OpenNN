from torch import nn
from .blocks import Features, Avgpool, Classifier


class AlexNet(nn.Module):
    def __init__(self, inc, nc):
        super().__init__()
        self.feat = Features(inc)
        self.avg = Avgpool()
        self.cls = Classifier(nc)

    def forward(self, x):
        return self.cls(self.avg(self.feat(x)))
