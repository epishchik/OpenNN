from torch import nn


class AlexNet(nn.Module):
    def __init__(self, inf, nc):
        super().__init__()
        self.fc1 = nn.Linear(inf, 4096, bias=True)
        self.fc2 = nn.Linear(4096, 4096, bias=True)
        self.fc3 = nn.Linear(4096, nc, bias=True)
        self.drop = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        b1 = self.relu(self.fc1(self.drop(x.view(x.shape[0], -1))))
        b2 = self.relu(self.fc2(self.drop(b1)))
        b3 = self.fc3(b2)
        return b3


class LeNet(nn.Module):
    def __init__(self, inf, nc):
        super().__init__()
        self.fc1 = nn.Linear(inf, 120, bias=True)
        self.fc2 = nn.Linear(120, 84, bias=True)
        self.fc3 = nn.Linear(84, nc, bias=True)
        self.drop = nn.Dropout(p=0.5)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b1 = self.tanh(self.fc1(self.drop(x.view(x.shape[0], -1))))
        b2 = self.tanh(self.fc2(self.drop(b1)))
        b3 = self.softmax(self.fc3(b2))
        return b3


class Linear(nn.Module):
    def __init__(self, inf, nc):
        super().__init__()
        self.fc = nn.Linear(inf, nc)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.fc(x.view(x.shape[0], -1)))
