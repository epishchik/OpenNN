from torch import nn


class AlexnetDecoder(nn.Module):
    '''
    Class used to decode features using alexnet architecture.
    https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf

    Methods
    -------
    forward(x)
        calculate output from input features x.
    '''
    def __init__(self, inf, nc):
        '''
        Create alexnet decoder layers.

        Parameters
        ----------
        inf : int
            number of input features.

        nc : int
            number output classes.
        '''
        super().__init__()
        self.fc1 = nn.Linear(inf, 4096, bias=True)
        self.fc2 = nn.Linear(4096, 4096, bias=True)
        self.fc3 = nn.Linear(4096, nc, bias=True)
        self.drop = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        '''
        Calculate output from input features x.

        Parameterts
        -----------
        x : torch.tensor
            input features.
        '''
        b1 = self.relu(self.fc1(self.drop(x.view(x.shape[0], -1))))
        b2 = self.relu(self.fc2(self.drop(b1)))
        b3 = self.fc3(b2)
        return b3
