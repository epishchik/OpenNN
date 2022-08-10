from torch import nn


class LenetDecoder(nn.Module):
    '''
    Class used to decode features using lenet architecture.
    http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf

    Methods
    -------
    forward(x)
        calculate output from input features x.
    '''
    def __init__(self, inf, nc):
        '''
        Create lenet decoder layers.

        Parameters
        ----------
        inf : int
            number of input features.

        nc : int
            number output classes.
        '''
        super().__init__()
        self.fc1 = nn.Linear(inf, 120, bias=True)
        self.fc2 = nn.Linear(120, 84, bias=True)
        self.fc3 = nn.Linear(84, nc, bias=True)
        self.drop = nn.Dropout(p=0.5)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        '''
        Calculate output from input features x.

        Parameterts
        -----------
        x : torch.tensor
            input features.
        '''
        b1 = self.tanh(self.fc1(self.drop(x.view(x.shape[0], -1))))
        b2 = self.tanh(self.fc2(self.drop(b1)))
        b3 = self.softmax(self.fc3(b2))
        return b3
