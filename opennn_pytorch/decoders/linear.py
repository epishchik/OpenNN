from torch import nn


class LinearDecoder(nn.Module):
    '''
    Class used to decode features using one linear layer.

    Methods
    -------
    forward(x)
        calculate output from input features x.
    '''
    def __init__(self, inf, nc):
        '''
        Create one linear decoder layer.

        Parameters
        ----------
        inf : int
            number of input features.

        nc : int
            number output classes.
        '''
        super().__init__()
        self.fc = nn.Linear(inf, nc)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        '''
        Calculate output from input features x.

        Parameterts
        -----------
        x : torch.tensor
            input features.
        '''
        return self.softmax(self.fc(x.view(x.shape[0], -1)))
