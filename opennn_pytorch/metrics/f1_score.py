from sklearn.metrics import recall_score, precision_score
import torch


class f1_score():
    '''
    Class used to calculate f1 metric.

    Attributes
    ----------
    nc : int
        classes number.

    Methods
    -------
    calc(preds, labels)
        calculate metric.

    name()
        return metric name.
    '''
    def __init__(self, nc):
        '''
        Parameters
        ----------
        nc : int
            classes number.
        '''
        self.nc = nc

    def calc(self, preds, labels):
        '''
        Calculate f1 metric.

        Parameterts
        -----------
        preds : torch.tensor
            model predictions.

        labels : torch.tensor
            ground-truth labels.
        '''
        shapes = preds.shape

        if len(shapes) == len(labels.shape) and shapes[1] != labels.shape[1]:
            preds = preds.argmax(dim=1).unsqueeze(1)
        elif len(shapes) > len(labels.shape):
            preds = preds.argmax(dim=1)

        if len(labels.shape) == 2:
            preds = preds.argmax(dim=1).float()
            labels = labels.argmax(dim=1).float()

        pr = precision_score(labels.detach().cpu(), preds.detach().cpu(), average='micro')
        re = recall_score(labels.detach().cpu(), preds.detach().cpu(), average='micro')
        f1 = 2 * re * pr / (re + pr + 1e-7)
        return torch.tensor(f1)

    def name(self):
        '''
        Return metric name.
        '''
        return 'f1_score'
