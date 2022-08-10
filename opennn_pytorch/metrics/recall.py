from sklearn.metrics import recall_score
import torch


class recall():
    '''
    Class used to calculate recall metric.

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
        Calculate recall metric.

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

        return torch.tensor(recall_score(labels.detach().cpu(), preds.detach().cpu(), average='micro'))

    def name(self):
        '''
        Return metric name.
        '''
        return 'recall'
