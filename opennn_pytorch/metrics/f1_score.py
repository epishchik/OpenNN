from sklearn.metrics import f1_score as sklearn_f1
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

        f1 = sklearn_f1(labels.detach().cpu(),
                        preds.detach().cpu(),
                        average='weighted')
        return torch.tensor(f1)

    def name(self):
        '''
        Return metric name.
        '''
        return 'f1_score'
