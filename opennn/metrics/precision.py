from sklearn.metrics import precision_score
import torch


class precision():
    def __init__(self, nc):
        self.nc = nc

    def calc(self, preds, labels):
        shapes = preds.shape

        if len(shapes) == len(labels.shape) and shapes[1] != labels.shape[1]:
            preds = preds.argmax(dim=1).unsqueeze(1)
        elif len(shapes) > len(labels.shape):
            preds = preds.argmax(dim=1)

        if len(labels.shape) == 2:
            preds = preds.argmax(dim=1).float()
            labels = labels.argmax(dim=1).float()

        return torch.tensor(precision_score(labels.detach().cpu(), preds.detach().cpu(), average='micro'))

    def name(self):
        return 'precision'
