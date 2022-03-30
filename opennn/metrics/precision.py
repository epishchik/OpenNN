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

        return torch.tensor(precision_score(labels.cpu(), preds.cpu(), average='micro'))

    def name(self):
        return 'precision'
