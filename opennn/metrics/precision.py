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
        # TODO сделать свой precision
        # tp = 0
        # tpfp = 0
        # for i in range(self.nc):
        #     preds_i = (preds.cpu() == i).float()
        #     labels_i = (labels.cpu() == i).float()
        #     tp += ((preds_i == labels_i).float() * preds_i).sum()
        #     tpfp += tp + ((preds_i != labels_i).float() * preds_i).sum()
        # return tp / tpfp

    def name(self):
        return 'precision'
