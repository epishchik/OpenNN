from sklearn import metrics
import torch


class sklearn_metric():
    def __init__(self, name, nc, metric_params):
        self.nc = nc
        self.metric = name
        self.metric_object = getattr(metrics, name + '_score')
        self.metric_params = metric_params

    def calc(self, preds, labels):
        shapes = preds.shape

        if len(shapes) == len(labels.shape) and shapes[1] != labels.shape[1]:
            preds = preds.argmax(dim=1).unsqueeze(1)
        elif len(shapes) > len(labels.shape):
            preds = preds.argmax(dim=1)

        if len(labels.shape) == 2:
            preds = preds.argmax(dim=1).float()
            labels = labels.argmax(dim=1).float()

        return torch.tensor(self.metric_object(labels.detach().cpu(),
                                               preds.detach().cpu(),
                                               **self.metric_params))

    def name(self):
        return self.metric
