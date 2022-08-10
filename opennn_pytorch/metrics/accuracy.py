class accuracy():
    '''
    Class used to calculate accuracy metric.

    Methods
    -------
    calc(preds, labels)
        calculate metric.

    name()
        return metric name.
    '''
    def calc(self, preds, labels):
        '''
        Calculate accuracy metric.

        Parameterts
        -----------
        preds : torch.tensor
            model predictions.

        labels : torch.tensor
            ground-truth labels.
        '''
        np = 1
        shapes = preds.shape

        if len(shapes) == len(labels.shape) and shapes[1] != labels.shape[1]:
            preds = preds.argmax(dim=1).unsqueeze(1)
        elif len(shapes) > len(labels.shape):
            preds = preds.argmax(dim=1)
        if len(labels.shape) == 2:
            preds = preds.argmax(dim=1).float()
            labels = labels.argmax(dim=1).float()

        for shape in labels.shape:
            np *= shape

        acc = (preds == labels).sum() / np
        return acc

    def name(self):
        '''
        Return metric name.
        '''
        return 'accuracy'
