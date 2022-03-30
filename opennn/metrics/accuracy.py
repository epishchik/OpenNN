class accuracy():
    def calc(self, preds, labels):
        np = 1
        shapes = preds.shape
        for shape in labels.shape:
            np *= shape
        if len(shapes) == len(labels.shape) and shapes[1] != labels.shape[1]:
            preds = preds.argmax(dim=1).unsqueeze(1)
        elif len(shapes) > len(labels.shape):
            preds = preds.argmax(dim=1)
        acc = (preds == labels).sum() / np
        return acc

    def name(self):
        return 'accuracy'
