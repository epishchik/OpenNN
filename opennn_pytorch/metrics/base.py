from .accuracy import accuracy
from .precision import precision
from .recall import recall
from .f1_score import f1_score


def get_metrics(metrics, nc=None):
    '''
    Return list of metrics by names.

    Parameterts
    -----------
    metrics : list[str]
        list of metrics names.

    nc : int
        number classes.
    '''
    metrics_fn = []
    for metric in metrics:
        if metric == 'accuracy':
            acc = accuracy()
            metrics_fn.append(acc)
        elif metric == 'precision':
            prec = precision(nc)
            metrics_fn.append(prec)
        elif metric == 'recall':
            rec = recall(nc)
            metrics_fn.append(rec)
        elif metric == 'f1_score':
            f1 = f1_score(nc)
            metrics_fn.append(f1)
        else:
            raise ValueError(f'no metric {metric}')
    return metrics_fn
