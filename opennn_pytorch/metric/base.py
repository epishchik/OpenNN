from .sklearn_metric import sklearn_metric


def get_metric(metrics, nc=None):
    metrics_fn = []

    for m in metrics:
        m_params = {} if m == 'accuracy' else {'average': 'weighted'}
        metrics_fn.append(sklearn_metric(m, nc, m_params))

    return metrics_fn
