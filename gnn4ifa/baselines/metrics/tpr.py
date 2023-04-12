from .utils import perf_measure


class Tpr:
    def __init__(self, data_mode='avg'):
        self.data_mode = data_mode

    def compute(self, y_true, y_pred):
        # Compute score and return it
        if self.data_mode in ['avg', 'cat']:
            true_pos, false_pos, true_neg, false_neg = perf_measure(y_true=y_true,
                                                                    y_pred=y_pred)
            score = true_pos / (true_pos + false_neg)
            return score
        else:
            raise ValueError('Data mode {} not available yet!'.format(self.data_mode))
