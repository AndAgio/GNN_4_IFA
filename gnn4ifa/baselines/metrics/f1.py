from sklearn.metrics import f1_score


class F1Score:
    def __init__(self, data_mode='avg'):
        self.data_mode = data_mode

    def compute(self, y_true, y_pred):
        # Compute score and return it
        if self.data_mode in ['avg', 'cat']:
            score = f1_score(y_true=y_true,
                                   y_pred=y_pred)
            return score
        else:
            raise ValueError('Data mode {} not available yet!'.format(self.data_mode))
