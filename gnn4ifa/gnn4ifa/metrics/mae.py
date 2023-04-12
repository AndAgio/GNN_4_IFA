import numpy as np
from sklearn.metrics import mean_absolute_error


class MAE:
    def __init__(self):
        pass

    @staticmethod
    def compute(y_pred, y_true):
        # Compute score using sklearn
        score = mean_absolute_error(y_true=y_true.detach().numpy(),
                                    y_pred=y_pred.detach().numpy())
        return score
