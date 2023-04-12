import numpy as np
from sklearn.metrics import mean_squared_error


class MSE:
    def __init__(self):
        pass

    @staticmethod
    def compute(y_pred, y_true):
        # Compute score using sklearn
        score = mean_squared_error(y_true=y_true.detach().numpy(),
                                   y_pred=y_pred.detach().numpy())
        return score
