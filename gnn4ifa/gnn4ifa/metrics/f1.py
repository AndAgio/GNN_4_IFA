import torch
from sklearn.metrics import f1_score


class F1Score():
    def __init__(self):
        pass

    @staticmethod
    def compute(y_true, y_pred):
        # Check if tensors have same shape
        if y_true.shape != y_pred.shape:
            y_pred = torch.argmax(y_pred, dim=1)
        # Detach tensors and convert the into numpy
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        # print('y_true: {}'.format(y_true))
        # print('y_pred: {}'.format(y_pred))
        # print('y_true.shape: {}'.format(y_true.shape))
        # print('y_pred.shape: {}'.format(y_pred.shape))
        # Compute AUC score and return it
        score = f1_score(y_true=y_true,
                         y_pred=y_pred)
        return score
