
def perf_measure(y_true, y_pred):
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0

    for i in range(len(y_pred)):
        if y_true[i] == y_pred[i] == 1:
            true_pos += 1
        if y_pred[i] == 1 and y_true[i] != y_pred[i]:
            false_pos += 1
        if y_true[i] == y_pred[i] == 0:
            true_neg += 1
        if y_pred[i ]== 0 and y_true[i] != y_pred[i]:
            false_neg += 1

    return true_pos, false_pos, true_neg, false_neg
