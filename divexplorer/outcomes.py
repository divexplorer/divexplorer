import numpy as np

def get_false_positive_rate_outcome(
    y_trues, y_preds, negative_value=0
):
    """Returns boolean outcome for the false positive rate. 1 if it is a false positive, 
    0 if it is a true negative, np.nan otherwhise.
    :param y_trues: true values (e.g., df['y_true'])
    :param y_preds: predicated values.
    :param negative_value: value of the negative class.
    """
    fp = np.array(
        get_false_positives(y_trues, y_preds, negative_value=negative_value)
    ).astype(int)
    tn = np.array(
        get_true_negatives(y_trues, y_preds, negative_value=negative_value)
    ).astype(int)
    fp_outcome = np.full(fp.shape, np.nan)
    fp_outcome[fp == 1] = 1
    fp_outcome[tn == 1] = 0
    return fp_outcome


def get_false_negative_rate_outcome(
    y_trues, y_preds, positive_value=1
):
    """Returns boolean outcome for the false negative rate. 
    1 if it is a false negative, 0 if it is a true positive, np.nan otherwhise.
    :param y_trues: true values (e.g., df['y_true'])
    :param y_preds: predicated values.
    :param positive_value: value of the positive class.
    """
    fn = np.array(
        get_false_negatives(y_trues, y_preds, positive_value=positive_value)
    ).astype(int)
    tp = np.array(
        get_true_positives(y_trues, y_preds, positive_value=positive_value)
    ).astype(int)
    fn_outcome = np.full(fn.shape, np.nan)
    fn_outcome[fn == 1] = 1
    fn_outcome[tp == 1] = 0
    return fn_outcome


def get_accuracy_outcome(y_trues, y_preds, negative_value=0, positive_value=1):
    """Returns boolean outcome for the accuracy rate. 1 if it is correct, 0 if it is incorrect.
    :param y_trues: true values (e.g., df['y_true'])
    :param y_preds: predicated values.
    :param negative_value: value of the negative class.
    :param positive_value: value of the positive class.
    """
    fn = np.array(
        get_false_negatives(y_trues, y_preds, positive_value=positive_value)
    ).astype(int)
    tp = np.array(
        get_true_positives(y_trues, y_preds, positive_value=positive_value)
    ).astype(int)
    fp = np.array(
        get_false_positives(y_trues, y_preds, negative_value=negative_value)
    ).astype(int)
    tn = np.array(
        get_true_negatives(y_trues, y_preds, negative_value=negative_value)
    ).astype(int)
    acc_outcome = np.full(y_trues.shape, np.nan)
    acc_outcome[(tp == 1) | (tn == 1)] = 1
    acc_outcome[(fp == 1) | (fn == 1)] = 0
    return acc_outcome


def get_true_positives(y_trues, y_preds, positive_value=1):
    """Returns true positives. True if it is a true positive, false otherwise.
    :param y_trues: true values (e.g., df['y_true'])
    :param y_preds: predicated values
    :param positive_value: value of the positive class. 
    """
    return (y_trues == y_preds) & (y_trues == positive_value)


def get_true_negatives(y_trues, y_preds, negative_value=1):
    """Returns true negatives. True if it is a true negative, false otherwise.
    :param y_trues: true values (e.g., df['y_true'])
    :param y_preds: predicated values.
    :param negative_value: value of the negative class.
    """
    return (y_trues == y_preds) & (y_trues == negative_value)


def get_false_positives(y_trues, y_preds, negative_value=1):
    """Returns false positives. True if it is a false positive, false otherwise
    :param y_trues: true values (e.g., df['y_true'])
    :param y_preds: predicated values.
    :param negative_value: value of the negative class.
    """
    return (y_trues != y_preds) & (y_trues == negative_value)


def get_false_negatives(y_trues, y_preds, positive_value=1):
    """Returns false negatives. True if it is a false negative, false otherwise.
    :param y_trues: true values (e.g., df['y_true'])
    :param y_preds: predicated values.
    :param positive_value: value of the positive class.
    """
    return (y_trues != y_preds) & (y_trues == positive_value)


