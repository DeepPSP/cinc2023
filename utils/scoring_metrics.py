from typing import Tuple

import numpy as np

from helper_code import is_nan


###########################################
# methods from the file evaluation_model.py
# of the official repository
###########################################


def compute_challenge_score(labels: np.ndarray, outputs: np.ndarray) -> float:
    """Compute the Challenge score.

    The Challenge score is the largest TPR such that FPR <= 0.05
    for `outcome`.

    Parameters
    ----------
    labels : np.ndarray
        The categorical ground truth labels for `outcome`,
        of shape ``(num_patients,)``.
    outputs : np.ndarray
        The probability outputs for `outcome`,
        of shape ``(num_patients, num_classes)``.

    Returns
    -------
    float
        The Challenge score.

    """
    assert len(labels) == len(outputs)
    num_patients = len(labels)

    # Use the unique output values as the thresholds for the positive and negative classes.
    thresholds = np.unique(outputs)
    thresholds = np.append(thresholds, thresholds[-1] + 1)
    thresholds = thresholds[::-1]
    num_thresholds = len(thresholds)

    idx = np.argsort(outputs)[::-1]

    # Initialize the TPs, FPs, FNs, and TNs with no positive outputs.
    tp = np.zeros(num_thresholds)
    fp = np.zeros(num_thresholds)
    fn = np.zeros(num_thresholds)
    tn = np.zeros(num_thresholds)

    tp[0] = 0
    fp[0] = 0
    fn[0] = np.sum(labels == 1)
    tn[0] = np.sum(labels == 0)

    # Update the TPs, FPs, FNs, and TNs using the values at the previous threshold.
    i = 0
    for j in range(1, num_thresholds):
        tp[j] = tp[j - 1]
        fp[j] = fp[j - 1]
        fn[j] = fn[j - 1]
        tn[j] = tn[j - 1]

        while i < num_patients and outputs[idx[i]] >= thresholds[j]:
            if labels[idx[i]]:
                tp[j] += 1
                fn[j] -= 1
            else:
                fp[j] += 1
                tn[j] -= 1
            i += 1

    # Compute the TPRs and FPRs.
    tpr = np.zeros(num_thresholds)
    fpr = np.zeros(num_thresholds)
    for j in range(num_thresholds):
        if tp[j] + fn[j] > 0:
            tpr[j] = float(tp[j]) / float(tp[j] + fn[j])
            fpr[j] = float(fp[j]) / float(fp[j] + tn[j])
        else:
            tpr[j] = float("nan")
            fpr[j] = float("nan")

    # Find the largest TPR such that FPR <= 0.05.
    max_fpr = 0.05
    max_tpr = float("nan")
    if np.any(fpr <= max_fpr):
        indices = np.where(fpr <= max_fpr)
        max_tpr = np.max(tpr[indices])

    return max_tpr


def compute_auc(labels: np.ndarray, outputs: np.ndarray) -> Tuple[float, float]:
    """Compute area under the receiver operating characteristic curve (AUROC)
    and area under the precision recall curve (AUPRC).

    Parameters
    ----------
    labels : np.ndarray
        The categorical ground truth labels for `outcome`,
        of shape ``(num_patients,)``.
    outputs : np.ndarray
        The probability outputs for `outcome`,
        of shape ``(num_patients, num_classes)``.

    Returns
    -------
    float
        The AUROC.
    float
        The AUPRC.

    """
    assert len(labels) == len(outputs)
    num_patients = len(labels)

    # Use the unique output values as the thresholds for the positive and negative classes.
    thresholds = np.unique(outputs)
    thresholds = np.append(thresholds, thresholds[-1] + 1)
    thresholds = thresholds[::-1]
    num_thresholds = len(thresholds)

    idx = np.argsort(outputs)[::-1]

    # Initialize the TPs, FPs, FNs, and TNs with no positive outputs.
    tp = np.zeros(num_thresholds)
    fp = np.zeros(num_thresholds)
    fn = np.zeros(num_thresholds)
    tn = np.zeros(num_thresholds)

    tp[0] = 0
    fp[0] = 0
    fn[0] = np.sum(labels == 1)
    tn[0] = np.sum(labels == 0)

    # Update the TPs, FPs, FNs, and TNs using the values at the previous threshold.
    i = 0
    for j in range(1, num_thresholds):
        tp[j] = tp[j - 1]
        fp[j] = fp[j - 1]
        fn[j] = fn[j - 1]
        tn[j] = tn[j - 1]

        while i < num_patients and outputs[idx[i]] >= thresholds[j]:
            if labels[idx[i]]:
                tp[j] += 1
                fn[j] -= 1
            else:
                fp[j] += 1
                tn[j] -= 1
            i += 1

    # Compute the TPRs, TNRs, and PPVs at each threshold.
    tpr = np.zeros(num_thresholds)
    tnr = np.zeros(num_thresholds)
    ppv = np.zeros(num_thresholds)
    for j in range(num_thresholds):
        if tp[j] + fn[j]:
            tpr[j] = float(tp[j]) / float(tp[j] + fn[j])
        else:
            tpr[j] = float("nan")
        if fp[j] + tn[j]:
            tnr[j] = float(tn[j]) / float(fp[j] + tn[j])
        else:
            tnr[j] = float("nan")
        if tp[j] + fp[j]:
            ppv[j] = float(tp[j]) / float(tp[j] + fp[j])
        else:
            ppv[j] = float("nan")

    # Compute AUROC as the area under a piecewise linear function
    # with TPR/sensitivity (x-axis) and TNR/specificity (y-axis) and
    # AUPRC as the area under a piecewise constant
    # with TPR/recall (x-axis) and PPV/precision (y-axis).
    auroc = 0.0
    auprc = 0.0
    for j in range(num_thresholds - 1):
        auroc += 0.5 * (tpr[j + 1] - tpr[j]) * (tnr[j + 1] + tnr[j])
        auprc += (tpr[j + 1] - tpr[j]) * ppv[j + 1]

    return auroc, auprc


def compute_one_hot_encoding(data: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Construct the one-hot encoding of data for the given classes.

    Parameters
    ----------
    data : np.ndarray
        The (categorical) data to encode,
        of shape ``(num_patients,)``.
    classes : np.ndarray
        The classes to use for the encoding,
        of shape ``(num_classes,)``.

    Returns
    -------
    np.ndarray
        The one-hot encoding of the data,
        of shape ``(num_patients, num_classes)``.

    """
    num_patients = len(data)
    num_classes = len(classes)

    one_hot_encoding = np.zeros((num_patients, num_classes), dtype=np.bool_)
    unencoded_data = list()
    for i, x in enumerate(data):
        for j, y in enumerate(classes):
            if (x == y) or (is_nan(x) and is_nan(y)):
                one_hot_encoding[i, j] = 1

    return one_hot_encoding


def compute_confusion_matrix(
    labels: np.ndarray, outputs: np.ndarray, classes: np.ndarray
) -> np.ndarray:
    """Compute the binary confusion matrix.

    The columns are the expert labels and
    the rows are the classifier labels for the given classes.

    Parameters
    ----------
    labels : np.ndarray
        The categorical ground truth labels for `outcome`,
        of shape ``(num_patients, num_classes)``.
    outputs : np.ndarray
        The binarized (one-hot encoded) classifier outputs for `outcome`,
        of shape ``(num_patients, num_classes)``.
    classes : np.ndarray
        The classes to use for the confusion matrix,
        of shape ``(num_classes,)``.

    Returns
    -------
    np.ndarray
        The confusion matrix,
        of shape ``(num_classes, num_classes)``.

    """
    assert np.shape(labels) == np.shape(outputs)

    num_patients = len(labels)
    num_classes = len(classes)

    A = np.zeros((num_classes, num_classes))
    for k in range(num_patients):
        for i in range(num_classes):
            for j in range(num_classes):
                if outputs[k, i] == 1 and labels[k, j] == 1:
                    A[i, j] += 1

    return A


def compute_one_vs_rest_confusion_matrix(
    labels: np.ndarray, outputs: np.ndarray, classes: np.ndarray
) -> np.ndarray:
    """Construct the binary one-vs-rest (OVR) confusion matrices.

    The columns are the expert labels and
    the rows are the classifier for the given classes.

    Parameters
    ----------
    labels : np.ndarray
        The binarized (one-hot encoded) ground truth labels for `outcome`,
        of shape ``(num_patients, num_classes)``.
    outputs : np.ndarray
        The binarized (one-hot encoded) classifier outputs for `outcome`,
    classes : np.ndarray
        The classes to use for the confusion matrices.

    Returns
    -------
    np.ndarray
        The one-vs-rest confusion matrices,
        of shape ``(num_classes, 2, 2)``.

    """
    assert np.shape(labels) == np.shape(outputs)

    num_patients = len(labels)
    num_classes = len(classes)

    A = np.zeros((num_classes, 2, 2))
    for i in range(num_patients):
        for j in range(num_classes):
            if labels[i, j] == 1 and outputs[i, j] == 1:  # TP
                A[j, 0, 0] += 1
            elif labels[i, j] == 0 and outputs[i, j] == 1:  # FP
                A[j, 0, 1] += 1
            elif labels[i, j] == 1 and outputs[i, j] == 0:  # FN
                A[j, 1, 0] += 1
            elif labels[i, j] == 0 and outputs[i, j] == 0:  # TN
                A[j, 1, 1] += 1

    return A


def compute_accuracy(
    labels: np.ndarray, outputs: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute the accuracy and per-class accuracy.

    Parameters
    ----------
    labels : np.ndarray
        The categorical ground truth labels for `outcome`,
        of shape ``(num_patients,)``.
    outputs : np.ndarray
        The categorical classifier outputs for `outcome`,
        of shape ``(num_patients,)``.

    Returns
    -------
    float
        The macro-averaged accuracy.
    np.ndarray
        The per-class accuracy,
        of shape ``(num_classes,)``.
    np.ndarray
        The array of classes,
        of shape ``(num_classes,)``.

    """
    # Compute the confusion matrix.
    classes = np.unique(np.concatenate((labels, outputs)))
    labels = compute_one_hot_encoding(labels, classes)
    outputs = compute_one_hot_encoding(outputs, classes)
    A = compute_confusion_matrix(labels, outputs, classes)

    # Compute accuracy.
    if np.sum(A) > 0:
        accuracy = np.trace(A) / np.sum(A)
    else:
        accuracy = float("nan")

    # Compute per-class accuracy.
    num_classes = len(classes)
    per_class_accuracy = np.zeros(num_classes)
    for i in range(num_classes):
        if np.sum(labels[:, i]) > 0:
            per_class_accuracy[i] = A[i, i] / np.sum(A[:, i])
        else:
            per_class_accuracy[i] = float("nan")

    return accuracy, per_class_accuracy, classes


def compute_f_measure(
    labels: np.ndarray, outputs: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute the F-measure and per-class F-measure.

    Parameters
    ----------
    labels : np.ndarray
        The categorical ground truth labels for `outcome`,
        of shape ``(num_patients,)``.
    outputs : np.ndarray
        The categorical classifier outputs for `outcome`,
        of shape ``(num_patients,)``.

    Returns
    -------
    float
        The macro-averaged F-measure.
    np.ndarray
        The per-class F-measure,
        of shape ``(num_classes,)``.
    np.ndarray
        The array of classes,
        of shape ``(num_classes,)``.

    """
    # Compute confusion matrix.
    classes = np.unique(np.concatenate((labels, outputs)))
    labels = compute_one_hot_encoding(labels, classes)
    outputs = compute_one_hot_encoding(outputs, classes)
    A = compute_one_vs_rest_confusion_matrix(labels, outputs, classes)

    num_classes = len(classes)
    per_class_f_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 0, 0], A[k, 0, 1], A[k, 1, 0], A[k, 1, 1]
        if 2 * tp + fp + fn > 0:
            per_class_f_measure[k] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            per_class_f_measure[k] = float("nan")

    if np.any(np.isfinite(per_class_f_measure)):
        macro_f_measure = np.nanmean(per_class_f_measure)
    else:
        macro_f_measure = float("nan")

    return macro_f_measure, per_class_f_measure, classes


def compute_mse(labels: np.ndarray, outputs: np.ndarray) -> float:
    """Compute the mean-squared error (MSE).

    Parameters
    ----------
    labels : np.ndarray
        The continuous (actually categorical) ground truth labels for `cpc`,
        of shape ``(num_patients,)``.
    outputs : np.ndarray
        The continuous (actually categorical) classifier outputs for `cpc`,
        of shape ``(num_patients,)``.

    Returns
    -------
    float
        The MSE.

    """
    assert len(labels) == len(outputs)

    labels = np.asarray(labels, dtype=np.float64)
    outputs = np.asarray(outputs, dtype=np.float64)
    mse = np.mean((labels - outputs) ** 2)

    return mse


# Compute mean-absolute error.
def compute_mae(labels: np.ndarray, outputs: np.ndarray) -> float:
    """Compute the mean-absolute error (MAE).

    Parameters
    ----------
    labels : np.ndarray
        The continuous (actually categorical) ground truth labels for `cpc`,
        of shape ``(num_patients,)``.
    outputs : np.ndarray
        The continuous (actually categorical) classifier outputs for `cpc`,
        of shape ``(num_patients,)``.

    Returns
    -------
    float
        The MAE.

    """
    assert len(labels) == len(outputs)

    labels = np.asarray(labels, dtype=np.float64)
    outputs = np.asarray(outputs, dtype=np.float64)
    mae = np.mean(np.abs(labels - outputs))

    return mae
