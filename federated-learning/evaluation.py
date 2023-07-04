import numpy as np
from collections import defaultdict
from sklearn.metrics import recall_score, f1_score, roc_auc_score, accuracy_score, balanced_accuracy_score
from copy import deepcopy

MIN_FLOAT = 1e-12

def get_recall_score(y_true, y_pred, average):
    # TP / (TP + FN)
    TP = MIN_FLOAT
    FN = 0
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            TP += 1
        elif true == 1 and pred == 0:
            FN += 1

    return TP/(TP+FN)


def get_precision_score(y_true, y_pred, average):
    # TP / (TP + FP)
    TP = MIN_FLOAT
    FP = 0
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            TP += 1
        elif true == 0 and pred == 1:
            FP += 1

    return TP/(TP+FP)

def get_ba_score(y_true, y_pred, average):
    sensitivity = get_recall_score(y_true=y_true, y_pred=y_pred, average='binary')
    specificity = get_recall_score(y_true=~y_true, y_pred=~y_pred, average='binary')
    return 0.5 * (sensitivity + specificity)

def get_f1_score(y_true, y_pred, average):
    # 2 * precision * recall / (precision + recall)
    rec = get_recall_score(y_true, y_pred, average)
    prec = get_precision_score(y_true, y_pred, average)
    return 2 * prec * rec / (prec + rec)


def get_roc_auc_score(y_true, y_score, threshold):
    i_true = deepcopy(y_true)
    i_score = deepcopy(y_score)

    existing_classes = np.unique(list(i_true))
    if len(existing_classes) == 1:
        missing_class = 1 - existing_classes[0]
        i_true = np.concatenate([i_true, [missing_class]]).astype(bool)
        i_score = np.concatenate([i_score, [threshold]])

    return roc_auc_score(y_true=i_true, y_score=i_score)


def calculate_MLC_metrics(y_true, y_pred, threshold):
    # sensitivity/recall, TPR=TP/(TP+FN)
    # specificity, TNR=TN/(TN+FP)
    # precision, precision=TP/(TP+FP)
    # balanced accuracy, BA=(TPR+TNR)/2

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    true = y_true == 1
    pred = y_pred > np.array(threshold)

    class_results = dict({
        'sensitivity': [],
        'specificity': [],
        'BA': [],
        'F1': [],
        'AUC': [],
        'ACC': []
    })

    for i in range(len(y_true[0])):
        i_true = true[:, i][valid_indices]
        i_pred = pred[:, i][valid_indices]
        i_score = y_pred[:, i][valid_indices]

        class_results['sensitivity'].append(get_recall_score(y_true=i_true, y_pred=i_pred, average='binary'))
        class_results['specificity'].append(get_recall_score(y_true=~i_true, y_pred=~i_pred, average='binary'))
        class_results['F1'].append(get_f1_score(y_true=i_true, y_pred=i_pred, average='binary'))
        class_results['AUC'].append(get_roc_auc_score(y_true=i_true, y_score=i_score, threshold=threshold[i]))
        class_results['ACC'].append(accuracy_score(y_true=i_true, y_pred=i_pred))

        for metric in class_results:
            class_results[metric].append(np.nan)

    class_results['sensitivity'] = np.array(class_results['sensitivity'])
    class_results['specificity'] = np.array(class_results['specificity'])
    class_results['BA'] = 0.5 * (class_results['sensitivity'] + class_results['specificity'])

    all_results = {
        'sensitivity': np.nanmean(class_results['sensitivity']),
        'specificity': np.nanmean(class_results['specificity']),
        'BA': np.nanmean(class_results['BA']),
        'AUC': np.nanmean(class_results['AUC']),
        'ACC': np.nanmean(class_results['ACC']),
        'F1': np.nanmean(class_results['F1'])
    }

    return all_results

def calculate_SLC_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    true = y_true == 1

    if len(y_true.shape) > 1:
        y_true_bool = y_true.argmax(-1)
    else:
        y_true_bool = y_true

    if len(y_pred.shape) > 1:
        y_pred_bool = y_pred.argmax(-1)
    else:
        y_pred_bool = y_pred

    class_results = dict({'AUC': []})
    for i in range(len(y_true[0])):
        i_true = true[:, i]
        i_score = y_pred[:, i]
        class_results['AUC'].append(get_roc_auc_score(y_true=i_true, y_score=i_score, threshold=0.5))

    all_results = {
        'F1': f1_score(y_true_bool, y_pred_bool, average='macro'),
        'BA': balanced_accuracy_score(y_true_bool, y_pred_bool),
        'AUC': np.nanmean(class_results['AUC']),
        'ACC': accuracy_score(y_true_bool, y_pred_bool)
    }

    return all_results

def display_results(results, metrics=['BA', 'F1', 'ACC'], logger=None):
    if logger is not None:
        logger.critical('{0:>10}'.format("Label") + ' '.join(['%10s'] * len(metrics)) % tuple([m for m in metrics]))
        logger.critical('{0:>10}'.format("AVG") + ' '.join(['%10.4f'] * len(metrics)) % tuple([results[m] for m in metrics]))
    else:
        print('{0:>20}'.format("Label") + ' '.join(['%10s'] * len(metrics)) % tuple([m for m in metrics]))
        print('{0:>20}'.format("AVG") + ' '.join(['%10.4f'] * len(metrics)) % tuple([results[m] for m in metrics]))

    return [results[m] for m in metrics]