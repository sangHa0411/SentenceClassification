import numpy as np
from sklearn.metrics import (accuracy_score, 
    precision_recall_curve,
    f1_score,
    auc
)

def auprc_score(probs, labels):
    labels = np.eye(3)[labels]
    score = np.zeros((3,))
    for c in range(3):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = precision_recall_curve(targets_c, preds_c)
        score[c] = auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    label_indices = list(range(3))
    f1 = f1_score(labels, preds, average="micro", labels=label_indices) * 100.0
    auprc = auprc_score(probs, labels)
    acc = accuracy_score(labels, preds)
    return {
        'micro f1 score': f1,
        'auprc' : auprc,
        'accuracy': acc,
    }
