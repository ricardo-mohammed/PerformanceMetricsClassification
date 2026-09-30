"""Calculate binary metrics and select thresholds from validation scores."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)


class ClassificationMetrics:
    """Evaluate predictions with a fixed negative/positive label order."""

    def __init__(self, y_true, y_pred):
        """Store aligned Boolean truth and predictions for one evaluation."""
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        if self.y_true.ndim != 1 or self.y_true.shape != self.y_pred.shape:
            raise ValueError("Truth and predictions must be matching 1D arrays.")
        if not len(self.y_true):
            raise ValueError("At least one observation is required.")
        for values in (self.y_true, self.y_pred):
            if not np.isin(values, [False, True]).all():
                raise ValueError("Use Boolean binary targets and predictions.")

    def confusion_matrix(self):
        """Return [[TN, FP], [FN, TP]], even if only one class was predicted."""
        return confusion_matrix(self.y_true, self.y_pred, labels=[False, True])

    def counts(self):
        """Extract the four confusion-matrix counts as named integers."""
        tn, fp, fn, tp = self.confusion_matrix().ravel()
        return dict(TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))

    @staticmethod
    def manual_precision(tp, fp):
        """Calculate TP / (TP + FP), returning zero for no positive predictions."""
        return tp / (tp + fp) if tp + fp else 0.0

    @staticmethod
    def manual_recall(tp, fn):
        """Calculate TP / (TP + FN), returning zero for no actual positives."""
        return tp / (tp + fn) if tp + fn else 0.0

    @staticmethod
    def manual_f1(tp, fp, fn):
        """Calculate 2*TP / (2*TP + FP + FN) directly from error counts."""
        denominator = 2 * tp + fp + fn
        return 2 * tp / denominator if denominator else 0.0

    def summary(self, scores=None):
        """Return accuracy, precision, recall, F1, and optional ranking metrics."""
        result = {
            "accuracy": accuracy_score(self.y_true, self.y_pred),
            "precision": precision_score(self.y_true, self.y_pred, zero_division=0),
            "recall": recall_score(self.y_true, self.y_pred, zero_division=0),
            "f1": f1_score(self.y_true, self.y_pred, zero_division=0),
        }
        if scores is not None:
            result["roc_auc"] = roc_auc_score(self.y_true, scores)
            result["average_precision"] = average_precision_score(self.y_true, scores)
        return result

    @staticmethod
    def threshold_for_precision(y_true, scores, target_precision=0.90):
        """Choose the highest-recall valid threshold meeting validation precision."""
        if not 0 < target_precision <= 1:
            raise ValueError("target_precision must be in (0, 1].")
        precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
        # Exclude the final artificial (precision=1, recall=0) endpoint.
        candidates = np.flatnonzero(precisions[:-1] >= target_precision)
        if not len(candidates):
            raise ValueError("No real threshold meets the requested precision.")
        best = candidates[np.argmax(recalls[candidates])]
        return float(thresholds[best])
