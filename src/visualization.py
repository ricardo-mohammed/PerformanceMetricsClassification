"""Draw image examples and classification evaluation charts."""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve


class ClassificationVisualizer:
    """Keep reusable plotting code outside the workshop notebook."""

    def __init__(self, figsize=(7, 4)):
        """Store the standard figure size used by evaluation charts."""
        self.figsize = figsize

    def plot_image(self, pixels, title="Image"):
        """Reshape one image's 784 pixels into a 28-by-28 grayscale display."""
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(np.asarray(pixels).reshape(28, 28), cmap="binary")
        ax.set_title(title)
        ax.axis("off")
        fig.tight_layout()
        return fig

    def plot_confusion_matrix(self, matrix, title="Confusion matrix"):
        """Show counts with actual labels on rows and predictions on columns."""
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(matrix, cmap="Blues")
        names = np.array([["TN", "FP"], ["FN", "TP"]])
        for row in range(2):
            for col in range(2):
                color = "white" if matrix[row, col] > np.max(matrix) / 2 else "black"
                ax.text(col, row, f"{names[row, col]}\n{matrix[row, col]:,}",
                        ha="center", va="center", color=color)
        ax.set(xticks=[0, 1], yticks=[0, 1],
               xticklabels=["False", "True"], yticklabels=["False", "True"],
               xlabel="Predicted label", ylabel="Actual label", title=title)
        fig.tight_layout()
        return fig

    def plot_threshold_curve(self, y_true, scores, threshold):
        """Plot precision and recall against real score thresholds."""
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(thresholds, precision[:-1], label="Precision")
        ax.plot(thresholds, recall[:-1], label="Recall")
        ax.axvline(threshold, color="black", linestyle=":", label="Selected threshold")
        ax.set(xlabel="Decision score threshold", ylabel="Metric value", ylim=(0, 1.05))
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_precision_recall(self, y_true, score_sets):
        """Compare named score arrays on the same precision-recall chart."""
        fig, ax = plt.subplots(figsize=self.figsize)
        for name, scores in score_sets.items():
            precision, recall, _ = precision_recall_curve(y_true, scores)
            ax.plot(recall, precision, label=name)
        ax.axhline(np.mean(y_true), linestyle=":", color="gray", label="Class prevalence")
        ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.05))
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_roc(self, y_true, scores, threshold=None):
        """Draw ROC and optionally mark the exact selected threshold's rates."""
        fpr, tpr, _ = roc_curve(y_true, scores)
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(fpr, tpr, label="SGD ROC")
        ax.plot([0, 1], [0, 1], "k:", label="Random ranking")
        if threshold is not None:
            pred = np.asarray(scores) >= threshold
            truth = np.asarray(y_true, dtype=bool)
            point_fpr = np.sum(pred & ~truth) / np.sum(~truth)
            point_tpr = np.sum(pred & truth) / np.sum(truth)
            ax.plot(point_fpr, point_tpr, "o", label="Selected threshold")
        ax.set(xlabel="False positive rate", ylabel="True positive rate (recall)",
               xlim=(0, 1), ylim=(0, 1.05))
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_mean_comparison(self):
        """Compare F1, arithmetic mean, and their gap over precision/recall pairs."""
        values = np.linspace(0.1, 1, 10)
        precision, recall = np.meshgrid(values, values)
        f1 = 2 * precision * recall / (precision + recall)
        arithmetic = (precision + recall) / 2
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
        for ax, data, title in zip(axes, [f1, arithmetic, arithmetic - f1],
                                   ["F1 (harmonic mean)", "Arithmetic mean", "Arithmetic minus F1"]):
            im = ax.imshow(data, origin="lower", extent=(0.05, 1.05, 0.05, 1.05),
                           vmin=0, vmax=1, cmap="viridis")
            ax.set(xlabel="Precision", ylabel="Recall", title=title)
        fig.colorbar(im, ax=axes, label="Score / gap", shrink=0.8)
        return fig
