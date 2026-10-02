# Performance Metrics Classification Workshop

This repository contains a hands-on workshop about binary classification, model evaluation, and the practical meaning of performance metrics. The main notebook uses MNIST images to classify whether an image is a handwritten `5`, then applies the same reasoning to a Fashion-MNIST comparison and several real-world decision scenarios.

The completed workshop is in [notebooks/PerformanceMetricsClassification-V1.ipynb](notebooks/PerformanceMetricsClassification-V1.ipynb). It follows all 13 "To the student" sections from the instructor material and moves from classifier fundamentals to threshold selection, precision-recall analysis, ROC curves, and a Random Forest comparison.

## Project structure

- `notebooks/PerformanceMetricsClassification-V1.ipynb` - solved workshop notebook.
- `src/data_loader.py` - downloads, caches, validates, and prepares image datasets.
- `src/classifier.py` - common interface for SGD, dummy, and Random Forest classifiers.
- `src/metrics_evaluator.py` - confusion matrices, metrics, and threshold utilities.
- `src/visualization.py` - plots for images, metrics, and model comparisons.
- `outputs/` - generated comparison tables and evaluation results.

The notebook downloads MNIST and Fashion-MNIST from OpenML on first use and caches the data under `.cache/openml`. Network access is required for that first download.

## Quick start

From the repository root:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install jupyter
jupyter notebook
```

Open `notebooks/PerformanceMetricsClassification-V1.ipynb` in Jupyter or VS Code and run the cells from top to bottom.

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead. The dependency file already includes the notebook kernel and scientific Python packages; the explicit `jupyter` install ensures the notebook server command is available.

## To the student / Talking points

### To the student / Talking points 1 — Classification is a decision process

- What are we trying to solve in a classification problem?
- What is a classifier?

A classifier learns patterns from labeled examples and uses those patterns to assign a label to new, unseen data. For MNIST, each image is represented by 784 grayscale pixel features. The model learns a decision rule for separating images of the digit `5` from all other digits.

The positive class must be defined explicitly. Here, `True` means "the image is a 5" and `False` means "the image is not a 5". That definition determines how the confusion matrix and every positive-class metric should be interpreted.

### To the student / Talking points 2 — Validation must reflect how the model will be used

- Why is cross-validation important?
- How do we avoid data leakage when evaluating a classifier?

Cross-validation gives a more reliable estimate than a single train/test split, but only when each validation prediction is produced by a model that did not train on that sample. The notebook therefore uses stratified folds and out-of-fold predictions for evaluation and for threshold analysis. Reusing training predictions would leak information and make the metrics look better than they really are.

The `DummyClassifier` is a useful baseline: a sophisticated model should be compared with a simple strategy, not judged in isolation. The notebook also compares an SGD classifier with a Random Forest while keeping the validation procedure consistent.

### To the student / Talking points 3 — Accuracy can hide the errors that matter

- Why can accuracy be misleading?
- How do confusion matrix entries help us interpret classification quality?

The confusion matrix separates predictions into true negatives, false positives, false negatives, and true positives. This is more informative than accuracy when the positive class is rare. In the security-drone example, 494 correct decisions out of 500 gives 98.8% accuracy, but the system still misses 2 of 10 actual intrusions, giving 80% recall.

The lesson is to choose metrics according to the consequences of errors:

| Metric | Question it answers | Useful when |
| --- | --- | --- |
| Precision | Of the predicted positives, how many are correct? | False alarms or false positive actions are costly. |
| Recall | Of the actual positives, how many did we find? | Missing a positive case is costly. |
| F1 score | How well are precision and recall balanced? | Both types of positive-class error matter and need one summary. |

F1 is the harmonic mean of precision and recall, so it penalizes an imbalance between them. It ignores true negatives and does not encode unequal error costs; it is not automatically the right metric for every application.

### To the student / Talking points 4 — Metric choice depends on the action

- Should we prioritize precision or recall for a screening system?
- What changes when the cost of a false positive is much higher than the cost of a false negative?

Medical pre-screening and security-alert triage generally prioritize recall because missed cases can be dangerous and additional false positives can be reviewed. A system that automatically deletes suspected spam or recommends only costly investment actions may prioritize precision because false positives are expensive. The right choice depends on what happens after a prediction, not on a universal ranking of metrics.

### To the student / Talking points 5 — Thresholds expose the precision-recall tradeoff

- What happens when the decision threshold is raised?
- What happens when the threshold is lowered?

The model's score is converted into a class prediction using a decision threshold. Raising the threshold makes the classifier more conservative: it predicts fewer positives, which usually improves precision while reducing recall. Lowering it makes the classifier more permissive: recall usually rises while precision falls.

Precision-recall curves help select a threshold for a stated objective, such as achieving at least 90% precision while retaining as much recall as possible. That threshold is a development result based on out-of-fold scores, so it should be checked on new or held-out data before deployment.

ROC curves and ROC AUC summarize how well a model ranks positives above negatives across thresholds. ROC AUC is not a guarantee of accuracy, calibration, or good precision at the threshold ultimately chosen. When positives are rare, a precision-recall curve often makes the operational tradeoff clearer.

## Reusable Python modules

The notebook adds the repository root to `sys.path`, so the project classes can also be imported from scripts run at the repository root:

```python
from src import (
	ImageDataLoader,
	ClassificationModel,
	ClassificationMetrics,
	ClassificationVisualizer,
)
```

This keeps the notebook focused on the workshop narrative while the data loading, modeling, metric calculations, and visualization logic remain reusable.

## References

- LeCun, Y., Cortes, C., and Burges, C. J. C. [MNIST handwritten digit database](http://yann.lecun.com/exdb/mnist/).
- Wikipedia contributors. [Category: Classification algorithms](https://en.wikipedia.org/wiki/Category:Classification_algorithms). Wikipedia.
- Wikipedia contributors. [Statistical classification](https://en.wikipedia.org/wiki/Statistical_classification). Wikipedia.
