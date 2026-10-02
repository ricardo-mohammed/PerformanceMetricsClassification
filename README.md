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
