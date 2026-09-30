"""Download image datasets and prepare reproducible binary classification data."""

from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml


class ImageDataLoader:
    """Reuse the same loading and preprocessing workflow for both datasets."""

    DATASETS = {"mnist": 554, "fashion_mnist": 40996}
    FASHION_LABELS = {
        "0": "T-shirt/top", "1": "Trouser", "2": "Pullover", "3": "Dress",
        "4": "Coat", "5": "Sandal", "6": "Shirt", "7": "Sneaker",
        "8": "Bag", "9": "Ankle boot",
    }

    def __init__(self, dataset="mnist", cache_dir=".cache/openml"):
        """Select a dataset and a local cache for repeated downloads."""
        if dataset not in self.DATASETS:
            raise ValueError(f"Choose one of {tuple(self.DATASETS)}.")
        self.dataset = dataset
        self.cache_dir = Path(cache_dir)

    def load_data(self):
        """Fetch 70,000 images and return pixel features and string labels."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        dataset = fetch_openml(
            data_id=self.DATASETS[self.dataset], as_frame=False,
            parser="auto", data_home=str(self.cache_dir),
        )
        # Keep the instructor's raw 0-255 pixel scale and numeric precision.
        X = np.asarray(dataset.data, dtype=np.float64)
        y = np.asarray(dataset.target).astype(str)
        if X.shape != (70000, 784) or y.shape != (70000,):
            raise ValueError("Expected 70,000 images with 784 pixels each.")
        if not np.isfinite(X).all():
            raise ValueError("The dataset contains missing or invalid pixels.")
        return X, y

    @staticmethod
    def split_data(X, y, train_size=60000):
        """Preserve the dataset order and reserve the final images for testing."""
        if len(X) != len(y) or not 0 < train_size < len(y):
            raise ValueError("Features, labels, and train_size are inconsistent.")
        return X[:train_size], X[train_size:], y[:train_size], y[train_size:]

    @staticmethod
    def prepare_binary_target(y, positive_label="5"):
        """Mark the selected class True and every other class False."""
        labels = np.asarray(y).astype(str)
        if str(positive_label) not in labels:
            raise ValueError("The requested positive class is not present.")
        return labels == str(positive_label)
