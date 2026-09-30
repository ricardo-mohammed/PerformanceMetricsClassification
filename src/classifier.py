"""Train classifiers and obtain validation predictions without data leakage."""

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict


class ClassificationModel:
    """Provide a common interface for SGD, Dummy, and Random Forest models."""

    def __init__(self, model_type="sgd", random_state=42):
        """Create the chosen estimator with reproducible training settings."""
        self.model_type = model_type
        if model_type == "sgd":
            self.estimator = SGDClassifier(random_state=random_state)
        elif model_type == "dummy":
            self.estimator = DummyClassifier(strategy="most_frequent")
        elif model_type == "random_forest":
            self.estimator = RandomForestClassifier(
                n_estimators=100, random_state=random_state, n_jobs=2,
            )
        else:
            raise ValueError("Choose sgd, dummy, or random_forest.")

    def train(self, X, y):
        """Fit the stored estimator using the supplied training data."""
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        """Return class labels from the fitted estimator."""
        return self.estimator.predict(X)

    def decision_scores(self, X):
        """Return signed SGD decision scores; these are not probabilities."""
        if not hasattr(self.estimator, "decision_function"):
            raise ValueError("This model has no decision_function.")
        return self.estimator.decision_function(X)

    def positive_probabilities(self, X):
        """Return the probability column for the True class of a fitted model."""
        if not hasattr(self.estimator, "predict_proba"):
            raise ValueError("This model has no predict_proba.")
        positive_index = list(self.estimator.classes_).index(True)
        return self.estimator.predict_proba(X)[:, positive_index]

    @staticmethod
    def make_folds(cv=3):
        """Create unshuffled stratified folds, matching the instructor's cv=3."""
        return StratifiedKFold(n_splits=cv, shuffle=False)

    def evaluate_cv(self, X, y, cv=3):
        """Return fold training time, scoring time, and validation accuracy."""
        return cross_validate(
            self.estimator, X, y, cv=self.make_folds(cv),
            scoring="accuracy", n_jobs=1, error_score="raise",
        )

    def out_of_fold(self, X, y, cv=3, method="predict"):
        """Predict each training sample using a clone that did not train on it."""
        return cross_val_predict(
            self.estimator, X, y, cv=self.make_folds(cv),
            method=method, n_jobs=1,
        )

    @staticmethod
    def threshold_predictions(scores, threshold=0.0):
        """Convert numeric scores into Boolean predictions at a threshold."""
        return np.asarray(scores) >= threshold
