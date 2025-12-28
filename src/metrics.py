import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, mean_absolute_error

class RatingMetrics:
    NAMES = ["1-star", "2-star", "3-star", "4-star", "5-star"]

    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "off_by_one_acc": float(np.mean(np.abs(y_true - y_pred) <= 1)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "report": classification_report(y_true, y_pred, target_names=RatingMetrics.NAMES, zero_division=0),
        }
